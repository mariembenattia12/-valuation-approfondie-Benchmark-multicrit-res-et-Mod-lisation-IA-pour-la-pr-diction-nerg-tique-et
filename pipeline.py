"""Load, clean, and prepare SINERT energy-meter exports for forecasting."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

import numpy as np
import pandas as pd

INTERVAL = pd.Timedelta(minutes=15)
STEPS_PER_DAY = 96
DEFAULT_WINDOW_STEPS = 672
PHYSICAL_COLUMNS = (
    "delta_ea_kwh",
    "delta_er_kvarh",
    "pmax_kw",
    "smax_kva",
)
CALENDAR_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_nonworking_day",
)
FEATURE_COLUMNS = PHYSICAL_COLUMNS + CALENDAR_COLUMNS
TARGET_COLUMNS = PHYSICAL_COLUMNS
RAW_COLUMNS = {
    "cumulative_active_kwh": ("kwh (kwh)", "[abs]"),
    "pmax_kw": ("kw dmd (kw)", "[max]"),
    "cumulative_reactive_kvarh": ("kvarh (kvarh)", "[abs]"),
    "smax_kva": ("kva dmd (kva)", "[max]"),
}


ExportSource = Path | tuple[ZipFile, str]


def _source_name(source: ExportSource) -> str:
    return source[1] if isinstance(source, tuple) else str(source)


def _source_suffix(source: ExportSource) -> str:
    return Path(_source_name(source)).suffix.casefold()


def _source_data(source: ExportSource) -> Path | io.BytesIO:
    if isinstance(source, tuple):
        return io.BytesIO(source[0].read(source[1]))
    return source


def _missing_quality(series: pd.Series, *, max_runs: int = 10) -> dict:
    missing = series.isna().to_numpy()
    positions = np.flatnonzero(missing)
    if len(positions):
        runs = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        longest_runs = [
            {
                "start": series.index[run[0]].isoformat(),
                "end": series.index[run[-1]].isoformat(),
                "steps": int(len(run)),
                "duration_hours": round(len(run) * INTERVAL.total_seconds() / 3600, 2),
            }
            for run in sorted(runs, key=len, reverse=True)[:max_runs]
        ]
    else:
        longest_runs = []

    valid_index = series.index[~missing]
    return {
        "missing_values": int(missing.sum()),
        "missing_percent": round(100 * float(missing.mean()), 4) if len(series) else 0.0,
        "first_valid_timestamp": valid_index[0].isoformat() if len(valid_index) else None,
        "last_valid_timestamp": valid_index[-1].isoformat() if len(valid_index) else None,
        "longest_missing_runs": longest_runs,
    }


def _column_map(source: ExportSource, meter: str) -> dict[str, str]:
    name = _source_name(source)
    if _source_suffix(source) == ".csv":
        headers = pd.read_csv(
            _source_data(source), sep=";", nrows=0, encoding="utf-8-sig"
        ).columns
    elif _source_suffix(source) == ".xlsx":
        headers = pd.read_excel(_source_data(source), sheet_name=0, nrows=0).columns
    else:
        raise ValueError(f"Unsupported export format: {name}")

    normalized = {str(name): " ".join(str(name).strip().split()).casefold() for name in headers}
    date_column = next((name for name, value in normalized.items() if value == "date"), None)
    time_column = next((name for name, value in normalized.items() if value == "time"), None)
    if date_column is None or time_column is None:
        raise ValueError(f"{name}: expected Date and Time columns")

    prefix = meter.strip().casefold()
    result = {"date": date_column, "time": time_column}
    for canonical, tokens in RAW_COLUMNS.items():
        matches = [
            name
            for name, value in normalized.items()
            if value.startswith(prefix) and all(token in value for token in tokens)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{name}: expected exactly one {canonical} column for meter "
                f"{meter!r}, found {len(matches)}"
            )
        result[canonical] = matches[0]
    return result


def _parse_timestamps(date_values: pd.Series, time_values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_values, format="mixed", dayfirst=True, errors="coerce")
    dates = dates.dt.normalize()
    times = time_values.astype("string").str.strip()
    end_of_day = times.str.fullmatch(r"24:00(?::00(?:\.0+)?)?", na=False)
    times = times.mask(end_of_day, "00:00:00")
    elapsed = pd.to_timedelta(times, errors="coerce")
    timestamps = dates + elapsed
    return timestamps + pd.to_timedelta(end_of_day.astype(int), unit="D")


def _read_export(source: ExportSource, meter: str) -> tuple[pd.DataFrame, int]:
    columns = _column_map(source, meter)
    selected = list(columns.values())
    if _source_suffix(source) == ".csv":
        frame = pd.read_csv(
            _source_data(source),
            sep=";",
            usecols=selected,
            dtype="string",
            encoding="utf-8-sig",
            na_values=["---", ""],
        )
    else:
        frame = pd.read_excel(
            _source_data(source), sheet_name=0, usecols=selected, dtype=str
        )

    timestamps = _parse_timestamps(
        frame[columns["date"]],
        frame[columns["time"]],
    )
    invalid_timestamps = int(timestamps.isna().sum())
    result = pd.DataFrame({"timestamp": timestamps})
    for canonical in RAW_COLUMNS:
        values = frame[columns[canonical]].astype("string").str.strip()
        values = values.str.replace(",", ".", regex=False)
        result[canonical] = pd.to_numeric(values, errors="coerce")
    return result, invalid_timestamps


def load_exports(input_path: Path, meter: str = "EM_VIRTUAL") -> tuple[pd.DataFrame, dict]:
    """Read CSV/XLSX exports from a folder, a single export, or a ZIP archive."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    def load_sources(sources: list[ExportSource]) -> tuple[pd.DataFrame, dict]:
        if not sources:
            raise FileNotFoundError(f"No CSV or XLSX exports found under {input_path}")
        frames = []
        invalid_timestamps = 0
        raw_missing_counts = dict.fromkeys(RAW_COLUMNS, 0)
        for source in sources:
            frame, invalid_count = _read_export(source, meter)
            invalid_timestamps += invalid_count
            for column in RAW_COLUMNS:
                raw_missing_counts[column] += int(frame[column].isna().sum())
            frames.append(frame)

        records = pd.concat(frames, ignore_index=True)
        raw_row_count = int(len(records))
        records = records.dropna(subset=["timestamp"]).sort_values("timestamp")
        duplicate_count = int(records["timestamp"].duplicated(keep="last").sum())
        records = records.drop_duplicates("timestamp", keep="last")
        records = records.set_index("timestamp").sort_index()
        return records, {
            "input_path": str(input_path),
            "meter": meter,
            "files_read": len(sources),
            "raw_rows_read": raw_row_count,
            "invalid_timestamps_dropped": invalid_timestamps,
            "duplicate_timestamps_dropped": duplicate_count,
            "raw_missing_values": raw_missing_counts,
            "raw_missing_percent": {
                column: round(100 * count / raw_row_count, 4)
                for column, count in raw_missing_counts.items()
            },
            "raw_measurement_quality": {
                column: _missing_quality(records[column]) for column in RAW_COLUMNS
            },
        }

    if input_path.is_file() and input_path.suffix.casefold() == ".zip":
        with ZipFile(input_path) as archive:
            sources = [
                (archive, name)
                for name in sorted(archive.namelist())
                if Path(name).suffix.casefold() in {".csv", ".xlsx"}
                and not name.endswith("/")
            ]
            return load_sources(sources)

    sources: list[ExportSource] = (
        [input_path]
        if input_path.is_file()
        else sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".csv", ".xlsx"}
        )
    )
    return load_sources(sources)


def _counter_differences(counter: pd.Series) -> tuple[pd.Series, int]:
    differences = counter.diff()
    consecutive = counter.index.to_series().diff().eq(INTERVAL)
    resets = differences.lt(0) & consecutive.to_numpy()
    result = differences.where(consecutive)
    result.loc[resets] = counter.loc[resets]
    return result, int(resets.sum())


def _stuck_active_counter_mask(
    counter: pd.Series, active_demand: pd.Series, *, min_repeated_intervals: int = 4
) -> pd.Series:
    unchanged_while_active = (
        counter.eq(counter.shift())
        & counter.notna()
        & active_demand.gt(1.0)
    )
    run_ids = (~unchanged_while_active).cumsum()
    run_lengths = unchanged_while_active.groupby(run_ids).transform("sum")
    repeated = unchanged_while_active & run_lengths.ge(min_repeated_intervals)
    # The first changed reading after a plateau contains energy accumulated over the plateau.
    return repeated | repeated.shift(1, fill_value=False)


def _impute_gaps(
    series: pd.Series, max_short_gap: int
) -> tuple[pd.Series, dict, pd.Series]:
    """Interpolate short bracketed gaps and fill longer gaps from prior daily/weekly slots."""
    original = series.copy()
    result = original.copy()
    future_boundary = pd.Series(pd.NaT, index=original.index, dtype="datetime64[ns]")
    missing_positions = np.flatnonzero(original.isna().to_numpy())
    runs = (
        np.split(
            missing_positions,
            np.flatnonzero(np.diff(missing_positions) > 1) + 1,
        )
        if len(missing_positions)
        else []
    )
    counts = {
        "linear_interpolation_short_gap": 0,
        "prior_day_or_week": 0,
        "unresolved": 0,
    }

    for run in runs:
        start, end = int(run[0]), int(run[-1])
        gap_length = len(run)
        left, right = start - 1, end + 1
        if (
            gap_length <= max_short_gap
            and left >= 0
            and right < len(original)
            and pd.notna(original.iloc[left])
            and pd.notna(original.iloc[right])
        ):
            left_value = float(original.iloc[left])
            right_value = float(original.iloc[right])
            step = (right_value - left_value) / (gap_length + 1)
            result.iloc[start : end + 1] = [
                left_value + step * offset
                for offset in range(1, gap_length + 1)
            ]
            future_boundary.iloc[start : end + 1] = original.index[right]
            counts["linear_interpolation_short_gap"] += gap_length

    for lag in (STEPS_PER_DAY, *(7 * STEPS_PER_DAY * week for week in range(1, 5))):
        donor = original.shift(lag)
        fill_mask = result.isna() & donor.notna()
        counts["prior_day_or_week"] += int(fill_mask.sum())
        result.loc[fill_mask] = donor.loc[fill_mask]
    counts["unresolved"] = int(result.isna().sum())
    return result, counts, future_boundary


def _calendar_features(
    index: pd.DatetimeIndex, holiday_dates: set
) -> pd.DataFrame:
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    weekday = index.dayofweek.to_numpy() + hour / 24.0
    days_in_month = index.days_in_month.to_numpy()
    month = index.month.to_numpy() - 1 + (index.day.to_numpy() - 1) / days_in_month
    is_holiday = np.fromiter(
        (timestamp.date() in holiday_dates for timestamp in index),
        dtype=bool,
        count=len(index),
    )
    nonworking_day = (index.dayofweek >= 5) | is_holiday
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "is_nonworking_day": nonworking_day.astype(np.float32),
        },
        index=index,
    )


def _rolling_outlier_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts = {}
    for column in PHYSICAL_COLUMNS:
        values = frame[column]
        history = values.shift(1)
        median = history.rolling(672, min_periods=96).median()
        deviation = (history - median).abs()
        mad = deviation.rolling(672, min_periods=96).median()
        robust_scale = 1.4826 * mad
        fallback_threshold = pd.concat(
            [median.abs() * 0.1, pd.Series(0.01, index=frame.index)],
            axis=1,
        ).max(axis=1)
        threshold = (8 * robust_scale).where(robust_scale.gt(0), fallback_threshold)
        scorable = median.notna()
        flagged = values.sub(median).abs().gt(threshold) & scorable
        counts[column] = {
            "flagged_not_removed": int(flagged.sum()),
            "scorable_intervals": int(scorable.sum()),
            "not_scorable_insufficient_history": int((~scorable).sum()),
        }
    return counts


def prepare_dataset(
    records: pd.DataFrame,
    audit: dict,
    *,
    max_short_gap: int = 4,
    holiday_dates: set | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build four physical series and seven calendar features on a 15-minute grid."""
    if max_short_gap < 0:
        raise ValueError("max_short_gap must be non-negative")
    if records.empty:
        raise ValueError("No timestamped measurements were loaded")

    off_grid = records.index != records.index.floor("15min")
    off_grid_count = int(off_grid.sum())
    if off_grid_count:
        records = records.loc[~off_grid]
        if records.empty:
            raise ValueError("No measurements fall on the expected 15-minute grid")

    start = records.index.min().floor("15min")
    end = records.index.max().ceil("15min")
    grid = pd.date_range(start, end, freq=INTERVAL, name="timestamp")
    source_missing_mask = ~grid.isin(records.index)
    missing_positions = np.flatnonzero(source_missing_mask)
    missing_runs = (
        np.split(missing_positions, np.flatnonzero(np.diff(missing_positions) > 1) + 1)
        if len(missing_positions)
        else []
    )
    source_gap_summary = [
        {
            "start": grid[run[0]].isoformat(),
            "end": grid[run[-1]].isoformat(),
            "steps": int(len(run)),
        }
        for run in sorted(missing_runs, key=len, reverse=True)[:20]
    ]
    observed = records.reindex(grid)
    active_delta, active_resets = _counter_differences(observed["cumulative_active_kwh"])
    stuck_active_counter = _stuck_active_counter_mask(
        observed["cumulative_active_kwh"], observed["pmax_kw"]
    )
    active_delta.loc[stuck_active_counter] = np.nan
    reactive_delta, reactive_resets = _counter_differences(
        observed["cumulative_reactive_kvarh"]
    )
    physical = pd.DataFrame(
        {
            "delta_ea_kwh": active_delta,
            "delta_er_kvarh": reactive_delta,
            "pmax_kw": observed["pmax_kw"],
            "smax_kva": observed["smax_kva"],
        },
        index=grid,
    )
    negative_power_counts = {}
    for column in ("pmax_kw", "smax_kva"):
        invalid = physical[column] < 0
        negative_power_counts[column] = int(invalid.sum())
        physical.loc[invalid, column] = np.nan

    physical_before_imputation = physical.copy()
    missing_before = physical.isna().sum().astype(int).to_dict()
    imputation = {}
    imputed_by_column = pd.DataFrame(False, index=grid, columns=PHYSICAL_COLUMNS)
    future_boundaries = pd.DataFrame(
        pd.NaT, index=grid, columns=PHYSICAL_COLUMNS, dtype="datetime64[ns]"
    )
    for column in PHYSICAL_COLUMNS:
        (
            physical[column],
            imputation[column],
            future_boundaries[column],
        ) = _impute_gaps(
            physical[column],
            max_short_gap,
        )
        imputed_by_column[column] = (
            physical_before_imputation[column].isna() & physical[column].notna()
        )

    calendar = _calendar_features(grid, holiday_dates or set())
    clean = pd.concat([physical, calendar], axis=1)
    clean["was_imputed"] = imputed_by_column.any(axis=1)
    clean["has_missing_physical"] = physical.isna().any(axis=1)
    for column in PHYSICAL_COLUMNS:
        clean[f"imputed_{column}"] = imputed_by_column[column]
    clean["imputation_future_boundary"] = future_boundaries.max(axis=1)

    suspicious_zero_measurements = {
        "zero_active_energy_with_active_demand": int(
            (
                (physical_before_imputation["delta_ea_kwh"] == 0)
                & (physical_before_imputation["pmax_kw"] > 1)
            ).sum()
        ),
        "zero_apparent_demand_with_active_demand": int(
            (
                (physical_before_imputation["smax_kva"] == 0)
                & (physical_before_imputation["pmax_kw"] > 1)
            ).sum()
        ),
    }
    active_exceeds_apparent = (
        physical_before_imputation["pmax_kw"].notna()
        & physical_before_imputation["smax_kva"].notna()
        & (
            physical_before_imputation["pmax_kw"]
            > physical_before_imputation["smax_kva"] + 0.5
        )
    )

    report = dict(audit)
    report.update(
        {
            "rows_after_timestamp_deduplication": int(len(records)),
            "off_grid_timestamps_dropped": off_grid_count,
            "grid_start": grid[0].isoformat(),
            "grid_end": grid[-1].isoformat(),
            "grid_rows": int(len(grid)),
            "grid_intervals_missing_from_source": int(source_missing_mask.sum()),
            "longest_source_gaps": source_gap_summary,
            "counter_resets": {
                "active": active_resets,
                "reactive": reactive_resets,
            },
            "active_counter_stuck_intervals_flagged": int(stuck_active_counter.sum()),
            "active_counter_stuck_rule": (
                "At least 4 repeated cumulative readings while Pmax > 1 kW; "
                "the first post-plateau difference is also marked missing."
            ),
            "negative_demand_values_replaced_with_missing": negative_power_counts,
            "suspicious_zero_measurements_not_removed": suspicious_zero_measurements,
            "active_demand_greater_than_apparent_demand_not_removed": int(
                active_exceeds_apparent.sum()
            ),
            "physical_missing_before_imputation": missing_before,
            "physical_missing_before_imputation_percent": {
                column: round(100 * count / len(grid), 4)
                for column, count in missing_before.items()
            },
            "physical_quality_before_imputation": {
                column: _missing_quality(physical_before_imputation[column])
                for column in PHYSICAL_COLUMNS
            },
            "imputation": imputation,
            "imputed_values_total": int(
                sum(
                    values["linear_interpolation_short_gap"]
                    + values["prior_day_or_week"]
                    for values in imputation.values()
                )
            ),
            "rolling_outlier_flags_not_removed": _rolling_outlier_counts(
                pd.DataFrame(
                    {
                        "delta_ea_kwh": active_delta,
                        "delta_er_kvarh": reactive_delta,
                        "pmax_kw": observed["pmax_kw"],
                        "smax_kva": observed["smax_kva"],
                    },
                    index=grid,
                )
            ),
            "outlier_detection_method": (
                "Trailing 672-step median and MAD; 8 robust scales, or a fallback "
                "threshold of max(10% of the median magnitude, 0.01) when MAD is zero. "
                "Flags are reported, not automatically removed."
            ),
            "physical_missing_after_imputation": {
                column: int(clean[column].isna().sum()) for column in PHYSICAL_COLUMNS
            },
            "physical_quality_after_imputation": {
                column: _missing_quality(clean[column]) for column in PHYSICAL_COLUMNS
            },
            "rows_with_any_imputed_physical_value": int(clean["was_imputed"].sum()),
            "imputed_values_by_column": {
                column: int(imputed_by_column[column].sum())
                for column in PHYSICAL_COLUMNS
            },
            "rows_with_unresolved_physical_missing": int(
                clean["has_missing_physical"].sum()
            ),
            "unresolved_physical_missing_percent_of_grid": round(
                100 * float(clean["has_missing_physical"].mean()), 4
            ),
            "window_features": list(FEATURE_COLUMNS),
            "target_columns": list(TARGET_COLUMNS),
            "calendar_note": (
                "is_nonworking_day is true on Saturdays, Sundays, and supplied holiday dates."
            ),
        }
    )
    return clean, report


def load_holiday_dates(path: Path | None) -> set:
    if path is None:
        return set()
    holidays = pd.read_csv(path)
    if holidays.empty:
        return set()
    date_column = next(
        (column for column in holidays.columns if str(column).strip().casefold() == "date"),
        holidays.columns[0],
    )
    dates = pd.to_datetime(holidays[date_column], dayfirst=True, errors="raise")
    return set(dates.dt.date)


def _split_bounds(
    length: int, train_fraction: float, validation_fraction: float
) -> tuple[int, int]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    return int(length * train_fraction), int(length * (train_fraction + validation_fraction))


def _window_references(
    frame: pd.DataFrame,
    *,
    window_steps: int,
    horizon_steps: int,
    train_fraction: float,
    validation_fraction: float,
    exclude_imputed_targets: bool = True,
) -> Iterator[tuple[int, int, str]]:
    if window_steps < 1 or horizon_steps < 1:
        raise ValueError("window_steps and horizon_steps must be positive")
    train_end, validation_end = _split_bounds(
        len(frame), train_fraction, validation_fraction
    )
    features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
    for origin in range(window_steps - 1, len(frame) - horizon_steps):
        target_start = origin + 1
        target_end = origin + horizon_steps
        if target_end < train_end:
            split = "train"
        elif target_start >= train_end and target_end < validation_end:
            split = "validation"
        elif target_start >= validation_end:
            split = "test"
        else:
            continue
        if not (
            np.isfinite(features[origin - window_steps + 1 : origin + 1]).all()
            and np.isfinite(targets[target_start : target_end + 1]).all()
        ):
            continue
        if "imputation_future_boundary" in frame:
            boundaries = frame["imputation_future_boundary"].iloc[
                origin - window_steps + 1 : origin + 1
            ]
            future_boundary = boundaries.max()
            if pd.notna(future_boundary) and future_boundary > frame.index[origin]:
                continue
        if (
            exclude_imputed_targets
            and "was_imputed" in frame
            and frame["was_imputed"].iloc[target_start : target_end + 1].any()
        ):
            continue
        yield origin, target_end, split


def build_window_index(
    frame: pd.DataFrame,
    *,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    horizon_steps: int = 1,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    exclude_imputed_targets: bool = True,
) -> pd.DataFrame:
    """Index chronological windows, excluding synthetic targets by default."""
    rows = []
    for origin, target_end, split in _window_references(
        frame,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        exclude_imputed_targets=exclude_imputed_targets,
    ):
        rows.append(
            {
                "input_start": frame.index[origin - window_steps + 1],
                "origin": frame.index[origin],
                "target_start": frame.index[origin + 1],
                "target_end": frame.index[target_end],
                "split": split,
            }
        )
    return pd.DataFrame(
        rows,
        columns=("input_start", "origin", "target_start", "target_end", "split"),
    )


def iter_window_batches(
    frame: pd.DataFrame,
    *,
    split: str,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    horizon_steps: int = 1,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    batch_size: int = 128,
    exclude_imputed_targets: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (X, y, target timestamps) batches, with y shaped (batch, horizon, 4)."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
    batch_x, batch_y, batch_times = [], [], []
    for origin, target_end, window_split in _window_references(
        frame,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        exclude_imputed_targets=exclude_imputed_targets,
    ):
        if window_split != split:
            continue
        target_start = origin + 1
        batch_x.append(features[origin - window_steps + 1 : origin + 1])
        batch_y.append(targets[target_start : target_end + 1])
        batch_times.append(frame.index[target_start : target_end + 1].to_numpy())
        if len(batch_x) == batch_size:
            yield (
                np.stack(batch_x),
                np.stack(batch_y),
                np.stack(batch_times),
            )
            batch_x, batch_y, batch_times = [], [], []
    if batch_x:
        yield (
            np.stack(batch_x),
            np.stack(batch_y),
            np.stack(batch_times),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Folder of CSV/XLSX exports")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--meter", default="EM_VIRTUAL")
    parser.add_argument("--holidays", type=Path, help="Optional CSV with a date column")
    parser.add_argument("--window-steps", type=int, default=DEFAULT_WINDOW_STEPS)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--max-short-gap", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, audit = load_exports(args.input, meter=args.meter)
    clean, report = prepare_dataset(
        records,
        audit,
        max_short_gap=args.max_short_gap,
        holiday_dates=load_holiday_dates(args.holidays),
    )
    window_index = build_window_index(
        clean,
        window_steps=args.window_steps,
        horizon_steps=args.horizon_steps,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    all_finite_windows = build_window_index(
        clean,
        window_steps=args.window_steps,
        horizon_steps=args.horizon_steps,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        exclude_imputed_targets=False,
    )
    clean.to_parquet(args.output / "sinert_clean.parquet", index=True)
    window_index.to_parquet(args.output / "window_index.parquet", index=False)
    report["window_configuration"] = {
        "window_steps": args.window_steps,
        "horizon_steps": args.horizon_steps,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": 1 - args.train_fraction - args.validation_fraction,
        "valid_windows_by_split": window_index["split"].value_counts().to_dict(),
        "windows_excluded_for_imputed_targets": int(
            len(all_finite_windows) - len(window_index)
        ),
    }
    report["window_index_file"] = "window_index.parquet"
    report["clean_data_file"] = "sinert_clean.parquet"
    with (args.output / "quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=str)
    print(
        f"Done: {len(clean):,} intervals, {len(window_index):,} valid windows. "
        f"Outputs: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
