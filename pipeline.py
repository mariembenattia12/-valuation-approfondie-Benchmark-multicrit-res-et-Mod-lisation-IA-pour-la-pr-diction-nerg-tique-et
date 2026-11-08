"""Load, clean, and prepare SINERT energy-meter exports for forecasting."""

from __future__ import annotations

import argparse
from collections import deque
import html
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
        result[canonical] = pd.to_numeric(values, errors="coerce").astype("float64")
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


def _rolling_outlier_masks(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.Series]]:
    flags = pd.DataFrame(False, index=frame.index, columns=PHYSICAL_COLUMNS)
    baselines = {}
    thresholds = {}
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
        flags[column] = values.sub(median).abs().gt(threshold) & scorable
        baselines[column] = median
        thresholds[column] = threshold
    return flags, baselines, thresholds


def _rolling_outlier_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    flags, baselines, _ = _rolling_outlier_masks(frame)
    counts = {}
    for column in PHYSICAL_COLUMNS:
        scorable = baselines[column].notna()
        counts[column] = {
            "flagged_not_removed": int(flags[column].sum()),
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
    original_power = {
        column: observed[column].copy()
        for column in ("pmax_kw", "smax_kva")
    }
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
    for column, values in original_power.items():
        clean[f"{column}_original"] = values
    for column in PHYSICAL_COLUMNS:
        clean[f"imputed_{column}"] = imputed_by_column[column]
        linear_imputation = (
            imputed_by_column[column] & future_boundaries[column].notna()
        )
        clean[f"imputed_{column}_linear"] = linear_imputation
        clean[f"imputed_{column}_prior_slot"] = (
            imputed_by_column[column] & ~linear_imputation
        )
    zero_power = (
        physical_before_imputation["pmax_kw"].eq(0)
        | physical_before_imputation["smax_kva"].eq(0)
    ).fillna(False).to_numpy(dtype=bool)
    positive_energy_with_zero_power = (
        zero_power & physical_before_imputation["delta_ea_kwh"].gt(0)
    ).fillna(False).to_numpy(dtype=bool)
    clean["power_zero_status"] = np.select(
        [
            positive_energy_with_zero_power,
            zero_power,
        ],
        [
            "suspect",
            "non_verifiable",
        ],
        default="normal",
    )
    clean["power_zero_reason"] = np.where(
        positive_energy_with_zero_power,
        "zero_power_with_positive_active_energy",
        np.where(zero_power, "zero_power_without_confirming_energy", ""),
    )
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
        "zero_pmax_with_positive_active_energy": int(
            (
                (physical_before_imputation["pmax_kw"] == 0)
                & (physical_before_imputation["delta_ea_kwh"] > 0)
            ).sum()
        ),
        "zero_smax_with_positive_active_energy": int(
            (
                (physical_before_imputation["smax_kva"] == 0)
                & (physical_before_imputation["delta_ea_kwh"] > 0)
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
    calendar = load_behavior_calendar(path)
    if "is_holiday" not in calendar:
        return set()
    return set(calendar.index[calendar["is_holiday"].fillna(False)].date)


def _calendar_boolean(values: pd.Series, column: str) -> pd.Series:
    normalized = values.astype("string").str.strip().str.casefold()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "oui": True,
        "non": False,
    }
    parsed = normalized.map(mapping)
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"Calendar column {column!r} contains invalid boolean values: {examples}"
        )
    return parsed.astype("boolean")


def load_behavior_calendar(path: Path | None) -> pd.DataFrame:
    """Load optional daily holiday, vacation, and study-week labels."""
    columns = ("is_holiday", "is_vacation", "is_study_week", "is_exam_period")
    if path is None:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    if path.suffix.casefold() == ".csv":
        calendar = pd.read_csv(path)
    elif path.suffix.casefold() in {".xlsx", ".xlsm"}:
        calendar = pd.read_excel(path, sheet_name=0)
    else:
        raise ValueError(f"{path}: calendar must be a .csv or .xlsx file")
    if calendar.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    date_column = next(
        (
            column
            for column in calendar.columns
            if str(column).strip().casefold() == "date"
        ),
        None,
    )
    if date_column is None:
        raise ValueError(f"{path}: calendar must contain a 'date' column")
    dates = pd.to_datetime(
        calendar[date_column], format="mixed", dayfirst=True, errors="coerce"
    )
    if dates.isna().any():
        raise ValueError(f"{path}: calendar contains invalid dates")
    dates = dates.dt.normalize()
    if dates.duplicated().any():
        duplicate_dates = dates.loc[dates.duplicated(keep=False)].dt.strftime(
            "%Y-%m-%d"
        ).unique()[:5]
        raise ValueError(
            f"{path}: calendar contains duplicate dates: {duplicate_dates.tolist()}"
        )

    result = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    for column in columns:
        if column in calendar:
            result[column] = _calendar_boolean(calendar[column], column).array
        elif column == "is_holiday" and not any(
            flag in calendar.columns
            for flag in ("is_vacation", "is_study_week", "is_exam_period")
        ):
            result[column] = True
    for column in (
        "calendar_confidence",
        "source_ids",
        "event_name",
        "calendar_notes",
        "source_statuses",
    ):
        if column in calendar:
            result[column] = calendar[column].to_numpy()
    return result.sort_index()


def _calendar_metadata(
    calendar: pd.DataFrame, timestamps: pd.DatetimeIndex
) -> dict:
    label_columns = [
        column
        for column in (
            "is_holiday",
            "is_vacation",
            "is_study_week",
            "is_exam_period",
        )
        if column in calendar
    ]
    labelled_rows = (
        calendar[label_columns].notna().any(axis=1)
        if label_columns
        else pd.Series(False, index=calendar.index)
    )
    labelled_dates = pd.DatetimeIndex(calendar.index[labelled_rows]).normalize()
    data_dates = pd.DatetimeIndex(timestamps).normalize().unique()
    metadata = {
        "labelled_days_in_data": int(data_dates.isin(labelled_dates).sum()),
        "holiday_labels_available": (
            "is_holiday" in calendar
            and bool(calendar["is_holiday"].notna().any())
        ),
        "vacation_labels_available": (
            "is_vacation" in calendar
            and bool(calendar["is_vacation"].notna().any())
        ),
        "study_week_labels_available": (
            "is_study_week" in calendar
            and bool(calendar["is_study_week"].notna().any())
        ),
    }
    if "calendar_confidence" in calendar:
        confidence = calendar.loc[labelled_rows, "calendar_confidence"]
        metadata["label_confidence_days"] = {
            str(key): int(value)
            for key, value in confidence.value_counts(dropna=True).items()
        }
    if "source_ids" in calendar:
        source_ids = {
            source_id
            for value in calendar.loc[labelled_rows, "source_ids"].dropna()
            for source_id in str(value).split("|")
            if source_id
        }
        metadata["calendar_source_ids"] = sorted(source_ids)
    return metadata


def _period_energy(
    energy: pd.Series, frequency: str, imputed_mask: pd.Series | None = None
) -> pd.DataFrame:
    rows = []
    period_labels = energy.index.to_period(frequency)
    if imputed_mask is None:
        imputed_mask = pd.Series(False, index=energy.index)
    else:
        imputed_mask = imputed_mask.reindex(energy.index).fillna(False).astype(bool)
    for period, values in energy.groupby(period_labels, sort=True):
        expected = int(
            (period.end_time.floor("15min") - period.start_time) / INTERVAL
        ) + 1
        imputed = imputed_mask.loc[values.index] & values.notna()
        observed = values[~imputed]
        valid = values.dropna()
        in_scope = len(values)
        expected_in_scope = int(
            (
                min(period.end_time.floor("15min"), energy.index.max())
                - max(period.start_time, energy.index.min())
            )
            / INTERVAL
        ) + 1
        expected_in_scope = max(0, expected_in_scope)
        rows.append(
            {
                "period_start": period.start_time,
                "period_end": period.end_time.floor("15min"),
                "energy_kwh": valid.sum() if len(valid) else np.nan,
                "valid_intervals": int(len(valid)),
                "observed_intervals": int(observed.notna().sum()),
                "imputed_intervals": int(imputed.sum()),
                "expected_intervals": expected,
                "measured_energy_kwh": observed.sum(min_count=1),
                "in_scope_intervals": in_scope,
                "expected_in_scope_intervals": expected_in_scope,
                "coverage_percent": 100 * observed.notna().sum() / expected,
                "coverage_of_scope_percent": (
                    100 * observed.notna().sum() / expected_in_scope
                    if expected_in_scope
                    else np.nan
                ),
                "missing_observed_intervals_in_scope": max(
                    0, expected_in_scope - int(observed.notna().sum())
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "period_start",
            "period_end",
            "energy_kwh",
            "valid_intervals",
            "observed_intervals",
            "imputed_intervals",
            "expected_intervals",
            "measured_energy_kwh",
            "in_scope_intervals",
            "expected_in_scope_intervals",
            "coverage_percent",
            "coverage_of_scope_percent",
            "missing_observed_intervals_in_scope",
        ),
    )


def _group_statistics(
    clean: pd.DataFrame, dimension: str, labels: pd.Series
) -> list[dict]:
    rows = []
    for category in labels.dropna().unique():
        subset = clean.loc[labels.eq(category)]
        row = {
            "dimension": dimension,
            "category": str(category),
            "intervals": int(len(subset)),
            "rows_with_imputation": int(
                subset["was_imputed"].sum() if "was_imputed" in subset else 0
            ),
        }
        for column in PHYSICAL_COLUMNS:
            values = subset[column].dropna()
            imputed_column = f"imputed_{column}"
            if imputed_column in subset:
                observed_values = subset[column].mask(
                    subset[imputed_column].fillna(False)
                ).dropna()
            else:
                observed_values = values
            row[f"{column}_valid_intervals"] = int(len(values))
            for statistic in ("mean", "median", "std", "var", "min", "max"):
                row[f"{column}_{statistic}"] = (
                    getattr(values, statistic)() if len(values) else np.nan
                )
            row[f"{column}_observed_intervals"] = int(len(observed_values))
            for statistic in ("mean", "std", "var"):
                row[f"{column}_observed_{statistic}"] = (
                    getattr(observed_values, statistic)()
                    if len(observed_values)
                    else np.nan
                )
        rows.append(row)
    return rows


def _circular_clock_statistics(timestamps: pd.Series) -> tuple[float, float, float]:
    valid = timestamps.dropna()
    if valid.empty:
        return np.nan, np.nan, np.nan
    minutes = (
        valid.dt.hour.to_numpy() * 60
        + valid.dt.minute.to_numpy()
        + valid.dt.second.to_numpy() / 60
    )
    angles = 2 * np.pi * minutes / (24 * 60)
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    resultant = np.hypot(np.sin(angles).mean(), np.cos(angles).mean())
    mean_minutes = (mean_angle % (2 * np.pi)) * 24 * 60 / (2 * np.pi)
    circular_std_hours = (
        np.sqrt(-2 * np.log(max(resultant, np.finfo(float).tiny)))
        * 24
        / (2 * np.pi)
    )
    return mean_minutes / 60, circular_std_hours, 1 - resultant


def _build_quality_reports(
    clean: pd.DataFrame,
    *,
    quality_report: dict | None = None,
    source_records: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    quality_report = quality_report or {}
    source_records = source_records.sort_index() if source_records is not None else None
    imputed_masks = {
        column: clean.get(
            f"imputed_{column}", pd.Series(False, index=clean.index)
        ).fillna(False)
        for column in PHYSICAL_COLUMNS
    }

    pmax = clean["pmax_kw"]
    smax = clean["smax_kva"]
    observed_pmax = pmax.mask(imputed_masks["pmax_kw"])
    observed_smax = smax.mask(imputed_masks["smax_kva"])
    observed_energy = clean["delta_ea_kwh"].mask(imputed_masks["delta_ea_kwh"])
    observed_reactive = clean["delta_er_kvarh"].mask(
        imputed_masks["delta_er_kvarh"]
    )
    zero_mask = pmax.eq(0).fillna(False) | smax.eq(0).fillna(False)
    zero_index = clean.index[zero_mask]
    zero_report = pd.DataFrame(index=zero_index)
    zero_report.index.name = "timestamp"
    zero_report["active_energy_kwh"] = observed_energy.reindex(zero_index)
    zero_report["processed_active_energy_kwh"] = clean["delta_ea_kwh"].reindex(
        zero_index
    )
    zero_report["active_energy_was_imputed"] = imputed_masks[
        "delta_ea_kwh"
    ].reindex(zero_index)
    zero_report["reactive_energy_kvarh"] = observed_reactive.reindex(zero_index)
    zero_report["processed_pmax_kw"] = pmax.reindex(zero_index)
    zero_report["processed_smax_kva"] = smax.reindex(zero_index)
    zero_report["original_pmax_kw"] = clean.get(
        "pmax_kw_original", observed_pmax
    ).reindex(zero_index)
    zero_report["original_smax_kva"] = clean.get(
        "smax_kva_original", observed_smax
    ).reindex(zero_index)
    zero_report["pmax_was_imputed"] = imputed_masks["pmax_kw"].reindex(zero_index)
    zero_report["smax_was_imputed"] = imputed_masks["smax_kva"].reindex(zero_index)
    for column, series in (
        ("active_energy_kwh", observed_energy),
        ("reactive_energy_kvarh", observed_reactive),
        ("pmax_kw", observed_pmax),
        ("smax_kva", observed_smax),
    ):
        zero_report[f"previous_{column}"] = series.shift(1).reindex(zero_index)
        zero_report[f"next_{column}"] = series.shift(-1).reindex(zero_index)
    zero_report["pmax_zero"] = zero_report["processed_pmax_kw"].eq(0)
    zero_report["smax_zero"] = zero_report["processed_smax_kva"].eq(0)
    zero_report["positive_active_energy"] = zero_report[
        "processed_active_energy_kwh"
    ].gt(0).fillna(False)
    zero_report["positive_measured_active_energy"] = zero_report[
        "active_energy_kwh"
    ].gt(0).fillna(False)
    zero_report["adjacent_pmax_kw_mean"] = zero_report[
        ["previous_pmax_kw", "next_pmax_kw"]
    ].mean(axis=1)
    zero_report["source_record_available"] = False
    zero_report["source_pmax_kw"] = np.nan
    zero_report["source_smax_kva"] = np.nan
    zero_report["source_cumulative_active_kwh"] = np.nan
    zero_report["source_cumulative_reactive_kvarh"] = np.nan
    zero_report["source_verification"] = "raw exports unavailable in workspace"
    if source_records is not None:
        matched = source_records.reindex(zero_index)
        zero_report["source_record_available"] = matched.index.isin(
            source_records.index
        )
        for source_column, report_column in (
            ("pmax_kw", "source_pmax_kw"),
            ("smax_kva", "source_smax_kva"),
            ("cumulative_active_kwh", "source_cumulative_active_kwh"),
            ("cumulative_reactive_kvarh", "source_cumulative_reactive_kvarh"),
        ):
            if source_column in matched:
                zero_report[report_column] = matched[source_column]
        matched_source = zero_report["source_record_available"]
        source_agrees = (
            zero_report["source_pmax_kw"].eq(zero_report["original_pmax_kw"])
            & zero_report["source_smax_kva"].eq(zero_report["original_smax_kva"])
        )
        zero_report["source_verification"] = np.where(
            matched_source & source_agrees,
            "zero confirmed in raw export; physical validity still unverified",
            np.where(matched_source, "raw export disagrees with processed values", "timestamp absent from raw export"),
        )
    zero_power_was_imputed = zero_report["pmax_was_imputed"] | zero_report[
        "smax_was_imputed"
    ]
    zero_report["status"] = np.where(
        zero_power_was_imputed,
        "non_verifiable",
        np.where(zero_report["positive_active_energy"], "suspect", "non_verifiable"),
    )
    zero_report["reason"] = np.where(
        zero_power_was_imputed,
        "processed zero came from imputation; original demand unavailable",
        np.where(
            zero_report["positive_measured_active_energy"],
            "recorded zero demand conflicts with measured positive active energy",
            np.where(
                zero_report["positive_active_energy"],
                "recorded zero demand conflicts with imputed positive active energy",
                "zero demand; corroborating active energy unavailable",
            ),
        ),
    )
    zero_report["decision"] = (
        "Retain original zero; do not correct without source-meter or operator validation"
    )
    zero_report = zero_report.reset_index()

    observed_frame = pd.DataFrame(index=clean.index)
    for column in PHYSICAL_COLUMNS:
        observed_frame[column] = clean[column].mask(imputed_masks[column])
    outlier_flags, baselines, thresholds = _rolling_outlier_masks(observed_frame)
    reactive_index = clean.index[outlier_flags["delta_er_kvarh"]]
    reactive_report = pd.DataFrame(index=reactive_index)
    reactive_report.index.name = "timestamp"
    reactive_report["delta_er_kvarh"] = observed_reactive.reindex(reactive_index)
    reactive_report["rolling_median"] = baselines["delta_er_kvarh"].reindex(
        reactive_index
    )
    reactive_report["rolling_threshold"] = thresholds["delta_er_kvarh"].reindex(
        reactive_index
    )
    reactive_report["absolute_deviation"] = (
        reactive_report["delta_er_kvarh"] - reactive_report["rolling_median"]
    ).abs()
    for column, values in (
        ("pmax_kw", observed_pmax),
        ("smax_kva", observed_smax),
        ("delta_ea_kwh", observed_energy),
    ):
        reactive_report[column] = values.reindex(reactive_index)
    if source_records is not None:
        source_reactive = source_records.reindex(reactive_index)
        if "cumulative_reactive_kvarh" in source_reactive:
            reactive_report["source_cumulative_reactive_kvarh"] = source_reactive[
                "cumulative_reactive_kvarh"
            ]
        reactive_report["source_record_available"] = reactive_index.isin(
            source_records.index
        )
    else:
        reactive_report["source_record_available"] = False
    reactive_report["status"] = "suspect_statistical_review_required"
    reactive_report["decision"] = (
        "Retain observation; robust rolling outlier is not proof of a physical error"
    )
    reactive_report = reactive_report.reset_index()

    reactive_values = observed_reactive.dropna()
    reactive_flagged = reactive_report["delta_er_kvarh"]
    reactive_summary = pd.DataFrame(
        [
            {
                "observed_intervals": int(len(reactive_values)),
                "missing_or_imputed_excluded": int(len(clean) - len(reactive_values)),
                "flagged_intervals": int(len(reactive_report)),
                "flagged_percent_of_observed": (
                    100 * len(reactive_report) / len(reactive_values)
                    if len(reactive_values)
                    else np.nan
                ),
                "minimum_kvarh": reactive_values.min(),
                "median_kvarh": reactive_values.median(),
                "p90_kvarh": reactive_values.quantile(0.90),
                "p95_kvarh": reactive_values.quantile(0.95),
                "p99_kvarh": reactive_values.quantile(0.99),
                "maximum_kvarh": reactive_values.max(),
                "flagged_minimum_kvarh": reactive_flagged.min(),
                "flagged_median_kvarh": reactive_flagged.median(),
                "flagged_maximum_kvarh": reactive_flagged.max(),
                "flagged_negative_values": int(reactive_flagged.lt(0).sum()),
                "flagged_zero_values": int(reactive_flagged.eq(0).sum()),
                "flagged_positive_values": int(reactive_flagged.gt(0).sum()),
                "correlation_with_pmax": observed_frame[
                    ["delta_er_kvarh", "pmax_kw"]
                ].corr().iloc[0, 1],
                "correlation_with_smax": observed_frame[
                    ["delta_er_kvarh", "smax_kva"]
                ].corr().iloc[0, 1],
                "method": "Trailing 672-step median/MAD; 8 robust scales with documented fallback",
                "decision": "No flagged value is removed automatically",
            }
        ]
    )

    missing_quality_rows = []
    missing_run_rows = []
    for column in PHYSICAL_COLUMNS:
        imputed = imputed_masks[column] & clean[column].notna()
        originally_missing = imputed | clean[column].isna()
        missing_after_positions = np.flatnonzero(clean[column].isna().to_numpy())
        missing_after_runs = (
            np.split(
                missing_after_positions,
                np.flatnonzero(np.diff(missing_after_positions) > 1) + 1,
            )
            if len(missing_after_positions)
            else []
        )
        longest_missing_run_after = max(
            (len(run) for run in missing_after_runs),
            default=0,
        )
        before_info = quality_report.get(
            "physical_quality_before_imputation", {}
        ).get(column, {})
        method_counts = quality_report.get("imputation", {}).get(column, {})
        missing_quality_rows.append(
            {
                "variable": column,
                "before_count": int(before_info.get("missing_values", originally_missing.sum())),
                "before_percent": before_info.get(
                    "missing_percent",
                    100 * float(originally_missing.mean()),
                ),
                "after_count": int(clean[column].isna().sum()),
                "after_percent": 100 * float(clean[column].isna().mean()),
                "linear_interpolation_count": int(
                    method_counts.get(
                        "linear_interpolation_short_gap",
                        clean.get(
                            f"imputed_{column}_linear",
                            pd.Series(False, index=clean.index),
                        ).sum(),
                    )
                ),
                "prior_day_or_week_count": int(
                    method_counts.get(
                        "prior_day_or_week",
                        clean.get(
                            f"imputed_{column}_prior_slot",
                            pd.Series(False, index=clean.index),
                        ).sum(),
                    )
                ),
                "imputed_total": int(imputed.sum()),
                "unresolved": int(clean[column].isna().sum()),
                "longest_missing_run_after_steps": int(longest_missing_run_after),
            }
        )
        missing_positions = np.flatnonzero(originally_missing.to_numpy())
        runs = (
            np.split(
                missing_positions,
                np.flatnonzero(np.diff(missing_positions) > 1) + 1,
            )
            if len(missing_positions)
            else []
        )
        for run in runs:
            run_index = clean.index[run]
            unresolved = clean[column].iloc[run].isna()
            linear_flag = clean.get(
                f"imputed_{column}_linear",
                clean.get(
                    "imputation_future_boundary",
                    pd.Series(pd.NaT, index=clean.index),
                ).notna()
                & imputed,
            )
            prior_flag = clean.get(
                f"imputed_{column}_prior_slot",
                imputed & ~linear_flag,
            )
            missing_run_rows.append(
                {
                    "variable": column,
                    "start": run_index[0],
                    "end": run_index[-1],
                    "steps": int(len(run)),
                    "duration_hours": len(run) * INTERVAL.total_seconds() / 3600,
                    "imputed_steps": int(imputed.iloc[run].sum()),
                    "unresolved_steps": int(unresolved.sum()),
                    "imputation_methods": (
                        "+".join(
                            method
                            for method, flags in (
                                ("linear", linear_flag),
                                ("prior_day_or_week", prior_flag),
                            )
                            if flags.iloc[run].any()
                        )
                        or "unresolved"
                    ),
                    "method_detail_confidence": (
                        "per-variable"
                        if f"imputed_{column}_linear" in clean
                        else "inferred from row-level interpolation boundary"
                    ),
                }
            )
    missing_quality = pd.DataFrame(missing_quality_rows)
    missing_runs = pd.DataFrame(missing_run_rows)

    energy = clean["delta_ea_kwh"]
    energy_imputed = imputed_masks["delta_ea_kwh"]
    coverage_month = _period_energy(energy, "M", energy_imputed)
    coverage_year = _period_energy(energy, "Y", energy_imputed)
    coverage_month.insert(0, "period", "month")
    coverage_year.insert(0, "period", "year")
    coverage_summary = pd.DataFrame(
        [
            {
                "data_start": clean.index.min(),
                "data_end": clean.index.max(),
                "grid_intervals": int(len(clean)),
                "energy_observed_kwh": energy.mask(energy_imputed).sum(min_count=1),
                "energy_including_imputation_kwh": energy.sum(min_count=1),
                "energy_imputed_kwh": energy[energy_imputed].sum(min_count=1),
                "imputed_energy_intervals": int(
                    (energy_imputed & energy.notna()).sum()
                ),
                "unresolved_energy_intervals": int(energy.isna().sum()),
                "months_reported": int(len(coverage_month)),
                "years_reported": int(len(coverage_year)),
                "interpretation": (
                    "Partial first/last periods and measured coverage must be considered "
                    "before comparing period totals."
                ),
            }
        ]
    )

    return {
        "suspect_power_zeros.csv": zero_report,
        "reactive_outliers.csv": reactive_report,
        "reactive_outlier_summary.csv": reactive_summary,
        "missing_imputation_quality.csv": missing_quality,
        "missing_runs.csv": missing_runs,
        "coverage_by_month.csv": coverage_month,
        "coverage_by_year.csv": coverage_year,
        "coverage_summary.csv": coverage_summary,
    }


def _write_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: np.ndarray,
    *,
    y_label: str,
    color: str,
    value_limit: float | None = None,
) -> None:
    width, height = 1200, 480
    left, right, top, bottom = 78, 28, 62, 92
    plot_width = width - left - right
    plot_height = height - top - bottom
    finite_values = values[np.isfinite(values)]
    maximum = value_limit or (
        float(finite_values.max()) if len(finite_values) else 1.0
    )
    maximum = max(maximum, 1.0)
    bar_width = max(1.0, plot_width / max(len(values), 1) * 0.72)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" font-weight="bold">{html.escape(title)}</text>',
    ]
    for step in range(6):
        y_value = maximum * step / 5
        y = top + plot_height * (1 - step / 5)
        pieces.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>'
        )
        pieces.append(
            f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.0f}</text>'
        )
    pieces.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#333"/>',
        ]
    )
    label_step = max(1, len(labels) // 12)
    for index, value in enumerate(values):
        x_center = left + plot_width * (index + 0.5) / max(len(values), 1)
        if np.isfinite(value):
            bar_height = max(0.0, min(float(value), maximum)) / maximum * plot_height
            pieces.append(
                f'<rect x="{x_center-bar_width/2:.1f}" y="{top+plot_height-bar_height:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
            )
        if index % label_step == 0 or index == len(labels) - 1:
            pieces.append(
                f'<text transform="translate({x_center:.1f},{top+plot_height+15}) rotate(55)" font-family="sans-serif" font-size="10">{html.escape(labels[index])}</text>'
            )
    pieces.extend(
        [
            f'<text transform="translate(20,{top+plot_height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces), encoding="utf-8")


def _write_line_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: np.ndarray,
    *,
    y_label: str,
    color: str,
) -> None:
    width, height = 1200, 480
    left, right, top, bottom = 82, 28, 62, 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    finite = np.isfinite(values)
    if finite.any():
        minimum = float(values[finite].min())
        maximum = float(values[finite].max())
        if minimum == maximum:
            padding = max(abs(minimum) * 0.05, 1.0)
            minimum -= padding
            maximum += padding
    else:
        minimum, maximum = 0.0, 1.0
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" font-weight="bold">{html.escape(title)}</text>',
    ]
    for step in range(6):
        fraction = step / 5
        value = minimum + (maximum - minimum) * fraction
        y = top + plot_height * (1 - fraction)
        pieces.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>',
                f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.3g}</text>',
            ]
        )
    pieces.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#333"/>',
            f'<text transform="translate(20,{top+plot_height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    if len(labels):
        for position in sorted({0, len(labels) // 2, len(labels) - 1}):
            x = left + plot_width * position / max(len(labels) - 1, 1)
            pieces.append(
                f'<text x="{x:.1f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(labels[position])}</text>'
            )
    run = []
    for position, value in enumerate(values):
        if not np.isfinite(value):
            if run:
                points = " ".join(run)
                pieces.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>'
                )
                run = []
            continue
        x = left + plot_width * position / max(len(values) - 1, 1)
        y = top + plot_height * (1 - (float(value) - minimum) / (maximum - minimum))
        run.append(f"{x:.1f},{y:.1f}")
    if run:
        points = " ".join(run)
        pieces.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
    pieces.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces), encoding="utf-8")


def write_phase1_visualizations(
    clean: pd.DataFrame, reports: dict[str, pd.DataFrame], output_dir: Path
) -> list[str]:
    """Write dependency-free SVG coverage charts and measured daily series."""
    figure_dir = output_dir / "figures"
    monthly = reports["coverage_by_month.csv"]
    yearly = reports["coverage_by_year.csv"]
    _write_bar_chart(
        figure_dir / "coverage_by_month.svg",
        "Measured energy coverage by month (in-scope)",
        pd.to_datetime(monthly["period_start"]).dt.strftime("%Y-%m").tolist(),
        monthly["coverage_of_scope_percent"].to_numpy(dtype=float),
        y_label="Measured coverage (%)",
        color="#3478bf",
        value_limit=100,
    )
    _write_bar_chart(
        figure_dir / "coverage_by_year.svg",
        "Measured energy coverage by year",
        pd.to_datetime(yearly["period_start"]).dt.strftime("%Y").tolist(),
        yearly["coverage_percent"].to_numpy(dtype=float),
        y_label="Measured coverage (%)",
        color="#498c52",
        value_limit=100,
    )
    imputed = clean.get(
        "imputed_pmax_kw", pd.Series(False, index=clean.index)
    ).fillna(False)
    observed_daily = clean["pmax_kw"].mask(imputed).resample("D").mean()
    _write_bar_chart(
        figure_dir / "pmax_daily_mean.svg",
        "Daily mean Pmax (observed intervals)",
        observed_daily.index.strftime("%Y-%m-%d").tolist(),
        observed_daily.to_numpy(dtype=float),
        y_label="Pmax (kW)",
        color="#d27a34",
    )
    daily_figures = []
    daily_series = (
        ("delta_ea_kwh", "Daily active energy (observed)", "kWh/day", "#3478bf", "sum"),
        ("delta_er_kvarh", "Daily reactive energy (observed)", "kVARh/day", "#8057a5", "sum"),
        ("pmax_kw", "Daily mean active demand (observed)", "kW", "#d27a34", "mean"),
        ("smax_kva", "Daily mean apparent demand (observed)", "kVA", "#498c52", "mean"),
    )
    for column, title, y_label, color, aggregation in daily_series:
        imputed = clean.get(
            f"imputed_{column}", pd.Series(False, index=clean.index)
        ).fillna(False)
        observed = clean[column].mask(imputed)
        resampler = observed.resample("D")
        daily = (
            resampler.sum(min_count=1)
            if aggregation == "sum"
            else resampler.mean()
        )
        filename = f"{column}_daily.svg"
        _write_line_chart(
            figure_dir / filename,
            title,
            daily.index.strftime("%Y-%m-%d").tolist(),
            daily.to_numpy(dtype=float),
            y_label=y_label,
            color=color,
        )
        daily_figures.append(f"figures/{filename}")
    return [
        "figures/coverage_by_month.svg",
        "figures/coverage_by_year.svg",
        "figures/pmax_daily_mean.svg",
        *daily_figures,
    ]


def build_behavior_reports(
    clean: pd.DataFrame,
    calendar: pd.DataFrame | None = None,
    *,
    peak_quantile: float = 0.90,
    activity_threshold_kw: float = 1.0,
    quality_report: dict | None = None,
    source_records: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Create reproducible energy, load, activity, and calendar summaries."""
    if not 0 < peak_quantile < 1:
        raise ValueError("peak_quantile must be between 0 and 1")
    if activity_threshold_kw < 0:
        raise ValueError("activity_threshold_kw must be non-negative")
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise TypeError("clean data must have a DatetimeIndex")
    missing_columns = set(PHYSICAL_COLUMNS) - set(clean.columns)
    if missing_columns:
        raise ValueError(f"clean data is missing columns: {sorted(missing_columns)}")

    energy = clean["delta_ea_kwh"]
    imputed_energy = clean.get(
        "imputed_delta_ea_kwh", pd.Series(False, index=clean.index)
    )
    energy_reports = {
        f"energy_{name}.csv": _period_energy(
            energy, frequency, imputed_energy
        )
        for name, frequency in (
            ("daily", "D"),
            ("weekly", "W-SUN"),
            ("monthly", "M"),
            ("annual", "Y"),
        )
    }

    power = clean["pmax_kw"]
    imputed_power = clean.get(
        "imputed_pmax_kw", pd.Series(False, index=clean.index)
    ).fillna(False)
    observed_power = power.mask(imputed_power)
    valid_power = observed_power.dropna()
    peak_threshold = (
        float(valid_power.quantile(peak_quantile)) if len(valid_power) else np.nan
    )
    all_peak_threshold = (
        float(power.dropna().quantile(peak_quantile))
        if power.notna().any()
        else np.nan
    )
    peak_mask = (
        observed_power.ge(peak_threshold).fillna(False)
        if np.isfinite(peak_threshold)
        else pd.Series(False, index=power.index)
    )
    all_peak_mask = (
        power.ge(all_peak_threshold).fillna(False)
        if np.isfinite(all_peak_threshold)
        else pd.Series(False, index=power.index)
    )
    peak_positions = np.flatnonzero(peak_mask.to_numpy())
    peak_runs = (
        np.split(peak_positions, np.flatnonzero(np.diff(peak_positions) > 1) + 1)
        if len(peak_positions)
        else []
    )
    peak_rows = []
    for run in peak_runs:
        episode = clean.iloc[run[0] : run[-1] + 1]
        peak_rows.append(
            {
                "start": episode.index[0],
                "last_interval_start": episode.index[-1],
                "end_exclusive": episode.index[-1] + INTERVAL,
                "duration_hours": len(episode) * INTERVAL.total_seconds() / 3600,
                "pmax_peak_kw": episode["pmax_kw"].max(),
                "smax_peak_kva": episode["smax_kva"].max(),
                "energy_kwh": episode["delta_ea_kwh"].sum(min_count=1),
                "intervals": int(len(episode)),
                "imputed_intervals": int(
                    episode["imputed_pmax_kw"].sum()
                    if "imputed_pmax_kw" in episode
                    else 0
                ),
            }
        )
    peak_columns = (
        "start",
        "last_interval_start",
        "end_exclusive",
        "duration_hours",
        "pmax_peak_kw",
        "smax_peak_kva",
        "energy_kwh",
        "intervals",
        "imputed_intervals",
    )
    peak_episodes = pd.DataFrame(peak_rows, columns=peak_columns)

    grouped_labels = []
    dates = clean.index.normalize()
    weekdays = pd.Series(
        np.where(clean.index.dayofweek >= 5, "weekend", "weekday"),
        index=clean.index,
    )
    grouped_labels.append(("day_type", weekdays))
    seasons = pd.Series(
        np.select(
            [
                clean.index.month.isin([12, 1, 2]),
                clean.index.month.isin([3, 4, 5]),
                clean.index.month.isin([6, 7, 8]),
            ],
            ["winter", "spring", "summer"],
            default="autumn",
        ),
        index=clean.index,
    )
    grouped_labels.append(("season", seasons))
    calendar = calendar if calendar is not None else pd.DataFrame()
    for column, dimension in (
        ("is_holiday", "holiday"),
        ("is_vacation", "vacation"),
        ("is_study_week", "study_week"),
        ("is_exam_period", "exam_period"),
    ):
        if column not in calendar.columns:
            continue
        label_map = calendar[column].to_dict()
        values = dates.map(label_map)
        values = np.asarray(values, dtype=object)
        known = pd.notna(values)
        label_values = np.full(len(clean), None, dtype=object)
        label_values[known] = np.where(values[known], "yes", "no")
        labels = pd.Series(label_values, index=clean.index)
        grouped_labels.append((dimension, labels))
    group_rows = []
    for dimension, labels in grouped_labels:
        group_rows.extend(_group_statistics(clean, dimension, labels))
    group_columns = ["dimension", "category", "intervals", "rows_with_imputation"]
    for column in PHYSICAL_COLUMNS:
        group_columns.extend(
            [f"{column}_valid_intervals"]
            + [f"{column}_{statistic}" for statistic in ("mean", "median", "std", "var", "min", "max")]
            + [f"{column}_observed_intervals"]
            + [f"{column}_observed_{statistic}" for statistic in ("mean", "std", "var")]
        )
    behavior_by_group = pd.DataFrame(group_rows, columns=group_columns)

    daily_rows = []
    for day, frame in clean.groupby(dates, sort=True):
        day_power = frame["pmax_kw"]
        available_power_intervals = int(day_power.notna().sum())
        imputed_power_intervals = int(
            (frame["imputed_pmax_kw"] & day_power.notna()).sum()
            if "imputed_pmax_kw" in frame
            else 0
        )
        active = day_power.gt(activity_threshold_kw)
        active_times = frame.index[active.fillna(False)]
        observed_day_power = day_power.mask(
            frame.get("imputed_pmax_kw", pd.Series(False, index=frame.index)).fillna(
                False
            )
        )
        observed_active = observed_day_power.gt(activity_threshold_kw)
        observed_active_times = frame.index[observed_active.fillna(False)]
        daily_rows.append(
            {
                "date": day,
                "energy_kwh": frame["delta_ea_kwh"].sum(min_count=1),
                "available_power_intervals": available_power_intervals,
                "observed_power_intervals": (
                    available_power_intervals - imputed_power_intervals
                ),
                "imputed_power_intervals": imputed_power_intervals,
                "active_intervals": int(active.sum()),
                "observed_active_intervals": int(observed_active.sum()),
                "coverage_percent": (
                    100
                    * (available_power_intervals - imputed_power_intervals)
                    / STEPS_PER_DAY
                ),
                "activation_time": active_times[0] if len(active_times) else pd.NaT,
                "deactivation_time": (
                    active_times[-1] + INTERVAL if len(active_times) else pd.NaT
                ),
                "active_duration_hours": (
                    active.sum() * INTERVAL.total_seconds() / 3600
                    if available_power_intervals
                    else np.nan
                ),
                "observed_active_duration_hours": (
                    observed_active.sum() * INTERVAL.total_seconds() / 3600
                    if available_power_intervals
                    else np.nan
                ),
                "pmax_mean_kw": day_power.mean(),
                "pmax_std_kw": day_power.std(),
                "pmax_variance_kw2": day_power.var(),
                "pmax_observed_mean_kw": observed_day_power.mean(),
                "pmax_observed_std_kw": observed_day_power.std(),
                "pmax_observed_variance_kw2": observed_day_power.var(),
                "smax_mean_kva": frame["smax_kva"].mean(),
                "smax_std_kva": frame["smax_kva"].std(),
                "smax_variance_kva2": frame["smax_kva"].var(),
                "activation_time_observed": (
                    observed_active_times[0]
                    if len(observed_active_times)
                    else pd.NaT
                ),
                "deactivation_time_observed": (
                    observed_active_times[-1] + INTERVAL
                    if len(observed_active_times)
                    else pd.NaT
                ),
            }
        )
    activity_by_day = pd.DataFrame(daily_rows)

    global_summary = {}
    for column in PHYSICAL_COLUMNS:
        values = clean[column].dropna()
        imputed_column = f"imputed_{column}"
        if imputed_column in clean:
            observed_values = clean[column].mask(
                clean[imputed_column].fillna(False)
            ).dropna()
        else:
            observed_values = values
        global_summary[f"{column}_mean"] = values.mean()
        global_summary[f"{column}_std"] = values.std()
        global_summary[f"{column}_variance"] = values.var()
        global_summary[f"{column}_min"] = values.min() if len(values) else np.nan
        global_summary[f"{column}_min_time"] = (
            values.idxmin() if len(values) else pd.NaT
        )
        global_summary[f"{column}_max"] = values.max() if len(values) else np.nan
        global_summary[f"{column}_max_time"] = (
            values.idxmax() if len(values) else pd.NaT
        )
        global_summary[f"{column}_p05"] = (
            values.quantile(0.05) if len(values) else np.nan
        )
        global_summary[f"{column}_observed_count"] = int(len(observed_values))
        global_summary[f"{column}_observed_mean"] = (
            observed_values.mean() if len(observed_values) else np.nan
        )
        global_summary[f"{column}_observed_std"] = (
            observed_values.std() if len(observed_values) else np.nan
        )
        global_summary[f"{column}_observed_variance"] = (
            observed_values.var() if len(observed_values) else np.nan
        )
        global_summary[f"{column}_observed_min"] = (
            observed_values.min() if len(observed_values) else np.nan
        )
        global_summary[f"{column}_observed_min_time"] = (
            observed_values.idxmin() if len(observed_values) else pd.NaT
        )
        global_summary[f"{column}_observed_max"] = (
            observed_values.max() if len(observed_values) else np.nan
        )
        global_summary[f"{column}_observed_max_time"] = (
            observed_values.idxmax() if len(observed_values) else pd.NaT
        )
        global_summary[f"{column}_observed_p05"] = (
            observed_values.quantile(0.05) if len(observed_values) else np.nan
        )
    (
        daily_activation_mean,
        daily_activation_std,
        daily_activation_variance,
    ) = _circular_clock_statistics(activity_by_day["activation_time"])
    (
        daily_deactivation_mean,
        daily_deactivation_std,
        daily_deactivation_variance,
    ) = _circular_clock_statistics(activity_by_day["deactivation_time"])
    (
        observed_activation_mean,
        observed_activation_std,
        observed_activation_variance,
    ) = _circular_clock_statistics(activity_by_day["activation_time_observed"])
    (
        observed_deactivation_mean,
        observed_deactivation_std,
        observed_deactivation_variance,
    ) = _circular_clock_statistics(activity_by_day["deactivation_time_observed"])
    global_summary.update(
        {
            "energy_total_kwh": energy.sum(min_count=1),
            "rows_with_any_imputed_physical_value": int(
                clean["was_imputed"].sum() if "was_imputed" in clean else 0
            ),
            "rows_with_unresolved_physical_missing": int(
                clean["has_missing_physical"].sum()
                if "has_missing_physical" in clean
                else 0
            ),
            "peak_quantile": peak_quantile,
            "peak_threshold_kw": peak_threshold,
            "peak_intervals": int(peak_mask.sum()),
            "peak_duration_hours": int(peak_mask.sum())
            * INTERVAL.total_seconds()
            / 3600,
            "peak_episodes": int(len(peak_episodes)),
            "peak_threshold_including_imputation_kw": all_peak_threshold,
            "peak_intervals_including_imputation": int(all_peak_mask.sum()),
            "peak_duration_including_imputation_hours": (
                int(all_peak_mask.sum()) * INTERVAL.total_seconds() / 3600
            ),
            "imputed_peak_intervals_at_observed_threshold": int(
                (power.ge(peak_threshold) & imputed_power).sum()
                if np.isfinite(peak_threshold)
                else 0
            ),
            "activity_threshold_kw": activity_threshold_kw,
            "mean_daily_active_duration_hours": activity_by_day[
                "active_duration_hours"
            ].mean(),
            "mean_daily_observed_active_duration_hours": activity_by_day[
                "observed_active_duration_hours"
            ].mean(),
            "activation_time_circular_mean_hour": daily_activation_mean,
            "activation_time_circular_std_hours": daily_activation_std,
            "activation_time_circular_variance": daily_activation_variance,
            "deactivation_time_circular_mean_hour": daily_deactivation_mean,
            "deactivation_time_circular_std_hours": daily_deactivation_std,
            "deactivation_time_circular_variance": daily_deactivation_variance,
            "observed_activation_time_circular_mean_hour": observed_activation_mean,
            "observed_activation_time_circular_std_hours": observed_activation_std,
            "observed_activation_time_circular_variance": observed_activation_variance,
            "observed_deactivation_time_circular_mean_hour": observed_deactivation_mean,
            "observed_deactivation_time_circular_std_hours": observed_deactivation_std,
            "observed_deactivation_time_circular_variance": observed_deactivation_variance,
            "calendar_days_in_data": int(dates.nunique()),
            "calendar_days_labelled": _calendar_metadata(
                calendar, clean.index
            )["labelled_days_in_data"],
        }
    )
    behavior_summary = pd.DataFrame([global_summary])

    return {
        **energy_reports,
        "behavior_by_group.csv": behavior_by_group,
        "activity_by_day.csv": activity_by_day,
        "peak_episodes.csv": peak_episodes,
        "behavior_summary.csv": behavior_summary,
        **_build_quality_reports(
            clean,
            quality_report=quality_report,
            source_records=source_records,
        ),
    }


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


def validate_window_index(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    window_steps: int,
    horizon_steps: int,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    """Audit chronology, lookback length, split boundaries, and imputation leakage."""
    checks: dict[str, bool | int | str] = {
        "window_count": int(len(windows)),
        "configured_window_steps": int(window_steps),
        "configured_horizon_steps": int(horizon_steps),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "features": ",".join(FEATURE_COLUMNS),
        "regular_15_minute_grid": bool(
            len(frame) < 2 or frame.index.to_series().diff().iloc[1:].eq(INTERVAL).all()
        ),
    }
    if windows.empty:
        checks.update(
            {
                "lookback_length_valid": False,
                "horizon_length_valid": False,
                "split_boundaries_valid": False,
                "target_timestamps_disjoint": True,
                "targets_exclude_imputations": False,
                "no_future_imputation_reference": False,
            }
        )
        return pd.DataFrame([checks])

    input_starts = frame.index.get_indexer(pd.DatetimeIndex(windows["input_start"]))
    origins = frame.index.get_indexer(pd.DatetimeIndex(windows["origin"]))
    target_starts = frame.index.get_indexer(pd.DatetimeIndex(windows["target_start"]))
    target_ends = frame.index.get_indexer(pd.DatetimeIndex(windows["target_end"]))
    positions_valid = bool(
        (input_starts >= 0).all()
        and (origins >= 0).all()
        and (target_starts >= 0).all()
        and (target_ends >= 0).all()
    )
    checks["lookback_length_valid"] = positions_valid and bool(
        np.all(origins - input_starts + 1 == window_steps)
    )
    checks["horizon_length_valid"] = positions_valid and bool(
        np.all(target_starts == origins + 1)
        and np.all(target_ends - target_starts + 1 == horizon_steps)
    )
    train_end, validation_end = _split_bounds(
        len(frame), train_fraction, validation_fraction
    )
    split_valid = positions_valid
    if positions_valid:
        split_names = windows["split"].to_numpy()
        split_valid = bool(
            np.all(target_ends[split_names == "train"] < train_end)
            and np.all(target_starts[split_names == "validation"] >= train_end)
            and np.all(target_ends[split_names == "validation"] < validation_end)
            and np.all(target_starts[split_names == "test"] >= validation_end)
        )
    checks["split_boundaries_valid"] = split_valid
    target_sets = {}
    split_names = windows["split"].to_numpy()
    for split in ("train", "validation", "test"):
        mask = split_names == split
        indices = [
            np.arange(start, end + 1)
            for start, end in zip(target_starts[mask], target_ends[mask])
        ]
        target_sets[split] = (
            set(np.concatenate(indices).tolist()) if indices else set()
        )
    checks["target_timestamps_disjoint"] = not any(
        target_sets[left] & target_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )
    if positions_valid and "was_imputed" in frame:
        imputed_targets = (
            frame["was_imputed"]
            .fillna(False)
            .astype(int)
            .rolling(horizon_steps, min_periods=horizon_steps)
            .sum()
        )
        checks["targets_exclude_imputations"] = bool(
            (imputed_targets.iloc[target_ends].to_numpy() == 0).all()
        )
    else:
        checks["targets_exclude_imputations"] = "was_imputed column unavailable"
    if positions_valid and "imputation_future_boundary" in frame:
        boundary_values = pd.to_datetime(
            frame["imputation_future_boundary"], errors="coerce"
        ).array.asi8
        latest_boundary = np.full(
            len(frame), np.iinfo(np.int64).min, dtype=np.int64
        )
        max_positions: deque[int] = deque()
        for position, boundary in enumerate(boundary_values):
            while max_positions and max_positions[0] <= position - window_steps:
                max_positions.popleft()
            while (
                max_positions
                and boundary_values[max_positions[-1]] <= boundary
            ):
                max_positions.pop()
            max_positions.append(position)
            if position >= window_steps - 1:
                latest_boundary[position] = boundary_values[max_positions[0]]
        checks["no_future_imputation_reference"] = bool(
            (latest_boundary[origins] <= frame.index.asi8[origins]).all()
        )
    else:
        checks["no_future_imputation_reference"] = (
            "imputation_future_boundary column unavailable"
        )
    return pd.DataFrame([checks])


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
    parser.add_argument(
        "--calendar",
        "--holidays",
        dest="calendar",
        type=Path,
        help=(
            "Optional daily CSV/XLSX calendar with date and optional is_holiday, "
            "is_vacation, and is_study_week columns"
        ),
    )
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
    calendar = load_behavior_calendar(args.calendar)
    holiday_dates = (
        set(calendar.index[calendar["is_holiday"].fillna(False)].date)
        if "is_holiday" in calendar
        else set()
    )
    clean, report = prepare_dataset(
        records,
        audit,
        max_short_gap=args.max_short_gap,
        holiday_dates=holiday_dates,
    )
    behavior_reports = build_behavior_reports(
        clean,
        calendar,
        quality_report=report,
        source_records=records,
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
    window_validation = validate_window_index(
        clean,
        window_index,
        window_steps=args.window_steps,
        horizon_steps=args.horizon_steps,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    validation_row = window_validation.iloc[0]
    required_checks = (
        "regular_15_minute_grid",
        "lookback_length_valid",
        "horizon_length_valid",
        "split_boundaries_valid",
        "target_timestamps_disjoint",
        "targets_exclude_imputations",
        "no_future_imputation_reference",
    )
    failed_checks = [
        check for check in required_checks if not bool(validation_row[check])
    ]
    if failed_checks:
        raise ValueError(f"Window validation failed: {failed_checks}")
    clean.to_parquet(args.output / "sinert_clean.parquet", index=True)
    window_index.to_parquet(args.output / "window_index.parquet", index=False)
    report_output = args.output / "reports"
    report_output.mkdir(parents=True, exist_ok=True)
    for filename, frame in behavior_reports.items():
        frame.to_csv(report_output / filename, index=False)
    window_validation.to_csv(report_output / "window_validation.csv", index=False)
    visualization_files = write_phase1_visualizations(
        clean, behavior_reports, args.output
    )
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
    report["window_validation"] = validation_row.to_dict()
    report["clean_data_file"] = "sinert_clean.parquet"
    report["behavior_report_files"] = {
        "directory": "reports",
        "files": sorted([*behavior_reports, "window_validation.csv"]),
    }
    report["visualization_files"] = visualization_files
    report["behavior_calendar"] = {
        "provided": args.calendar is not None,
        "file": str(args.calendar) if args.calendar is not None else None,
        **_calendar_metadata(calendar, clean.index),
    }
    with (args.output / "quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=str)
    print(
        f"Done: {len(clean):,} intervals, {len(window_index):,} valid windows. "
        f"Outputs: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
