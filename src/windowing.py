"""Construction de fenêtres temporelles pour les modèles séquentiels."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_sliding_windows(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    timestamp_column: str = "timestamp",
    window_size: int = 672,
    stride: int = 96,
    horizon: int = 1,
    target_column: str | None = None,
    target_columns: Sequence[str] | None = None,
    frequency: str = "15min",
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Construit des fenêtres contiguës et leur cible optionnelle.

    Une fenêtre n'est créée que si tous ses pas sont espacés de ``frequency``.
    Le ``stride`` de 96 produit une fenêtre par jour tout en conservant les
    672 pas historiques, soit exactement sept jours de contexte.
    """

    if window_size < 1 or stride < 1 or horizon < 1:
        raise ValueError("window_size, stride et horizon doivent être positifs")

    if target_column is not None and target_columns is not None:
        raise ValueError("Utiliser target_column ou target_columns, pas les deux")

    columns = list(feature_columns)
    output_columns = list(target_columns or [])
    if target_column is not None:
        output_columns = [target_column]
    required = [timestamp_column, *columns]
    required.extend(output_columns)
    missing_columns = [column for column in required if column not in frame]
    if missing_columns:
        raise KeyError(f"Colonnes introuvables : {missing_columns}")

    working = frame[required].copy()
    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
    )
    working = (
        working.dropna(subset=[timestamp_column])
        .sort_values(timestamp_column)
        .drop_duplicates(subset=[timestamp_column])
        .reset_index(drop=True)
    )

    values = working[columns].to_numpy(dtype=np.float32)
    target = (
        working[output_columns].to_numpy(dtype=np.float32)
        if output_columns
        else None
    )
    timestamps = working[timestamp_column].to_numpy()
    expected_delta = pd.Timedelta(frequency).to_timedelta64()
    time_delta = np.diff(timestamps)
    segment_start = np.r_[True, time_delta != expected_delta]
    segment_id = np.cumsum(segment_start) - 1

    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for current_segment in np.unique(segment_id):
        row_indices = np.flatnonzero(segment_id == current_segment)
        segment_values = values[row_indices]
        segment_target = target[row_indices] if target is not None else None
        segment_timestamps = timestamps[row_indices]
        max_start = len(row_indices) - window_size - horizon + 1
        for start in range(0, max_start, stride):
            end = start + window_size
            target_index = end + horizon - 1
            window = segment_values[start:end]
            if not np.isfinite(window).all():
                continue
            if segment_target is not None and not np.isfinite(segment_target[target_index]).all():
                continue

            windows.append(window)
            if segment_target is not None:
                targets.append(np.asarray(segment_target[target_index], dtype=np.float32))
            metadata.append(
                {
                    "window_start": pd.Timestamp(segment_timestamps[start]),
                    "window_end": pd.Timestamp(segment_timestamps[end - 1]),
                    "target_timestamp": pd.Timestamp(segment_timestamps[target_index]),
                    "segment_id": int(current_segment),
                    "window_size": window_size,
                    "horizon": horizon,
                }
            )

    if windows:
        window_array = np.stack(windows).astype(np.float32, copy=False)
    else:
        window_array = np.empty((0, window_size, len(columns)), dtype=np.float32)
    if target is None:
        target_array = None
    elif targets:
        target_array = np.stack(targets).astype(np.float32, copy=False)
        if target_array.shape[1] == 1:
            target_array = target_array[:, 0]
    else:
        target_array = np.empty((0, len(output_columns)), dtype=np.float32)
        if len(output_columns) == 1:
            target_array = target_array[:, 0]
    return window_array, target_array, pd.DataFrame(metadata)
