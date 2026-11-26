"""Création des variables retardées, glissantes et calendaires."""

import numpy as np
import pandas as pd


def add_calendar_features(
    frame: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """Ajoute des variables calendaires et des encodages cycliques."""

    result = frame.copy()
    timestamp = pd.to_datetime(result[timestamp_column])
    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    day_of_week = timestamp.dt.dayofweek
    month = timestamp.dt.month

    result["hour"] = hour
    result["day_of_week"] = day_of_week
    result["month"] = month
    result["is_weekend"] = (day_of_week >= 5).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    return result


def add_lag_features(
    frame: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = (1, 4, 96, 672),
) -> pd.DataFrame:
    """Ajoute des retards de 15 minutes, 1 heure, 1 jour et 1 semaine."""

    result = frame.copy()
    for column in columns:
        for lag in lags:
            result[f"{column}_lag_{lag}"] = result[column].shift(lag)
    return result
