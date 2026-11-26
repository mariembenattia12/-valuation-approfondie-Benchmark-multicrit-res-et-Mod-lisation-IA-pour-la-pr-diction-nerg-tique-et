"""Outils de séparation temporelle et de benchmark des fenêtres énergétiques."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def temporal_split_indices(
    timestamps: Sequence[object],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    """Retourne des indices train/validation/test sans mélanger le temps."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction doit être compris entre 0 et 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction doit être positif ou nul")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction doit être inférieur à 1")

    parsed = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    if parsed.isna().any():
        raise ValueError("Les timestamps contiennent des dates invalides")
    order = np.argsort(parsed.to_numpy())
    n_samples = len(order)
    train_end = max(1, int(n_samples * train_fraction))
    validation_end = max(train_end + 1, int(n_samples * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, n_samples - 1)

    return {
        "train": order[:train_end],
        "validation": order[train_end:validation_end],
        "test": order[validation_end:],
    }


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """Aplati des fenêtres ``(n, pas, variables)`` pour les modèles sklearn."""

    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("windows doit avoir la forme (n_fenêtres, n_pas, n_variables)")
    return values.reshape(values.shape[0], -1)


def seasonal_persistence_forecast(
    known_timestamps: Sequence[object],
    known_values: Sequence[float],
    forecast_timestamps: Sequence[object],
    *,
    season: str = "1D",
) -> np.ndarray:
    """Prévoit une valeur saisonnière, avec repli sur la dernière valeur connue.

    Seules les observations explicitement passées dans ``known_values`` sont
    utilisées. Cela évite toute fuite d'information lors de la validation et du test.
    """

    known_dates = pd.to_datetime(pd.Series(known_timestamps), errors="raise")
    forecast_dates = pd.to_datetime(pd.Series(forecast_timestamps), errors="raise")
    values = np.asarray(known_values, dtype=np.float32)
    if len(known_dates) != len(values) or len(values) == 0:
        raise ValueError("Les observations connues doivent être non vides et alignées")

    lookup = dict(zip(known_dates, values, strict=True))
    fallback = float(values[-1])
    season_delta = pd.Timedelta(season)
    predictions: list[float] = []
    for timestamp in forecast_dates:
        seasonal_timestamp = timestamp - season_delta
        prediction = lookup.get(seasonal_timestamp, fallback)
        predictions.append(float(prediction))
    return np.asarray(predictions, dtype=np.float32)


def mean_forecast(known_values: Sequence[float], n_predictions: int) -> np.ndarray:
    """Construit une prévision constante égale à la moyenne d'entraînement."""

    values = np.asarray(known_values, dtype=np.float32)
    if values.size == 0 or n_predictions < 0:
        raise ValueError("Les valeurs connues doivent être non vides")
    return np.full(n_predictions, float(np.mean(values)), dtype=np.float32)
