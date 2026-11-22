"""Métriques de régression et métriques spécifiques aux pics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcule MAE, RMSE, R² et MAPE avec protection contre les zéros."""

    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    denominator = np.maximum(np.abs(true), 1e-8)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)),
        "mape": float(np.mean(np.abs((true - pred) / denominator)) * 100),
    }
