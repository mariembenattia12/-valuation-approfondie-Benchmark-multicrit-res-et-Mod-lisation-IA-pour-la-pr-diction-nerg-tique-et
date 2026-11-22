"""Fabriques de modèles et conventions communes du benchmark."""

from sklearn.ensemble import RandomForestRegressor


def build_random_forest(
    *,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Construit un modèle Random Forest reproductible."""

    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def seasonal_persistence(values, lag: int = 96):
    """Produit une prédiction naïve basée sur une saisonnalité passée."""

    return values.shift(lag)
