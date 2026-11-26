"""Nettoyage et préparation temporelle des mesures énergétiques."""

import pandas as pd


def parse_timestamp(
    frame: pd.DataFrame,
    timestamp_column: str,
    *,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Convertit la colonne temporelle et trie les observations."""

    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
        dayfirst=dayfirst,
    )
    result = result.dropna(subset=[timestamp_column])
    return result.sort_values(timestamp_column).reset_index(drop=True)


def combine_date_and_time(
    frame: pd.DataFrame,
    *,
    date_column: str = "Date",
    time_column: str = "Time",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Construit un horodatage à partir des colonnes Date et Time de SINERT."""

    result = frame.copy()
    date_values = pd.to_datetime(
        result[date_column],
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()
    time_values = result[time_column].astype("string")
    is_next_day = time_values.str.startswith("24:", na=False)
    normalized_time = time_values.str.replace(r"^24:", "00:", regex=True)
    result[timestamp_column] = date_values + pd.to_timedelta(
        normalized_time,
        errors="coerce",
    ) + pd.to_timedelta(
        is_next_day.astype("int64"),
        unit="D",
    )
    return result


def add_estimated_power_factor(
    frame: pd.DataFrame,
    *,
    active_power_column: str = "EM_VIRTUAL kW sys (kW) [AVG]",
    apparent_power_column: str = "EM_VIRTUAL kVA sys (kVA) [AVG]",
    output_column: str = "power_factor_estimated",
) -> pd.DataFrame:
    """Estime le facteur de puissance à partir des puissances disponibles.

    Le facteur de puissance ``EM_VIRTUAL PF sys [AVG]`` étant vide dans les
    exports observés, on utilise ``abs(kW) / kVA`` lorsque ``kVA`` est non nul.
    Les valeurs sont bornées entre 0 et 1 pour limiter les artefacts numériques.
    """

    result = frame.copy()
    apparent_power = result[apparent_power_column].replace(0, pd.NA)
    result[output_column] = (
        result[active_power_column].abs()
        .div(apparent_power)
        .clip(lower=0, upper=1)
    )
    return result


def remove_duplicates(
    frame: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """Supprime les doublons temporels en conservant la première mesure."""

    return frame.drop_duplicates(subset=[timestamp_column]).reset_index(drop=True)


def impute_short_gaps(
    frame: pd.DataFrame,
    timestamp_column: str,
    *,
    columns: list[str] | None = None,
    max_consecutive_missing: int = 4,
) -> pd.DataFrame:
    """Interpole uniquement les petits trous temporels entre deux mesures.

    Les trous de plus de ``max_consecutive_missing`` points restent manquants.
    Cette règle évite de fabriquer plusieurs heures ou plusieurs jours de
    mesures lorsque l'export SINERT est incomplet.
    """

    if max_consecutive_missing < 1:
        raise ValueError("max_consecutive_missing doit être positif")

    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )
    result = result.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
    result = result.set_index(timestamp_column)

    target_columns = columns or list(result.select_dtypes(include="number").columns)
    missing_columns = [column for column in target_columns if column not in result]
    if missing_columns:
        raise KeyError(f"Colonnes introuvables pour l'imputation : {missing_columns}")

    for column in target_columns:
        series = result[column]
        missing = series.isna()
        groups = missing.ne(missing.shift()).cumsum()
        run_lengths = missing.groupby(groups).transform("sum")
        interpolated = series.interpolate(method="time", limit_area="inside")
        eligible = missing & (run_lengths <= max_consecutive_missing)
        result[column] = series.where(~eligible, interpolated)
    return result.reset_index()


def regularize_time_grid(
    frame: pd.DataFrame,
    timestamp_column: str,
    *,
    frequency: str = "15min",
) -> pd.DataFrame:
    """Réindexe les mesures sur une grille temporelle régulière."""

    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )
    result = result.dropna(subset=[timestamp_column])
    result = result.sort_values(timestamp_column).drop_duplicates(
        subset=[timestamp_column],
        keep="first",
    )
    result = result.set_index(timestamp_column)
    grid = pd.date_range(
        start=result.index.min().floor(frequency),
        end=result.index.max().ceil(frequency),
        freq=frequency,
    )
    return result.reindex(grid).rename_axis(timestamp_column).reset_index()


def resample_to_15_minutes(
    frame: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """Rééchantillonne les colonnes numériques sur une grille de 15 minutes."""

    result = frame.copy().set_index(timestamp_column)
    numeric = result.select_dtypes(include="number")
    return numeric.resample("15min").mean().reset_index()
