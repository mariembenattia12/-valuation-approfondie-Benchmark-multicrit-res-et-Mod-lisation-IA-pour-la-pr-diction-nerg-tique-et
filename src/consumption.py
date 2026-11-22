"""Extraction des indicateurs de consommation et de comportement."""

import unicodedata
from pathlib import Path

import pandas as pd


def compute_interval_energy(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    cumulative_column: str,
    output_column: str = "energy_kwh",
    frequency: str = "15min",
) -> pd.DataFrame:
    """Calcule l'énergie entre deux relevés consécutifs valides.

    Les différences qui traversent une période sans relevé sont exclues afin
    de ne pas attribuer plusieurs heures ou jours d'absence à un seul pas.
    """

    result = frame[[timestamp_column, cumulative_column]].copy()
    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )
    result = result.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
    result = result.drop_duplicates(subset=[timestamp_column])

    timestamps = result[timestamp_column]
    delta = result[cumulative_column].diff()
    expected_delta = pd.Timedelta(frequency)
    valid = timestamps.diff().eq(expected_delta) & delta.ge(0)
    result[output_column] = delta.where(valid)
    return result.reset_index(drop=True)


def aggregate_energy_consumption(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    cumulative_column: str,
    period: str,
) -> pd.DataFrame:
    """Agrège l'énergie mesurée par jour, semaine, mois ou année."""

    interval_energy = compute_interval_energy(
        frame,
        timestamp_column=timestamp_column,
        cumulative_column=cumulative_column,
    )
    return (
        interval_energy.set_index(timestamp_column)["energy_kwh"]
        .resample(period)
        .sum(min_count=1)
        .rename("consumption_kwh")
        .reset_index()
    )


def load_academic_breaks(
    calendar_path: str | Path,
    *,
    sheet_name: str = "Calendrier",
) -> list[tuple[str, str]]:
    """Lit les intervalles de vacances depuis le calendrier SUP'COM Excel."""

    calendar = pd.read_excel(calendar_path, sheet_name=sheet_name)

    def normalize(value: object) -> str:
        text = unicodedata.normalize("NFKD", str(value))
        return "".join(character for character in text if not unicodedata.combining(character)).lower()

    normalized_columns = {normalize(column): column for column in calendar.columns}
    start_column = next(
        column for name, column in normalized_columns.items() if name.startswith("debut")
    )
    end_column = next(
        column for name, column in normalized_columns.items() if name.startswith("fin")
    )
    type_column = next(
        column for name, column in normalized_columns.items() if name == "type"
    )
    vacation_rows = calendar.loc[
        calendar[type_column].map(normalize).eq("vacances")
    ].copy()
    vacation_rows[start_column] = pd.to_datetime(vacation_rows[start_column])
    vacation_rows[end_column] = pd.to_datetime(vacation_rows[end_column])
    return [
        (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        for start, end in zip(vacation_rows[start_column], vacation_rows[end_column])
    ]


def classify_behavior_periods(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    academic_breaks: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Ajoute les catégories jour/semaine universitaire/saison.

    ``academic_breaks`` doit contenir les intervalles réels des vacances
    universitaires validés par l'établissement. Une liste vide signifie que
    toutes les dates sont provisoirement classées en semaine d'études.
    """

    result = frame.copy()
    timestamp = pd.to_datetime(result[timestamp_column])
    day_of_week = timestamp.dt.day_of_week
    month = timestamp.dt.month

    result["day_type"] = day_of_week.map(
        lambda value: "weekend" if value >= 5 else "jour_semaine"
    )
    result["season"] = month.map(
        {
            12: "hiver",
            1: "hiver",
            2: "hiver",
            3: "printemps",
            4: "printemps",
            5: "printemps",
            6: "ete",
            7: "ete",
            8: "ete",
            9: "automne",
            10: "automne",
            11: "automne",
        }
    )
    result["academic_period"] = "semaine_etudes"

    for start, end in academic_breaks or []:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        in_break = timestamp.between(start_date, end_date)
        result.loc[in_break, "academic_period"] = "vacances_universitaires"

    return result


def summarize_behavior(
    frame: pd.DataFrame,
    *,
    group_column: str,
    active_power_column: str,
    apparent_power_column: str,
) -> pd.DataFrame:
    """Calcule moyenne, écart-type, variance, minimum et maximum par groupe."""

    summary = (
        frame.groupby(group_column)[[active_power_column, apparent_power_column]]
        .agg(["mean", "std", "var", "min", "max", "count"])
    )
    summary.columns = [f"{column}_{statistic}" for column, statistic in summary.columns]
    return summary.reset_index()


def extract_peak_episodes(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    active_power_column: str,
    threshold_quantile: float = 0.95,
    frequency: str = "15min",
) -> pd.DataFrame:
    """Retourne les épisodes où la puissance dépasse le seuil de pointe.

    Par défaut, le seuil est le 95e percentile de la puissance active. Deux
    points séparés par un trou temporel ne sont jamais considérés comme un
    même épisode.
    """

    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile doit être compris entre 0 et 1")

    working = frame[[timestamp_column, active_power_column]].copy()
    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
    )
    working = (
        working.dropna()
        .sort_values(timestamp_column)
        .drop_duplicates(subset=[timestamp_column])
        .reset_index(drop=True)
    )
    threshold = working[active_power_column].quantile(threshold_quantile)
    is_peak = working[active_power_column].ge(threshold)
    is_consecutive = working[timestamp_column].diff().eq(pd.Timedelta(frequency))
    episode_group = ((~is_peak) | (~is_consecutive)).cumsum()

    peaks = working.loc[is_peak].copy()
    peaks["episode"] = episode_group.loc[is_peak].to_numpy()
    episodes = (
        peaks.groupby("episode", as_index=False)
        .agg(
            start=(timestamp_column, "min"),
            end=(timestamp_column, "max"),
            points=(timestamp_column, "size"),
            peak_kw=(active_power_column, "max"),
        )
    )
    episodes["duration_hours"] = (
        episodes["points"] * pd.Timedelta(frequency).total_seconds() / 3600
    )
    episodes["threshold_kw"] = threshold
    return episodes.sort_values("start").reset_index(drop=True)
