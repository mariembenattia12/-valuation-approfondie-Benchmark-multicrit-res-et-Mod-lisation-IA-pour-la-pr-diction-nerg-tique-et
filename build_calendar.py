"""Build a traceable daily calendar from reviewed event and source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_EVENTS = ROOT / "data" / "calendar_events.csv"
DEFAULT_OUTPUT = ROOT / "data" / "calendar_daily_features.csv"
DEFAULT_START = pd.Timestamp("2022-06-01")
DEFAULT_END = pd.Timestamp("2026-02-01")
DEFAULT_WORKBOOK = ROOT / "data" / "calendrier_complete_SUPCOM_2022-2027.xlsx"
DEFAULT_SOURCES = ROOT / "data" / "calendar_sources.csv"
DEFAULT_MISSING = ROOT / "data" / "calendar_missing_information.csv"
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
STATUS_METADATA = {
    "confirmé (sup'com)": (
        "SC_SUPCOM_CONFIRMED",
        "high",
        "Dates qualifiées de confirmées SUP'COM dans le classeur fourni.",
    ),
    "date fixe nationale": (
        "SC_NATIONAL_FIXED",
        "high",
        "Dates fixes nationales selon la légende du classeur.",
    ),
    "ministère (officiel)": (
        "SC_MINISTRY_REPORTED",
        "medium",
        "Dates présentées comme ministérielles/officielles par la légende; "
        "le classeur indique des circulaires relayées par la presse.",
    ),
    "source secondaire": (
        "SC_SECONDARY",
        "medium",
        "Date rapportée par une source secondaire selon la légende du classeur.",
    ),
    "probable (presse)": (
        "SC_PRESS_PROBABLE",
        "low",
        "Date signalée comme probable par la presse selon le classeur.",
    ),
    "estimation": (
        "SC_ESTIMATE",
        "low",
        "Date estimée; la légende précise qu'elle doit être confirmée par SUP'COM.",
    ),
    "hypothèse": (
        "SC_HYPOTHESIS",
        "low",
        "Date hypothétique, non confirmée par SUP'COM.",
    ),
}
TYPE_METADATA = {
    "férié": ("public_holiday", "is_holiday"),
    "vacances": ("university_vacation", "is_vacation"),
    "examens": ("exam_period", "is_exam_period"),
    "rentrée": ("academic_event", None),
    "fin d'enseignement": ("academic_event", None),
    "activité spéciale": ("special_activity", None),
    "événement": ("academic_event", None),
}


def build_calendar_daily_features(
    events: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Expand reviewed events to daily labels, leaving unknown dates unlabeled."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must not precede start_date")

    required = {
        "event_id",
        "event",
        "event_type",
        "start_date",
        "end_date",
        "recurrence_month_day",
        "source_id",
        "confidence",
        "notes",
    }
    missing_columns = required - set(events.columns)
    if missing_columns:
        raise ValueError(f"Event table is missing columns: {sorted(missing_columns)}")
    if events["event_id"].duplicated().any():
        raise ValueError("Event table contains duplicate event_id values")

    dates = pd.date_range(start, end, freq="D", name="date")
    year = dates.year - (dates.month < 9).astype(int)
    result = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "academic_year": [f"{value}-{value + 1}" for value in year],
            "day_of_week": dates.dayofweek,
            "is_weekend": dates.dayofweek >= 5,
            "is_holiday": pd.Series(pd.NA, index=range(len(dates)), dtype="object"),
            "is_exam_period": pd.Series(pd.NA, index=range(len(dates)), dtype="object"),
            "is_study_week": pd.Series(pd.NA, index=range(len(dates)), dtype="object"),
            "is_university_closure": pd.Series(
                pd.NA, index=range(len(dates)), dtype="object"
            ),
            "is_academic_day": pd.Series(pd.NA, index=range(len(dates)), dtype="object"),
            "is_vacation": pd.Series(pd.NA, index=range(len(dates)), dtype="object"),
            "event_name": "",
            "calendar_confidence": "unknown",
            "source_ids": "",
            "calendar_notes": "",
        }
    )
    date_positions = {date: position for position, date in enumerate(dates)}
    events_by_date: dict[pd.Timestamp, list[pd.Series]] = {}

    for _, event in events.iterrows():
        event_dates: list[pd.Timestamp] = []
        if pd.notna(event["recurrence_month_day"]):
            month_day = str(event["recurrence_month_day"]).strip()
            try:
                month, day = (int(value) for value in month_day.split("-", maxsplit=1))
                for event_year in range(start.year, end.year + 1):
                    try:
                        event_date = pd.Timestamp(
                            year=event_year, month=month, day=day
                        )
                    except ValueError:
                        continue
                    if start <= event_date <= end:
                        event_dates.append(event_date)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{event['event_id']}: invalid recurrence_month_day "
                    f"{month_day!r}"
                ) from error
        elif pd.notna(event["start_date"]):
            event_start = pd.to_datetime(event["start_date"], errors="coerce")
            event_end = pd.to_datetime(
                event["end_date"] if pd.notna(event["end_date"]) else event_start,
                errors="coerce",
            )
            if pd.isna(event_start) or pd.isna(event_end):
                raise ValueError(f"{event['event_id']}: invalid event date")
            if event_end < event_start:
                raise ValueError(f"{event['event_id']}: end_date precedes start_date")
            clipped_start = max(start, event_start.normalize())
            clipped_end = min(end, event_end.normalize())
            if clipped_start <= clipped_end:
                event_dates = list(pd.date_range(clipped_start, clipped_end, freq="D"))
        else:
            raise ValueError(
                f"{event['event_id']}: provide a start_date or recurrence_month_day"
            )

        for event_date in event_dates:
            events_by_date.setdefault(event_date, []).append(event)

    for event_date, daily_events in events_by_date.items():
        position = date_positions[event_date]
        names = sorted({str(event["event"]) for event in daily_events})
        source_ids = sorted({str(event["source_id"]) for event in daily_events})
        notes = sorted(
            {str(event["notes"]) for event in daily_events if pd.notna(event["notes"])}
        )
        source_statuses = sorted(
            {
                str(event["source_status"])
                for event in daily_events
                if "source_status" in event and pd.notna(event["source_status"])
            }
        )
        confidences = [
            str(event["confidence"]).casefold()
            for event in daily_events
            if pd.notna(event["confidence"])
        ]
        confidence = max(
            confidences,
            key=lambda value: CONFIDENCE_RANK.get(value, -1),
            default="unknown",
        )
        result.loc[position, "event_name"] = " | ".join(names)
        result.loc[position, "calendar_confidence"] = confidence
        result.loc[position, "source_ids"] = "|".join(source_ids)
        result.loc[position, "calendar_notes"] = " | ".join(notes)
        result.loc[position, "source_statuses"] = " | ".join(source_statuses)
        for event in daily_events:
            eligible = event.get("analysis_eligible", True)
            if pd.notna(eligible) and str(eligible).strip().casefold() in {
                "false",
                "0",
                "no",
            }:
                continue
            if event["event_type"] == "public_holiday":
                result.loc[position, "is_holiday"] = True
            elif event["event_type"] == "university_vacation":
                result.loc[position, "is_vacation"] = True
            elif event["event_type"] == "exam_period":
                result.loc[position, "is_exam_period"] = True
            elif event["event_type"] == "university_closure":
                result.loc[position, "is_university_closure"] = True

    return result


def load_supcom_calendar_events(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize the supplied SUP'COM schedule while retaining its evidence status."""
    calendar = pd.read_excel(path, sheet_name="Calendrier")
    required = {"Année univ.", "Début", "Fin", "Type", "Libellé", "Statut"}
    missing = required - set(calendar.columns)
    if missing:
        raise ValueError(f"{path}: calendar sheet is missing columns: {sorted(missing)}")
    calendar = calendar.dropna(how="all").copy()
    events = []
    sources = []
    source_ids_seen = set()
    for row_number, row in calendar.iterrows():
        event_type_label = str(row["Type"]).strip()
        normalized_type = event_type_label.casefold()
        if normalized_type not in TYPE_METADATA:
            raise ValueError(
                f"{path}: unsupported event type at spreadsheet row "
                f"{row_number + 2}: {event_type_label!r}"
            )
        status = str(row["Statut"]).strip()
        status_metadata = STATUS_METADATA.get(status.casefold())
        if status_metadata is None:
            raise ValueError(
                f"{path}: unsupported status at spreadsheet row "
                f"{row_number + 2}: {status!r}"
            )
        start = pd.to_datetime(row["Début"], errors="coerce")
        end = pd.to_datetime(row["Fin"], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            raise ValueError(
                f"{path}: invalid event dates at spreadsheet row {row_number + 2}"
            )
        start = start.normalize()
        end = end.normalize()
        if end < start:
            raise ValueError(
                f"{path}: event end precedes start at spreadsheet row {row_number + 2}"
            )

        source_id, confidence, source_description = status_metadata
        canonical_type, _ = TYPE_METADATA[normalized_type]
        events.append(
            {
                "event_id": f"SC{row_number + 1:04d}",
                "academic_year": str(row["Année univ."]).strip(),
                "event": str(row["Libellé"]).strip(),
                "event_type": canonical_type,
                "original_event_type": event_type_label,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "recurrence_month_day": pd.NA,
                "source_id": source_id,
                "source_status": status,
                "confidence": confidence,
                "analysis_eligible": confidence != "low",
                "notes": source_description,
            }
        )
        if source_id not in source_ids_seen:
            sources.append(
                {
                    "source_id": source_id,
                    "organization": "SUP'COM / source décrite dans le classeur fourni",
                    "title": f"Calendrier SUP'COM 2022-2027 - {status}",
                    "url": "",
                    "publication_date": "",
                    "date_consulted": pd.Timestamp.now().date().isoformat(),
                    "source_type": status,
                    "reliability": confidence,
                    "description": source_description,
                    "source_file": path.name,
                }
            )
            source_ids_seen.add(source_id)

    return pd.DataFrame(events), pd.DataFrame(sources)


def _resolve_calendar_range(
    start: str | None, end: str | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if (start is None) != (end is None):
        raise ValueError("Provide both --start and --end, or neither")
    if start is not None and end is not None:
        return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()

    processed_data = ROOT / "data" / "processed" / "sinert_clean.parquet"
    if not processed_data.exists():
        raise FileNotFoundError(
            "Calendar range is not specified and the processed SINERT Parquet "
            f"file is unavailable: {processed_data}"
        )
    index = pd.read_parquet(processed_data).index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError(f"{processed_data}: expected a non-empty datetime index")
    return index.min().normalize(), index.max().normalize()


def build_missing_information(events: pd.DataFrame) -> pd.DataFrame:
    """Record calendar gaps that prevent a complete academic activity comparison."""
    low_confidence_count = int(events["confidence"].eq("low").sum())
    return pd.DataFrame(
        [
            {
                "missing_id": "M01",
                "academic_year": "all",
                "missing_information": "Semaines d'études et de révision",
                "why_missing": (
                    "Le classeur ne contient aucune période de type semaine d'études; "
                    "is_study_week reste inconnu."
                ),
                "impact": (
                    "Impossible de comparer les moyennes entre semaines d'études "
                    "et autres périodes."
                ),
                "action_required": "Obtenir les calendriers SUP'COM détaillés par année.",
                "priority": "high",
            },
            {
                "missing_id": "M02",
                "academic_year": "2022-2026",
                "missing_information": "Vacances d'été et bornes complètes des périodes de cours",
                "why_missing": (
                    "Le classeur décrit des événements et congés sélectionnés, mais "
                    "ne fournit pas les bornes complètes des périodes d'activité."
                ),
                "impact": (
                    "Les jours sans événement ne peuvent pas être classés comme "
                    "jours de cours avec certitude."
                ),
                "action_required": "Faire confirmer les calendriers académiques complets par la scolarité.",
                "priority": "high",
            },
            {
                "missing_id": "M03",
                "academic_year": "2022-2026",
                "missing_information": "Confirmation SUP'COM des événements estimés ou hypothétiques",
                "why_missing": (
                    f"{low_confidence_count} lignes du classeur sont marquées "
                    "Estimation, Hypothèse ou Probable (presse)."
                ),
                "impact": (
                    "Ces périodes sont conservées dans le registre d'événements, "
                    "mais exclues des comparaisons et des variables de jours fériés."
                ),
                "action_required": "Remplacer les dates estimées par des dates confirmées par SUP'COM.",
                "priority": "high",
            },
            {
                "missing_id": "M04",
                "academic_year": "all",
                "missing_information": "Liens et références primaires par événement",
                "why_missing": (
                    "La légende du classeur décrit les catégories de sources mais "
                    "ne fournit pas d'URL ou de référence primaire pour chaque ligne."
                ),
                "impact": "Les dates ne peuvent pas toutes être vérifiées indépendamment.",
                "action_required": "Ajouter les circulaires ou liens SUP'COM/MESRS dans le registre des sources.",
                "priority": "medium",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path)
    parser.add_argument(
        "--supcom-workbook",
        type=Path,
        help="Normalize the supplied SUP'COM calendar workbook.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources-output", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--missing-output", type=Path, default=DEFAULT_MISSING)
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    supcom_workbook = args.supcom_workbook
    if supcom_workbook is None and args.events is None and DEFAULT_WORKBOOK.exists():
        supcom_workbook = DEFAULT_WORKBOOK
    if supcom_workbook is not None:
        events, sources = load_supcom_calendar_events(supcom_workbook)
        start, end = _resolve_calendar_range(args.start, args.end)
        daily = build_calendar_daily_features(events, start, end)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(args.output, index=False)
        events.to_csv(DEFAULT_EVENTS, index=False)
        sources.to_csv(args.sources_output, index=False)
        build_missing_information(events).to_csv(args.missing_output, index=False)
    else:
        events_path = args.events or DEFAULT_EVENTS
        start = args.start or DEFAULT_START.strftime("%Y-%m-%d")
        end = args.end or DEFAULT_END.strftime("%Y-%m-%d")
        events = pd.read_csv(events_path, dtype="string")
        daily = build_calendar_daily_features(events, start, end)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(args.output, index=False)
    print(
        f"Built {len(daily):,} days from {start} to {end}; "
        f"{int(daily['is_holiday'].notna().sum()):,} days have holiday labels, "
        f"{int(daily['is_vacation'].notna().sum()):,} have vacation labels, and "
        f"{int(daily['is_exam_period'].notna().sum()):,} have exam labels. "
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()
