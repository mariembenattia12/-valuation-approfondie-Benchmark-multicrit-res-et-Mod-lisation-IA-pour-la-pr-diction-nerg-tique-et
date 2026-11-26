"""Chargement des exports SINERT placés dans ``data/raw``.

Les exports observés sont organisés par mois, utilisent ``;`` comme séparateur
CSV et représentent les valeurs absentes par ``---``. Les fichiers contiennent
beaucoup de colonnes ; le chargeur permet donc de sélectionner uniquement les
mesures nécessaires.
"""

from pathlib import Path
import re

import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".parquet", ".xlsx", ".xls"}
MISSING_VALUES = ["---", ""]

SINERT_MAIN_COLUMNS = [
    "Date",
    "Time",
    "EM_VIRTUAL kWh (kWh) [ABS]",
    "EM_VIRTUAL kvarh (kvarh) [ABS]",
    "EM_VIRTUAL kW dmd (kW) [MAX]",
    "EM_VIRTUAL kVA dmd (kVA) [MAX]",
    "EM_VIRTUAL kW sys (kW) [AVG]",
    "EM_VIRTUAL kVA sys (kVA) [AVG]",
]


def list_data_files(raw_dir: str | Path = "data/raw") -> list[Path]:
    """Retourne les fichiers de données supportés présents dans ``raw_dir``."""

    directory = Path(raw_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Dossier introuvable : {directory}")
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _strip_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Supprime les espaces superflus autour des noms de colonnes."""

    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def _is_requested_column(column: object, wanted: set[str]) -> bool:
    """Accepte les noms CSV et les suffixes numériques des fichiers Excel."""

    name = str(column).strip()
    return name in wanted or any(
        name.startswith(f"{requested} ")
        and name[len(requested) + 1 :].isdigit()
        for requested in wanted
    )


def _canonicalize_selected_columns(
    frame: pd.DataFrame,
    requested_columns: list[str] | None,
) -> pd.DataFrame:
    """Ramène les en-têtes Excel suffixés à leurs noms SINERT canoniques."""

    result = _strip_column_names(frame)
    if requested_columns is None:
        return result

    requested = [column.strip() for column in requested_columns]
    renamed = {}
    for column in result.columns:
        name = str(column).strip()
        for wanted in requested:
            if name == wanted or re.fullmatch(rf"{re.escape(wanted)} \d+", name):
                renamed[column] = wanted
                break
    return result.rename(columns=renamed)


def read_data_file(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Lit un fichier SINERT CSV, Parquet ou Excel selon son extension.

    ``columns`` contient les noms après suppression des espaces périphériques.
    Une sélection par fonction est utilisée afin de rester compatible avec les
    espaces présents dans les en-têtes bruts des exports CSV.
    """

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    usecols = None
    if columns is not None:
        wanted = {column.strip() for column in columns}
        usecols = lambda column: _is_requested_column(column, wanted)

    if suffix == ".csv":
        frame = pd.read_csv(
            file_path,
            sep=";",
            na_values=MISSING_VALUES,
            usecols=usecols,
            nrows=nrows,
            low_memory=False,
        )
    elif suffix == ".parquet":
        frame = pd.read_parquet(file_path, columns=columns)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(
            file_path,
            na_values=MISSING_VALUES,
            usecols=usecols,
            nrows=nrows,
        )
    else:
        raise ValueError(f"Format non supporté : {file_path.suffix}")

    return _canonicalize_selected_columns(frame, columns)


def _read_files(
    files: list[Path],
    *,
    columns: list[str] | None,
    include_source: bool,
) -> pd.DataFrame:
    """Lit plusieurs fichiers et conserve l'origine de chaque ligne."""

    frames = []
    for file_path in files:
        frame = read_data_file(file_path, columns=columns)
        if include_source:
            frame["source_file"] = str(file_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_raw_data(
    raw_dir: str | Path = "data/raw",
    *,
    columns: list[str] | None = None,
    include_source: bool = True,
) -> pd.DataFrame:
    """Charge récursivement les fichiers de données brutes compatibles.

    Pour les exports SINERT, il est recommandé de fournir ``columns`` ou
    d'utiliser :func:`load_sinert_measurements`, car les fichiers contiennent
    environ 1 500 colonnes.
    """

    files = list_data_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier CSV, Parquet ou Excel trouvé dans {Path(raw_dir)}"
        )
    return _read_files(
        files,
        columns=columns,
        include_source=include_source,
    )


def load_sinert_measurements(
    raw_dir: str | Path = "data/raw/SINERT_DATA_CONCENTRATOR",
    *,
    include_source: bool = True,
) -> pd.DataFrame:
    """Charge les mesures principales utiles au projet SINERT."""

    frame = load_raw_data(
        raw_dir,
        columns=SINERT_MAIN_COLUMNS,
        include_source=include_source,
    )
    expected = [column.strip() for column in SINERT_MAIN_COLUMNS]
    if include_source:
        expected.append("source_file")
    return frame.reindex(columns=expected)
