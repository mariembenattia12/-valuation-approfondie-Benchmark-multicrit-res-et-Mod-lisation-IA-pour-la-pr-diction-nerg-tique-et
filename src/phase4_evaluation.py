"""Évaluation multi-critères des modèles de la phase 2.

Les métriques sont calculées sur le split ``test`` de Phase 2. La MAPE exclut
les valeurs réelles nulles afin d'éviter une division artificielle par zéro.
La facture provient du résumé de Phase 3. Les ressources de calcul sont
rapportées seulement lorsqu'elles ont été exportées par Phase 2 ; aucune durée
d'entraînement n'est reconstruite à partir de la date des fichiers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .phase3_billing import _select_phase2_predictions
except ImportError:  # pragma: no cover
    from phase3_billing import _select_phase2_predictions


TARGETS = ["AEa", "AEr", "Pmax", "Smax"]


def _mape_without_zero(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    mask = np.abs(y_true) > 1e-6
    if not np.any(mask):
        return float("nan"), 0
    value = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return float(value), int(mask.sum())


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator <= 1e-12:
        return float("nan")
    return float(1 - np.sum((y_true - y_pred) ** 2) / denominator)


def _load_resources(phase2_dir: Path, models: list[str]) -> pd.DataFrame:
    candidates = sorted(
        phase2_dir.glob("benchmark_resource_usage*.csv"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if candidates:
        resources = pd.read_csv(candidates[0])
        if "model" in resources.columns:
            if "parameter_source" not in resources:
                resources["parameter_source"] = "phase2_export"
            return resources

    return pd.DataFrame(
        {
            "model": models,
            "parameter_count": np.nan,
            "parameter_source": "not_exported_by_phase2",
            "training_time_seconds": np.nan,
            "training_time_source": "not_exported_by_phase2",
            "resource_status": "not_exported",
        }
    )


def _normalise(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    minimum = values.min()
    maximum = values.max()
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum == minimum:
        return pd.Series(1.0, index=values.index)
    if higher_is_better:
        return (values - minimum) / (maximum - minimum)
    return (maximum - values) / (maximum - minimum)


def run_phase4(project_root: str | Path = ".") -> dict[str, pd.DataFrame]:
    root = Path(project_root).resolve()
    phase2_dir = root / "results" / "resultats_phase2"
    phase3_dir = root / "results" / "resultats_phase3"
    output_dir = root / "results" / "resultats_phase4"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = _select_phase2_predictions(phase2_dir)
    predictions = pd.read_csv(predictions_path, parse_dates=["target_timestamp"])
    test = predictions.loc[predictions["split"].eq("test")].copy()
    if test.empty:
        raise ValueError("Le fichier Phase 2 ne contient aucune prédiction de test.")

    detail_rows: list[dict[str, object]] = []
    for model, group in test.groupby("model", sort=True):
        for target in TARGETS:
            y_true = pd.to_numeric(group[f"real_{target}"], errors="coerce").to_numpy(dtype=float)
            y_pred = pd.to_numeric(group[f"pred_{target}"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[valid]
            y_pred = y_pred[valid]
            mape, mape_count = _mape_without_zero(y_true, y_pred)
            peak_mask = y_true >= np.quantile(y_true, 0.90)
            detail_rows.append(
                {
                    "model": model,
                    "target": target,
                    "n_test": len(y_true),
                    "mape_pct": mape,
                    "mape_nonzero_count": mape_count,
                    "mae": float(np.mean(np.abs(y_true - y_pred))),
                    "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                    "r2": _r2(y_true, y_pred),
                    "real_peak_p90_mean": float(np.mean(y_true[peak_mask])),
                    "predicted_peak_p90_mean": float(np.mean(y_pred[peak_mask])),
                }
            )

    detail = pd.DataFrame(detail_rows)
    aggregate = (
        detail.groupby("model", as_index=False)
        .agg(
            mape_pct=("mape_pct", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            test_points=("n_test", "min"),
        )
    )

    peak = detail.loc[detail["target"].isin(["Pmax", "Smax"])].copy()
    peak["pde_pct"] = (
        (peak["predicted_peak_p90_mean"] - peak["real_peak_p90_mean"]).abs()
        / peak["real_peak_p90_mean"].abs().clip(lower=1e-12)
        * 100
    )
    peak_wide = peak.pivot(index="model", columns="target", values="pde_pct").reset_index()
    peak_wide = peak_wide.rename(columns={"Pmax": "pde_pmax_pct", "Smax": "pde_smax_pct"})
    peak_wide["pde_pct"] = peak_wide[["pde_pmax_pct", "pde_smax_pct"]].mean(axis=1)
    aggregate = aggregate.merge(peak_wide, on="model", how="left")

    billing_path = phase3_dir / "phase3_facturation_resume.csv"
    billing = pd.read_csv(billing_path)
    billing = billing.rename(
        columns={
            "absolute_error_tnd": "afacture_abs_tnd",
            "relative_error_pct": "afacture_pct",
        }
    )
    aggregate = aggregate.merge(
        billing[["model", "afacture_abs_tnd", "afacture_pct", "billing_scope"]],
        on="model",
        how="left",
    )

    resources = _load_resources(phase2_dir, aggregate["model"].tolist())
    resources.to_csv(output_dir / "phase4_ressources.csv", index=False)
    aggregate = aggregate.merge(resources, on="model", how="left")

    criteria = {
        "mape_pct": False,
        "pde_pct": False,
        "afacture_pct": False,
        "rmse": False,
        "mae": False,
        "r2": True,
    }
    resource_criteria = ["training_time_seconds", "parameter_count"]
    for column in resource_criteria:
        if column in aggregate and aggregate[column].notna().all():
            criteria[column] = False
    for column, higher_is_better in criteria.items():
        aggregate[f"score_{column}"] = _normalise(
            aggregate[column], higher_is_better=higher_is_better
        )
    score_columns = [f"score_{column}" for column in criteria]
    aggregate["multi_criteria_score"] = aggregate[score_columns].mean(axis=1)
    aggregate["rank"] = aggregate["multi_criteria_score"].rank(
        method="min", ascending=False
    ).astype(int)
    ranking = aggregate.sort_values(["rank", "model"]).reset_index(drop=True)

    detail.to_csv(output_dir / "phase4_metriques_detail.csv", index=False)
    ranking.to_csv(output_dir / "phase4_classement.csv", index=False)

    figure, axis = plt.subplots(figsize=(13, 7))
    ordered = ranking.sort_values("multi_criteria_score", ascending=True)
    axis.barh(ordered["model"], ordered["multi_criteria_score"], color="#2563eb")
    axis.set_xlabel("Score multi-critères normalisé (plus haut = meilleur)")
    axis.set_title("Phase 4 — Classement multi-critères")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "phase4_classement.png", dpi=160)
    plt.close(figure)

    matrix = ranking.set_index("model")[score_columns].rename(
        columns={column: column.replace("score_", "") for column in score_columns}
    )
    figure, axis = plt.subplots(figsize=(11, 8))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_yticks(np.arange(len(matrix)), matrix.index)
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_title("Phase 4 — Scores normalisés par critère")
    figure.colorbar(image, ax=axis, label="Score")
    figure.tight_layout()
    figure.savefig(output_dir / "phase4_scores_normalises.png", dpi=160)
    plt.close(figure)

    configuration = {
        "source_predictions": str(predictions_path),
        "source_billing": str(billing_path),
        "split": "test",
        "models": sorted(test["model"].unique().tolist()),
        "criteria_used_for_ranking": list(criteria),
        "resource_criteria_excluded": [
            column for column in resource_criteria if column not in criteria
        ],
        "resource_note": (
            "Training time enters the ranking when available for every model. "
            "Parameter count is excluded when classical-model values are not comparable or unavailable."
        ),
    }
    (output_dir / "phase4_configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"detail": detail, "ranking": ranking, "resources": resources}


if __name__ == "__main__":  # pragma: no cover
    results = run_phase4(Path(__file__).resolve().parents[1])
    print(results["ranking"][["rank", "model", "multi_criteria_score"]].to_string(index=False))
