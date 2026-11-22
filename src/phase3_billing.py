"""Exécution reproductible de la phase 3 : facturation STEG MT.

Le module compare la facture calculée sur les valeurs réelles du test avec
la facture calculée sur les prédictions de chaque modèle de la phase 2.
Comme la phase 2 produit une prédiction à chaque fenêtre (stride 96), le
résultat est explicitement marqué comme estimation sur l'échantillon prédit,
et non comme facture mensuelle complète à 15 minutes.
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
    from .tariff import (
        TariffConfig,
        calculate_interval_costs,
        calculate_monthly_bill,
    )
except ImportError:  # pragma: no cover - permet l'exécution directe du fichier
    from tariff import TariffConfig, calculate_interval_costs, calculate_monthly_bill


TARGET_COLUMNS = ["AEa", "AEr", "Pmax", "Smax"]


def _select_phase2_predictions(phase2_dir: Path) -> Path:
    """Sélectionne le fichier de prédictions Phase 2 le plus récent et valide.

    Les téléchargements depuis Kaggle peuvent créer des noms comme
    ``benchmark_predictions_all_models (1).csv``.  Le fichier canonique peut
    alors être plus ancien que ce téléchargement ; on choisit donc le dernier
    fichier dont l'en-tête contient toutes les colonnes nécessaires.
    """

    required = {
        "model",
        "split",
        "target_timestamp",
        *(f"{kind}_{target}" for kind in ("real", "pred") for target in TARGET_COLUMNS),
    }
    candidates = sorted(
        phase2_dir.glob("benchmark_predictions_all_models*.csv"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for candidate in candidates:
        columns = set(pd.read_csv(candidate, nrows=0).columns)
        if required.issubset(columns):
            return candidate
    expected = phase2_dir / "benchmark_predictions_all_models.csv"
    raise FileNotFoundError(
        f"Aucun fichier de prédictions Phase 2 valide dans {phase2_dir}. "
        f"Fichier attendu notamment : {expected}"
    )


def _load_official_tariff(config_path: Path) -> tuple[TariffConfig, float]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prices = config["energy_prices_tnd_per_kwh"]
    tariff = TariffConfig(
        peak_price=prices["peak_morning"],
        full_price=prices["full"],
        offpeak_price=prices["offpeak"],
        evening_peak_price=prices["peak_evening"],
        subscribed_power_price=config["power_charge_tnd_per_kw_month"],
        vat_rate=config["vat_rate"],
        municipal_surcharge=config["municipal_surcharge_tnd_per_kwh"],
        excess_reduction_rate=config.get("excess_reduction_rate", 0.10),
        tariff_mode=config["tariff_mode"],
    )
    return tariff, float(config["subscribed_power_kw"])


def _invoice_tariff() -> TariffConfig:
    """Configuration observée sur la facture de septembre 2026."""

    return TariffConfig(
        peak_price=0.291,
        full_price=0.291,
        offpeak_price=0.291,
        evening_peak_price=0.291,
        subscribed_power_price=5.0,
        vat_rate=0.19,
        municipal_surcharge=0.010,
        rit_charge=3.500,
        tariff_mode="uniforme_facture",
    )


def _series_frame(group: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = {
        "target_timestamp": "target_timestamp",
        f"{prefix}_AEa": "AEa",
        f"{prefix}_AEr": "AEr",
        f"{prefix}_Pmax": "Pmax",
        f"{prefix}_Smax": "Smax",
    }
    return group[list(columns)].rename(columns=columns).reset_index(drop=True)


def _bill_pair(
    real: pd.DataFrame,
    predicted: pd.DataFrame,
    tariff: TariffConfig,
    subscribed_power_kw: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_costs = calculate_interval_costs(real, tariff)
    predicted_costs = calculate_interval_costs(predicted, tariff)
    real_bill = calculate_monthly_bill(
        real_costs,
        tariff,
        active_power_column="Pmax",
        subscribed_power_kw=subscribed_power_kw,
    )
    predicted_bill = calculate_monthly_bill(
        predicted_costs,
        tariff,
        active_power_column="Pmax",
        subscribed_power_kw=subscribed_power_kw,
    )
    return real_bill, predicted_bill


def _save_invoice_reconciliation(output_dir: Path) -> pd.DataFrame:
    tariff = _invoice_tariff()
    invoice_row = pd.DataFrame(
        {
            "target_timestamp": [pd.Timestamp("2026-09-15 12:00")],
            "AEa": [101065.0],
            "AEr": [38162.0],
            "Pmax": [358.0],
            "Smax": [np.nan],
            "invoice_pf": [0.94],
        }
    )
    costs = calculate_interval_costs(
        invoice_row,
        tariff,
        power_factor_column="invoice_pf",
    )
    bill = calculate_monthly_bill(
        costs,
        tariff,
        active_power_column="Pmax",
        subscribed_power_kw=600.0,
    )
    expected = 38881.993
    output = bill.assign(
        invoice_number="61096187",
        observed_net_to_pay_tnd=expected,
    )
    output["reconciliation_error_tnd"] = output["total_bill_tnd"] - expected
    output.to_csv(output_dir / "phase3_facture_steg_reconciliation.csv", index=False)
    return output


def run_phase3(project_root: str | Path = ".") -> dict[str, pd.DataFrame]:
    """Exécute le calcul réel/prédit et écrit les livrables de la phase 3."""

    root = Path(project_root).resolve()
    output_dir = root / "results" / "resultats_phase3"
    output_dir.mkdir(parents=True, exist_ok=True)
    phase2_dir = root / "results" / "resultats_phase2"
    tariff_path = root / "config" / "tariff_steg_mt_2026.json"
    phase2_path = _select_phase2_predictions(phase2_dir)

    predictions = pd.read_csv(phase2_path)
    required = {
        "model",
        "split",
        "target_timestamp",
        *(f"{kind}_{target}" for kind in ("real", "pred") for target in TARGET_COLUMNS),
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"Colonnes manquantes dans les prédictions : {missing}")

    test = predictions.loc[predictions["split"].eq("test")].copy()
    test["target_timestamp"] = pd.to_datetime(test["target_timestamp"], errors="coerce")
    test = test.dropna(subset=["target_timestamp"])
    official_tariff, subscribed_power_kw = _load_official_tariff(tariff_path)
    tariff_metadata = json.loads(tariff_path.read_text(encoding="utf-8"))
    expected_invoice_month = tariff_metadata["invoice_reference"]["month"]
    available_prediction_months = set(test["target_timestamp"].dt.to_period("M").astype(str))
    invoice_period_available = expected_invoice_month in available_prediction_months

    timestamp_steps = (
        test[["model", "target_timestamp"]]
        .drop_duplicates()
        .sort_values(["model", "target_timestamp"])
        .groupby("model")["target_timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
    )
    median_step_minutes = float(timestamp_steps.median()) if not timestamp_steps.empty else np.nan
    coverage_ratio = float(15 / median_step_minutes) if median_step_minutes > 0 else np.nan
    is_complete_15min = bool(
        np.isfinite(median_step_minutes) and abs(median_step_minutes - 15.0) < 1e-9
    )

    summary_rows = []
    monthly_rows = []
    detail_rows = []
    for model_name, group in test.groupby("model", sort=True):
        group = group.sort_values("target_timestamp").reset_index(drop=True)
        real = _series_frame(group, "real")
        predicted = _series_frame(group, "pred")
        real_bill, predicted_bill = _bill_pair(
            real,
            predicted,
            official_tariff,
            subscribed_power_kw,
        )
        real_total = float(real_bill["total_bill_tnd"].sum())
        predicted_total = float(predicted_bill["total_bill_tnd"].sum())
        absolute_error = abs(predicted_total - real_total)
        relative_error = 100 * absolute_error / max(abs(real_total), 1e-12)
        summary_rows.append(
            {
                "model": model_name,
                "tariff_mode": official_tariff.tariff_mode,
                "real_bill_tnd": real_total,
                "predicted_bill_tnd": predicted_total,
                "absolute_error_tnd": absolute_error,
                "relative_error_pct": relative_error,
                "test_points": len(group),
                "median_step_minutes": median_step_minutes,
                "estimated_15min_coverage_ratio": coverage_ratio,
                "billing_scope": (
                    "phase2_test_predictions_complete_15min"
                    if is_complete_15min
                    else "phase2_test_predictions_sparse_windows"
                ),
            }
        )
        for kind, bill in (("real", real_bill), ("predicted", predicted_bill)):
            monthly_rows.append(bill.assign(model=model_name, series=kind))
        for kind, frame in (("real", real), ("predicted", predicted)):
            costs = calculate_interval_costs(frame, official_tariff)
            detail_rows.append(costs.assign(model=model_name, series=kind))

    summary = pd.DataFrame(summary_rows).sort_values("relative_error_pct")
    monthly = pd.concat(monthly_rows, ignore_index=True)
    detail = pd.concat(detail_rows, ignore_index=True)
    summary.to_csv(output_dir / "phase3_facturation_resume.csv", index=False)
    monthly.to_csv(output_dir / "phase3_facturation_mensuelle.csv", index=False)
    detail.to_csv(output_dir / "phase3_facturation_detail.csv", index=False)

    figure, axis = plt.subplots(figsize=(14, 7))
    positions = np.arange(len(summary))
    width = 0.38
    axis.bar(positions - width / 2, summary["real_bill_tnd"], width, label="Réel")
    axis.bar(positions + width / 2, summary["predicted_bill_tnd"], width, label="Prédit")
    axis.set_xticks(positions, summary["model"], rotation=45, ha="right")
    axis.set_ylabel("Facture estimée (TND)")
    axis.set_title("Phase 3 — Facturation réelle vs prédite")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "phase3_facturation_reel_vs_predit.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(summary["model"], summary["relative_error_pct"], color="#d97706")
    axis.set_ylabel("Écart financier relatif (%)")
    axis.set_title("Phase 3 — Écart financier par modèle")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "phase3_ecart_financier_pct.png", dpi=160)
    plt.close(figure)

    reconciliation = _save_invoice_reconciliation(output_dir)
    configuration = {
        "tariff_config": str(tariff_path),
        "source_predictions": str(phase2_path),
        "official_tariff_mode": official_tariff.tariff_mode,
        "subscribed_power_kw": subscribed_power_kw,
        "test_models": sorted(test["model"].unique().tolist()),
        "test_points_per_model": int(test.groupby("model").size().min()),
        "median_step_minutes": median_step_minutes,
        "estimated_15min_coverage_ratio": coverage_ratio,
        "complete_15min_billing": is_complete_15min,
        "expected_invoice_month": expected_invoice_month,
        "invoice_period_available_in_predictions": invoice_period_available,
        "prediction_date_min": str(test["target_timestamp"].min()),
        "prediction_date_max": str(test["target_timestamp"].max()),
        "warning": (
            "Les prédictions phase 2 ont un stride de 96; cette estimation ne remplace pas une facture complète à 15 minutes."
            if not is_complete_15min
            else ""
        ),
        "invoice_reconciliation_error_tnd": float(reconciliation["reconciliation_error_tnd"].iloc[0]),
    }
    (output_dir / "phase3_facturation_configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "monthly": monthly,
        "detail": detail,
        "invoice_reconciliation": reconciliation,
    }


if __name__ == "__main__":  # pragma: no cover
    result = run_phase3(Path(__file__).resolve().parents[1])
    print(result["summary"].to_string(index=False))
    print(result["invoice_reconciliation"].to_string(index=False))
