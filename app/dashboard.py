"""Tableau de bord Streamlit SCADA pour le projet SUP'COM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "sinert_features.csv"
PHASE2_DIR = PROJECT_ROOT / "results" / "resultats_phase2"
PHASE3_DIR = PROJECT_ROOT / "results" / "resultats_phase3"
PHASE4_DIR = PROJECT_ROOT / "results" / "resultats_phase4"
TARIFF_PATH = PROJECT_ROOT / "config" / "tariff_steg_mt_2026.json"

RAW_AEA = "EM_VIRTUAL kWh (kWh) [ABS]"
RAW_AER = "EM_VIRTUAL kvarh (kvarh) [ABS]"
RAW_PMAX = "EM_VIRTUAL kW dmd (kW) [MAX]"
RAW_SMAX = "EM_VIRTUAL kVA dmd (kVA) [MAX]"
RAW_KW_AVG = "EM_VIRTUAL kW sys (kW) [AVG]"
RAW_KVA_AVG = "EM_VIRTUAL kVA sys (kVA) [AVG]"

METRIC_LABELS = {
    "AEa": "Énergie active — incrément (kWh)",
    "AEr": "Énergie réactive — incrément (kVARh)",
    "Pmax": "Demande active crête (kW)",
    "Smax": "Demande apparente crête (kVA)",
    "kW_avg": "Puissance active moyenne (kW)",
    "kVA_avg": "Puissance apparente moyenne (kVA)",
    "power_factor_estimated": "Facteur de puissance estimé",
}


st.set_page_config(
    page_title="SUP'COM — SCADA énergétique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_processed() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        return pd.DataFrame()
    columns = [
        "timestamp",
        RAW_AEA,
        RAW_AER,
        RAW_PMAX,
        RAW_SMAX,
        RAW_KW_AVG,
        RAW_KVA_AVG,
        "power_factor_estimated",
    ]
    frame = pd.read_csv(PROCESSED_PATH, usecols=lambda column: column in columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    for column in columns[1:]:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["AEa"] = frame[RAW_AEA].diff().clip(lower=0)
    frame["AEr"] = frame[RAW_AER].diff().clip(lower=0)
    frame["Pmax"] = frame[RAW_PMAX]
    frame["Smax"] = frame[RAW_SMAX]
    frame["kW_avg"] = frame[RAW_KW_AVG]
    frame["kVA_avg"] = frame[RAW_KVA_AVG]
    return frame.reset_index(drop=True)


def newest_file(pattern: str) -> Path | None:
    candidates = sorted(
        PHASE2_DIR.glob(pattern),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    path = newest_file("benchmark_predictions_all_models*.csv")
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["target_timestamp"])
    return frame.sort_values("target_timestamp")


@st.cache_data(show_spinner=False)
def load_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_tariff() -> dict:
    return json.loads(TARIFF_PATH.read_text(encoding="utf-8")) if TARIFF_PATH.exists() else {}


def format_tnd(value: object) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.2f} TND".replace(",", " ")


def metric_card_row(frame: pd.DataFrame, tariff: dict) -> None:
    subscribed = tariff.get("subscribed_power_kw", np.nan)
    max_power = frame["Pmax"].max() if not frame.empty else np.nan
    min_pf = frame["power_factor_estimated"].min() if not frame.empty else np.nan
    active_total = frame["AEa"].sum() if not frame.empty else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Énergie active disponible", f"{active_total:,.0f} kWh" if pd.notna(active_total) else "—")
    c2.metric("Pmax observée", f"{max_power:,.2f} kW" if pd.notna(max_power) else "—")
    c3.metric("Puissance souscrite", f"{subscribed:,.0f} kW" if pd.notna(subscribed) else "—")
    c4.metric("PF minimal estimé", f"{min_pf:.3f}" if pd.notna(min_pf) else "—")


def historical_module(frame: pd.DataFrame) -> None:
    st.header("Analyse historique")
    if frame.empty:
        st.warning("Le fichier traité est absent. Exécutez le prétraitement avant d'utiliser ce module.")
        return

    min_date = frame["timestamp"].min().date()
    max_date = frame["timestamp"].max().date()
    selected_dates = st.sidebar.date_input(
        "Période historique",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start, end = selected_dates
    else:
        start = end = selected_dates
    filtered = frame.loc[frame["timestamp"].dt.date.between(start, end)].copy()
    if filtered.empty:
        st.warning("Aucune mesure sur cette période.")
        return

    metric = st.selectbox("Paramètre à analyser", list(METRIC_LABELS), format_func=METRIC_LABELS.get)
    metric_card_row(filtered, load_tariff())

    filtered["minute_of_day"] = filtered["timestamp"].dt.hour * 60 + filtered["timestamp"].dt.minute
    profile = filtered.groupby("minute_of_day", as_index=False)[metric].mean()
    profile["heure"] = pd.to_datetime(profile["minute_of_day"], unit="m", origin="2026-01-01")
    profile = profile.rename(columns={metric: "moyenne"})
    fig = px.line(
        profile,
        x="heure",
        y="moyenne",
        title=f"Profil journalier moyen — {METRIC_LABELS[metric]}",
        labels={"heure": "Heure", "moyenne": METRIC_LABELS[metric]},
    )
    fig.update_xaxes(tickformat="%H:%M")
    st.plotly_chart(fig, use_container_width=True)

    timeline = filtered[["timestamp", metric]].dropna().copy()
    if len(timeline) > 20000:
        timeline = timeline.set_index("timestamp").resample("1h").mean().reset_index()
    fig = px.line(
        timeline,
        x="timestamp",
        y=metric,
        title=f"Évolution temporelle — {METRIC_LABELS[metric]}",
        labels={"timestamp": "Date", metric: METRIC_LABELS[metric]},
    )
    st.plotly_chart(fig, use_container_width=True)

    peak = filtered.loc[filtered[metric].idxmax()]
    trough = filtered.loc[filtered[metric].idxmin()]
    c1, c2 = st.columns(2)
    c1.success(f"Pic : {peak[metric]:,.3f} à {peak['timestamp']}")
    c2.info(f"Creux : {trough[metric]:,.3f} à {trough['timestamp']}")


def benchmark_module(predictions: pd.DataFrame) -> None:
    st.header("Arène de benchmarking IA")
    if predictions.empty:
        st.warning("Les prédictions Phase 2 sont absentes.")
        return
    test = predictions.loc[predictions["split"].eq("test")].copy()
    models = sorted(test["model"].dropna().unique())
    default_models = models[: min(3, len(models))]
    selected_models = st.multiselect("Modèles à superposer", models, default=default_models)
    target = st.selectbox("Sortie à visualiser", ["AEa", "AEr", "Pmax", "Smax"])
    if not selected_models:
        st.info("Sélectionnez au moins un modèle.")
        return

    fig = go.Figure()
    real = test.drop_duplicates("target_timestamp").sort_values("target_timestamp")
    fig.add_trace(
        go.Scatter(
            x=real["target_timestamp"],
            y=real[f"real_{target}"],
            name="Réel",
            mode="lines",
            line={"width": 3, "color": "#111827"},
        )
    )
    for model in selected_models:
        part = test.loc[test["model"].eq(model)].sort_values("target_timestamp")
        fig.add_trace(
            go.Scatter(
                x=part["target_timestamp"],
                y=part[f"pred_{target}"],
                name=model,
                mode="lines",
            )
        )
    fig.update_layout(
        title=f"Réel vs prédit — {target}",
        xaxis_title="Date cible",
        yaxis_title=METRIC_LABELS.get(target, target),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    ranking = load_csv(str(PHASE4_DIR / "phase4_classement.csv"))
    if not ranking.empty:
        st.subheader("Matrice des métriques")
        visible = [
            "rank", "model", "mape_pct", "pde_pct", "afacture_pct",
            "rmse", "mae", "r2", "multi_criteria_score",
        ]
        st.dataframe(
            ranking.loc[ranking["model"].isin(selected_models), visible]
            .sort_values("rank")
            .style.format(
                {
                    "mape_pct": "{:.2f}%", "pde_pct": "{:.2f}%",
                    "afacture_pct": "{:.4f}%", "rmse": "{:.3f}",
                    "mae": "{:.3f}", "r2": "{:.3f}",
                    "multi_criteria_score": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def financial_module() -> None:
    st.header("Module financier")
    summary = load_csv(str(PHASE3_DIR / "phase3_facturation_resume.csv"))
    monthly = load_csv(str(PHASE3_DIR / "phase3_facturation_mensuelle.csv"))
    if summary.empty:
        st.warning("Les résultats de facturation Phase 3 sont absents.")
        return

    chart_data = summary.melt(
        id_vars="model",
        value_vars=["real_bill_tnd", "predicted_bill_tnd"],
        var_name="série",
        value_name="facture_tnd",
    )
    chart_data["série"] = chart_data["série"].map({"real_bill_tnd": "Réelle", "predicted_bill_tnd": "Prédite"})
    fig = px.bar(
        chart_data,
        x="model",
        y="facture_tnd",
        color="série",
        barmode="group",
        title="Facturation réelle et prédite par modèle",
        labels={"facture_tnd": "Facture (TND)", "model": "Modèle"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        summary[["model", "real_bill_tnd", "predicted_bill_tnd", "absolute_error_tnd", "relative_error_pct"]]
        .style.format(
            {
                "real_bill_tnd": format_tnd, "predicted_bill_tnd": format_tnd,
                "absolute_error_tnd": format_tnd, "relative_error_pct": "{:.4f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if monthly.empty:
        return
    model = st.selectbox("Modèle pour la décomposition mensuelle", sorted(monthly["model"].unique()))
    series = st.radio("Série", ["real", "predicted"], format_func={"real": "Réelle", "predicted": "Prédite"}.get, horizontal=True)
    part = monthly.loc[(monthly["model"].eq(model)) & (monthly["series"].eq(series))].copy()
    if part.empty:
        st.info("Aucune décomposition disponible pour cette sélection.")
        return
    cost_columns = [
        "energy_cost_before_tax_tnd", "vat_energy_tnd", "municipal_surcharge_tnd",
        "subscribed_power_fee_tnd", "vat_power_tnd", "other_fixed_charges_tnd",
    ]
    long = part.melt(
        id_vars="billing_month",
        value_vars=[column for column in cost_columns if column in part],
        var_name="composante",
        value_name="montant_tnd",
    )
    fig = px.bar(
        long,
        x="billing_month",
        y="montant_tnd",
        color="composante",
        title=f"Décomposition mensuelle — {model} ({'réelle' if series == 'real' else 'prédite'})",
        labels={"montant_tnd": "Montant (TND)", "billing_month": "Mois"},
    )
    fig.update_layout(barmode="stack")
    st.plotly_chart(fig, use_container_width=True)


def alerts_module(frame: pd.DataFrame, predictions: pd.DataFrame, tariff: dict) -> None:
    st.header("Système d'alerte SCADA")
    if frame.empty:
        st.warning("Les données historiques sont absentes.")
        return
    subscribed = float(tariff.get("subscribed_power_kw", np.nan))
    max_power = float(frame["Pmax"].max())
    min_pf = float(frame["power_factor_estimated"].min())
    c1, c2 = st.columns(2)
    if np.isfinite(subscribed) and max_power > subscribed:
        c1.error(f"Dépassement de puissance : {max_power:.2f} kW > {subscribed:.2f} kW")
    else:
        c1.success(f"Puissance sous contrôle : {max_power:.2f} kW / {subscribed:.2f} kW")
    if min_pf < 0.9:
        c2.warning(f"Facteur de puissance dégradé : PF minimal {min_pf:.3f} < 0.900")
    else:
        c2.success(f"Facteur de puissance conforme : PF minimal {min_pf:.3f}")

    alerts = []
    over_power = frame.loc[frame["Pmax"].gt(subscribed), ["timestamp", "Pmax"]].copy()
    for _, row in over_power.nlargest(10, "Pmax").iterrows():
        alerts.append({"type": "Puissance", "niveau": "Critique", "timestamp": row["timestamp"], "valeur": row["Pmax"], "unité": "kW"})
    low_pf = frame.loc[frame["power_factor_estimated"].lt(0.9), ["timestamp", "power_factor_estimated"]].copy()
    for _, row in low_pf.nsmallest(10, "power_factor_estimated").iterrows():
        alerts.append({"type": "Facteur de puissance", "niveau": "Attention", "timestamp": row["timestamp"], "valeur": row["power_factor_estimated"], "unité": "PF"})
    if alerts:
        st.subheader("Événements détectés")
        st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
    else:
        st.info("Aucune alerte sur les données historiques disponibles.")

    if not predictions.empty:
        st.subheader("Risque sur les prédictions")
        model = st.selectbox("Modèle à surveiller", sorted(predictions["model"].unique()), key="alert_model")
        predicted = predictions.loc[(predictions["model"].eq(model)) & (predictions["split"].eq("test"))].copy()
        predicted["pf_predicted"] = predicted.apply(
            lambda row: abs(row["pred_AEa"]) / np.hypot(row["pred_AEa"], row["pred_AEr"])
            if np.hypot(row["pred_AEa"], row["pred_AEr"]) > 0 else 1.0,
            axis=1,
        )
        predicted_max = predicted["pred_Pmax"].max()
        predicted_pf = predicted["pf_predicted"].min()
        st.write({"Pmax prédit maximal (kW)": round(float(predicted_max), 3), "PF prédit minimal": round(float(predicted_pf), 4)})


frame = load_processed()
predictions = load_predictions()
tariff = load_tariff()

st.title("SUP'COM — Tableau de bord SCADA énergétique")
st.caption("Analyse historique • Benchmark IA • Facturation • Alertes")
if frame.empty:
    st.warning("Aucune donnée traitée détectée. Vérifiez data/processed/sinert_features.csv.")
else:
    st.sidebar.success(f"Données chargées : {len(frame):,} lignes")
    st.sidebar.caption(f"Période : {frame['timestamp'].min():%Y-%m-%d} → {frame['timestamp'].max():%Y-%m-%d}")

module = st.sidebar.radio(
    "Module",
    ["Analyse historique", "Benchmarking IA", "Module financier", "Système d'alerte"],
)

if module == "Analyse historique":
    historical_module(frame)
elif module == "Benchmarking IA":
    benchmark_module(predictions)
elif module == "Module financier":
    financial_module()
else:
    alerts_module(frame, predictions, tariff)
