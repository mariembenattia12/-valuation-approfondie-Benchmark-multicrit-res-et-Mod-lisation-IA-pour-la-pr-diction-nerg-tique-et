"""Modèle tarifaire STEG moyenne tension.

Les montants monétaires sont exprimés en TND. Les tarifs STEG publiés en
millimes sont convertis en TND dans la configuration du projet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TariffConfig:
    """Paramètres d'une grille STEG MT à postes horaires ou uniforme."""

    peak_price: float = 0.417
    full_price: float = 0.290
    offpeak_price: float = 0.222
    evening_peak_price: float = 0.377
    subscribed_power_price: float = 11.0
    power_factor_threshold: float = 0.9
    vat_rate: float = 0.19
    municipal_surcharge: float = 0.005
    fixed_monthly_charge: float = 0.0
    rit_charge: float = 0.0
    excess_reduction_rate: float = 0.10
    tariff_mode: str = "postes_horaires"
    sunday_offpeak: bool = True


def calculate_energy_cost(
    peak_kwh: float,
    full_kwh: float,
    offpeak_kwh: float,
    tariff: TariffConfig,
    *,
    evening_peak_kwh: float = 0.0,
) -> float:
    """Calcule le coût énergie hors TVA, pénalités et puissance souscrite."""

    return float(
        peak_kwh * tariff.peak_price
        + full_kwh * tariff.full_price
        + offpeak_kwh * tariff.offpeak_price
        + evening_peak_kwh * tariff.evening_peak_price
    )


def classify_time_period(timestamp: object, *, sunday_offpeak: bool = True) -> str:
    """Retourne ``full``, ``peak_morning``, ``peak_evening`` ou ``offpeak``.

    Les horaires suivent le contrat MT fourni : hiver de septembre à mai et
    été de juin à août. Le dimanche est entièrement classé en nuit/creux.
    """

    value = pd.Timestamp(timestamp)
    hour = value.hour + value.minute / 60
    if sunday_offpeak and value.day_of_week == 6:
        return "offpeak"

    if value.month in (6, 7, 8):
        if 6.5 <= hour < 8.5 or 13.5 <= hour < 19:
            return "full"
        if 8.5 <= hour < 13.5:
            return "peak_morning"
        if 19 <= hour < 22:
            return "peak_evening"
        return "offpeak"

    if 7 <= hour < 18:
        return "full"
    if 18 <= hour < 21:
        return "peak_evening"
    return "offpeak"


def tariff_price_for_period(period: str, tariff: TariffConfig) -> float:
    """Retourne le prix énergie en TND/kWh pour une période tarifaire."""

    if tariff.tariff_mode == "uniforme_facture":
        return tariff.full_price
    return {
        "full": tariff.full_price,
        "peak_morning": tariff.peak_price,
        "peak_evening": tariff.evening_peak_price,
        "offpeak": tariff.offpeak_price,
    }[period]


def power_factor_from_energy(
    active_kwh: Iterable[float] | np.ndarray,
    reactive_kvarh: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Estime le facteur de puissance à partir des incréments d'énergie."""

    active = np.abs(np.asarray(active_kwh, dtype=float))
    reactive = np.abs(np.asarray(reactive_kvarh, dtype=float))
    denominator = np.hypot(active, reactive)
    return np.divide(active, denominator, out=np.ones_like(active), where=denominator > 0)


def power_factor_adjustment_rate(power_factor: Iterable[float] | np.ndarray) -> np.ndarray:
    """Calcule le bonus/malus contractuel sur le prix de l'énergie.

    Entre 0,80 et 0,90, le taux est nul. Au-dessus de 0,90, le contrat
    prévoit une bonification de 0,5 % par centième. Sous 0,80, les tranches
    cumulatives du contrat sont appliquées.
    """

    values = np.clip(np.asarray(power_factor, dtype=float), 0.0, 1.0)
    adjustment = np.zeros_like(values)

    bonus = values > 0.90
    adjustment[bonus] = -0.005 * (values[bonus] - 0.90) / 0.01

    first = (values >= 0.75) & (values < 0.80)
    adjustment[first] = 0.005 * (0.79 - values[first]) / 0.01

    second = (values >= 0.70) & (values < 0.75)
    adjustment[second] = 0.02 + 0.01 * (0.74 - values[second]) / 0.01

    third = (values >= 0.60) & (values < 0.70)
    adjustment[third] = 0.06 + 0.015 * (0.69 - values[third]) / 0.01

    fourth = values < 0.60
    adjustment[fourth] = 0.195 + 0.02 * (0.60 - values[fourth]) / 0.01
    return adjustment


def calculate_interval_costs(
    frame: pd.DataFrame,
    tariff: TariffConfig,
    *,
    timestamp_column: str = "target_timestamp",
    active_energy_column: str = "AEa",
    reactive_energy_column: str = "AEr",
    power_factor_column: str | None = None,
) -> pd.DataFrame:
    """Calcule le coût énergie de chaque observation réelle ou prédite."""

    required = [timestamp_column, active_energy_column, reactive_energy_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Colonnes absentes pour le calcul tarifaire : {missing}")

    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="coerce")
    result = result.dropna(subset=[timestamp_column]).reset_index(drop=True)
    active = pd.to_numeric(result[active_energy_column], errors="coerce").fillna(0).clip(lower=0)
    reactive = pd.to_numeric(result[reactive_energy_column], errors="coerce").fillna(0).clip(lower=0)
    result["tariff_period"] = result[timestamp_column].map(
        lambda value: classify_time_period(value, sunday_offpeak=tariff.sunday_offpeak)
    )
    result["energy_price_tnd_per_kwh"] = result["tariff_period"].map(
        lambda period: tariff_price_for_period(period, tariff)
    )
    if power_factor_column and power_factor_column in result:
        result["power_factor"] = pd.to_numeric(
            result[power_factor_column], errors="coerce"
        ).fillna(pd.Series(power_factor_from_energy(active, reactive)))
    else:
        result["power_factor"] = power_factor_from_energy(active, reactive)
    result["pf_adjustment_rate"] = power_factor_adjustment_rate(result["power_factor"])
    result["active_energy_kwh"] = active
    result["reactive_energy_kvarh"] = reactive
    result["energy_cost_before_tax_tnd"] = (
        active * result["energy_price_tnd_per_kwh"] * (1 + result["pf_adjustment_rate"])
    )
    result["vat_energy_tnd"] = result["energy_cost_before_tax_tnd"] * tariff.vat_rate
    result["municipal_surcharge_tnd"] = active * tariff.municipal_surcharge
    result["energy_cost_total_tnd"] = (
        result["energy_cost_before_tax_tnd"]
        + result["vat_energy_tnd"]
        + result["municipal_surcharge_tnd"]
    )
    return result


def calculate_monthly_bill(
    interval_costs: pd.DataFrame,
    tariff: TariffConfig,
    *,
    timestamp_column: str = "target_timestamp",
    active_power_column: str | None = None,
    subscribed_power_kw: float = 0.0,
) -> pd.DataFrame:
    """Agrège les coûts par mois et ajoute puissance, TVA et charges fixes."""

    if interval_costs.empty:
        return pd.DataFrame()
    working = interval_costs.copy()
    working["billing_month"] = pd.to_datetime(working[timestamp_column]).dt.to_period("M").astype(str)
    rows = []
    for month, group in working.groupby("billing_month", sort=True):
        energy_before_tax = float(group["energy_cost_before_tax_tnd"].sum())
        vat_energy = float(group["vat_energy_tnd"].sum())
        municipal = float(group["municipal_surcharge_tnd"].sum())
        subscribed_fee = float(subscribed_power_kw * tariff.subscribed_power_price)
        vat_power = subscribed_fee * tariff.vat_rate
        if active_power_column and active_power_column in group:
            max_demand = float(pd.to_numeric(group[active_power_column], errors="coerce").max())
        else:
            max_demand = np.nan
        raw_excess = max(0.0, max_demand - subscribed_power_kw) if np.isfinite(max_demand) else 0.0
        excess = raw_excess * (1 - tariff.excess_reduction_rate)
        excess_fee = excess * tariff.subscribed_power_price / 3
        fixed = tariff.fixed_monthly_charge + tariff.rit_charge
        energy_total = energy_before_tax + vat_energy + municipal
        total = energy_total + subscribed_fee + vat_power + excess_fee + fixed
        rows.append({
            "billing_month": month,
            "active_energy_kwh": float(group["active_energy_kwh"].sum()),
            "reactive_energy_kvarh": float(group["reactive_energy_kvarh"].sum()),
            "energy_cost_before_tax_tnd": energy_before_tax,
            "vat_energy_tnd": vat_energy,
            "municipal_surcharge_tnd": municipal,
            "subscribed_power_kw": subscribed_power_kw,
            "maximum_demand_kw": max_demand,
            "power_excess_tnd": excess_fee,
            "subscribed_power_fee_tnd": subscribed_fee,
            "vat_power_tnd": vat_power,
            "other_fixed_charges_tnd": fixed,
            "total_bill_tnd": total,
        })
    return pd.DataFrame(rows)
