import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


# Default environmental and economic factors (Campus / Commercial baseline)
GRID_EMISSIONS_FACTOR_KG_PER_KWH = 0.420     # kg CO2e per kWh
ELECTRICITY_COST_PER_KWH = 0.140             # USD per kWh
WATER_COST_PER_M3 = 3.80                     # USD per m3
LANDFILL_EMISSIONS_FACTOR_KG_PER_KG = 0.580  # kg CO2e per kg landfill waste
LANDFILL_HAULING_COST_PER_KG = 0.120         # USD per kg


def calculate_energy_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes energy performance indicators aligning with SDG 7.
    """
    if "energy_kwh" not in df.columns:
        return {}

    total_kwh = float(df["energy_kwh"].sum())
    total_mwh = total_kwh / 1000.0
    mean_kwh = float(df["energy_kwh"].mean())
    peak_row = df.loc[df["energy_kwh"].idxmax()] if not df.empty else None

    # Off-hours baseload fraction (hours 22:00 to 05:59)
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    hours = df["timestamp"].dt.hour if "timestamp" in df.columns else pd.Series(dtype=int)
    off_hours_mask = (hours >= 22) | (hours < 6)
    off_hours_kwh = float(df.loc[off_hours_mask, "energy_kwh"].sum()) if not hours.empty else 0.0
    baseload_fraction = (off_hours_kwh / total_kwh * 100) if total_kwh > 0 else 0.0

    # Carbon emissions and cost
    emissions_tonnes = (total_kwh * GRID_EMISSIONS_FACTOR_KG_PER_KWH) / 1000.0
    cost_usd = total_kwh * ELECTRICITY_COST_PER_KWH

    # Building breakdown
    bldg_breakdown = {}
    if "building" in df.columns:
        grouped = df.groupby("building")["energy_kwh"].agg(["sum", "mean", "max"]).reset_index()
        for _, row in grouped.iterrows():
            b_total = float(row["sum"])
            bldg_breakdown[row["building"]] = {
                "total_kwh": round(b_total, 2),
                "total_mwh": round(b_total / 1000.0, 3),
                "mean_kwh": round(float(row["mean"]), 2),
                "peak_kwh": round(float(row["max"]), 2),
                "share_pct": round((b_total / total_kwh * 100), 1) if total_kwh > 0 else 0.0,
            }

    return {
        "total_kwh": round(total_kwh, 2),
        "total_mwh": round(total_mwh, 3),
        "mean_hourly_kwh": round(mean_kwh, 2),
        "peak_kw": round(float(peak_row["energy_kwh"]), 2) if peak_row is not None else 0.0,
        "peak_timestamp": str(peak_row["timestamp"]) if peak_row is not None and "timestamp" in peak_row else "N/A",
        "peak_building": str(peak_row["building"]) if peak_row is not None and "building" in peak_row else "N/A",
        "off_hours_baseload_pct": round(baseload_fraction, 1),
        "carbon_emissions_tco2e": round(emissions_tonnes, 2),
        "estimated_cost_usd": round(cost_usd, 2),
        "building_breakdown": bldg_breakdown,
    }


def calculate_water_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes water stewardship and flow indicators aligning with SDG 6.
    """
    if "water_m3" not in df.columns:
        return {}

    total_m3 = float(df["water_m3"].sum())
    total_liters = total_m3 * 1000.0
    mean_flow = float(df["water_m3"].mean())
    peak_row = df.loc[df["water_m3"].idxmax()] if not df.empty else None

    # Night flow index (02:00 - 05:00) - key indicator for pipe leaks
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    hours = df["timestamp"].dt.hour if "timestamp" in df.columns else pd.Series(dtype=int)
    night_mask = (hours >= 2) & (hours <= 5)
    night_mean_m3 = float(df.loc[night_mask, "water_m3"].mean()) if not hours.empty and night_mask.any() else 0.0
    night_flow_ratio = (night_mean_m3 / mean_flow * 100) if mean_flow > 0 else 0.0

    cost_usd = total_m3 * WATER_COST_PER_M3

    bldg_breakdown = {}
    if "building" in df.columns:
        grouped = df.groupby("building")["water_m3"].agg(["sum", "mean", "max"]).reset_index()
        for _, row in grouped.iterrows():
            b_total = float(row["sum"])
            bldg_breakdown[row["building"]] = {
                "total_m3": round(b_total, 2),
                "mean_flow_m3_h": round(float(row["mean"]), 2),
                "peak_flow_m3_h": round(float(row["max"]), 2),
                "share_pct": round((b_total / total_m3 * 100), 1) if total_m3 > 0 else 0.0,
            }

    return {
        "total_m3": round(total_m3, 2),
        "total_liters": round(total_liters, 1),
        "mean_flow_m3_h": round(mean_flow, 2),
        "peak_flow_m3_h": round(float(peak_row["water_m3"]), 2) if peak_row is not None else 0.0,
        "peak_timestamp": str(peak_row["timestamp"]) if peak_row is not None and "timestamp" in peak_row else "N/A",
        "peak_building": str(peak_row["building"]) if peak_row is not None and "building" in peak_row else "N/A",
        "min_night_flow_m3_h": round(night_mean_m3, 2),
        "night_to_day_ratio_pct": round(night_flow_ratio, 1),
        "estimated_cost_usd": round(cost_usd, 2),
        "building_breakdown": bldg_breakdown,
    }


def calculate_waste_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes resource consumption & circularity metrics aligning with SDG 12.
    """
    if "waste_kg" not in df.columns:
        return {}

    total_waste = float(df["waste_kg"].sum())
    diverted_waste = float(df["diverted_kg"].sum()) if "diverted_kg" in df.columns else (
        float((df["waste_kg"] * (df["diversion_rate"] / 100.0)).sum()) if "diversion_rate" in df.columns else 0.0
    )
    landfill_waste = float(df["landfill_kg"].sum()) if "landfill_kg" in df.columns else max(total_waste - diverted_waste, 0.0)
    
    overall_diversion_rate = (diverted_waste / total_waste * 100) if total_waste > 0 else 0.0
    emissions_tonnes = (landfill_waste * LANDFILL_EMISSIONS_FACTOR_KG_PER_KG) / 1000.0
    hauling_cost_usd = landfill_waste * LANDFILL_HAULING_COST_PER_KG

    bldg_breakdown = {}
    if "building" in df.columns:
        grouped = df.groupby("building")[["waste_kg", "diverted_kg" if "diverted_kg" in df.columns else "waste_kg"]].sum().reset_index()
        for _, row in grouped.iterrows():
            b_total = float(row["waste_kg"])
            b_div = float(row.get("diverted_kg", 0.0))
            bldg_breakdown[row["building"]] = {
                "total_waste_kg": round(b_total, 1),
                "diverted_kg": round(b_div, 1),
                "landfill_kg": round(max(b_total - b_div, 0.0), 1),
                "diversion_rate_pct": round((b_div / b_total * 100), 1) if b_total > 0 else 0.0,
            }

    return {
        "total_waste_kg": round(total_waste, 1),
        "total_waste_tonnes": round(total_waste / 1000.0, 2),
        "diverted_waste_kg": round(diverted_waste, 1),
        "landfill_waste_kg": round(landfill_waste, 1),
        "overall_diversion_rate_pct": round(overall_diversion_rate, 1),
        "landfill_emissions_tco2e": round(emissions_tonnes, 2),
        "hauling_cost_usd": round(hauling_cost_usd, 2),
        "building_breakdown": bldg_breakdown,
    }


def calculate_unified_campus_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Combines Energy, Water, and Waste metrics into an integrated campus sustainability scorecard.
    """
    energy = calculate_energy_kpis(df)
    water = calculate_water_kpis(df)
    waste = calculate_waste_kpis(df)

    total_cost = (
        energy.get("estimated_cost_usd", 0.0)
        + water.get("estimated_cost_usd", 0.0)
        + waste.get("hauling_cost_usd", 0.0)
    )

    total_carbon = (
        energy.get("carbon_emissions_tco2e", 0.0)
        + waste.get("landfill_emissions_tco2e", 0.0)
    )

    return {
        "energy": energy,
        "water": water,
        "waste": waste,
        "total_operational_cost_usd": round(total_cost, 2),
        "total_carbon_footprint_tco2e": round(total_carbon, 2),
        "facility_count": df["building"].nunique() if "building" in df.columns else 1,
        "total_records": len(df),
    }
