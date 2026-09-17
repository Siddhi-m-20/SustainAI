import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.analytics.kpi_engine import (
    GRID_EMISSIONS_FACTOR_KG_PER_KWH,
    ELECTRICITY_COST_PER_KWH,
    WATER_COST_PER_M3,
    LANDFILL_EMISSIONS_FACTOR_KG_PER_KG,
    LANDFILL_HAULING_COST_PER_KG,
)

# Conversion factors for human-centric storytelling:
# 1 passenger vehicle emits ~4.6 tonnes CO2e per year (~0.38 tCO2e/month)
# 1 mature urban tree sequesters ~21.77 kg CO2e per year (~1.81 kg CO2e/month)
TREE_SEQUESTRATION_KG_YEAR = 21.77
VEHICLE_ANNUAL_EMISSIONS_TONNES = 4.60


@dataclass
class SimulationResult:
    # Resource Savings
    energy_saved_kwh: float
    energy_saved_mwh: float
    water_saved_m3: float
    water_saved_liters: float
    waste_diverted_kg: float
    waste_avoided_kg: float

    # Financial Savings
    energy_cost_saved_usd: float
    water_cost_saved_usd: float
    waste_cost_saved_usd: float
    total_cost_saved_usd: float

    # Environmental Impact
    carbon_avoided_tco2e: float
    equivalent_cars_removed_annual: float
    equivalent_trees_planted: float

    # SDG Performance Impacts
    sdg7_energy_intensity_change_pct: float
    sdg6_water_reduction_pct: float
    sdg12_diversion_rate_projected_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy_saved_kwh": round(self.energy_saved_kwh, 2),
            "energy_saved_mwh": round(self.energy_saved_mwh, 3),
            "water_saved_m3": round(self.water_saved_m3, 2),
            "water_saved_liters": round(self.water_saved_liters, 0),
            "waste_diverted_kg": round(self.waste_diverted_kg, 2),
            "waste_avoided_kg": round(self.waste_avoided_kg, 2),
            "energy_cost_saved_usd": round(self.energy_cost_saved_usd, 2),
            "water_cost_saved_usd": round(self.water_cost_saved_usd, 2),
            "waste_cost_saved_usd": round(self.waste_cost_saved_usd, 2),
            "total_cost_saved_usd": round(self.total_cost_saved_usd, 2),
            "carbon_avoided_tco2e": round(self.carbon_avoided_tco2e, 3),
            "equivalent_cars_removed_annual": round(self.equivalent_cars_removed_annual, 1),
            "equivalent_trees_planted": round(self.equivalent_trees_planted, 0),
            "sdg7_energy_intensity_change_pct": round(self.sdg7_energy_intensity_change_pct, 1),
            "sdg6_water_reduction_pct": round(self.sdg6_water_reduction_pct, 1),
            "sdg12_diversion_rate_projected_pct": round(self.sdg12_diversion_rate_projected_pct, 1),
        }


class WhatIfSimulator:
    """
    Simulates operational sustainability interventions and computes
    SDG-aligned resource, financial, and decarbonization outcomes.
    """

    def __init__(self):
        pass

    def simulate(
        self,
        df: pd.DataFrame,
        building: Optional[str] = None,
        off_hours_reduction_pct: float = 20.0,
        peak_shaving_pct: float = 10.0,
        night_water_leak_fix_pct: float = 40.0,
        greywater_reuse_pct: float = 0.0,
        compost_diversion_target_pct: float = 30.0,
        source_reduction_pct: float = 5.0,
    ) -> SimulationResult:
        """
        Runs counterfactual scenario simulation against historical baseline data.
        """
        data = df.copy()
        if building and building != "ALL" and "building" in data.columns:
            data = data[data["building"] == building]

        # Datetime processing
        ts_col = "timestamp" if "timestamp" in data.columns else None
        if ts_col and not pd.api.types.is_datetime64_any_dtype(data[ts_col]):
            data[ts_col] = pd.to_datetime(data[ts_col], errors="coerce")

        hours = data[ts_col].dt.hour if ts_col else pd.Series(0, index=data.index)
        is_off_hours = (hours >= 22) | (hours < 6)

        # -------------------------------------------------------------
        # 1. SDG 7: Energy Interventions
        # -------------------------------------------------------------
        total_kwh = float(data["energy_kwh"].sum()) if "energy_kwh" in data.columns else 0.0
        off_hours_kwh = float(data.loc[is_off_hours, "energy_kwh"].sum()) if "energy_kwh" in data.columns else 0.0
        peak_threshold = float(np.percentile(data["energy_kwh"], 95)) if ("energy_kwh" in data.columns and len(data) > 20) else 0.0
        is_peak = data["energy_kwh"] > peak_threshold if "energy_kwh" in data.columns else pd.Series(False, index=data.index)
        peak_excess_kwh = float((data.loc[is_peak, "energy_kwh"] - peak_threshold).sum()) if "energy_kwh" in data.columns else 0.0

        off_hours_savings = off_hours_kwh * (off_hours_reduction_pct / 100.0)
        peak_savings = peak_excess_kwh * (peak_shaving_pct / 100.0)
        total_energy_saved_kwh = off_hours_savings + peak_savings
        energy_saved_mwh = total_energy_saved_kwh / 1000.0
        energy_cost_saved_usd = total_energy_saved_kwh * ELECTRICITY_COST_PER_KWH
        energy_carbon_avoided_tco2e = (total_energy_saved_kwh * GRID_EMISSIONS_FACTOR_KG_PER_KWH) / 1000.0

        energy_intensity_change_pct = (total_energy_saved_kwh / total_kwh * 100.0) if total_kwh > 0 else 0.0

        # -------------------------------------------------------------
        # 2. SDG 6: Water Interventions
        # -------------------------------------------------------------
        total_water_m3 = float(data["water_m3"].sum()) if "water_m3" in data.columns else 0.0
        night_water_m3 = float(data.loc[is_off_hours, "water_m3"].sum()) if "water_m3" in data.columns else 0.0

        # Estimating avoidable night flow (unoccupied flow above base minimal maintenance)
        night_flow_leak_savings = night_water_m3 * (night_water_leak_fix_pct / 100.0)
        greywater_savings = total_water_m3 * (greywater_reuse_pct / 100.0)
        total_water_saved_m3 = night_flow_leak_savings + greywater_savings
        water_saved_liters = total_water_saved_m3 * 1000.0
        water_cost_saved_usd = total_water_saved_m3 * WATER_COST_PER_M3

        water_reduction_pct = (total_water_saved_m3 / total_water_m3 * 100.0) if total_water_m3 > 0 else 0.0

        # -------------------------------------------------------------
        # 3. SDG 12: Resource / Waste Interventions
        # -------------------------------------------------------------
        waste_col = "waste_kg" if "waste_kg" in data.columns else ("waste_total_kg" if "waste_total_kg" in data.columns else None)
        total_waste_kg = float(data[waste_col].sum()) if waste_col else 0.0
        recycled_kg = float(data["waste_recycled_kg"].sum()) if "waste_recycled_kg" in data.columns else (
            float(data["diverted_kg"].sum()) if "diverted_kg" in data.columns else 0.0
        )
        compost_kg = float(data["waste_compost_kg"].sum()) if "waste_compost_kg" in data.columns else 0.0
        landfill_kg = float(data["waste_landfill_kg"].sum()) if "waste_landfill_kg" in data.columns else (
            float(data["landfill_kg"].sum()) if "landfill_kg" in data.columns else max(total_waste_kg - recycled_kg - compost_kg, 0.0)
        )

        waste_avoided_kg = total_waste_kg * (source_reduction_pct / 100.0)
        current_landfill_kg = max(landfill_kg - waste_avoided_kg, 0.0)
        additional_diverted_to_compost_kg = current_landfill_kg * (compost_diversion_target_pct / 100.0)
        total_waste_diverted_kg = additional_diverted_to_compost_kg
        new_landfill_kg = max(current_landfill_kg - additional_diverted_to_compost_kg, 0.0)

        # Landfill hauling and emissions savings
        landfill_avoided_kg = waste_avoided_kg + total_waste_diverted_kg
        waste_cost_saved_usd = landfill_avoided_kg * LANDFILL_HAULING_COST_PER_KG
        waste_carbon_avoided_tco2e = (landfill_avoided_kg * LANDFILL_EMISSIONS_FACTOR_KG_PER_KG) / 1000.0

        new_total_waste_kg = max(total_waste_kg - waste_avoided_kg, 1.0)
        new_diverted_total = recycled_kg + compost_kg + additional_diverted_to_compost_kg
        projected_diversion_rate = (new_diverted_total / new_total_waste_kg) * 100.0

        # -------------------------------------------------------------
        # 4. Total Impact Synthesis
        # -------------------------------------------------------------
        total_cost_saved_usd = energy_cost_saved_usd + water_cost_saved_usd + waste_cost_saved_usd
        total_carbon_avoided_tco2e = energy_carbon_avoided_tco2e + waste_carbon_avoided_tco2e

        cars_removed = total_carbon_avoided_tco2e / VEHICLE_ANNUAL_EMISSIONS_TONNES
        trees_planted = (total_carbon_avoided_tco2e * 1000.0) / TREE_SEQUESTRATION_KG_YEAR

        return SimulationResult(
            energy_saved_kwh=total_energy_saved_kwh,
            energy_saved_mwh=energy_saved_mwh,
            water_saved_m3=total_water_saved_m3,
            water_saved_liters=water_saved_liters,
            waste_diverted_kg=total_waste_diverted_kg,
            waste_avoided_kg=waste_avoided_kg,
            energy_cost_saved_usd=energy_cost_saved_usd,
            water_cost_saved_usd=water_cost_saved_usd,
            waste_cost_saved_usd=waste_cost_saved_usd,
            total_cost_saved_usd=total_cost_saved_usd,
            carbon_avoided_tco2e=total_carbon_avoided_tco2e,
            equivalent_cars_removed_annual=cars_removed,
            equivalent_trees_planted=trees_planted,
            sdg7_energy_intensity_change_pct=energy_intensity_change_pct,
            sdg6_water_reduction_pct=water_reduction_pct,
            sdg12_diversion_rate_projected_pct=min(projected_diversion_rate, 100.0),
        )
