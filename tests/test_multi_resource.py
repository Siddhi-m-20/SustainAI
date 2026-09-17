import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.validate_data import (
    validate_energy_data,
    validate_multi_resource_data,
    detect_dataset_schema,
)
from src.data.multi_resource_generator import generate_campus_multi_resource_data
from src.analytics.kpi_engine import (
    calculate_energy_kpis,
    calculate_water_kpis,
    calculate_waste_kpis,
    calculate_unified_campus_kpis,
)


def test_schema_detection():
    # Energy schema
    df_energy = pd.DataFrame({"timestamp": ["2026-01-01"], "building": ["B1"], "energy_kwh": [100.0]})
    assert detect_dataset_schema(df_energy) == "energy"

    # Water schema
    df_water = pd.DataFrame({"timestamp": ["2026-01-01"], "building": ["B1"], "water_m3": [10.0]})
    assert detect_dataset_schema(df_water) == "water"

    # Waste schema
    df_waste = pd.DataFrame({"timestamp": ["2026-01-01"], "building": ["B1"], "waste_kg": [50.0]})
    assert detect_dataset_schema(df_waste) == "waste"

    # Multi-resource schema
    df_multi = pd.DataFrame({
        "timestamp": ["2026-01-01"],
        "building": ["B1"],
        "energy_kwh": [100.0],
        "water_m3": [10.0],
        "waste_kg": [50.0],
    })
    assert detect_dataset_schema(df_multi) == "unified_multi_resource"


def test_multi_resource_validation():
    # Valid multi-resource dataframe
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
        "building": ["Building_A"] * 10,
        "energy_kwh": [50.0 + i for i in range(10)],
        "water_m3": [3.0 + i * 0.1 for i in range(10)],
        "waste_kg": [10.0 + i * 0.5 for i in range(10)],
    })

    result = validate_multi_resource_data(df)
    assert result["is_valid"] is True
    assert result["row_count"] == 10
    assert "Building_A" in result["buildings"]
    assert set(result["resource_columns"]) == {"energy_kwh", "water_m3", "waste_kg"}


def test_validation_handles_nulls_and_negatives():
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"],
        "building": ["Building_A", "Building_A", "Building_A"],
        "energy_kwh": [100.0, -15.0, np.nan],  # negative and null
    })

    result = validate_multi_resource_data(df)
    assert result["is_valid"] is True
    cleaned = result["cleaned_df"]
    # Negative should be clipped to 0
    assert cleaned["energy_kwh"].iloc[1] == 0.0
    # Null should be imputed with median
    assert not cleaned["energy_kwh"].isna().any()


def test_kpi_engine_calculations():
    timestamps = pd.date_range("2026-01-01", periods=24, freq="h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "building": ["Building_A"] * 24,
        "energy_kwh": [100.0] * 24,
        "water_m3": [5.0] * 24,
        "waste_kg": [20.0] * 24,
        "diverted_kg": [12.0] * 24,
        "landfill_kg": [8.0] * 24,
        "diversion_rate": [60.0] * 24,
    })

    energy_kpi = calculate_energy_kpis(df)
    assert energy_kpi["total_kwh"] == 2400.0
    assert energy_kpi["peak_kw"] == 100.0
    assert energy_kpi["carbon_emissions_tco2e"] > 0
    assert energy_kpi["estimated_cost_usd"] > 0

    water_kpi = calculate_water_kpis(df)
    assert water_kpi["total_m3"] == 120.0
    assert water_kpi["estimated_cost_usd"] > 0

    waste_kpi = calculate_waste_kpis(df)
    assert waste_kpi["total_waste_kg"] == 480.0
    assert waste_kpi["overall_diversion_rate_pct"] == 60.0

    unified = calculate_unified_campus_kpis(df)
    assert unified["total_operational_cost_usd"] > 0
    assert unified["total_carbon_footprint_tco2e"] > 0
