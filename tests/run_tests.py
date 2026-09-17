import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

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
from src.anomaly.anomaly_service import AnomalyService
from tests.test_ecosync_services import TestEcoSyncServices


class TestMultiResourceDataEngine(unittest.TestCase):
    def test_schema_detection(self):
        df_energy = pd.DataFrame({"timestamp": ["2026-01-01"], "building": ["B1"], "energy_kwh": [100.0]})
        self.assertEqual(detect_dataset_schema(df_energy), "energy")

        df_water = pd.DataFrame({"timestamp": ["2026-01-01"], "building": ["B1"], "water_m3": [10.0]})
        self.assertEqual(detect_dataset_schema(df_water), "water")

        df_waste = pd.DataFrame({"timestamp": ["2026-01-01"], "building": ["B1"], "waste_kg": [50.0]})
        self.assertEqual(detect_dataset_schema(df_waste), "waste")

        df_multi = pd.DataFrame({
            "timestamp": ["2026-01-01"],
            "building": ["B1"],
            "energy_kwh": [100.0],
            "water_m3": [10.0],
            "waste_kg": [50.0],
        })
        self.assertEqual(detect_dataset_schema(df_multi), "unified_multi_resource")

    def test_multi_resource_validation(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
            "building": ["Building_A"] * 10,
            "energy_kwh": [50.0 + i for i in range(10)],
            "water_m3": [3.0 + i * 0.1 for i in range(10)],
            "waste_kg": [10.0 + i * 0.5 for i in range(10)],
        })

        result = validate_multi_resource_data(df)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["row_count"], 10)
        self.assertIn("Building_A", result["buildings"])
        self.assertEqual(set(result["resource_columns"]), {"energy_kwh", "water_m3", "waste_kg"})

    def test_validation_handles_nulls_and_negatives(self):
        df = pd.DataFrame({
            "timestamp": ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"],
            "building": ["Building_A", "Building_A", "Building_A"],
            "energy_kwh": [100.0, -15.0, np.nan],
        })

        result = validate_multi_resource_data(df)
        self.assertTrue(result["is_valid"])
        cleaned = result["cleaned_df"]
        self.assertEqual(cleaned["energy_kwh"].iloc[1], 0.0)
        self.assertFalse(cleaned["energy_kwh"].isna().any())

    def test_kpi_engine_calculations(self):
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
        self.assertEqual(energy_kpi["total_kwh"], 2400.0)
        self.assertEqual(energy_kpi["peak_kw"], 100.0)
        self.assertGreater(energy_kpi["carbon_emissions_tco2e"], 0)
        self.assertGreater(energy_kpi["estimated_cost_usd"], 0)

        water_kpi = calculate_water_kpis(df)
        self.assertEqual(water_kpi["total_m3"], 120.0)
        self.assertGreater(water_kpi["estimated_cost_usd"], 0)

        waste_kpi = calculate_waste_kpis(df)
        self.assertEqual(waste_kpi["total_waste_kg"], 480.0)
        self.assertEqual(waste_kpi["overall_diversion_rate_pct"], 60.0)

        unified = calculate_unified_campus_kpis(df)
        self.assertGreater(unified["total_operational_cost_usd"], 0)
        self.assertGreater(unified["total_carbon_footprint_tco2e"], 0)


class TestAnomalyService(unittest.TestCase):
    def test_feature_extraction(self):
        service = AnomalyService(contamination=0.03)
        dates = pd.date_range("2026-01-01", periods=50, freq="h")
        df = pd.DataFrame({
            "timestamp": dates,
            "building": ["Building_A"] * 50,
            "energy_kwh": [50.0 + (i % 24) * 2.0 for i in range(50)],
        })

        feats = service.extract_features(df, resource_col="energy_kwh")
        self.assertIn("expected_energy_kwh", feats.columns)
        self.assertIn("deviation_percent_energy_kwh", feats.columns)
        self.assertIn("rolling_mean_6h", feats.columns)
        self.assertIn("z_score_energy_kwh", feats.columns)
        self.assertIn("hour_sin", feats.columns)
        self.assertIn("hour_cos", feats.columns)

    def test_detection_and_incidents(self):
        service = AnomalyService(contamination=0.05)
        dates = pd.date_range("2026-01-01", periods=60, freq="h")
        energy_vals = [50.0 + (i % 24) * 1.5 for i in range(60)]
        energy_vals[30] = 350.0  # Massive injected spike

        df = pd.DataFrame({
            "timestamp": dates,
            "building": ["Building_A"] * 60,
            "energy_kwh": energy_vals,
        })

        scored_df = service.detect(df, resource_col="energy_kwh")
        self.assertIn("is_anomaly_energy_kwh", scored_df.columns)
        self.assertIn("severity_energy_kwh", scored_df.columns)
        self.assertIn("severity_tier_energy_kwh", scored_df.columns)

        spike_row = scored_df.iloc[30]
        self.assertTrue(spike_row["is_anomaly_energy_kwh"])
        self.assertGreaterEqual(spike_row["severity_energy_kwh"], 50.0)

        incidents = service.extract_incidents(scored_df, resource_col="energy_kwh", min_severity=40.0)
        self.assertFalse(incidents.empty)
        self.assertIn("incident_id", incidents.columns)
        self.assertIn("severity_score", incidents.columns)
        self.assertIn("pattern", incidents.columns)

    def test_multi_resource_detection(self):
        service = AnomalyService(contamination=0.04)
        dates = pd.date_range("2026-01-01", periods=40, freq="h")
        df = pd.DataFrame({
            "timestamp": dates,
            "building": ["Building_A"] * 40,
            "energy_kwh": [60.0 + (i % 24) for i in range(40)],
            "water_m3": [4.0 + (i % 12) * 0.2 for i in range(40)],
            "waste_kg": [15.0 for _ in range(40)],
        })
        df.loc[20, "water_m3"] = 45.0  # Injected water anomaly

        scored, ledger = service.detect_multi_resource(df)
        self.assertIn("is_anomaly_energy_kwh", scored.columns)
        self.assertIn("is_anomaly_water_m3", scored.columns)
        self.assertIn("is_anomaly_waste_kg", scored.columns)
        self.assertFalse(ledger.empty)
        self.assertIn("Water", ledger["resource"].values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
