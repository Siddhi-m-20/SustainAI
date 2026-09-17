import pytest
import pandas as pd
import numpy as np

from src.anomaly.anomaly_service import AnomalyService


def test_anomaly_service_feature_extraction():
    service = AnomalyService(contamination=0.03)

    # Synthetic time series
    dates = pd.date_range("2026-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "building": ["Building_A"] * 100,
        "energy_kwh": [50.0 + (i % 24) * 2.0 for i in range(100)],
    })

    feats = service.extract_features(df, resource_col="energy_kwh")
    assert "expected_energy_kwh" in feats.columns
    assert "deviation_percent_energy_kwh" in feats.columns
    assert "rolling_mean_6h" in feats.columns
    assert "rolling_z_score" in feats.columns or "z_score_energy_kwh" in feats.columns
    assert "hour_sin" in feats.columns
    assert "hour_cos" in feats.columns


def test_anomaly_service_detection_and_incidents():
    service = AnomalyService(contamination=0.04)

    dates = pd.date_range("2026-01-01", periods=120, freq="h")
    energy_vals = [50.0 + (i % 24) * 1.5 for i in range(120)]
    # Inject obvious spike at index 50
    energy_vals[50] = 300.0

    df = pd.DataFrame({
        "timestamp": dates,
        "building": ["Building_A"] * 120,
        "energy_kwh": energy_vals,
    })

    scored_df = service.detect(df, resource_col="energy_kwh")
    assert "is_anomaly_energy_kwh" in scored_df.columns
    assert "severity_energy_kwh" in scored_df.columns
    assert "severity_tier_energy_kwh" in scored_df.columns

    # Check that the injected extreme spike is detected with high severity
    spike_row = scored_df.iloc[50]
    assert spike_row["is_anomaly_energy_kwh"] == True
    assert spike_row["severity_energy_kwh"] >= 50.0

    incidents = service.extract_incidents(scored_df, resource_col="energy_kwh", min_severity=40.0)
    assert not incidents.empty
    assert "incident_id" in incidents.columns
    assert "severity_score" in incidents.columns
    assert "pattern" in incidents.columns
    assert "excess_consumption" in incidents.columns


def test_multi_resource_detection():
    service = AnomalyService(contamination=0.03)

    dates = pd.date_range("2026-01-01", periods=50, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "building": ["Building_A"] * 50,
        "energy_kwh": [60.0 + (i % 24) for i in range(50)],
        "water_m3": [4.0 + (i % 12) * 0.2 for i in range(50)],
        "waste_kg": [15.0 for _ in range(50)],
    })
    # Inject water anomaly
    df.loc[25, "water_m3"] = 40.0

    scored, ledger = service.detect_multi_resource(df)
    assert "is_anomaly_energy_kwh" in scored.columns
    assert "is_anomaly_water_m3" in scored.columns
    assert "is_anomaly_waste_kg" in scored.columns
    assert isinstance(ledger, pd.DataFrame)
    assert not ledger.empty
    assert "Water" in ledger["resource"].values
