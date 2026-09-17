import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "campus_energy_sample.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "campus_energy_with_anomalies.csv"
)


def inject_anomalies():
    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_injected_anomaly"] = False
    df["anomaly_type"] = "normal"

    # 1. Sudden major spike
    condition = (
        (df["building"] == "Building_A")
        & (df["timestamp"] >= "2026-02-10 09:00")
        & (df["timestamp"] <= "2026-02-10 17:00")
    )
    df.loc[condition, "energy_kwh"] *= 2.2
    df.loc[condition, "is_injected_anomaly"] = True
    df.loc[condition, "anomaly_type"] = "sudden_spike"

    # 2. Moderate spike
    condition = (
        (df["building"] == "Building_B")
        & (df["timestamp"] >= "2026-02-20 11:00")
        & (df["timestamp"] <= "2026-02-20 16:00")
    )
    df.loc[condition, "energy_kwh"] *= 1.5
    df.loc[condition, "is_injected_anomaly"] = True
    df.loc[condition, "anomaly_type"] = "moderate_spike"

    # 3. Gradual consumption drift
    condition = (
        (df["building"] == "Building_C")
        & (df["timestamp"] >= "2026-03-01")
        & (df["timestamp"] <= "2026-03-07 23:00")
    )

    drift = (
        (df.loc[condition, "timestamp"].dt.day - 1) * 0.02
        + 1.10
    )

    df.loc[condition, "energy_kwh"] *= drift.values
    df.loc[condition, "is_injected_anomaly"] = True
    df.loc[condition, "anomaly_type"] = "gradual_drift"

    # 4. Off-hours anomaly
    condition = (
        (df["building"] == "Building_D")
        & (df["timestamp"] >= "2026-03-15")
        & (df["timestamp"] <= "2026-03-17 23:00")
        & (df["timestamp"].dt.hour.isin([1, 2, 3, 4]))
    )

    df.loc[condition, "energy_kwh"] *= 1.8
    df.loc[condition, "is_injected_anomaly"] = True
    df.loc[condition, "anomaly_type"] = "off_hours"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Total anomalies: {df['is_injected_anomaly'].sum():,}")
    print("\nAnomaly types:")
    print(df[df["is_injected_anomaly"]]["anomaly_type"].value_counts())
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    inject_anomalies()