import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def generate_campus_multi_resource_data(
    start_date: str = "2026-01-01",
    end_date: str = "2026-03-31 23:00",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synchronized multi-resource campus data (Energy, Water, Waste)
    for Buildings A, B, C, D across hourly timestamps.
    Returns:
        (clean_df, anomaly_df)
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")

    # Building archetypes:
    # Building_A: Academic Lecture Halls & Classrooms
    # Building_B: Scientific Research Labs (high 24/7 baseload)
    # Building_C: Campus Dining & Student Center (high water & food waste)
    # Building_D: Administration & Faculty Offices (standard 8-18 schedule)
    building_profiles = {
        "Building_A": {"energy_base": 85.0, "water_base": 4.5, "waste_base": 18.0, "diversion_base": 0.55},
        "Building_B": {"energy_base": 120.0, "water_base": 9.0, "waste_base": 14.0, "diversion_base": 0.42},
        "Building_C": {"energy_base": 95.0, "water_base": 8.0, "waste_base": 32.0, "diversion_base": 0.65},
        "Building_D": {"energy_base": 65.0, "water_base": 3.0, "waste_base": 10.0, "diversion_base": 0.48},
    }

    rows = []

    for building, profile in building_profiles.items():
        building_factor = rng.uniform(0.92, 1.08)

        for ts in timestamps:
            hour = ts.hour
            weekday = ts.weekday()
            is_weekend = weekday >= 5

            # -------------------------------------------------------------
            # 1. Occupancy & Schedule Multipliers
            # -------------------------------------------------------------
            if not is_weekend and 8 <= hour <= 18:
                occ_factor = 1.55
                activity_water = 1.70
                activity_waste = 2.10
            elif not is_weekend and (18 < hour <= 22 or 6 <= hour < 8):
                occ_factor = 0.90
                activity_water = 0.85
                activity_waste = 0.80
            else:  # Night or weekend
                occ_factor = 0.50 if is_weekend and 10 <= hour <= 18 else 0.35
                activity_water = 0.30 if is_weekend and 10 <= hour <= 18 else 0.15
                activity_waste = 0.35 if is_weekend and 10 <= hour <= 18 else 0.05

            # -------------------------------------------------------------
            # 2. Energy Consumption (kWh)
            # -------------------------------------------------------------
            energy_noise = rng.normal(0, 4.0)
            energy = (profile["energy_base"] * building_factor * occ_factor) + energy_noise
            energy = max(round(energy, 2), 5.0)

            # -------------------------------------------------------------
            # 3. Water Consumption (m3/hour)
            # -------------------------------------------------------------
            # Cafeteria meal surges
            dining_boost = 1.0
            if building == "Building_C":
                if hour in [11, 12, 13]:  # Lunch
                    dining_boost = 2.4
                elif hour in [17, 18, 19]:  # Dinner
                    dining_boost = 2.0

            water_noise = rng.normal(0, 0.35)
            water = (profile["water_base"] * building_factor * activity_water * dining_boost) + water_noise
            water = max(round(water, 2), 0.20)

            # -------------------------------------------------------------
            # 4. Resource / Waste Generation (kg/hour)
            # -------------------------------------------------------------
            waste_noise = rng.normal(0, 1.2)
            total_waste = (profile["waste_base"] * building_factor * activity_waste * dining_boost) + waste_noise
            total_waste = max(round(total_waste, 2), 0.5)

            # Diversion rate (recycling + composting)
            div_rate = np.clip(profile["diversion_base"] + rng.normal(0, 0.04), 0.15, 0.90)
            diverted = round(total_waste * div_rate, 2)
            landfill = round(max(total_waste - diverted, 0.0), 2)

            rows.append({
                "timestamp": ts,
                "building": building,
                "energy_kwh": energy,
                "water_m3": water,
                "waste_kg": total_waste,
                "diverted_kg": diverted,
                "landfill_kg": landfill,
                "diversion_rate": round(div_rate * 100, 1),
            })

    clean_df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Multi-Resource Anomaly Injections
    # -----------------------------------------------------------------
    anomaly_df = clean_df.copy()
    anomaly_df["is_injected_anomaly"] = False
    anomaly_df["anomaly_resource"] = "none"
    anomaly_df["anomaly_type"] = "normal"

    # Energy Anomalies (retaining exact archetypes from inject_anomalies.py)
    # E1: Sudden major spike in Building_A
    e1_cond = (
        (anomaly_df["building"] == "Building_A")
        & (anomaly_df["timestamp"] >= "2026-02-10 09:00")
        & (anomaly_df["timestamp"] <= "2026-02-10 17:00")
    )
    anomaly_df.loc[e1_cond, "energy_kwh"] = (anomaly_df.loc[e1_cond, "energy_kwh"] * 2.2).round(2)
    anomaly_df.loc[e1_cond, "is_injected_anomaly"] = True
    anomaly_df.loc[e1_cond, "anomaly_resource"] = "energy"
    anomaly_df.loc[e1_cond, "anomaly_type"] = "sudden_spike"

    # E2: Moderate spike in Building_B
    e2_cond = (
        (anomaly_df["building"] == "Building_B")
        & (anomaly_df["timestamp"] >= "2026-02-20 11:00")
        & (anomaly_df["timestamp"] <= "2026-02-20 16:00")
    )
    anomaly_df.loc[e2_cond, "energy_kwh"] = (anomaly_df.loc[e2_cond, "energy_kwh"] * 1.5).round(2)
    anomaly_df.loc[e2_cond, "is_injected_anomaly"] = True
    anomaly_df.loc[e2_cond, "anomaly_resource"] = "energy"
    anomaly_df.loc[e2_cond, "anomaly_type"] = "moderate_spike"

    # E3: Gradual consumption drift in Building_C
    e3_cond = (
        (anomaly_df["building"] == "Building_C")
        & (anomaly_df["timestamp"] >= "2026-03-01")
        & (anomaly_df["timestamp"] <= "2026-03-07 23:00")
    )
    drift = (anomaly_df.loc[e3_cond, "timestamp"].dt.day - 1) * 0.02 + 1.10
    anomaly_df.loc[e3_cond, "energy_kwh"] = (anomaly_df.loc[e3_cond, "energy_kwh"] * drift.values).round(2)
    anomaly_df.loc[e3_cond, "is_injected_anomaly"] = True
    anomaly_df.loc[e3_cond, "anomaly_resource"] = "energy"
    anomaly_df.loc[e3_cond, "anomaly_type"] = "gradual_drift"

    # E4: Off-hours energy surge in Building_D
    e4_cond = (
        (anomaly_df["building"] == "Building_D")
        & (anomaly_df["timestamp"] >= "2026-03-15")
        & (anomaly_df["timestamp"] <= "2026-03-17 23:00")
        & (anomaly_df["timestamp"].dt.hour.isin([1, 2, 3, 4]))
    )
    anomaly_df.loc[e4_cond, "energy_kwh"] = (anomaly_df.loc[e4_cond, "energy_kwh"] * 1.8).round(2)
    anomaly_df.loc[e4_cond, "is_injected_anomaly"] = True
    anomaly_df.loc[e4_cond, "anomaly_resource"] = "energy"
    anomaly_df.loc[e4_cond, "anomaly_type"] = "off_hours"

    # Water Anomalies
    # W1: Continuous main pipe leak in Building_A (constant night & day 3.2x flow)
    w1_cond = (
        (anomaly_df["building"] == "Building_A")
        & (anomaly_df["timestamp"] >= "2026-01-22 00:00")
        & (anomaly_df["timestamp"] <= "2026-01-25 18:00")
    )
    anomaly_df.loc[w1_cond, "water_m3"] = (anomaly_df.loc[w1_cond, "water_m3"] * 3.2 + 5.0).round(2)
    anomaly_df.loc[w1_cond, "is_injected_anomaly"] = True
    anomaly_df.loc[w1_cond, "anomaly_resource"] = "water"
    anomaly_df.loc[w1_cond, "anomaly_type"] = "pipe_leak"

    # W2: Cooling tower fill-valve malfunction in Building_B (overnight surge)
    w2_cond = (
        (anomaly_df["building"] == "Building_B")
        & (anomaly_df["timestamp"] >= "2026-02-14")
        & (anomaly_df["timestamp"] <= "2026-02-16 23:00")
        & (anomaly_df["timestamp"].dt.hour.isin([22, 23, 0, 1, 2, 3, 4]))
    )
    anomaly_df.loc[w2_cond, "water_m3"] = (anomaly_df.loc[w2_cond, "water_m3"] * 2.8).round(2)
    anomaly_df.loc[w2_cond, "is_injected_anomaly"] = True
    anomaly_df.loc[w2_cond, "anomaly_resource"] = "water"
    anomaly_df.loc[w2_cond, "anomaly_type"] = "cooling_tower_surge"

    # Waste Anomalies
    # R1: Post-event symposium waste surge in Building_A
    r1_cond = (
        (anomaly_df["building"] == "Building_A")
        & (anomaly_df["timestamp"] >= "2026-02-27 12:00")
        & (anomaly_df["timestamp"] <= "2026-02-27 20:00")
    )
    anomaly_df.loc[r1_cond, "waste_kg"] = (anomaly_df.loc[r1_cond, "waste_kg"] * 3.5).round(2)
    anomaly_df.loc[r1_cond, "landfill_kg"] = (anomaly_df.loc[r1_cond, "waste_kg"] * 0.85).round(2)
    anomaly_df.loc[r1_cond, "diverted_kg"] = (anomaly_df.loc[r1_cond, "waste_kg"] * 0.15).round(2)
    anomaly_df.loc[r1_cond, "diversion_rate"] = 15.0
    anomaly_df.loc[r1_cond, "is_injected_anomaly"] = True
    anomaly_df.loc[r1_cond, "anomaly_resource"] = "waste"
    anomaly_df.loc[r1_cond, "anomaly_type"] = "waste_surge"

    # R2: Recycling stream contamination drop in Building_C
    r2_cond = (
        (anomaly_df["building"] == "Building_C")
        & (anomaly_df["timestamp"] >= "2026-03-20 08:00")
        & (anomaly_df["timestamp"] <= "2026-03-24 18:00")
    )
    anomaly_df.loc[r2_cond, "diverted_kg"] = (anomaly_df.loc[r2_cond, "diverted_kg"] * 0.25).round(2)
    anomaly_df.loc[r2_cond, "landfill_kg"] = (anomaly_df.loc[r2_cond, "waste_kg"] - anomaly_df.loc[r2_cond, "diverted_kg"]).round(2)
    anomaly_df.loc[r2_cond, "diversion_rate"] = (
        (anomaly_df.loc[r2_cond, "diverted_kg"] / anomaly_df.loc[r2_cond, "waste_kg"]) * 100
    ).round(1)
    anomaly_df.loc[r2_cond, "is_injected_anomaly"] = True
    anomaly_df.loc[r2_cond, "anomaly_resource"] = "waste"
    anomaly_df.loc[r2_cond, "anomaly_type"] = "diversion_drop"

    # Save outputs
    clean_path = RAW_DIR / "campus_multi_resource_sample.csv"
    anomaly_path = PROCESSED_DIR / "campus_multi_resource_with_anomalies.csv"

    clean_df.to_csv(clean_path, index=False)
    anomaly_df.to_csv(anomaly_path, index=False)

    print(f"Generated {len(clean_df):,} records for {clean_df['building'].nunique()} buildings.")
    print(f"Clean data saved to: {clean_path}")
    print(f"Anomaly-injected data saved to: {anomaly_path}")
    print(f"Total injected anomalies: {anomaly_df['is_injected_anomaly'].sum():,}")

    return clean_df, anomaly_df


if __name__ == "__main__":
    generate_campus_multi_resource_data()
