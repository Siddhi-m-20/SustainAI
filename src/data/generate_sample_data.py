import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_energy_data():
    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        start="2026-01-01",
        end="2026-03-31 23:00",
        freq="h"
    )

    buildings = [
        "Building_A",
        "Building_B",
        "Building_C",
        "Building_D"
    ]

    rows = []

    for building in buildings:
        building_factor = rng.uniform(0.8, 1.2)

        for timestamp in timestamps:
            hour = timestamp.hour
            weekday = timestamp.weekday()

            # Base consumption
            base = 80 * building_factor

            # Higher usage during working hours
            if 8 <= hour <= 18 and weekday < 5:
                occupancy_factor = 1.5
            else:
                occupancy_factor = 0.65

            # Daily variation
            noise = rng.normal(0, 5)

            energy = (
                base
                * occupancy_factor
                + noise
            )

            energy = max(energy, 5)

            rows.append(
                {
                    "timestamp": timestamp,
                    "building": building,
                    "energy_kwh": round(energy, 2),
                }
            )

    df = pd.DataFrame(rows)

    output_file = OUTPUT_DIR / "campus_energy_sample.csv"
    df.to_csv(output_file, index=False)

    print(f"Generated {len(df):,} records.")
    print(f"Saved to: {output_file}")

    return df


if __name__ == "__main__":
    generate_energy_data()