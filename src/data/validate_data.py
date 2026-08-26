import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "campus_energy_sample.csv"
)


def validate_energy_data():
    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "timestamp",
        "building",
        "energy_kwh",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp values detected."
        )

    if df["energy_kwh"].isna().any():
        raise ValueError(
            "Missing energy consumption values detected."
        )

    if (df["energy_kwh"] < 0).any():
        raise ValueError(
            "Negative energy consumption detected."
        )

    print("Data validation successful.")
    print(f"Rows: {len(df):,}")
    print(f"Buildings: {df['building'].nunique()}")
    print(
        f"Time range: "
        f"{df['timestamp'].min()} "
        f"to "
        f"{df['timestamp'].max()}"
    )


if __name__ == "__main__":
    validate_energy_data()