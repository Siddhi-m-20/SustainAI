import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "campus_energy_with_anomalies.csv"
)


def create_baseline(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Use non-anomalous observations to construct the baseline.
    normal_data = df[
        df["is_injected_anomaly"] == False
    ].copy()

    baseline = (
        normal_data
        .groupby(
            ["building", "day_of_week", "hour"]
        )["energy_kwh"]
        .median()
        .reset_index()
    )

    baseline = baseline.rename(
        columns={
            "energy_kwh": "expected_energy_kwh"
        }
    )

    df = df.merge(
        baseline,
        on=[
            "building",
            "day_of_week",
            "hour",
        ],
        how="left",
    )

    df["deviation_percent"] = (
        (
            df["energy_kwh"]
            - df["expected_energy_kwh"]
        )
        / df["expected_energy_kwh"]
    ) * 100

    df["absolute_deviation_percent"] = (
        df["deviation_percent"].abs()
    )

    return df


def main():
    df = pd.read_csv(INPUT_FILE)

    result = create_baseline(df)

    print(
        result[
            [
                "timestamp",
                "building",
                "energy_kwh",
                "expected_energy_kwh",
                "deviation_percent",
                "is_injected_anomaly",
            ]
        ].head(20)
    )

    print("\nAverage absolute deviation:")
    print(
        result[
            "absolute_deviation_percent"
        ].mean()
    )


if __name__ == "__main__":
    main()