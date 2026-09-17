import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


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

    return df


def evaluate_threshold(df, threshold):
    df["predicted_anomaly"] = (
        df["deviation_percent"].abs()
        >= threshold
    )

    print(
        f"\n===== Threshold: ±{threshold}% ====="
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            df["is_injected_anomaly"],
            df["predicted_anomaly"],
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            df["is_injected_anomaly"],
            df["predicted_anomaly"],
            zero_division=0,
        )
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    df = create_baseline(df)

    for threshold in [10, 15, 20, 25, 30, 40]:
        evaluate_threshold(
            df.copy(),
            threshold
        )


if __name__ == "__main__":
    main()