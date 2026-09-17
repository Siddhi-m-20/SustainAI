import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
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


def prepare_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Cyclical representation of hour.
    df["hour_sin"] = __import__("numpy").sin(
        2 * __import__("numpy").pi
        * df["hour"] / 24
    )

    df["hour_cos"] = __import__("numpy").cos(
        2 * __import__("numpy").pi
        * df["hour"] / 24
    )

    return df


def detect_anomalies():
    df = pd.read_csv(INPUT_FILE)

    df = prepare_features(df)

    feature_columns = [
        "energy_kwh",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "month",
    ]

    X = df[feature_columns]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
    )

    predictions = model.fit_predict(X)

    # Isolation Forest:
    # 1  = normal
    # -1 = anomaly
    df["predicted_anomaly"] = (
        predictions == -1
    )

    print("\nDetected anomalies:")
    print(
        df["predicted_anomaly"]
        .value_counts()
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

    return df


if __name__ == "__main__":
    detect_anomalies()