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
    / "energy_features.csv"
)


def main():
    print("Loading engineered features...")

    df = pd.read_csv(INPUT_FILE)

    # Features used by the ML model.
    feature_columns = [
        "energy_kwh",
        "expected_energy_kwh",
        "deviation_percent",
        "absolute_deviation_percent",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "is_weekend",
        "is_off_hours",
        "rolling_mean_6h",
        "rolling_std_6h",
        "rolling_mean_24h",
        "rolling_std_24h",
        "lag_1h",
        "lag_24h",
        "change_from_1h",
        "change_from_24h",
        "rolling_z_score",
    ]

    # Remove rows where rolling/lag calculations
    # have not produced values yet.
    model_data = df.dropna(
        subset=feature_columns
    ).copy()

    X = model_data[feature_columns]

    print(
        f"Records available for ML: "
        f"{len(model_data):,}"
    )

    # Try several contamination levels.
    contamination_values = [
        0.01,
        0.02,
        0.03,
        0.05,
    ]

    for contamination in contamination_values:

        print(
            f"\n{'=' * 55}"
        )

        print(
            f"Contamination: {contamination}"
        )

        model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )

        predictions = model.fit_predict(X)

        model_data["predicted_anomaly"] = (
            predictions == -1
        )

        print("\nConfusion Matrix:")

        print(
            confusion_matrix(
                model_data["is_injected_anomaly"],
                model_data["predicted_anomaly"],
            )
        )

        print("\nClassification Report:")

        print(
            classification_report(
                model_data["is_injected_anomaly"],
                model_data["predicted_anomaly"],
                zero_division=0,
            )
        )

    # Save the final run for inspection.
    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "improved_anomaly_results.csv"
    )

    model_data.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nResults saved to: {output_file}"
    )


if __name__ == "__main__":
    main()