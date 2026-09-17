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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "energy_features.csv"
)


def create_features(df):
    df = df.copy()

    # ---------------------------------------------------------
    # 1. Time features
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    # Weekend indicator
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Off-hours indicator
    df["is_off_hours"] = (
        (df["hour"] < 7)
        | (df["hour"] > 19)
    ).astype(int)

    # ---------------------------------------------------------
    # 2. Cyclical time features
    # ---------------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # ---------------------------------------------------------
    # 3. Building-specific expected consumption
    # ---------------------------------------------------------

    normal_data = df[
        df["is_injected_anomaly"] == False
    ]

    baseline = (
        normal_data
        .groupby(
            [
                "building",
                "day_of_week",
                "hour",
            ]
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

    # ---------------------------------------------------------
    # 4. Deviation features
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 5. Rolling statistics
    # ---------------------------------------------------------

    df = df.sort_values(
        ["building", "timestamp"]
    )

    grouped = df.groupby(
        "building"
    )["energy_kwh"]

    df["rolling_mean_6h"] = (
        grouped
        .transform(
            lambda x: x.shift(1)
            .rolling(6)
            .mean()
        )
    )

    df["rolling_std_6h"] = (
        grouped
        .transform(
            lambda x: x.shift(1)
            .rolling(6)
            .std()
        )
    )

    df["rolling_mean_24h"] = (
        grouped
        .transform(
            lambda x: x.shift(1)
            .rolling(24)
            .mean()
        )
    )

    df["rolling_std_24h"] = (
        grouped
        .transform(
            lambda x: x.shift(1)
            .rolling(24)
            .std()
        )
    )

    # ---------------------------------------------------------
    # 6. Lag features
    # ---------------------------------------------------------

    df["lag_1h"] = (
        grouped.transform(
            lambda x: x.shift(1)
        )
    )

    df["lag_24h"] = (
        grouped.transform(
            lambda x: x.shift(24)
        )
    )

    # ---------------------------------------------------------
    # 7. Difference from recent behavior
    # ---------------------------------------------------------

    df["change_from_1h"] = (
        df["energy_kwh"]
        - df["lag_1h"]
    )

    df["change_from_24h"] = (
        df["energy_kwh"]
        - df["lag_24h"]
    )

    # ---------------------------------------------------------
    # 8. Rolling z-score
    # ---------------------------------------------------------

    df["rolling_z_score"] = (
        (
            df["energy_kwh"]
            - df["rolling_mean_24h"]
        )
        / df["rolling_std_24h"]
    )

    # Avoid infinite values
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return df


def main():
    print("Loading energy data...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Input records: {len(df):,}"
    )

    df = create_features(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Feature dataset created: "
        f"{len(df):,} records"
    )

    print("\nFeatures created:")

    feature_columns = [
        "hour",
        "day_of_week",
        "is_weekend",
        "is_off_hours",
        "hour_sin",
        "hour_cos",
        "expected_energy_kwh",
        "deviation_percent",
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

    for feature in feature_columns:
        print(f"  ✓ {feature}")

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()