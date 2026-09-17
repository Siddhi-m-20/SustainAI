import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AnomalyService:
    """
    Unified Multi-Resource Anomaly Detection and Diagnostic Service.
    Integrates contextual median baselines, high-dimensional feature engineering,
    Isolation Forest ML, and multi-factor severity scoring.
    """

    def __init__(self, contamination: float = 0.025, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.models: Dict[str, IsolationForest] = {}
        self.baselines: Dict[str, pd.DataFrame] = {}

    def extract_features(
        self,
        df: pd.DataFrame,
        resource_col: str = "energy_kwh",
        building_col: str = "building",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Calculates time, cyclical, baseline, rolling, and lag features
        following the proven mathematical pipeline from feature_engineering.py.
        """
        data = df.copy()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
            data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")

        data = data.sort_values([building_col, timestamp_col]).reset_index(drop=True)

        # 1. Time & Cyclical features
        data["hour"] = data[timestamp_col].dt.hour
        data["day_of_week"] = data[timestamp_col].dt.dayofweek
        data["month"] = data[timestamp_col].dt.month
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
        data["is_off_hours"] = ((data["hour"] < 7) | (data["hour"] > 19)).astype(int)

        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        # 2. Contextual Median Baseline (by building, day_of_week, hour)
        # If ground-truth anomaly column exists, filter normal points; otherwise use full series
        if "is_injected_anomaly" in data.columns:
            normal_data = data[data["is_injected_anomaly"] == False]
        else:
            normal_data = data

        expected_col = f"expected_{resource_col}"
        baseline = (
            normal_data
            .groupby([building_col, "day_of_week", "hour"])[resource_col]
            .median()
            .reset_index()
            .rename(columns={resource_col: expected_col})
        )

        data = data.merge(baseline, on=[building_col, "day_of_week", "hour"], how="left")

        # Fallback if any baseline is missing
        if data[expected_col].isna().any():
            overall_med = data[resource_col].median()
            data[expected_col] = data[expected_col].fillna(overall_med)

        # Prevent zero-division in baseline
        data[expected_col] = data[expected_col].replace(0, 0.001)

        # 3. Deviations
        dev_col = f"deviation_percent_{resource_col}"
        abs_dev_col = f"abs_deviation_percent_{resource_col}"
        data[dev_col] = ((data[resource_col] - data[expected_col]) / data[expected_col]) * 100.0
        data[abs_dev_col] = data[dev_col].abs()

        # 4. Rolling Statistics and Lags (by building)
        grouped = data.groupby(building_col)[resource_col]

        data["rolling_mean_6h"] = grouped.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
        data["rolling_std_6h"] = grouped.transform(lambda s: s.shift(1).rolling(6, min_periods=1).std()).fillna(0.0)
        data["rolling_mean_24h"] = grouped.transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
        data["rolling_std_24h"] = grouped.transform(lambda s: s.shift(1).rolling(24, min_periods=1).std()).fillna(0.0)

        data["lag_1h"] = grouped.transform(lambda s: s.shift(1)).fillna(data[resource_col])
        data["lag_24h"] = grouped.transform(lambda s: s.shift(24)).fillna(data[resource_col])

        data["change_from_1h"] = data[resource_col] - data["lag_1h"]
        data["change_from_24h"] = data[resource_col] - data["lag_24h"]

        # 5. Rolling Z-Score
        std_safe = data["rolling_std_24h"].replace(0, 0.001)
        z_col = f"z_score_{resource_col}"
        data[z_col] = (data[resource_col] - data["rolling_mean_24h"]) / std_safe
        data[z_col] = data[z_col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return data

    def detect(
        self,
        df: pd.DataFrame,
        resource_col: str = "energy_kwh",
        building_col: str = "building",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Runs full feature engineering + IsolationForest + multi-factor scoring
        on a specific resource column.
        """
        data = self.extract_features(df, resource_col, building_col, timestamp_col)

        feature_cols = [
            resource_col,
            f"expected_{resource_col}",
            f"deviation_percent_{resource_col}",
            f"abs_deviation_percent_{resource_col}",
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
            f"z_score_{resource_col}",
        ]

        X = data[feature_cols].copy()
        X = X.fillna(0.0)

        # Isolation Forest Model
        model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )
        model.fit(X)
        self.models[resource_col] = model

        # Predictions: 1 is normal, -1 is anomaly
        raw_pred = model.predict(X)
        anomaly_scores = model.decision_function(X)  # Lower is more abnormal

        # Invert score so higher is more anomalous
        ml_score = (1.0 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-6)) * 100.0

        # Multi-factor Severity Score (0 - 100)
        abs_dev = data[f"abs_deviation_percent_{resource_col}"].clip(upper=150.0)
        z_mag = data[f"z_score_{resource_col}"].abs().clip(upper=6.0)

        # Composite severity formulation
        severity = (
            (abs_dev / 150.0) * 40.0
            + (z_mag / 6.0) * 35.0
            + (ml_score / 100.0) * 25.0
        ).clip(0.0, 100.0).round(1)

        is_anomaly = (raw_pred == -1) | (severity >= 50.0)

        pred_col = f"is_anomaly_{resource_col}"
        sev_col = f"severity_{resource_col}"
        tier_col = f"severity_tier_{resource_col}"
        pattern_col = f"pattern_{resource_col}"

        data[pred_col] = is_anomaly
        data[sev_col] = severity

        # Severity Tiers
        conditions = [
            data[sev_col] >= 75.0,
            data[sev_col] >= 50.0,
            data[sev_col] >= 30.0,
        ]
        choices = ["Critical", "High", "Medium"]
        data[tier_col] = np.select(conditions, choices, default="Low")

        # Classify Anomaly Pattern Archetype
        patterns = []
        for _, row in data.iterrows():
            if not row[pred_col]:
                patterns.append("Normal")
                continue

            dev = row[f"deviation_percent_{resource_col}"]
            off_hrs = row["is_off_hours"]
            chg1 = row["change_from_1h"]
            z = row[f"z_score_{resource_col}"]

            if off_hrs and dev > 25.0:
                patterns.append("Off-Hours Excess")
            elif chg1 > (row[f"expected_{resource_col}"] * 0.4) and dev > 40.0:
                patterns.append("Sudden Surge / Spike")
            elif dev > 20.0 and z > 1.8:
                patterns.append("Continuous Baseline Drift")
            elif dev < -30.0:
                patterns.append("Unusual Drop / Disruption")
            else:
                patterns.append("Erratic Fluctuation")

        data[pattern_col] = patterns
        return data

    def extract_incidents(
        self,
        scored_df: pd.DataFrame,
        resource_col: str = "energy_kwh",
        resource_label: str = "Energy",
        min_severity: float = 40.0,
    ) -> pd.DataFrame:
        """
        Extracts a clean, ranked incident ledger for the Investigate view.
        """
        pred_col = f"is_anomaly_{resource_col}"
        sev_col = f"severity_{resource_col}"

        if pred_col not in scored_df.columns:
            return pd.DataFrame()

        mask = (scored_df[pred_col] == True) & (scored_df[sev_col] >= min_severity)
        anomalies = scored_df[mask].copy()

        if anomalies.empty:
            return pd.DataFrame()

        incidents = []
        for idx, row in anomalies.iterrows():
            actual = float(row[resource_col])
            expected = float(row[f"expected_{resource_col}"])
            delta = actual - expected
            dev_pct = float(row[f"deviation_percent_{resource_col}"])
            sev = float(row[sev_col])
            tier = row[f"severity_tier_{resource_col}"]
            pattern = row[f"pattern_{resource_col}"]
            building = row.get("building", "Campus")
            ts = row["timestamp"]

            # Cost and impact estimation
            if "energy" in resource_col:
                cost_impact = max(delta, 0) * 0.14
                carbon_impact_kg = max(delta, 0) * 0.42
                unit = "kWh"
            elif "water" in resource_col:
                cost_impact = max(delta, 0) * 3.80
                carbon_impact_kg = max(delta, 0) * 0.15
                unit = "m³"
            else:  # waste
                cost_impact = max(delta, 0) * 0.12
                carbon_impact_kg = max(delta, 0) * 0.58
                unit = "kg"

            incidents.append({
                "incident_id": f"INC-{resource_label[:3].upper()}-{idx:05d}",
                "timestamp": ts,
                "building": building,
                "resource": resource_label,
                "actual_value": round(actual, 2),
                "expected_value": round(expected, 2),
                "excess_consumption": round(max(delta, 0.0), 2),
                "unit": unit,
                "deviation_percent": round(dev_pct, 1),
                "severity_score": round(sev, 1),
                "severity_tier": tier,
                "pattern": pattern,
                "estimated_cost_loss_usd": round(cost_impact, 2),
                "excess_carbon_kg": round(carbon_impact_kg, 2),
            })

        incident_df = pd.DataFrame(incidents)
        incident_df = incident_df.sort_values(by="severity_score", ascending=False).reset_index(drop=True)
        return incident_df

    def detect_multi_resource(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs anomaly detection across all available resources in the dataset.
        Returns:
            (fully_scored_df, unified_incident_ledger)
        """
        scored_df = df.copy()
        all_incidents = []

        resource_configs = [
            ("energy_kwh", "Energy"),
            ("water_m3", "Water"),
            ("waste_kg", "Waste"),
        ]

        for col, label in resource_configs:
            if col in scored_df.columns:
                scored_df = self.detect(scored_df, resource_col=col)
                incidents = self.extract_incidents(scored_df, resource_col=col, resource_label=label)
                if not incidents.empty:
                    all_incidents.append(incidents)

        if all_incidents:
            unified_ledger = pd.concat(all_incidents, ignore_index=True)
            unified_ledger = unified_ledger.sort_values(by="severity_score", ascending=False).reset_index(drop=True)
        else:
            unified_ledger = pd.DataFrame()

        return scored_df, unified_ledger

    def detect_multi_resource_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience wrapper returning the unified anomaly ledger with normalized column names.
        """
        _, ledger = self.detect_multi_resource(df)
        if not ledger.empty and "deviation_percent" in ledger.columns:
            ledger["deviation_pct"] = ledger["deviation_percent"]
        return ledger

