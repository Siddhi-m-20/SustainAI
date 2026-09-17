import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


class ForecastService:
    """
    Demand Forecasting Service for EcoSync Resource Manager.
    Predicts multi-resource demand (Energy kWh, Water m³, Waste kg)
    for individual facilities or campus-wide over 24h, 48h, or 7d horizons.
    Provides point estimates, uncertainty confidence intervals, and peak alerts.
    """

    RESOURCE_MAP = {
        "energy": "energy_kwh",
        "water": "water_m3",
        "waste": "waste_kg",
        "waste_kg": "waste_kg",
        "waste_total_kg": "waste_total_kg",
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.residual_stds: Dict[str, float] = {}

    def _engineer_features(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Creates time-based and cyclical features from timestamps.
        """
        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
            data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")

        data["hour"] = data[timestamp_col].dt.hour
        data["day_of_week"] = data[timestamp_col].dt.dayofweek
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(float)
        data["is_off_hours"] = ((data["hour"] < 7) | (data["hour"] > 19)).astype(float)

        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)

        return data

    def fit(
        self,
        df: pd.DataFrame,
        resource: str = "energy",
        building: Optional[str] = None,
        model_type: str = "ridge",
    ) -> None:
        """
        Fits a demand model on historical data.
        """
        target_col = self.RESOURCE_MAP.get(resource.lower(), resource)
        if target_col not in df.columns:
            if target_col == "waste_kg" and "waste_total_kg" in df.columns:
                target_col = "waste_total_kg"
            elif target_col == "waste_total_kg" and "waste_kg" in df.columns:
                target_col = "waste_kg"
            else:
                raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        train_data = df.copy()
        if building and building != "ALL" and "building" in train_data.columns:
            train_data = train_data[train_data["building"] == building]

        train_data = self._engineer_features(train_data)
        train_data = train_data.dropna(subset=[target_col, "hour_sin"])

        features = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend", "is_off_hours"]
        X = train_data[features]
        y = train_data[target_col]

        if model_type == "rf":
            model = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=self.random_state)
        else:
            model = Ridge(alpha=1.0)

        model.fit(X, y)
        preds = model.predict(X)
        residuals = y - preds
        std_resid = float(np.std(residuals)) if len(residuals) > 1 else 1.0

        key = f"{resource}_{building or 'ALL'}"
        self.models[key] = model
        self.residual_stds[key] = max(std_resid, 0.05 * float(y.mean() if y.mean() > 0 else 1.0))

    def predict(
        self,
        df: pd.DataFrame,
        resource: str = "energy",
        building: Optional[str] = None,
        horizon_hours: int = 48,
    ) -> pd.DataFrame:
        """
        Generates future predictions for the given horizon.
        Returns a DataFrame with timestamp, predicted_mean, lower_95, upper_95, and is_peak_demand.
        """
        target_col = self.RESOURCE_MAP.get(resource.lower(), resource)
        key = f"{resource}_{building or 'ALL'}"

        if key not in self.models:
            self.fit(df, resource=resource, building=building)

        model = self.models[key]
        std_resid = self.residual_stds.get(key, 1.0)

        # Determine last timestamp
        ts_col = "timestamp" if "timestamp" in df.columns else df.select_dtypes(include=["datetime"]).columns[0]
        last_ts = pd.to_datetime(df[ts_col]).max()

        future_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")
        future_df = pd.DataFrame({"timestamp": future_ts})
        future_feat = self._engineer_features(future_df)

        features = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend", "is_off_hours"]
        X_future = future_feat[features]

        y_pred = model.predict(X_future)
        # Demand cannot be negative
        y_pred = np.maximum(y_pred, 0.0)

        lower_95 = np.maximum(y_pred - 1.96 * std_resid, 0.0)
        upper_95 = y_pred + 1.96 * std_resid

        peak_threshold = np.percentile(y_pred, 90) if len(y_pred) > 5 else np.max(y_pred)
        is_peak = y_pred >= peak_threshold

        res_df = pd.DataFrame({
            "timestamp": future_ts,
            "predicted_demand": np.round(y_pred, 2),
            "lower_95": np.round(lower_95, 2),
            "upper_95": np.round(upper_95, 2),
            "is_peak_forecast": is_peak,
            "resource": resource,
            "building": building or "ALL",
        })

        return res_df

    def get_forecast_kpis(self, forecast_df: pd.DataFrame, resource: str = "energy") -> Dict[str, Any]:
        """
        Calculates high-level forecast summary metrics.
        """
        total = float(forecast_df["predicted_demand"].sum())
        peak = float(forecast_df["predicted_demand"].max())
        mean_demand = float(forecast_df["predicted_demand"].mean())
        peak_time = str(forecast_df.loc[forecast_df["predicted_demand"].idxmax(), "timestamp"])
        peak_hours_count = int(forecast_df["is_peak_forecast"].sum())

        summary = {
            "total_predicted": round(total, 2),
            "peak_predicted": round(peak, 2),
            "mean_hourly_predicted": round(mean_demand, 2),
            "peak_forecast_time": peak_time,
            "peak_alert_hours": peak_hours_count,
            "horizon_hours": len(forecast_df),
        }

        if resource == "energy":
            summary["projected_cost_usd"] = round(total * 0.14, 2)
            summary["projected_emissions_tco2e"] = round((total * 0.42) / 1000.0, 3)
            summary["unit"] = "kWh"
        elif resource == "water":
            summary["projected_cost_usd"] = round(total * 3.80, 2)
            summary["unit"] = "m³"
        else:
            summary["projected_cost_usd"] = round(total * 0.12, 2)
            summary["unit"] = "kg"

        return summary
