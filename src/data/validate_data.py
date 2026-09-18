import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "campus_energy_sample.csv"
)


def validate_energy_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Validates legacy energy data from file.
    Maintains full backward compatibility with the original baseline script.
    """
    target_path = Path(file_path) if file_path else DATA_FILE
    df = pd.read_csv(target_path)

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
    return df


def detect_dataset_schema(df: pd.DataFrame) -> str:
    """
    Infers whether a dataframe contains energy, water, waste, or unified multi-resource data.
    """
    cols = set(c.lower() for c in df.columns)
    
    has_energy = any(c in cols for c in ["energy_kwh", "power_kwh", "kwh", "energy"])
    has_water = any(c in cols for c in ["water_m3", "water_liters", "flow_rate", "water"])
    has_waste = any(c in cols for c in ["waste_kg", "total_waste_kg", "waste", "diverted_kg"])

    resource_count = sum([has_energy, has_water, has_waste])
    if resource_count >= 2:
        return "unified_multi_resource"
    elif has_energy:
        return "energy"
    elif has_water:
        return "water"
    elif has_waste:
        return "waste"
    return "custom"


def validate_multi_resource_data(
    data: Union[pd.DataFrame, str, Path, Any],
    expected_resource: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive multi-resource validator for CSV uploads or campus datasets.
    Supports Energy, Water, and Waste streams.
    Returns cleaned DataFrame, schema identification, metrics, and any validation errors/warnings.
    """
    errors = []
    warnings = []

    # 1. Load data
    if hasattr(data, "read"):
        try:
            # If it's a file-like object (e.g. Streamlit UploadedFile)
            data.seek(0)
            df = pd.read_csv(data)
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Failed to read CSV file: {str(e)}"],
                "warnings": [],
                "cleaned_df": None,
            }
    elif isinstance(data, (str, Path)):
        try:
            df = pd.read_csv(data)
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Failed to read CSV file: {str(e)}"],
                "warnings": [],
                "cleaned_df": None,
            }
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return {
            "is_valid": False,
            "errors": ["Input data must be a pandas DataFrame, file path, or file-like object."],
            "warnings": [],
            "cleaned_df": None,
        }

    if df.empty:
        return {
            "is_valid": False,
            "errors": ["Uploaded dataset is empty."],
            "warnings": [],
            "cleaned_df": None,
        }

    # Normalize column names for flexible matching
    col_mapping = {}
    for c in df.columns:
        c_clean = c.strip()
        c_lower = c_clean.lower()
        if c_lower in ["time", "datetime", "date_time", "timestamp"]:
            col_mapping[c] = "timestamp"
        elif c_lower in ["facility", "site", "location", "building", "building_name"]:
            col_mapping[c] = "building"
        elif c_lower in ["energy", "energy_kwh", "kwh", "electricity_kwh"]:
            col_mapping[c] = "energy_kwh"
        elif c_lower in ["water", "water_m3", "m3", "consumption_m3"]:
            col_mapping[c] = "water_m3"
        elif c_lower in ["waste", "waste_kg", "kg_waste", "total_waste_kg"]:
            col_mapping[c] = "waste_kg"

    df = df.rename(columns=col_mapping)

    # Check required core identifiers
    if "timestamp" not in df.columns:
        errors.append("Dataset is missing a 'timestamp' column.")
    if "building" not in df.columns:
        warnings.append("No 'building' column found. Defaulting building to 'Campus_Main'.")
        df["building"] = "Campus_Main"

    # Identify schema
    schema = detect_dataset_schema(df)

    # Check that at least one resource metric exists
    resource_columns = [c for c in ["energy_kwh", "water_m3", "waste_kg"] if c in df.columns]
    if not resource_columns:
        errors.append("No recognized resource column found (expected 'energy_kwh', 'water_m3', or 'waste_kg').")

    # If errors so far, return early
    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "detected_schema": schema,
            "cleaned_df": None,
        }

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_dates = df["timestamp"].isna().sum()
    if invalid_dates > 0:
        warnings.append(f"Found {invalid_dates:,} unparseable timestamps which were dropped.")
        df = df.dropna(subset=["timestamp"])

    if df.empty:
        return {
            "is_valid": False,
            "errors": ["All records contained invalid timestamps."],
            "warnings": warnings,
            "cleaned_df": None,
        }

    # Clean and validate numerical resource values
    for res_col in resource_columns:
        df[res_col] = pd.to_numeric(df[res_col], errors="coerce")
        null_count = df[res_col].isna().sum()
        if null_count > 0:
            median_val = df[res_col].median()
            df[res_col] = df[res_col].fillna(median_val)
            warnings.append(f"Column '{res_col}' had {null_count} null/non-numeric values; imputed with median ({median_val:.2f}).")

        negative_count = (df[res_col] < 0).sum()
        if negative_count > 0:
            df[res_col] = df[res_col].clip(lower=0)
            warnings.append(f"Column '{res_col}' contained {negative_count} negative values; clipped to 0.")

    # Sort sequentially
    df = df.sort_values(by=["building", "timestamp"]).reset_index(drop=True)

    buildings = sorted(df["building"].unique().tolist())
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()

    return {
        "is_valid": True,
        "errors": [],
        "warnings": warnings,
        "detected_schema": schema,
        "resource_columns": resource_columns,
        "row_count": len(df),
        "buildings": buildings,
        "date_range": (min_date, max_date),
        "cleaned_df": df,
    }


if __name__ == "__main__":
    validate_energy_data()