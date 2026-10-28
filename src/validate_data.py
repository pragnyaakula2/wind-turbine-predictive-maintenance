"""Data validation script for wind turbine sensor measurements."""

import os
from pathlib import Path
import pandas as pd

INPUT_FILE = "data/raw/turbine_5yr_labeled_data.csv"
OUTPUT_FILE = "data/processed/validated_data.csv"

SENSOR_RANGES = {
    "gearbox_oil_temp": (-20, 150),
    "gearbox_bearing_temp": (-20, 150),
    "vibration_x": (0, 10),
    "vibration_y": (0, 10),
    "vibration_z": (0, 10),
    "oil_pressure": (0, 20),
    "particle_count": (0, 100000),
}


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates timestamps, ordering, and checks sensor value ranges."""
    df = df.copy()

    # 1. Convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_timestamps = int(df["timestamp"].isna().sum())
    print("Invalid timestamps:", invalid_timestamps)

    # Remove records with invalid timestamps
    df = df.dropna(subset=["timestamp"])

    # 2. Check duplicate timestamps
    duplicates = int(df.duplicated(subset=["timestamp"], keep=False).sum())
    print("Rows with duplicate timestamps:", duplicates)

    # 3. Chronological sorting
    print("Already chronological:", bool(df["timestamp"].is_monotonic_increasing))
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 4. Check sensor ranges
    for column, (low, high) in SENSOR_RANGES.items():
        if column in df.columns:
            invalid = ~df[column].between(low, high)
            print(f"{column} out-of-range rows:", int(invalid.sum()))

    return df


def run_validation(input_path: str = INPUT_FILE, output_path: str = OUTPUT_FILE):
    """Loads input CSV, validates data, and saves validated output."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {input_path}")

    df = pd.read_csv(input_path)
    print("Original rows:", len(df))

    validated_df = validate_data(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(output_path, index=False)

    print("\nValidated dataset saved to:", output_path)
    print("Final rows:", len(validated_df))
    return validated_df


if __name__ == "__main__":
    run_validation()
