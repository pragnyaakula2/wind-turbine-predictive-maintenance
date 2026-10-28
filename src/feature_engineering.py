"""Feature engineering module for wind turbine sensor datasets."""

import os
from pathlib import Path
import pandas as pd

INPUT_FILE = "data/processed/validated_data.csv"
OUTPUT_FILE = "data/processed/features.csv"

SENSOR_COLUMNS = [
    "gearbox_oil_temp",
    "gearbox_bearing_temp",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "oil_pressure",
    "particle_count",
]


def engineer_features(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Computes temperature differential and 1-hour rolling statistics."""
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Temperature difference
    if "gearbox_oil_temp" in df.columns and "gearbox_bearing_temp" in df.columns:
        df["oil_bearing_temp_diff"] = (
            df["gearbox_oil_temp"] - df["gearbox_bearing_temp"]
        )

    # Rolling statistics: 1 hour = 6 readings at 10-minute intervals
    for column in SENSOR_COLUMNS:
        if column in df.columns:
            rolling = df[column].rolling(window=window, min_periods=window)
            df[f"{column}_mean_1h"] = rolling.mean()
            df[f"{column}_std_1h"] = rolling.std()
            df[f"{column}_min_1h"] = rolling.min()
            df[f"{column}_max_1h"] = rolling.max()

    return df


def run_feature_engineering(
    input_path: str = INPUT_FILE, output_path: str = OUTPUT_FILE
):
    """Loads validated dataset, applies feature engineering, and saves output."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    features_df = engineer_features(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print("Feature engineering complete.")
    print("Rows:", len(features_df))
    print("Columns:", len(features_df.columns))
    print("Saved to:", output_path)
    return features_df


if __name__ == "__main__":
    run_feature_engineering()
