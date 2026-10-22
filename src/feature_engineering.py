
import pandas as pd

INPUT_FILE = "data/processed/validated_data.csv"
OUTPUT_FILE = "data/processed/features.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

sensor_columns = [
    "gearbox_oil_temp",
    "gearbox_bearing_temp",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "oil_pressure",
    "particle_count",
]

# Temperature difference
df["oil_bearing_temp_diff"] = (
    df["gearbox_oil_temp"] - df["gearbox_bearing_temp"]
)

# 1 hour = 6 readings at 10-minute intervals
window = 6

for column in sensor_columns:
    rolling = df[column].rolling(window=window, min_periods=window)

    df[f"{column}_mean_1h"] = rolling.mean()
    df[f"{column}_std_1h"] = rolling.std()
    df[f"{column}_min_1h"] = rolling.min()
    df[f"{column}_max_1h"] = rolling.max()

df.to_csv(OUTPUT_FILE, index=False)

print("Feature engineering complete.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved to:", OUTPUT_FILE)

