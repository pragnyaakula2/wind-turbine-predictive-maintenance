
import pandas as pd

INPUT_FILE = "data/raw/turbine_5yr_labeled_data.csv"
OUTPUT_FILE = "data/processed/validated_data.csv"

df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))

# 1. Convert timestamps
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamps = df["timestamp"].isna().sum()
print("Invalid timestamps:", invalid_timestamps)

# Remove records with invalid timestamps
df = df.dropna(subset=["timestamp"])

# 2. Check duplicate timestamps
duplicates = df.duplicated(
    subset=["timestamp"],
    keep=False
)

print("Rows with duplicate timestamps:", duplicates.sum())

# 3. Check chronological order
print(
    "Already chronological:",
    df["timestamp"].is_monotonic_increasing
)

# Sort by timestamp
df = df.sort_values("timestamp").reset_index(drop=True)

# 4. Check sensor ranges
sensor_ranges = {
    "gearbox_oil_temp": (-20, 150),
    "gearbox_bearing_temp": (-20, 150),
    "vibration_x": (0, 10),
    "vibration_y": (0, 10),
    "vibration_z": (0, 10),
    "oil_pressure": (0, 20),
    "particle_count": (0, 100000),
}

for column, (low, high) in sensor_ranges.items():
    invalid = ~df[column].between(low, high)
    print(f"{column} out-of-range rows:", invalid.sum())

# Save a validated, timestamp-sorted copy.
# Do not delete duplicate rows or alter sensor values yet.
df.to_csv(OUTPUT_FILE, index=False)

print("\nValidated dataset saved to:", OUTPUT_FILE)
print("Final rows:", len(df))

