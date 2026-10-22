
import pandas as pd

INPUT_FILE = "data/processed/validated_data.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

# Calculate the time difference between consecutive records
df["time_gap"] = df["timestamp"].diff()

print("Time gap distribution:")
print(df["time_gap"].value_counts().head(10))

# Find gaps greater than the expected 10 minutes
expected_gap = pd.Timedelta(minutes=10)
gaps = df.loc[df["time_gap"] > expected_gap, [
    "timestamp", "time_gap"
]]

print("\nNumber of gaps greater than 10 minutes:", len(gaps))
print("\nFirst 10 gaps:")
print(gaps.head(10))

# Estimate how many 10-minute intervals are missing
missing_intervals = (
    (gaps["time_gap"] / expected_gap).round() - 1
).clip(lower=0).sum()

print("\nEstimated missing intervals:", int(missing_intervals))

