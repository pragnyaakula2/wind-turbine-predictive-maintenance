
import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/features.csv"
OUTPUT_FILE = "data/processed/training_data.csv"

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# Remove rows where rolling features are unavailable.
rolling_columns = [
    c for c in df.columns
    if c.endswith(("_mean_1h", "_std_1h", "_min_1h", "_max_1h"))
]

df = df.dropna(subset=rolling_columns).copy()

# Check for infinite values.
numeric_columns = df.select_dtypes(include="number").columns
df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows with missing values in required model inputs or labels.
feature_columns = [
    c for c in numeric_columns
    if c != "is_anomaly"
]

df = df.dropna(
    subset=feature_columns + ["is_anomaly"]
).copy()

# Ensure labels are binary.
df = df[df["is_anomaly"].isin([0, 1])].copy()
df["is_anomaly"] = df["is_anomaly"].astype(int)

df.to_csv(OUTPUT_FILE, index=False)

print("Final shape:", df.shape)
print("\nAnomaly distribution:")
print(df["is_anomaly"].value_counts())

print("\nSaved to:", OUTPUT_FILE)

