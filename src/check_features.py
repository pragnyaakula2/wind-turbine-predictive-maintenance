
import numpy as np
import pandas as pd

FILE = "data/processed/features.csv"
df = pd.read_csv(FILE)

print("Dataset shape:", df.shape)

# Check missing values
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

# Check infinite values in numeric columns
numeric = df.select_dtypes(include="number")
infinite_counts = np.isinf(numeric).sum()

print("\nColumns with infinite values:")
print(infinite_counts[infinite_counts > 0])

# Check anomaly labels
print("\nAnomaly label distribution:")
print(df["is_anomaly"].value_counts(dropna=False))

# Verify the temperature difference
expected_diff = (
    df["gearbox_oil_temp"] - df["gearbox_bearing_temp"]
)

print(
    "\nTemperature difference correct:",
    np.allclose(
        df["oil_bearing_temp_diff"],
        expected_diff,
        equal_nan=True,
    ),
)

# Check that rolling features exist
rolling_columns = [
    c for c in df.columns if c.endswith("_mean_1h")
]

print("\nNumber of 1-hour rolling means:", len(rolling_columns))
print("\nFeature validation complete.")

