
import pandas as pd

file_path = "data/raw/turbine_5yr_labeled_data.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nAnomaly label counts:")
print(df["is_anomaly"].value_counts(dropna=False))

