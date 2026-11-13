
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load("models/isolation_forest.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

# Inspect the final 15%, matching the original test period.
test = df.iloc[int(len(df) * 0.85):].copy()

test["if_score"] = -model.decision_function(test[features])
test["temp_score"] = test["gearbox_bearing_temp"]

print("Saved Isolation Forest threshold:", threshold)

print("\nIsolation Forest score by actual label:")
print(
    test.groupby("is_anomaly")["if_score"]
    .agg(["count", "min", "median", "mean", "max"])
    .round(4)
)

print("\nBearing temperature by actual label:")
print(
    test.groupby("is_anomaly")["temp_score"]
    .agg(["count", "min", "median", "mean", "max"])
    .round(3)
)

print("\nIsolation Forest threshold and test scores:")
print("Normal records above threshold:",
      ((test["is_anomaly"] == 0) &
       (test["if_score"] >= threshold)).sum())
print("Anomalies below threshold:",
      ((test["is_anomaly"] == 1) &
       (test["if_score"] < threshold)).sum())

