
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

df = pd.read_csv(
    "data/processed/training_data.csv",
    parse_dates=["timestamp"],
).sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load("models/isolation_forest.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

# Recreate the exact chronological test split
test_start = int(len(df) * 0.85)
test = df.iloc[test_start:].copy()

X_test = test[features]
y_test = test["is_anomaly"]

scores = -model.decision_function(X_test)
pred = (scores >= threshold).astype(int)

print("Test period:")
print(test["timestamp"].min(), "to", test["timestamp"].max())

print("\nAnomaly rate by chronological split:")
for name, start, end in [
    ("Train", 0, int(len(df) * 0.70)),
    ("Validation", int(len(df) * 0.70), test_start),
    ("Test", test_start, len(df)),
]:
    part = df.iloc[start:end]
    print(name, part["is_anomaly"].mean())

print("\nScore summary by actual label:")
print(pd.DataFrame({
    "score": scores,
    "actual_label": y_test.to_numpy(),
}).groupby("actual_label")["score"].describe())

print("\nROC-AUC:", roc_auc_score(y_test, scores))

# Find the sensor features most different between classes
print("\nLargest differences in sensor means:")
summary = df.groupby("is_anomaly")[features].mean().T
summary["absolute_difference"] = (
    summary[1] - summary[0]
).abs()
print(summary.sort_values(
    "absolute_difference", ascending=False
).head(10))

