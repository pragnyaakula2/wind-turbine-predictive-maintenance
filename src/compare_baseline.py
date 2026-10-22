
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Use the same final 15% test period as the existing model.
test_start = int(len(df) * 0.85)
test = df.iloc[test_start:].copy()

y_true = test["is_anomaly"].astype(int)

# Simple baseline: flag unusually high bearing temperature.
# Threshold is learned from the earlier 85% only.
earlier = df.iloc[:test_start]
normal_bearing = earlier.loc[
    earlier["is_anomaly"] == 0, "gearbox_bearing_temp"
]

threshold = normal_bearing.quantile(0.99)
y_pred = (
    test["gearbox_bearing_temp"] > threshold
).astype(int)

print(f"Test period: {test['timestamp'].min()} to {test['timestamp'].max()}")
print(f"Bearing temperature threshold: {threshold:.3f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print(f"\nPrecision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")

print(
    "\nROC-AUC:",
    f"{roc_auc_score(y_true, test['gearbox_bearing_temp']):.3f}"
)

