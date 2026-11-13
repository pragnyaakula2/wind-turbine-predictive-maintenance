
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import precision_score, recall_score, f1_score

# Load the prepared dataset and saved model
df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load("models/isolation_forest.joblib")
model = bundle["model"]
features = bundle["features"]

# Use the same final 15% test period as the original experiment
test_start = int(len(df) * 0.85)
test = df.iloc[test_start:].copy()

# IMPORTANT: this script only examines thresholds on the
# existing test set for diagnosis. Do not use this test set
# to choose your final threshold.
scores = -model.decision_function(test[features])
actual = test["is_anomaly"].astype(int)

print("Test anomalies:", int(actual.sum()))
print("Test records:", len(test))
print("\nThreshold comparison (diagnostic only):")

for threshold in np.quantile(scores, [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]):
    predicted = (scores >= threshold).astype(int)

    print(
        f"Threshold={threshold:.4f} | "
        f"Alerts={predicted.sum():5d} | "
        f"Precision={precision_score(actual, predicted, zero_division=0):.3f} | "
        f"Recall={recall_score(actual, predicted, zero_division=0):.3f} | "
        f"F1={f1_score(actual, predicted, zero_division=0):.3f}"
    )

