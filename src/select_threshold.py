
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load("models/isolation_forest.joblib")
model = bundle["model"]
features = bundle["features"]

# Use the first 70% for model training,
# the next 15% for validation, and keep the final 15% untouched.
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

validation = df.iloc[train_end:val_end].copy()

scores = -model.decision_function(validation[features])
actual = validation["is_anomaly"].astype(int)

print("Validation anomalies:", int(actual.sum()))
print("Validation records:", len(validation))

best = None

for threshold in np.quantile(scores, np.linspace(0.50, 0.995, 200)):
    predicted = (scores >= threshold).astype(int)

    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)

    # Require at least 85% recall, then maximize precision.
    if recall >= 0.85:
        candidate = (precision, f1, threshold, recall, predicted.sum())
        if best is None or candidate[0] > best[0]:
            best = candidate

if best is None:
    print("No tested threshold achieved 85% validation recall.")
else:
    precision, f1, threshold, recall, alerts = best
    print("\nSelected using validation data only:")
    print(f"Threshold: {threshold:.5f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"Alerts: {alerts}")

