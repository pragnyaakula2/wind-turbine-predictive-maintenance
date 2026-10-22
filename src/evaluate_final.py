
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

THRESHOLD = 0.12737

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load("models/isolation_forest.joblib")
model = bundle["model"]
features = bundle["features"]

# Keep the final 15% as the untouched test period.
test_start = int(len(df) * 0.85)
test = df.iloc[test_start:].copy()

actual = test["is_anomaly"].astype(int)
scores = -model.decision_function(test[features])
predicted = (scores >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(
    actual, predicted, labels=[0, 1]
).ravel()

print("TEST PERIOD")
print(test["timestamp"].min(), "to", test["timestamp"].max())
print("\nConfusion matrix:")
print(confusion_matrix(actual, predicted, labels=[0, 1]))
print(f"\nTrue positives: {tp}")
print(f"False positives: {fp}")
print(f"True negatives: {tn}")
print(f"False negatives: {fn}")
print(f"Precision: {precision_score(actual, predicted, zero_division=0):.3f}")
print(f"Recall: {recall_score(actual, predicted, zero_division=0):.3f}")
print(f"F1-score: {f1_score(actual, predicted, zero_division=0):.3f}")
print(f"ROC-AUC: {roc_auc_score(actual, scores):.3f}")
print(f"Total alerts: {predicted.sum()}")

