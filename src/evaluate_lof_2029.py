
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

bundle = joblib.load("models/lof_anomaly_detector.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

# Evaluate the complete 2029 period, including normal records.
test = df[
    (df["timestamp"] >= "2029-01-01") &
    (df["timestamp"] < "2030-01-01")
].copy()

X = test[features]
y = test["is_anomaly"].astype(int)

scores = -model.decision_function(X)
pred = (scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y, pred, labels=[0, 1]
).ravel()

print("2029 EVENT EVALUATION")
print("Records:", len(test))
print("Actual anomalies:", int(y.sum()))
print("True positives:", tp)
print("False alarms:", fp)
print("Missed anomalies:", fn)
print("Precision:", round(precision_score(y, pred, zero_division=0), 3))
print("Recall:", round(recall_score(y, pred, zero_division=0), 3))
print("F1-score:", round(f1_score(y, pred, zero_division=0), 3))
print("ROC-AUC:", round(roc_auc_score(y, scores), 3))
print("Total alerts:", int(pred.sum()))

