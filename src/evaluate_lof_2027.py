
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

bundle = joblib.load("models/lof_anomaly_detector.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

# Evaluate only the 2027 anomaly event
test = df[
    (df["timestamp"] >= "2027-04-15 04:00:00")
    & (df["timestamp"] <= "2027-04-25 13:50:00")
].copy()

X = test[features]
y = test["is_anomaly"].astype(int)

scores = -model.decision_function(X)
predictions = (scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y, predictions, labels=[0, 1]
).ravel()

print("2027 EVENT EVALUATION")
print("Records:", len(test))
print("Actual anomalies:", int(y.sum()))
print("Threshold:", round(threshold, 5))
print("True positives:", tp)
print("False positives:", fp)
print("Missed anomalies:", fn)
print("Precision:", round(
    precision_score(y, predictions, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y, predictions, zero_division=0), 3
))
print("F1-score:", round(
    f1_score(y, predictions, zero_division=0), 3
))

