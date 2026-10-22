
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

df = pd.read_csv("data/processed/training_data.csv")

bundle = joblib.load("models/lof_anomaly_detector.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

X = df[features]
y = df["is_anomaly"].astype(int)

scores = -model.decision_function(X)
predictions = (scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y, predictions, labels=[0, 1]
).ravel()

print("FULL DATASET EVALUATION")
print("Total records:", len(df))
print("Actual anomalies:", int(y.sum()))
print("True positives:", tp)
print("False alarms:", fp)
print("Missed anomalies:", fn)
print("True negatives:", tn)

print("\nClassification report:")
print(classification_report(
    y, predictions,
    target_names=["NORMAL", "ANOMALY"],
    zero_division=0
))

print("ROC-AUC:", round(roc_auc_score(y, scores), 4))


