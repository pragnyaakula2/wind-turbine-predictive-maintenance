
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
df = df.sort_values("timestamp").reset_index(drop=True)

# Load thresholds selected using validation data
if_model = joblib.load("models/final_anomaly_detector.joblib")
lof_model = joblib.load("models/lof_anomaly_detector.joblib")

# Use normal records before April 2027 for training
train = df[
    (df["timestamp"] < "2027-04-01")
    & (df["is_anomaly"] == 0)
]

# Use March 2027 normal records for threshold selection
validation = df[
    (df["timestamp"] >= "2027-03-01")
    & (df["timestamp"] < "2027-04-01")
    & (df["is_anomaly"] == 0)
]

# Evaluate the event and surrounding records
test = df[
    (df["timestamp"] >= "2027-04-01")
    & (df["timestamp"] <= "2027-05-01")
].copy()

excluded = {"timestamp", "is_anomaly"}
features = [
    c for c in df.select_dtypes(include="number").columns
    if c not in excluded
]

X_train = train[features]
X_val = validation[features]
X_test = test[features]
y_test = test["is_anomaly"].astype(int)

models = {
    "Isolation Forest": if_model["model"],
    "Local Outlier Factor": lof_model["model"],
}

for name, bundle in [
    ("Isolation Forest", if_model),
    ("Local Outlier Factor", lof_model),
]:
    model = bundle["model"]
    threshold = bundle["threshold"]

    # Reuse saved models; do not retrain or change thresholds.
    scores = -model.decision_function(X_test)
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("Threshold:", round(threshold, 5))
    print("Actual anomalies:", int(y_test.sum()))
    print("True positives:", tp)
    print("False alarms:", fp)
    print("Missed anomalies:", fn)
    print("Precision:", round(
        precision_score(y_test, predictions, zero_division=0), 3
    ))
    print("Recall:", round(
        recall_score(y_test, predictions, zero_division=0), 3
    ))
    print("F1-score:", round(
        f1_score(y_test, predictions, zero_division=0), 3
    ))
    if y_test.nunique() == 2:
        print("ROC-AUC:", round(
            roc_auc_score(y_test, scores), 3
        ))

