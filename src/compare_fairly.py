
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Same chronological split as the original experiment.
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

# Load the previously trained Isolation Forest.
bundle = joblib.load("models/isolation_forest.joblib")
model = bundle["model"]
features = bundle["features"]

def choose_threshold(scores, labels):
    best_threshold = None
    best_f1 = -1

    for threshold in np.quantile(
        scores, np.linspace(0.50, 0.995, 200)
    ):
        predictions = (scores >= threshold).astype(int)
        score = f1_score(labels, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold

def evaluate(name, val_scores, test_scores):
    threshold = choose_threshold(
        val_scores, val["is_anomaly"].to_numpy()
    )

    predictions = (test_scores >= threshold).astype(int)
    labels = test["is_anomaly"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    auc = (
        roc_auc_score(labels, test_scores)
        if len(np.unique(labels)) == 2 else float("nan")
    )

    print(f"\n{name}")
    print(f"Validation-selected threshold: {threshold:.5f}")
    print(f"Precision: {precision_score(labels, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(labels, predictions, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(labels, predictions, zero_division=0):.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print(f"False alarms: {fp}")
    print(f"Missed anomalies: {fn}")

# Both scores increase as the reading becomes more suspicious.
if_val = -model.decision_function(val[features])
if_test = -model.decision_function(test[features])

temp_val = val["gearbox_bearing_temp"].to_numpy()
temp_test = test["gearbox_bearing_temp"].to_numpy()

print("Validation period:", val["timestamp"].min(), "to", val["timestamp"].max())
print("Test period:", test["timestamp"].min(), "to", test["timestamp"].max())
print("Validation anomalies:", int(val["is_anomaly"].sum()))
print("Test anomalies:", int(test["is_anomaly"].sum()))

evaluate("Isolation Forest", if_val, if_test)
evaluate("Bearing temperature baseline", temp_val, temp_test)

