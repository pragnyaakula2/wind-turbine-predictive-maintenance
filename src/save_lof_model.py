
import pandas as pd
import numpy as np
import joblib

from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

features = [
    c for c in df.select_dtypes(include="number").columns
    if c != "is_anomaly"
]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

X_train = train.loc[train["is_anomaly"] == 0, features]
X_val = val[features]
X_test = test[features]

model = make_pipeline(
    StandardScaler(),
    LocalOutlierFactor(
        n_neighbors=35,
        novelty=True,
        contamination="auto",
    ),
)
model.fit(X_train)

val_scores = -model.decision_function(X_val)
y_val = val["is_anomaly"].to_numpy()

best = None
for threshold in np.quantile(
    val_scores, np.linspace(0.50, 0.995, 200)
):
    pred = (val_scores >= threshold).astype(int)
    recall = recall_score(y_val, pred, zero_division=0)
    precision = precision_score(y_val, pred, zero_division=0)
    f1 = f1_score(y_val, pred, zero_division=0)

    if recall >= 0.85 and (
        best is None or precision > best[0]
    ):
        best = (precision, f1, threshold, recall)

if best is None:
    raise RuntimeError("No validation threshold reached 85% recall.")

precision, f1, threshold, recall = best

test_scores = -model.decision_function(X_test)
y_test = test["is_anomaly"].to_numpy()
test_pred = (test_scores >= threshold).astype(int)

print("Validation-selected threshold:", round(threshold, 5))
print("Validation precision:", round(precision, 3))
print("Validation recall:", round(recall, 3))
print("Validation F1:", round(f1, 3))
print("Test precision:", round(precision_score(y_test, test_pred, zero_division=0), 3))
print("Test recall:", round(recall_score(y_test, test_pred, zero_division=0), 3))
print("Test F1:", round(f1_score(y_test, test_pred, zero_division=0), 3))

joblib.dump(
    {
        "model": model,
        "features": features,
        "threshold": float(threshold),
    },
    "models/lof_anomaly_detector.joblib",
)

print("Saved: models/lof_anomaly_detector.joblib")

