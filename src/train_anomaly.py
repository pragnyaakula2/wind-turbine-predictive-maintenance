
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_FILE = "data/processed/training_data.csv"
MODEL_FILE = "models/lof_anomaly_detector.joblib"

df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Exclude timestamp and target label to prevent target leakage.
feature_columns = [
    c for c in df.select_dtypes(include="number").columns
    if c != "is_anomaly"
]

X = df[feature_columns]
y = df["is_anomaly"]

# Chronological split: 70% train, 15% validation, 15% test.
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

# Train only on normal records.
X_train_normal = X_train[y_train == 0]

model = make_pipeline(
    StandardScaler(),
    LocalOutlierFactor(
        n_neighbors=20,
        contamination="auto",
        novelty=True,
        n_jobs=-1,
    ),
)

model.fit(X_train_normal)

# Higher scores indicate more anomalous observations.
validation_scores = -model.decision_function(X_val)

# Select threshold using validation labels only.
threshold = np.inf
best_f1 = -1.0

for candidate in np.quantile(
    validation_scores, np.linspace(0.50, 0.995, 100)
):
    predictions = (validation_scores >= candidate).astype(int)
    score = f1_score(y_val, predictions, zero_division=0)

    if score > best_f1:
        best_f1 = score
        threshold = candidate

# Evaluate on the held-out test period.
test_scores = -model.decision_function(X_test)
test_predictions = (test_scores >= threshold).astype(int)

print("Algorithm: Local Outlier Factor (LOF)")
print("Threshold:", threshold)
print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification report:")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

if y_test.nunique() == 2:
    print("ROC-AUC:", roc_auc_score(y_test, test_scores))

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": feature_columns,
        "threshold": threshold,
    },
    MODEL_FILE,
)

print("\nLOF model saved to:", MODEL_FILE)