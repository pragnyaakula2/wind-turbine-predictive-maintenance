
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_FILE = "data/processed/training_data.csv"
MODEL_FILE = "models/isolation_forest.joblib"

df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Exclude timestamps and the target label from model inputs.
# Using the anomaly label as an input would cause target leakage.
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

# Train only on records labelled normal.
X_train_normal = X_train[y_train == 0]

model = make_pipeline(
    StandardScaler(),
    IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    ),
)

model.fit(X_train_normal)

# Isolation Forest: lower decision_function values are more anomalous.
validation_scores = -model.decision_function(X_val)

# Choose a threshold using validation labels only.
threshold = np.inf
best_f1 = -1.0

from sklearn.metrics import f1_score

for candidate in np.quantile(
    validation_scores, np.linspace(0.50, 0.995, 100)
):
    predictions = (validation_scores >= candidate).astype(int)
    score = f1_score(y_val, predictions, zero_division=0)

    if score > best_f1:
        best_f1 = score
        threshold = candidate

# Evaluate once on the held-out test period.
test_scores = -model.decision_function(X_test)
test_predictions = (test_scores >= threshold).astype(int)

print("Threshold:", threshold)
print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification report:")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

    if y_test.nunique() == 2:
        auc = roc_auc_score(y_test, test_scores)
        print("ROC-AUC:", auc)
    else:
        auc = None

    import os
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "threshold": threshold,
        },
        MODEL_FILE,
    )

    print("\nModel saved to:", MODEL_FILE)

    # MLflow experiment tracking
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment("wind-turbine-anomaly-detection")
        with mlflow.start_run(run_name="isolation_forest"):
            mlflow.log_params({
                "model_type": "IsolationForest",
                "n_estimators": 200,
                "contamination": "auto",
                "random_state": 42,
                "threshold": float(threshold),
                "num_features": len(feature_columns),
            })

            metrics_to_log = {
                "val_best_f1": float(best_f1),
                "test_f1": float(f1_score(y_test, test_predictions, zero_division=0)),
            }
            if auc is not None:
                metrics_to_log["test_roc_auc"] = float(auc)

            mlflow.log_metrics(metrics_to_log)
            mlflow.sklearn.log_model(model, "isolation_forest_model")
            print("Logged experiment run to MLflow.")
    except Exception as exc:
        print(f"MLflow logging notice: {exc}")

