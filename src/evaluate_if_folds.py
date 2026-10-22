
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

features = [
    c for c in df.select_dtypes(include=np.number).columns
    if c != "is_anomaly"
]

n = len(df)
fold_size = n // 5

for fold in range(1, 4):
    train_end = fold_size * (fold + 1)
    test_end = fold_size * (fold + 2) if fold < 3 else n

    train = df.iloc[:train_end]
    test = df.iloc[train_end:test_end]

    # Train only on normal records in the earlier period.
    normal_train = train[train["is_anomaly"] == 0]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(normal_train[features])

    # Higher scores indicate more anomalous records.
    val_start = int(len(train) * 0.8)
    validation = train.iloc[val_start:]
    validation_scores = -model.decision_function(
        validation[features]
    )

    # Select threshold using validation labels only.
    best_threshold = None
    best_f1 = -1

    for threshold in np.quantile(
        validation_scores, np.linspace(0.50, 0.995, 100)
    ):
        pred = (validation_scores >= threshold).astype(int)
        score = f1_score(
            validation["is_anomaly"], pred, zero_division=0
        )
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    test_scores = -model.decision_function(test[features])
    y_true = test["is_anomaly"].astype(int)
    y_pred = (test_scores >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    auc = (
        roc_auc_score(y_true, test_scores)
        if y_true.nunique() == 2 else float("nan")
    )

    print(f"\nFOLD {fold}")
    print(f"Train through: {train['timestamp'].max()}")
    print(f"Test: {test['timestamp'].min()} to {test['timestamp'].max()}")
    print(f"Test anomalies: {int(y_true.sum())}")
    print(f"Threshold: {best_threshold:.5f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score: {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"False alarms: {fp}")
    print(f"Missed anomalies: {fn}")

