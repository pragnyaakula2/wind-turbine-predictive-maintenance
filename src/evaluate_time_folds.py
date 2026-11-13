
import pandas as pd
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

# Three expanding-window evaluations.
# Each test fold is a later, non-overlapping time period.
n = len(df)
fold_size = n // 5
results = []

for fold in range(1, 4):
    train_end = fold_size * (fold + 1)
    test_end = fold_size * (fold + 2) if fold < 3 else n

    train = df.iloc[:train_end]
    test = df.iloc[train_end:test_end]

    normal_temps = train.loc[
        train["is_anomaly"] == 0, "gearbox_bearing_temp"
    ]

    threshold = normal_temps.quantile(0.99)

    y_true = test["is_anomaly"].astype(int)
    scores = test["gearbox_bearing_temp"]
    y_pred = (scores > threshold).astype(int)

    if y_true.nunique() < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, scores)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "fold": fold,
        "train_end": train["timestamp"].max(),
        "test_start": test["timestamp"].min(),
        "test_end": test["timestamp"].max(),
        "threshold": threshold,
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "false_alarms": fp,
        "missed_anomalies": fn,
        "test_anomalies": int(y_true.sum()),
    })

print(pd.DataFrame(results).round(3).to_string(index=False))

