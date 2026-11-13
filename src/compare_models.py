
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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

# Train only on normal records from the training period.
X_train = train.loc[train["is_anomaly"] == 0, features]
X_val = val[features]
X_test = test[features]

models = {
    "Isolation Forest": make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "Local Outlier Factor": make_pipeline(
        StandardScaler(),
        LocalOutlierFactor(
            n_neighbors=35,
            novelty=True,
            contamination="auto",
        ),
    ),
}

for name, model in models.items():
    model.fit(X_train)

    val_scores = -model.decision_function(X_val)
    test_scores = -model.decision_function(X_test)

    # Choose threshold on validation data only.
    candidates = np.quantile(
        val_scores, np.linspace(0.50, 0.995, 200)
    )

    best = None
    for threshold in candidates:
        pred = (val_scores >= threshold).astype(int)
        y = val["is_anomaly"].to_numpy()

        recall = recall_score(y, pred, zero_division=0)
        precision = precision_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)

        if recall >= 0.85 and (
            best is None or precision > best[0]
        ):
            best = (precision, recall, f1, threshold)

    print(f"\n{name}")
    if best is None:
        print("No validation threshold achieved 85% recall.")
        continue

    precision, recall, f1, threshold = best
    test_pred = (test_scores >= threshold).astype(int)
    y_test = test["is_anomaly"].to_numpy()

    print(f"Validation threshold: {threshold:.5f}")
    print(
        "Test precision:",
        round(precision_score(y_test, test_pred, zero_division=0), 3),
    )
    print(
        "Test recall:",
        round(recall_score(y_test, test_pred, zero_division=0), 3),
    )
    print(
        "Test F1:",
        round(f1_score(y_test, test_pred, zero_division=0), 3),
    )

    event_2027 = test[
        (test["timestamp"].dt.year == 2027)
        & (test["is_anomaly"] == 1)
    ]
    print("2027 anomalies in test period:", len(event_2027))

