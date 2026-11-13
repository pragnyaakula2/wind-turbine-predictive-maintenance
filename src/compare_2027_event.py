
import pandas as pd
import joblib

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

# Train using normal records from before the 2027 event.
train = df[
    (df["timestamp"] < "2027-04-15 04:00:00")
    & (df["is_anomaly"] == 0)
]

# Evaluate on the 2027 event and surrounding normal records.
test = df[
    (df["timestamp"] >= "2027-04-01")
    & (df["timestamp"] <= "2027-05-01")
].copy()

X_train = train[features]
X_test = test[features]
y_test = test["is_anomaly"].astype(int)

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

    scores = -model.decision_function(X_test)

    # Diagnostic only: use the known event labels to choose a threshold.
    # These results must not be reported as unbiased test performance.
    threshold = None
    best_f1 = -1

    for candidate in sorted(set(scores)):
        pred = (scores >= candidate).astype(int)
        f1 = f1_score(y_test, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            threshold = candidate

    pred = (scores >= threshold).astype(int)

    print(f"\n{name}")
    print(f"Diagnostic threshold: {threshold:.5f}")
    print(f"Precision: {precision_score(y_test, pred, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_test, pred, zero_division=0):.3f}")
    print(f"F1-score: {f1_score(y_test, pred, zero_division=0):.3f}")
    print(f"False alarms: {int(((pred == 1) & (y_test == 0)).sum())}")
    print(f"Missed anomalies: {int(((pred == 0) & (y_test == 1)).sum())}")

