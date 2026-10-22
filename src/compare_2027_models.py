
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

event = df[
    (df["timestamp"] >= "2027-04-15 04:00:00")
    & (df["timestamp"] <= "2027-04-25 13:50:00")
].copy()

y = event["is_anomaly"].astype(int)

models = [
    ("Isolation Forest", "models/isolation_forest.joblib"),
    ("Local Outlier Factor", "models/lof_anomaly_detector.joblib"),
]

for name, path in models:
    bundle = joblib.load(path)
    model = bundle["model"]
    features = bundle["features"]
    threshold = bundle["threshold"]

    scores = -model.decision_function(event[features])
    pred = (scores >= threshold).astype(int)

    print(f"\n{name}")
    print("Threshold:", round(threshold, 5))
    print("Precision:", round(precision_score(y, pred, zero_division=0), 3))
    print("Recall:", round(recall_score(y, pred, zero_division=0), 3))
    print("F1-score:", round(f1_score(y, pred, zero_division=0), 3))
    print("False alarms:", int(((pred == 1) & (y == 0)).sum()))
    print("Missed anomalies:", int(((pred == 0) & (y == 1)).sum()))

