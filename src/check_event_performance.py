
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load("models/final_anomaly_detector.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

df["score"] = -model.decision_function(df[features])
df["prediction"] = (df["score"] >= threshold).astype(int)

# Group consecutive anomalous records into events.
anomaly = df["is_anomaly"].eq(1)
event_start = anomaly & ~anomaly.shift(1, fill_value=False)
df["event_id"] = event_start.cumsum().where(anomaly)

events = df[anomaly].groupby("event_id").agg(
    start=("timestamp", "min"),
    end=("timestamp", "max"),
    records=("is_anomaly", "size"),
    detected=("prediction", "sum"),
)

events["recall"] = events["detected"] / events["records"]

print("ANOMALY EVENTS")
print(events.to_string())

print("\nOVERALL PERFORMANCE BY YEAR")
df["year"] = df["timestamp"].dt.year

for year, group in df.groupby("year"):
    actual = group["is_anomaly"]
    predicted = group["prediction"]

    print(
        f"{year}: records={len(group)}, "
        f"anomalies={int(actual.sum())}, "
        f"precision={precision_score(actual, predicted, zero_division=0):.3f}, "
        f"recall={recall_score(actual, predicted, zero_division=0):.3f}, "
        f"F1={f1_score(actual, predicted, zero_division=0):.3f}"
    )

