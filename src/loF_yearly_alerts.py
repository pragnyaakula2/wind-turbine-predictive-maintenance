
import pandas as pd
import joblib

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

bundle = joblib.load("models/lof_anomaly_detector.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

df["score"] = -model.decision_function(df[features])
df["predicted_anomaly"] = (df["score"] >= threshold).astype(int)
df["year"] = df["timestamp"].dt.year

print("YEARLY ALERT SUMMARY")
print(
    df.groupby("year").agg(
        records=("is_anomaly", "size"),
        actual_anomalies=("is_anomaly", "sum"),
        predicted_alerts=("predicted_anomaly", "sum"),
    ).to_string()
)

