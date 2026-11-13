
import pandas as pd
import joblib

# Load the saved LOF model
bundle = joblib.load("models/lof_anomaly_detector.joblib")

model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

# Load processed sensor data
df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Select a small sample from the latest available records
sample = df[df["is_anomaly"] == 0].tail(10).copy()

# Calculate anomaly scores
sample["anomaly_score"] = -model.decision_function(
    sample[features]
)

# Predict anomalies using the saved threshold
sample["predicted_anomaly"] = (
    sample["anomaly_score"] >= threshold
).astype(int)

sample["status"] = sample["predicted_anomaly"].map({
    0: "NORMAL",
    1: "ANOMALY ALERT"
})

print(sample[[
    "timestamp",
    "is_anomaly",
    "anomaly_score",
    "predicted_anomaly",
    "status"
]].to_string(index=False))

