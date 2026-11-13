
import pandas as pd
import joblib

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load("models/lof_anomaly_detector.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

df["score"] = -model.decision_function(df[features])
df["predicted_anomaly"] = (
    df["score"] >= threshold
).astype(int)

# A new alert event starts when a predicted alert
# follows a normal prediction.
df["alert_start"] = (
    df["predicted_anomaly"].eq(1)
    & df["predicted_anomaly"].shift(fill_value=0).eq(0)
)

# Count alert events that occur during labelled normal records.
false_alarm_starts = df[
    df["alert_start"] & df["is_anomaly"].eq(0)
]

print("FALSE ALARM EVENT SUMMARY")
print("Total alert events:", int(df["alert_start"].sum()))
print("Alert events starting during normal records:",
      len(false_alarm_starts))

print("\nFalse alarm start times:")
print(
    false_alarm_starts["timestamp"]
    .to_string(index=False)
)

