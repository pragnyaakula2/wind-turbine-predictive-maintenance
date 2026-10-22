
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
df["predicted_anomaly"] = (df["score"] >= threshold).astype(int)

df["event_group"] = (df["is_anomaly"].diff().fillna(0) != 0).cumsum()
events = df[df["is_anomaly"] == 1].groupby("event_group")

print("FIRST ALERT DURING EACH ANOMALY EVENT")

for event_id, event in events:
    start = event["timestamp"].min()
    end = event["timestamp"].max()

    alerts = event[event["predicted_anomaly"] == 1]

    print(f"\nEvent {event_id}")
    print("Actual anomaly starts:", start)
    print("Actual anomaly ends:", end)

    if alerts.empty:
        print("No alerts detected.")
    else:
        first_alert = alerts["timestamp"].min()
        delay = (first_alert - start).total_seconds() / 3600

        print("First alert:", first_alert)
        print("Hours after anomaly started:", round(delay, 2))

