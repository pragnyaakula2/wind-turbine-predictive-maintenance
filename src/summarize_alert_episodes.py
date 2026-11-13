
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
df["alert"] = (df["score"] >= threshold).astype(int)

# Merge alert periods separated by gaps of 30 minutes or less.
alert_rows = df[df["alert"] == 1].copy()

if alert_rows.empty:
    print("No alerts found.")
    raise SystemExit

alert_rows["gap_minutes"] = (
    alert_rows["timestamp"].diff().dt.total_seconds() / 60
)

alert_rows["episode"] = (
    alert_rows["gap_minutes"].isna()
    | (alert_rows["gap_minutes"] > 30)
).cumsum()

episodes = alert_rows.groupby("episode").agg(
    start=("timestamp", "min"),
    end=("timestamp", "max"),
    alert_records=("alert", "size"),
    actual_anomalies=("is_anomaly", "sum"),
)

episodes["duration_hours"] = (
    episodes["end"] - episodes["start"]
).dt.total_seconds() / 3600

episodes["contains_normal_records"] = (
    episodes["actual_anomalies"] < episodes["alert_records"]
)

print("ALERT EPISODE SUMMARY")
print("Raw alert records:", len(alert_rows))
print("Episodes after merging 30-minute gaps:", len(episodes))
print("Episodes containing normal-labelled alerts:",
      int(episodes["contains_normal_records"].sum()))

print("\nLONGEST EPISODES")
print(
    episodes.sort_values("duration_hours", ascending=False)
    .head(15)
    .to_string()
)

episodes.to_csv(
    "data/processed/merged_alert_episodes.csv",
    index=False,
)
print("\nSaved: data/processed/merged_alert_episodes.csv")

