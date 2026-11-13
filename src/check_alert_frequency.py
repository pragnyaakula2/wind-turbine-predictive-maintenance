
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

# Count alert records and separate them into continuous alert episodes.
df["episode"] = (
    df["alert"].ne(df["alert"].shift()).cumsum()
)

alerts = df[df["alert"] == 1].copy()

print("ALERT FREQUENCY")
print("Total alert records:", len(alerts))
print("Actual anomaly records:", int(alerts["is_anomaly"].sum()))
print("Alerts on normal records:",
      int((alerts["is_anomaly"] == 0).sum()))

print("\nYEARLY ALERT COUNTS")
print(
    df.groupby(df["timestamp"].dt.year)
      .agg(
          records=("alert", "size"),
          alerts=("alert", "sum"),
          actual_anomalies=("is_anomaly", "sum"),
      )
      .to_string()
)

print("\nCONTINUOUS ALERT EPISODES")
episodes = (
    alerts.groupby("episode")
    .agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        alert_records=("alert", "size"),
        actual_anomalies=("is_anomaly", "sum"),
    )
)

print("Total alert episodes:", len(episodes))
print(episodes.to_string())

