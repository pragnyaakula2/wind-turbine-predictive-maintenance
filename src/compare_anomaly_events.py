
import pandas as pd

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

sensors = [
    "particle_count",
    "gearbox_bearing_temp",
    "gearbox_oil_temp",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "oil_pressure",
]

anomalies = df[df["is_anomaly"] == 1].copy()

anomalies["event"] = anomalies["timestamp"].dt.year.astype(str)

print("ANOMALY SENSOR AVERAGES BY YEAR")
print(
    anomalies.groupby("event")[sensors]
    .mean()
    .round(3)
    .to_string()
)

print("\nNORMAL SENSOR AVERAGES")
print(
    df[df["is_anomaly"] == 0][sensors]
    .mean()
    .round(3)
    .to_string()
)

print("\nANOMALY COUNTS BY YEAR")
print(anomalies.groupby("event").size().to_string())

