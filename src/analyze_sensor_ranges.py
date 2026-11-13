
import pandas as pd

df = pd.read_csv("data/processed/training_data.csv")

sensors = [
    "particle_count",
    "gearbox_bearing_temp",
    "gearbox_oil_temp",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "oil_pressure",
]

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year

for year in [2025, 2026, 2027, 2028, 2029]:
    normal = df[
        (df["year"] == year) & (df["is_anomaly"] == 0)
    ]
    anomalous = df[
        (df["year"] == year) & (df["is_anomaly"] == 1)
    ]

    print(f"\nYEAR {year}")
    print("Normal records:", len(normal))
    print("Anomalous records:", len(anomalous))

    if len(anomalous) > 0:
        print("\nNormal sensor averages:")
        print(normal[sensors].mean().round(3))

        print("\nAnomaly sensor averages:")
        print(anomalous[sensors].mean().round(3))

