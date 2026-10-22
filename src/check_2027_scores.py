
import pandas as pd
import joblib

df = pd.read_csv("data/processed/training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

bundle = joblib.load("models/final_anomaly_detector.joblib")
model = bundle["model"]
features = bundle["features"]

df["score"] = -model.decision_function(df[features])

groups = {
    "Normal records": df[df["is_anomaly"] == 0],
    "2027 anomalies": df[
        (df["is_anomaly"] == 1)
        & (df["timestamp"].dt.year == 2027)
    ],
    "2029 anomalies": df[
        (df["is_anomaly"] == 1)
        & (df["timestamp"].dt.year == 2029)
    ],
}

for name, group in groups.items():
    print(f"\n{name}")
    print("Records:", len(group))
    print("Mean score:", round(group["score"].mean(), 4))
    print("Median score:", round(group["score"].median(), 4))
    print("Maximum score:", round(group["score"].max(), 4))
    print(
        "Above threshold:",
        int((group["score"] >= bundle["threshold"]).sum()),
    )

