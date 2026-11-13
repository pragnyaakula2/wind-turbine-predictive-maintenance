
import pandas as pd

df = pd.read_csv("data/processed/validated_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

print("\n1. SENSOR VALUES BY LABEL")
for col in [
    "particle_count",
    "gearbox_bearing_temp",
    "gearbox_oil_temp",
    "oil_pressure",
]:
    if col in df.columns:
        print(f"\n{col}")
        print(
            df.groupby("is_anomaly")[col]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .round(3)
        )

print("\n2. ANOMALY COUNTS BY MONTH")
df["month"] = df["timestamp"].dt.to_period("M")
print(
    df.groupby("month")["is_anomaly"]
    .agg(["count", "sum", "mean"])
    .to_string()
)

print("\n3. FIRST 20 ANOMALOUS RECORDS")
print(
    df.loc[
        df["is_anomaly"] == 1,
        ["timestamp", "particle_count",
         "gearbox_bearing_temp", "gearbox_oil_temp",
         "oil_pressure"]
    ].head(20).to_string(index=False)
)

print("\n4. RECORDS AROUND THE FIRST ANOMALY PERIOD")

start = pd.Timestamp("2027-04-14 23:00:00")
end = pd.Timestamp("2027-04-15 08:00:00")

cols = [
    "timestamp",
    "particle_count",
    "gearbox_bearing_temp",
    "gearbox_oil_temp",
    "oil_pressure",
    "is_anomaly",
]

print(
    df.loc[
        df["timestamp"].between(start, end),
        cols
    ].to_string(index=False)
)

