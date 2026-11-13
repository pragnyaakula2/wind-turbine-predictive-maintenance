
import pandas as pd

df = pd.read_csv("data/raw/turbine_5yr_labeled_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Start a new event when an anomaly begins,
# or when the time gap between records changes.
anomaly = df["is_anomaly"].eq(1)
new_event = anomaly & (
    ~anomaly.shift(fill_value=False)
    | df["timestamp"].diff().ne(pd.Timedelta(minutes=10))
)

df["event_id"] = new_event.cumsum().where(anomaly)

events = df[anomaly].groupby("event_id").agg(
    start=("timestamp", "min"),
    end=("timestamp", "max"),
    records=("timestamp", "size"),
)

print("ANOMALY EVENTS")
print(events.to_string())
print("\nTotal events:", len(events))

