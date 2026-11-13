
import pandas as pd

events = pd.read_csv(
    "data/processed/merged_alert_episodes.csv",
    parse_dates=["start", "end"],
)

events = events.sort_values("start")

print("DETECTION DELAY FOR LABELLED EVENTS")

for event_id, event_start, event_end in [
    (1, pd.Timestamp("2027-04-15 04:00:00"),
     pd.Timestamp("2027-04-25 13:50:00")),
    (2, pd.Timestamp("2029-02-03 17:20:00"),
     pd.Timestamp("2029-02-21 01:50:00")),
    (3, pd.Timestamp("2029-12-20 14:00:00"),
     pd.Timestamp("2029-12-30 23:50:00")),
]:
    matches = events[
        (events["start"] <= event_end)
        & (events["end"] >= event_start)
        & (~events["contains_normal_records"].astype(bool))
    ]

    if matches.empty:
        print(f"\nEvent {event_id}: No matching alert episode")
        continue

    first_alert = matches["start"].min()
    delay_hours = (
        first_alert - event_start
    ).total_seconds() / 3600

    print(f"\nEvent {event_id}")
    print("Actual start:", event_start)
    print("First alert:", first_alert)
    print("Detection delay (hours):", round(delay_hours, 2))
    print(
        "Detected before event:",
        first_alert < event_start
    )

