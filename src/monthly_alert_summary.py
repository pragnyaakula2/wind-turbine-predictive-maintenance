
import pandas as pd

episodes = pd.read_csv(
    "data/processed/merged_alert_episodes.csv",
    parse_dates=["start", "end"],
)

# Count episodes by the month in which each episode starts.
episodes["month"] = episodes["start"].dt.to_period("M")

monthly = episodes.groupby("month").agg(
    alert_episodes=("start", "size"),
    alert_records=("alert_records", "sum"),
    episodes_with_normal_records=(
        "contains_normal_records", "sum"
    ),
)

print("MONTHLY ALERT SUMMARY")
print(monthly.to_string())

print("\nMonths with more than 1 alert episode:")
print(monthly[monthly["alert_episodes"] > 1].to_string())

monthly.to_csv(
    "data/processed/monthly_alert_summary.csv"
)

print("\nSaved: data/processed/monthly_alert_summary.csv")

