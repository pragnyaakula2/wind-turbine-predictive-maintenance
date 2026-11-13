
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATA_PATH = "data/processed/training_data.csv"
MODEL_PATH = "models/lof_anomaly_detector.joblib"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

# Define each known anomaly event.
events = [
    ("Event 1", "2027-04-15", "2027-04-25 23:59:59"),
    ("Event 2", "2029-02-03", "2029-02-21 23:59:59"),
    ("Event 3", "2029-12-20", "2029-12-30 23:59:59"),
]

# Use only normal data before the first event to select a threshold.
# Do not use event labels to tune the threshold.
calibration = df[
    (df["timestamp"] < pd.Timestamp("2027-04-15"))
    & (df["is_anomaly"] == 0)
].copy()

if calibration.empty:
    raise ValueError("No normal calibration records found.")

calibration_scores = -model.decision_function(
    calibration[features]
)

# Choose a threshold that flags at most approximately 1% of
# calibration records. This is a diagnostic operating point,
# not a proven operational false-alarm rate.
threshold = np.quantile(calibration_scores, 0.99)

print("EVENT-BASED EVALUATION")
print("Calibration records:", len(calibration))
print("Threshold:", round(float(threshold), 6))
print("Calibration alert rate:",
      round(float((calibration_scores >= threshold).mean()), 4))

all_results = []

for name, start, end in events:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    event_df = df[
        (df["timestamp"] >= start)
        & (df["timestamp"] <= end)
    ].copy()

    if event_df.empty:
        print(f"\n{name}: no records found")
        continue

    scores = -model.decision_function(event_df[features])
    predictions = (scores >= threshold).astype(int)
    actual = event_df["is_anomaly"].astype(int).to_numpy()

    tp = int(((predictions == 1) & (actual == 1)).sum())
    fp = int(((predictions == 1) & (actual == 0)).sum())
    fn = int(((predictions == 0) & (actual == 1)).sum())

    result = {
        "event": name,
        "records": len(event_df),
        "actual_anomalies": int(actual.sum()),
        "true_positives": tp,
        "false_alarms": fp,
        "missed_anomalies": fn,
        "precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "f1": f1_score(
            actual, predictions, zero_division=0
        ),
    }

    if len(np.unique(actual)) == 2:
        result["roc_auc"] = roc_auc_score(actual, scores)
    else:
        result["roc_auc"] = np.nan

    all_results.append(result)

    print(f"\n{name}")
    for key, value in result.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if all_results:
    output = pd.DataFrame(all_results)
    output.to_csv(
        "data/processed/event_evaluation_results.csv",
        index=False,
    )
    print("\nSaved: data/processed/event_evaluation_results.csv")
else:
    print("\nNo event results were produced.")

