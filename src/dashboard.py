
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Wind Turbine Monitoring",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Wind Turbine Gearbox Monitoring")
st.caption("Prototype for sensor monitoring and anomaly detection")

DATA_PATH = Path("data/processed/training_data.csv")

# Use the model file that exists in your project.
MODEL_CANDIDATES = [
    Path("models/lof_anomaly_detector.joblib"),
]
MODEL_PATH = next(
    (path for path in MODEL_CANDIDATES if path.exists()),
    None
)

MAX_DISPLAY_ROWS = 5000

SENSORS = [
    "gearbox_oil_temp",
    "gearbox_bearing_temp",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "oil_pressure",
    "particle_count",
]

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Check that data/processed/training_data.csv exists.")
    st.stop()

if MODEL_PATH is None:
    st.error("No supported saved model file was found in models/.")
    st.write("Expected one of:")
    st.code("\n".join(str(p) for p in MODEL_CANDIDATES))
    st.stop()


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data = data.dropna(subset=["timestamp"])
    return data.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    df = load_data(str(DATA_PATH))
    bundle = load_model(str(MODEL_PATH))

    # Support the dictionary format used by our saved model scripts.
    model = bundle["model"]
    features = bundle["features"]
    threshold = float(bundle["threshold"])

except Exception as error:
    st.error("Could not load the dataset or saved model.")
    st.exception(error)
    st.stop()


missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    st.error(f"Model features missing from dataset: {missing_features}")
    st.stop()

if "is_anomaly" not in df.columns:
    st.error("The dataset does not contain the is_anomaly column.")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Filters")

minimum_date = df["timestamp"].min().date()
maximum_date = df["timestamp"].max().date()

start_date = st.sidebar.date_input(
    "Start date",
    value=minimum_date,
    min_value=minimum_date,
    max_value=maximum_date
)

end_date = st.sidebar.date_input(
    "End date",
    value=maximum_date,
    min_value=minimum_date,
    max_value=maximum_date
)

if start_date > end_date:
    st.error("Start date must be on or before the end date.")
    st.stop()

date_filtered = df[
    (df["timestamp"].dt.date >= start_date)
    & (df["timestamp"].dt.date <= end_date)
].copy()

if date_filtered.empty:
    st.warning("No records exist in this date range.")
    st.stop()

# Score only the latest records in the selected range.
# This is a display limit, not a full-period evaluation.
filtered = date_filtered.tail(MAX_DISPLAY_ROWS).copy()

# --------------------------------------------------
# ANOMALY PREDICTIONS
# --------------------------------------------------

try:
    model_input = filtered[features]

    if model_input.isna().any().any():
        st.error("Selected model features contain missing values.")
        st.stop()

    filtered["anomaly_score"] = -model.decision_function(model_input)

    filtered["predicted_anomaly"] = (
        filtered["anomaly_score"] >= threshold
    ).astype(int)

    filtered["status"] = filtered["predicted_anomaly"].map({
        0: "NORMAL",
        1: "ANOMALY ALERT"
    })

except Exception as error:
    st.error("Prediction failed. Check the saved model and its features.")
    st.exception(error)
    st.stop()

# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------

st.subheader("Monitoring Summary")

st.caption(
    f"Showing the latest {len(filtered):,} records from the selected "
    f"period. Metrics below describe these records only."
)

total_records = len(filtered)
predicted_alerts = int(filtered["predicted_anomaly"].sum())
normal_predictions = total_records - predicted_alerts
actual_anomalies = int(filtered["is_anomaly"].sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Sensor Records", f"{total_records:,}")
col2.metric("Normal Predictions", f"{normal_predictions:,}")
col3.metric("Anomaly Alerts", f"{predicted_alerts:,}")
col4.metric("Actual Labelled Anomalies", f"{actual_anomalies:,}")

st.divider()

# --------------------------------------------------
# SENSOR TRENDS
# --------------------------------------------------

st.subheader("Sensor Trends")

available_sensors = [
    sensor for sensor in SENSORS
    if sensor in filtered.columns
]

if available_sensors:
    sensor = st.selectbox("Choose a sensor", available_sensors)

    chart_data = filtered.set_index("timestamp")[[sensor]]

    st.line_chart(chart_data, height=350)
else:
    st.warning("No sensor columns are available to plot.")

st.divider()

# --------------------------------------------------
# PREDICTION RESULTS
# --------------------------------------------------

st.subheader("Prediction Results")

show_only_alerts = st.checkbox("Show anomaly alerts only")

columns_to_show = [
    "timestamp",
    "is_anomaly",
    "anomaly_score",
    "predicted_anomaly",
    "status",
]

if show_only_alerts:
    results = filtered[
        filtered["predicted_anomaly"] == 1
    ]
else:
    results = filtered

st.dataframe(
    results[columns_to_show].sort_values(
        "timestamp", ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download displayed results as CSV",
    data=csv_data,
    file_name="wind_turbine_predictions.csv",
    mime="text/csv"
)

st.caption(
    "This prototype detects unusual sensor behaviour. "
    "It does not yet predict the time until gearbox failure. "
    "Predictions depend on the selected saved model and threshold."
)

