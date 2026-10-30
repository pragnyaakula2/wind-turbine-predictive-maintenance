"""Unified Interactive Streamlit Dashboard for Wind Turbine Predictive Maintenance.

Supports Classical ML (LOF/iForest), Deep Learning (LSTM Autoencoder), and
Quantum ML (PennyLane VQC) with single model scoring or multi-model comparison.
"""

import os
from pathlib import Path
import time
import joblib
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Wind Turbine Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Wind Turbine Gearbox Predictive Maintenance")
st.caption("Multi-Model Anomaly Detection & Telemetry Monitoring Dashboard")

DATA_PATH = Path("data/processed/training_data.csv")

MODEL_CANDIDATES = [
    Path("models/lof_anomaly_detector.joblib"),
    Path("models/final_anomaly_detector.joblib"),
    Path("models/isolation_forest.joblib"),
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)

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

# Service Endpoints (configurable via env variables for Docker & Kubernetes)
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")
DL_SERVICE_URL = os.getenv("DL_SERVICE_URL", "http://localhost:8002")
QML_SERVICE_URL = os.getenv("QML_SERVICE_URL", "http://localhost:8003")


# --------------------------------------------------
# LOAD DATA & CLASSICAL MODEL CACHE
# --------------------------------------------------

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"])
    return data.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource
def load_classical_model(path):
    if path and Path(path).exists():
        return joblib.load(path)
    return None


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Ensure that data/processed/training_data.csv exists.")
    st.stop()

df = load_data(str(DATA_PATH))
bundle = load_classical_model(str(MODEL_PATH)) if MODEL_PATH else None


# --------------------------------------------------
# MICROSERVICE HELPER CLIENTS
# --------------------------------------------------

def check_service_health(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=0.8)
        return resp.status_code == 200
    except Exception:
        return False


def call_service_predict(base_url: str, records: list) -> dict:
    resp = requests.post(
        f"{base_url}/predict",
        json={"records": records},
        timeout=10.0,
    )
    if resp.status_code == 200:
        return resp.json()
    raise RuntimeError(f"Service at {base_url} returned error: {resp.text}")


# Fallback inference if microservice is offline
def fallback_classical_predict(sample_df: pd.DataFrame) -> dict:
    start = time.perf_counter()
    if bundle is None:
        raise RuntimeError("No saved classical model bundle found on disk.")
    m = bundle["model"]
    feats = bundle["features"]
    thresh = float(bundle.get("threshold", 0.12))
    scores = -m.decision_function(sample_df[feats])
    preds = (scores >= thresh).astype(int)
    results = [
        {
            "anomaly_score": round(float(s), 5),
            "is_anomaly": int(p),
            "status": "ANOMALY ALERT" if p == 1 else "NORMAL",
        }
        for s, p in zip(scores, preds)
    ]
    latency = (time.perf_counter() - start) * 1000.0
    return {
        "service": "Classical ML (Local)",
        "algorithm": type(m.steps[-1][1]).__name__ if hasattr(m, "steps") else type(m).__name__,
        "latency_ms": round(latency, 2),
        "predictions": results,
    }


def fallback_dl_predict(sample_df: pd.DataFrame) -> dict:
    from src.services.dl_service import predict as dl_pred, SensorDataInput
    records = sample_df.to_dict(orient="records")
    return dl_pred(SensorDataInput(records=records))


def fallback_qml_predict(sample_df: pd.DataFrame) -> dict:
    from src.services.qml_service import predict as qml_pred, SensorDataInput
    records = sample_df.to_dict(orient="records")
    return qml_pred(SensorDataInput(records=records))


# --------------------------------------------------
# SIDEBAR: CONTROLS & MICROSERVICE STATUS
# --------------------------------------------------

st.sidebar.header("Microservices Status")
ml_online = check_service_health(ML_SERVICE_URL)
dl_online = check_service_health(DL_SERVICE_URL)
qml_online = check_service_health(QML_SERVICE_URL)

st.sidebar.markdown(
    f"- **Classical ML (Port 8001):** {'🟢 Online' if ml_online else '🟡 Local Engine'}\n"
    f"- **Deep Learning (Port 8002):** {'🟢 Online' if dl_online else '🟡 Local Engine'}\n"
    f"- **Quantum ML (Port 8003):** {'🟢 Online' if qml_online else '🟡 Local Engine'}"
)

st.sidebar.divider()
st.sidebar.header("Model Selection")
selected_mode = st.sidebar.selectbox(
    "Choose Active Microservice",
    [
        "Triple Model Comparison (All 3 Side-by-Side)",
        "Classical ML (Isolation Forest / LOF)",
        "Deep Learning (PyTorch LSTM Autoencoder)",
        "Quantum ML (PennyLane 4-Qubit VQC)",
    ],
)

st.sidebar.divider()
st.sidebar.header("Date Range Filter")
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

start_date = st.sidebar.date_input(
    "Start date", value=min_date, min_value=min_date, max_value=max_date
)
end_date = st.sidebar.date_input(
    "End date", value=max_date, min_value=min_date, max_value=max_date
)

if start_date > end_date:
    st.error("Start date must be on or before the end date.")
    st.stop()

date_filtered = df[
    (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
].copy()

if date_filtered.empty:
    st.warning("No records exist in this date range.")
    st.stop()

st.sidebar.divider()
st.sidebar.header("Data Sampling")
max_available = min(len(date_filtered), 5000)
default_val = min(1000, max_available)
sample_size_slider = st.sidebar.slider(
    "Number of Recent Samples to Analyze",
    min_value=100,
    max_value=max_available,
    value=default_val,
    step=100,
    help="Select how many recent sensor records to visualize and score.",
)

# Keep reasonable sample size for real-time dashboard interaction
limit_rows = 500 if "Quantum" in selected_mode else sample_size_slider
display_sample_size = min(len(date_filtered), limit_rows)
filtered = date_filtered.tail(display_sample_size).copy()


# --------------------------------------------------
# INFERENCE LOGIC
# --------------------------------------------------

# Prepare clean numeric records for API payloads without non-serializable Timestamps
numeric_cols = [c for c in filtered.columns if c not in ["timestamp", "is_anomaly"]]
sample_records = filtered[numeric_cols].to_dict(orient="records")


def get_ml_predictions():
    if ml_online:
        return call_service_predict(ML_SERVICE_URL, sample_records)
    return fallback_classical_predict(filtered)


def get_dl_predictions():
    if dl_online:
        return call_service_predict(DL_SERVICE_URL, sample_records)
    return fallback_dl_predict(filtered)


def get_qml_predictions():
    if qml_online:
        return call_service_predict(QML_SERVICE_URL, sample_records)
    return fallback_qml_predict(filtered)


# --------------------------------------------------
# DISPLAY: TRIPLE MODEL COMPARISON
# --------------------------------------------------

if selected_mode == "Triple Model Comparison (All 3 Side-by-Side)":
    st.subheader("🔬 Multi-Model Comparative Evaluation")
    st.caption(
        f"Evaluating latest {len(filtered):,} records across Classical ML, "
        "Deep Learning (LSTM), and Quantum ML (VQC)."
    )

    with st.spinner("Executing predictions across Classical ML, DL, and QML services..."):
        ml_res = get_ml_predictions()
        dl_res = get_dl_predictions()
        # Evaluate subset for QML simulation speed
        qml_sample_size = min(len(filtered), 100)
        qml_sample = filtered.head(qml_sample_size)
        qml_records = qml_sample[numeric_cols].to_dict(orient="records")
        if qml_online:
            qml_res = call_service_predict(QML_SERVICE_URL, qml_records)
        else:
            qml_res = fallback_qml_predict(qml_sample)

    col1, col2, col3, col4 = st.columns(4)
    ml_alerts = sum(p["is_anomaly"] for p in ml_res["predictions"])
    dl_alerts = sum(p["is_anomaly"] for p in dl_res["predictions"])
    qml_alerts = sum(p["is_anomaly"] for p in qml_res["predictions"])

    col1.metric("1️⃣ Classical ML Alerts", f"{ml_alerts:,}", f"Latency: {ml_res['latency_ms']} ms")
    col2.metric("2️⃣ Deep Learning Alerts", f"{dl_alerts:,}", f"Latency: {dl_res['latency_ms']} ms")
    col3.metric(
        "3️⃣ Quantum ML Alerts",
        f"{qml_alerts}/{qml_sample_size}",
        f"Latency: {qml_res['latency_ms']} ms",
    )
    col4.metric("Actual Labelled Faults", f"{int(filtered['is_anomaly'].sum()):,}")

    st.divider()
    st.subheader("Comparison Table")

    comp_df = pd.DataFrame({
        "timestamp": filtered["timestamp"].iloc[:qml_sample_size],
        "Actual Anomaly": filtered["is_anomaly"].iloc[:qml_sample_size].map({0: "No", 1: "YES"}),
        "Classical ML": [p["status"] for p in ml_res["predictions"][:qml_sample_size]],
        "Deep Learning (LSTM)": [p["status"] for p in dl_res["predictions"][:qml_sample_size]],
        "Quantum ML (VQC)": [p["status"] for p in qml_res["predictions"]],
    })

    # Majority voting / Consensus
    def compute_agreement(row):
        votes = [
            row["Classical ML"] == "ANOMALY ALERT",
            row["Deep Learning (LSTM)"] == "ANOMALY ALERT",
            row["Quantum ML (VQC)"] == "ANOMALY ALERT",
        ]
        return "CRITICAL ALERT (Majority)" if sum(votes) >= 2 else "NORMAL"

    comp_df["Ensemble Consensus"] = comp_df.apply(compute_agreement, axis=1)

    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# --------------------------------------------------
# DISPLAY: INDIVIDUAL MODEL MODES
# --------------------------------------------------

else:
    st.subheader(f"📊 {selected_mode}")
    with st.spinner("Scoring records..."):
        if "Classical" in selected_mode:
            pred_res = get_ml_predictions()
            preds = pred_res["predictions"]
            filtered["anomaly_score"] = [p["anomaly_score"] for p in preds]
            filtered["predicted_anomaly"] = [p["is_anomaly"] for p in preds]
            filtered["status"] = [p["status"] for p in preds]
        elif "Deep Learning" in selected_mode:
            pred_res = get_dl_predictions()
            preds = pred_res["predictions"]
            filtered["anomaly_score"] = [p["reconstruction_mse"] for p in preds]
            filtered["predicted_anomaly"] = [p["is_anomaly"] for p in preds]
            filtered["status"] = [p["status"] for p in preds]
        else:
            qml_n = min(len(filtered), 200)
            filtered = filtered.tail(qml_n).copy()
            qml_sample_records = filtered[numeric_cols].to_dict(orient="records")
            if qml_online:
                pred_res = call_service_predict(QML_SERVICE_URL, qml_sample_records)
            else:
                pred_res = fallback_qml_predict(filtered)
            preds = pred_res["predictions"]
            filtered["anomaly_score"] = [p["quantum_anomaly_score"] for p in preds]
            filtered["predicted_anomaly"] = [p["is_anomaly"] for p in preds]
            filtered["status"] = [p["status"] for p in preds]

    total_rec = len(filtered)
    alerts = int(filtered["predicted_anomaly"].sum())
    normal = total_rec - alerts

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated Records", f"{total_rec:,}")
    c2.metric("Normal Predictions", f"{normal:,}")
    c3.metric("Anomaly Alerts", f"{alerts:,}")
    c4.metric("Inference Latency", f"{pred_res['latency_ms']} ms")

    st.divider()
    st.subheader("Prediction Results")
    only_alerts = st.checkbox("Show anomaly alerts only")
    cols = ["timestamp", "is_anomaly", "anomaly_score", "predicted_anomaly", "status"]
    table_data = filtered[filtered["predicted_anomaly"] == 1] if only_alerts else filtered

    st.dataframe(
        table_data[cols].sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------
# SENSOR TREND CHARTS & CSV EXPORT
# --------------------------------------------------

st.divider()
st.subheader("Sensor Telemetry Trends")
available = [s for s in SENSORS if s in filtered.columns]
if available:
    chosen_sensor = st.selectbox("Select sensor channel", available)
    st.line_chart(filtered.set_index("timestamp")[[chosen_sensor]], height=320)

csv_export = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Results as CSV",
    data=csv_export,
    file_name="wind_turbine_multi_model_predictions.csv",
    mime="text/csv",
)
