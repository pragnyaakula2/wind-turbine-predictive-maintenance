"""Unified Interactive Streamlit Dashboard for Wind Turbine Predictive Maintenance.

Supports Classical ML (LOF/iForest), Deep Learning (LSTM Autoencoder), and
Quantum ML (PennyLane VQC) with single model scoring, multi-model comparison,
MongoDB-backed authentication, and real-time manual sensor diagnostic simulation.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import joblib
import pandas as pd
import requests
import streamlit as st

try:
    from src.database import (
        verify_user,
        register_user,
        log_prediction,
        get_prediction_history,
        is_mongodb_connected,
    )
except ImportError:
    from database import (
        verify_user,
        register_user,
        log_prediction,
        get_prediction_history,
        is_mongodb_connected,
    )

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Wind Turbine Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

# --------------------------------------------------
# SECURITY: AUTHENTICATION GATE (MONGODB / LOCAL)
# --------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user_info"] = None


def render_login_page():
    """Renders the secure engineer authentication portal (Sign In / Register)."""
    st.title("🔒 Wind Turbine SCADA Security Portal")
    st.caption("Secure Role-Based Access Control for Maintenance Engineers & Operators")

    mongo_online = is_mongodb_connected()
    if mongo_online:
        st.success(
            "🟢 Connected to MongoDB (Port 27017) — Enterprise Database Authentication Active"
        )
    else:
        st.info("🟡 MongoDB Offline — Operating in Secure Cryptographic Local Mode")

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        auth_tab1, auth_tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])

        with auth_tab1:
            with st.form("login_form"):
                st.subheader("Engineer Sign In")
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submit = st.form_submit_button("Sign In", type="primary")

                if submit:
                    user = verify_user(username.strip(), password)
                    if user:
                        st.session_state["authenticated"] = True
                        st.session_state["user_info"] = user
                        st.success(f"Welcome, {user['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Please try again.")

        with auth_tab2:
            with st.form("register_form"):
                st.subheader("Create New Engineer Account")
                reg_name = st.text_input("Full Name (e.g. Dr. Alex Mercer)", key="reg_name")
                reg_username = st.text_input("Username (min 3 chars)", key="reg_username")
                reg_role = st.selectbox(
                    "Role",
                    ["Lead Engineer", "Reliability Engineer", "Field Operator", "SCADA Analyst"],
                    key="reg_role",
                )
                reg_password = st.text_input(
                    "Password (min 6 chars)", type="password", key="reg_pw"
                )
                reg_confirm = st.text_input(
                    "Confirm Password", type="password", key="reg_pw_confirm"
                )
                reg_submit = st.form_submit_button("Create Account", type="primary")

                if reg_submit:
                    if reg_password != reg_confirm:
                        st.error("Passwords do not match. Please re-enter.")
                    else:
                        ok, msg = register_user(
                            username=reg_username,
                            password=reg_password,
                            full_name=reg_name,
                            role=reg_role,
                        )
                        if ok:
                            st.success(f"✅ {msg}")
                            st.info("Switch to the 'Sign In' tab above to log in.")
                        else:
                            st.error(f"❌ {msg}")


if not st.session_state["authenticated"]:
    render_login_page()
    st.stop()

# --------------------------------------------------
# SYSTEM CONFIGURATION & CONSTANTS
# --------------------------------------------------

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
    if not path.exists():
        fallback_csv = Path("data/raw/turbine_5yr_labeled_data.csv")
        if fallback_csv.exists():
            data = pd.read_csv(fallback_csv)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            return data
        return pd.DataFrame()
    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data


@st.cache_resource
def load_classical_model(path):
    if path is None or not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


df = load_data(DATA_PATH)
bundle = load_classical_model(MODEL_PATH)

if df.empty:
    st.error(f"Dataset not found at {DATA_PATH}. Place data and restart.")
    st.stop()


# --------------------------------------------------
# MICROSERVICE CLIENT FUNCTIONS
# --------------------------------------------------

def check_service_health(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=0.8)
        return resp.status_code == 200
    except Exception:
        return False


def call_service_predict(
    base_url: str,
    records: list,
    algorithm: str = None,
    timeout: float = 60.0,
) -> dict:
    payload = {"records": records}
    if algorithm:
        payload["algorithm"] = algorithm
    resp = requests.post(
        f"{base_url}/predict",
        json=payload,
        timeout=timeout,
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

    # Guard for missing engineered features
    missing_cols = [col for col in feats if col not in sample_df.columns]
    eval_df = sample_df.copy()
    for col in missing_cols:
        eval_df[col] = 0.0

    scores = -m.decision_function(eval_df[feats])
    preds = (scores >= thresh).astype(int)
    latency = (time.perf_counter() - start) * 1000.0

    results = []
    for s, p in zip(scores, preds):
        results.append({
            "anomaly_score": round(float(s), 5),
            "is_anomaly": int(p),
            "status": "ANOMALY ALERT" if p == 1 else "NORMAL",
        })
    return {
        "service": "Classical ML (Local)",
        "algorithm": type(m.steps[-1][1]).__name__ if hasattr(m, "steps") else type(m).__name__,
        "latency_ms": round(latency, 2),
        "predictions": results,
    }


def fallback_dl_predict(sample_df: pd.DataFrame, algorithm: str = "lstm") -> dict:
    from src.services.dl_service import predict as dl_pred, SensorDataInput
    records = sample_df.to_dict(orient="records")
    return dl_pred(SensorDataInput(records=records, algorithm=algorithm))


def fallback_qml_predict(sample_df: pd.DataFrame, algorithm: str = "vqc") -> dict:
    from src.services.qml_service import predict as qml_pred, SensorDataInput
    records = sample_df.to_dict(orient="records")
    return qml_pred(SensorDataInput(records=records, algorithm=algorithm))


# --------------------------------------------------
# SIDEBAR: USER INFO, STATUS & GLOBAL FILTERS
# --------------------------------------------------

user_info = st.session_state.get("user_info", {})
st.sidebar.markdown(f"👤 **{user_info.get('full_name', 'Engineer')}**")
role_text = user_info.get("role", "Lead Engineer")
auth_text = user_info.get("backend", "MongoDB")
st.sidebar.caption(f"Role: {role_text} | Auth: {auth_text}")
if st.sidebar.button("🚪 Logout", key="logout_btn"):
    st.session_state["authenticated"] = False
    st.session_state["user_info"] = None
    st.rerun()

st.sidebar.divider()
st.sidebar.header("System Connectivity")
ml_online = check_service_health(ML_SERVICE_URL)
dl_online = check_service_health(DL_SERVICE_URL)
qml_online = check_service_health(QML_SERVICE_URL)
mongo_online = is_mongodb_connected()

st.sidebar.markdown(
    f"- **Database (MongoDB 27017):** {'🟢 Connected' if mongo_online else '🟡 Local Fallback'}\n"
    f"- **Classical ML (Port 8001):** {'🟢 Online' if ml_online else '🟡 Local Engine'}\n"
    f"- **Deep Learning (Port 8002):** {'🟢 Online' if dl_online else '🟡 Local Engine'}\n"
    f"- **Quantum ML (Port 8003):** {'🟢 Online' if qml_online else '🟡 Local Engine'}"
)

# --------------------------------------------------
# MAIN APP HEADER & TAB NAVIGATION
# --------------------------------------------------

st.title("⚙️ Wind Turbine Gearbox Predictive Maintenance")
st.caption("Multi-Model Anomaly Detection, Real-Time Telemetry & Consensus Voting")

tab_monitor, tab_manual = st.tabs([
    "📊 Historical SCADA Telemetry & Batch Analysis",
    "🎯 Manual Diagnostic & What-If Simulator",
])

# ==================================================
# TAB 1: HISTORICAL SCADA BATCH MONITORING
# ==================================================

with tab_monitor:
    st.sidebar.divider()
    st.sidebar.header("Batch Model Selection")
    selected_mode = st.sidebar.selectbox(
        "Choose Active Microservice Mode",
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

    display_sample_size = min(len(date_filtered), sample_size_slider)
    filtered = date_filtered.tail(display_sample_size).copy()

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

    if selected_mode == "Triple Model Comparison (All 3 Side-by-Side)":
        st.subheader("🔬 Multi-Model Comparative Evaluation")
        st.caption(
            f"Evaluating latest {len(filtered):,} records across Classical ML, "
            "Deep Learning (LSTM), and Quantum ML (VQC)."
        )

        with st.spinner("Executing predictions across Classical ML, DL, and QML services..."):
            ml_res = get_ml_predictions()
            dl_res = get_dl_predictions()
            qml_res = get_qml_predictions()

        col1, col2, col3, col4 = st.columns(4)
        ml_alerts = sum(p["is_anomaly"] for p in ml_res["predictions"])
        dl_alerts = sum(p["is_anomaly"] for p in dl_res["predictions"])
        qml_alerts = sum(p["is_anomaly"] for p in qml_res["predictions"])

        col1.metric(
            "1️⃣ Classical ML Alerts", f"{ml_alerts:,}", f"Latency: {ml_res['latency_ms']} ms"
        )
        col2.metric(
            "2️⃣ Deep Learning Alerts", f"{dl_alerts:,}", f"Latency: {dl_res['latency_ms']} ms"
        )
        col3.metric(
            "3️⃣ Quantum ML Alerts", f"{qml_alerts:,}", f"Latency: {qml_res['latency_ms']} ms"
        )
        col4.metric("Actual Labelled Faults", f"{int(filtered['is_anomaly'].sum()):,}")

        st.divider()
        st.subheader("Comparison Table")

        comp_df = pd.DataFrame({
            "timestamp": filtered["timestamp"].values,
            "Actual Anomaly": filtered["is_anomaly"].map({0: "No", 1: "YES"}).values,
            "Classical ML": [p["status"] for p in ml_res["predictions"]],
            "Deep Learning (LSTM)": [p["status"] for p in dl_res["predictions"]],
            "Quantum ML (VQC)": [p["status"] for p in qml_res["predictions"]],
        })

        def compute_agreement(row):
            votes = [
                row["Classical ML"] == "ANOMALY ALERT",
                row["Deep Learning (LSTM)"] == "ANOMALY ALERT",
                row["Quantum ML (VQC)"] == "ANOMALY ALERT",
            ]
            return "CRITICAL ALERT (Majority)" if sum(votes) >= 2 else "NORMAL"

        comp_df["Ensemble Consensus"] = comp_df.apply(compute_agreement, axis=1)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

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


# ==================================================
# TAB 2: MANUAL DIAGNOSTIC & WHAT-IF SIMULATOR
# ==================================================

with tab_manual:
    st.subheader("🎯 Real-Time Manual Sensor Diagnostic & Simulation")
    st.caption(
        "Input live SCADA telemetry or select a diagnostic preset. Predictions are evaluated "
        "across all 3 AI paradigms and audit-logged into MongoDB."
    )

    # Initialize preset session state defaults if not present
    if "man_oil_temp" not in st.session_state:
        st.session_state["man_oil_temp"] = 60.87
        st.session_state["man_bearing_temp"] = 67.83
        st.session_state["man_vib_x"] = 0.021
        st.session_state["man_vib_y"] = 0.019
        st.session_state["man_vib_z"] = 0.024
        st.session_state["man_pressure"] = 4.28
        st.session_state["man_particles"] = 918.0

    # Preset Quick Buttons
    st.markdown("**Quick-Load Diagnostic Presets:**")
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("🟢 Load Healthy Turbine Baseline", use_container_width=True):
            st.session_state["man_oil_temp"] = 60.87
            st.session_state["man_bearing_temp"] = 67.83
            st.session_state["man_vib_x"] = 0.021
            st.session_state["man_vib_y"] = 0.019
            st.session_state["man_vib_z"] = 0.024
            st.session_state["man_pressure"] = 4.28
            st.session_state["man_particles"] = 918.0
            st.rerun()

    with p2:
        if st.button("🔴 Simulate Bearing Overheat & Seizure", use_container_width=True):
            st.session_state["man_oil_temp"] = 88.50
            st.session_state["man_bearing_temp"] = 94.70
            st.session_state["man_vib_x"] = 0.086
            st.session_state["man_vib_y"] = 0.079
            st.session_state["man_vib_z"] = 0.099
            st.session_state["man_pressure"] = 2.92
            st.session_state["man_particles"] = 764.0
            st.rerun()

    with p3:
        if st.button("🟡 Simulate Extreme Vibration Anomaly", use_container_width=True):
            st.session_state["man_oil_temp"] = 72.00
            st.session_state["man_bearing_temp"] = 75.00
            st.session_state["man_vib_x"] = 0.095
            st.session_state["man_vib_y"] = 0.088
            st.session_state["man_vib_z"] = 0.105
            st.session_state["man_pressure"] = 3.65
            st.session_state["man_particles"] = 1450.0
            st.rerun()

    st.divider()

    # Manual Input Form
    with st.form("manual_diagnostic_form"):
        st.markdown("### 🎛️ Physical SCADA Sensor Inputs")
        c_temp, c_vib, c_oil = st.columns(3)

        with c_temp:
            st.markdown("**Thermal Channels (°C)**")
            inp_oil_temp = st.number_input(
                "Gearbox Oil Temp (°C)",
                min_value=30.0,
                max_value=120.0,
                value=float(st.session_state["man_oil_temp"]),
                step=0.5,
            )
            inp_bearing_temp = st.number_input(
                "Gearbox Bearing Temp (°C)",
                min_value=30.0,
                max_value=130.0,
                value=float(st.session_state["man_bearing_temp"]),
                step=0.5,
            )

        with c_vib:
            st.markdown("**Vibration Channels (g)**")
            inp_vib_x = st.number_input(
                "Vibration X (g)",
                min_value=0.005,
                max_value=0.200,
                value=float(st.session_state["man_vib_x"]),
                step=0.002,
                format="%.4f",
            )
            inp_vib_y = st.number_input(
                "Vibration Y (g)",
                min_value=0.005,
                max_value=0.200,
                value=float(st.session_state["man_vib_y"]),
                step=0.002,
                format="%.4f",
            )
            inp_vib_z = st.number_input(
                "Vibration Z (g)",
                min_value=0.005,
                max_value=0.200,
                value=float(st.session_state["man_vib_z"]),
                step=0.002,
                format="%.4f",
            )

        with c_oil:
            st.markdown("**Fluid & Particulate Channels**")
            inp_pressure = st.number_input(
                "Oil Pressure (bar)",
                min_value=1.5,
                max_value=6.0,
                value=float(st.session_state["man_pressure"]),
                step=0.05,
            )
            inp_particles = st.number_input(
                "Particle Count (particles/mL)",
                min_value=50.0,
                max_value=3000.0,
                value=float(st.session_state["man_particles"]),
                step=25.0,
            )

        with st.expander("⚙️ Algorithm Architecture Selector (2 Models per Microservice)"):
            a_col1, a_col2 = st.columns(2)
            with a_col1:
                selected_dl = st.selectbox(
                    "Deep Learning Model",
                    ["LSTM Autoencoder (Recurrent)", "MLP Dense Autoencoder (Feedforward)"],
                    index=0,
                )
            with a_col2:
                selected_qml = st.selectbox(
                    "Quantum ML Model",
                    ["Variational Quantum Classifier (VQC)", "Quantum State Fidelity Metric"],
                    index=0,
                )

        dl_algo_code = "mlp" if "MLP" in selected_dl else "lstm"
        qml_algo_code = "fidelity" if "Fidelity" in selected_qml else "vqc"

        run_diag = st.form_submit_button("⚡ Run Real-Time Multi-Model Prediction", type="primary")

    if run_diag:
        # Build standard record
        diag_record = {
            "gearbox_oil_temp": float(inp_oil_temp),
            "gearbox_bearing_temp": float(inp_bearing_temp),
            "vibration_x": float(inp_vib_x),
            "vibration_y": float(inp_vib_y),
            "vibration_z": float(inp_vib_z),
            "oil_pressure": float(inp_pressure),
            "particle_count": float(inp_particles),
        }

        # 1. Classical ML Evaluation
        classical_record = dict(diag_record)
        classical_record["oil_bearing_temp_diff"] = inp_oil_temp - inp_bearing_temp
        for s in SENSORS:
            classical_record[f"{s}_mean_1h"] = diag_record[s]
            classical_record[f"{s}_min_1h"] = diag_record[s] - 0.05
            classical_record[f"{s}_max_1h"] = diag_record[s] + 0.05
            classical_record[f"{s}_std_1h"] = 0.001

        try:
            if ml_online:
                m_out = call_service_predict(ML_SERVICE_URL, [classical_record])
            else:
                m_out = fallback_classical_predict(pd.DataFrame([classical_record]))
            ml_pred = m_out["predictions"][0]
            ml_status = ml_pred["status"]
            ml_score = ml_pred["anomaly_score"]
        except Exception:
            ml_status = "ERROR"
            ml_score = 0.0

        # 2. Deep Learning Evaluation
        try:
            if dl_online:
                d_out = call_service_predict(
                    DL_SERVICE_URL, [diag_record], algorithm=dl_algo_code
                )
            else:
                d_out = fallback_dl_predict(
                    pd.DataFrame([diag_record]), algorithm=dl_algo_code
                )
            dl_pred = d_out["predictions"][0]
            dl_status = dl_pred["status"]
            dl_mse = dl_pred["reconstruction_mse"]
        except Exception:
            dl_status = "ERROR"
            dl_mse = 0.0

        # 3. Quantum ML Evaluation
        try:
            if qml_online:
                q_out = call_service_predict(
                    QML_SERVICE_URL, [diag_record], algorithm=qml_algo_code
                )
            else:
                q_out = fallback_qml_predict(
                    pd.DataFrame([diag_record]), algorithm=qml_algo_code
                )
            qml_pred = q_out["predictions"][0]
            qml_status = qml_pred["status"]
            qml_score = qml_pred["quantum_anomaly_score"]
        except Exception:
            qml_status = "ERROR"
            qml_score = 0.0

        # 4. Consensus Decision Logic
        votes = [
            ml_status == "ANOMALY ALERT",
            dl_status == "ANOMALY ALERT",
            qml_status == "ANOMALY ALERT",
        ]
        vote_count = sum(votes)
        consensus_result = "CRITICAL ALERT (Majority)" if vote_count >= 2 else "NORMAL"

        st.divider()
        st.markdown("### 📋 Real-Time Diagnostic Verdict")

        if vote_count >= 2:
            st.error(
                f"🚨 **CRITICAL ANOMALY ALERT: High Risk Detected!**  \n"
                f"Ensemble Majority Consensus achieved "
                f"(**{vote_count}/3 Models Agree on Anomaly**). "
                "Immediate inspection of gearbox lubrication and bearings recommended."
            )
        else:
            st.success(
                f"✅ **NORMAL OPERATIONAL STATE: Healthy Gearbox**  \n"
                f"Ensemble Majority Consensus confirms healthy operation "
                f"(**{3 - vote_count}/3 Models Confirm Normal**). "
                "Sensor signatures are within nominal physical tolerances."
            )

        # Multi-Model Diagnostic Breakdown Cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "1️⃣ Classical ML (LOF)",
                ml_status,
                f"Score: {ml_score} (Thresh: 0.328)",
            )
        with m2:
            st.metric(
                f"2️⃣ DL ({'MLP' if dl_algo_code == 'mlp' else 'LSTM'})",
                dl_status,
                f"MSE: {dl_mse} (Thresh: 2.00)",
            )
        with m3:
            qml_lbl = "Fidelity" if qml_algo_code == "fidelity" else "VQC"
            qml_thr = 0.50 if qml_algo_code == "fidelity" else 0.52
            st.metric(
                f"3️⃣ QML ({qml_lbl})",
                qml_status,
                f"Score: {qml_score} (Thresh: {qml_thr})",
            )

        # Log prediction to MongoDB (or local fallback)
        operator_name = user_info.get("username", "admin")
        log_prediction(
            username=operator_name,
            sensor_inputs=diag_record,
            predictions={
                "ml_status": ml_status,
                "dl_status": dl_status,
                "qml_status": qml_status,
            },
            consensus=consensus_result,
        )
        st.toast("Diagnostic event logged to MongoDB!", icon="💾")

    # Diagnostic Audit History Table
    st.divider()
    st.subheader("📜 Diagnostic Audit History (MongoDB: `prediction_history`)")
    history_logs = get_prediction_history(limit=10)

    if history_logs:
        table_rows = []
        for h in history_logs:
            s_in = h.get("sensor_inputs", {})
            table_rows.append({
                "Timestamp (UTC)": h.get("timestamp"),
                "Operator": h.get("operator"),
                "Oil Temp (°C)": s_in.get("gearbox_oil_temp"),
                "Bearing Temp (°C)": s_in.get("gearbox_bearing_temp"),
                "Vibration X (g)": s_in.get("vibration_x"),
                "Pressure (bar)": s_in.get("oil_pressure"),
                "Classical ML": h.get("classical_ml"),
                "Deep Learning": h.get("deep_learning"),
                "Quantum ML": h.get("quantum_ml"),
                "Consensus": h.get("consensus_verdict"),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No manual predictions recorded yet. Run a prediction above to "
            "populate the MongoDB audit history."
        )
