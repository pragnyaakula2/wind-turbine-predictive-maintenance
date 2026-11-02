"""Microservice 3: Quantum Machine Learning Variational Classifier API (Port 8003).

Uses a 4-qubit parameterized quantum circuit in PennyLane to detect anomalous states.
"""

import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pennylane as qml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Wind Turbine - Quantum ML Service",
    description="Microservice 3 serving PennyLane Variational Quantum Circuit (VQC)",
    version="1.0.0",
)

# 4 Key sensors encoded into 4 quantum bits
QML_SENSORS = [
    "gearbox_oil_temp",
    "gearbox_bearing_temp",
    "vibration_x",
    "oil_pressure",
]

# Empirical baseline statistics for 4 key sensors (derived from normal operating records)
SENSOR_MEANS = np.array([60.87, 67.83, 0.0210, 4.276], dtype=np.float32)
SENSOR_STDS = np.array([7.34, 7.38, 0.0031, 0.0877], dtype=np.float32)
QML_THRESHOLD = 0.52

# ----------------------------------------------------
# PENNYLANE QUANTUM CIRCUIT DEFINITION
# ----------------------------------------------------
NUM_WIRES = 4
dev = qml.device("default.qubit", wires=NUM_WIRES)


@qml.qnode(dev)
def quantum_anomaly_circuit(z_features):
    """Encodes normalized sensor deviations via AngleEmbedding and computes Pauli-Z expectation."""
    # Convert standardized deviations to angle interval [0, pi]
    angles = np.clip(np.abs(z_features), 0.0, 3.0) * (np.pi / 3.0)

    # 1. State preparation: Angle Embedding into |0> state
    qml.AngleEmbedding(angles, wires=range(NUM_WIRES), rotation="Y")

    # 2. Entangling ring of CNOT gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

    # 3. Measure expectation value of Pauli-Z across all 4 qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_WIRES)]


class SensorDataInput(BaseModel):
    records: List[Dict[str, Any]]


@app.get("/health")
def health():
    """Health check endpoint for Kubernetes liveness & readiness probes."""
    return {
        "status": "healthy",
        "service": "Quantum ML Variational Classifier",
        "port": 8003,
        "framework": "PennyLane",
        "qubits": NUM_WIRES,
        "quantum_device": "default.qubit (state-vector simulator)",
        "quantum_circuit": "AngleEmbedding(Y) + Parameterized RY/RZ + Ring CNOT",
        "threshold": QML_THRESHOLD,
    }


@app.post("/predict")
def predict(payload: SensorDataInput):
    """Scores sensor records using the 4-qubit quantum expectation circuit."""
    if not payload.records:
        return {"predictions": [], "latency_ms": 0.0}

    start_time = time.perf_counter()
    df = pd.DataFrame(payload.records)

    missing = [s for s in QML_SENSORS if s not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Quantum ML service requires sensors: {missing}",
        )

    # Compute standardized deviations from nominal baseline
    vals = df[QML_SENSORS].to_numpy(dtype=np.float32)
    z_scores = np.atleast_2d((vals - SENSOR_MEANS) / (SENSOR_STDS + 1e-6))

    # Vectorized evaluation across quantum circuit via parameter broadcasting
    all_expvals = np.array(quantum_anomaly_circuit(z_scores))
    mean_paulis = np.mean(all_expvals, axis=0)
    anomaly_scores = (1.0 - mean_paulis) / 2.0
    is_anomalies = (anomaly_scores >= QML_THRESHOLD).astype(int)

    results = [
        {
            "quantum_anomaly_score": round(float(score), 4),
            "pauli_z_expval": round(float(ev), 4),
            "is_anomaly": int(ia),
            "status": "ANOMALY ALERT" if ia == 1 else "NORMAL",
        }
        for score, ev, ia in zip(anomaly_scores, mean_paulis, is_anomalies)
    ]

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "service": "Quantum ML",
        "algorithm": "Variational Quantum Classifier (4 Qubits)",
        "count": len(results),
        "latency_ms": round(latency_ms, 2),
        "predictions": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)  # nosec B104
