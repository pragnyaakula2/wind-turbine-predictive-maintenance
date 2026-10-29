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

SENSOR_MINS = np.array([30.0, 30.0, 0.0, 0.0], dtype=np.float32)
SENSOR_MAXS = np.array([110.0, 105.0, 5.0, 10.0], dtype=np.float32)
QML_THRESHOLD = 0.45

# ----------------------------------------------------
# PENNYLANE QUANTUM CIRCUIT DEFINITION
# ----------------------------------------------------
NUM_WIRES = 4
dev = qml.device("default.qubit", wires=NUM_WIRES)

# Pre-tuned variational parameters (rotation angles)
QUANTUM_WEIGHTS = np.array([
    [0.54, 0.32, 0.76, 0.19],
    [0.87, 0.45, 0.23, 0.61],
], dtype=np.float32)


@qml.qnode(dev)
def quantum_anomaly_circuit(features, weights):
    """Encodes normalized sensor values via AngleEmbedding and computes Pauli-Z expectation."""
    # 1. State preparation: Angle Embedding into |0> state
    qml.AngleEmbedding(features, wires=range(NUM_WIRES), rotation="Y")

    # 2. Variational rotation layer
    for i in range(NUM_WIRES):
        qml.RY(weights[0, i], wires=i)

    # 3. Entangling ring of CNOT gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

    # 4. Secondary rotation layer
    for i in range(NUM_WIRES):
        qml.RZ(weights[1, i], wires=i)

    # 5. Measure expectation value of Pauli-Z on qubit 0
    return qml.expval(qml.PauliZ(0))


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

    # Normalize sensor features to [0, pi] for angle embedding
    vals = df[QML_SENSORS].to_numpy(dtype=np.float32)
    norm = np.clip((vals - SENSOR_MINS) / (SENSOR_MAXS - SENSOR_MINS + 1e-6), 0.0, 1.0)
    angle_features = norm * np.pi

    results = []
    for angles in angle_features:
        # Run quantum circuit
        expval = float(quantum_anomaly_circuit(angles, QUANTUM_WEIGHTS))
        # Map PauliZ expectation (-1 to 1) to anomaly probability (0 to 1)
        anomaly_score = (1.0 - expval) / 2.0
        is_anomaly = int(anomaly_score >= QML_THRESHOLD)

        results.append({
            "quantum_anomaly_score": round(anomaly_score, 4),
            "pauli_z_expval": round(expval, 4),
            "is_anomaly": is_anomaly,
            "status": "ANOMALY ALERT" if is_anomaly == 1 else "NORMAL",
        })

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
    uvicorn.run(app, host="0.0.0.0", port=8003)
