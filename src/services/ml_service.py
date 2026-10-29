"""Microservice 1: Classical Machine Learning Anomaly Detection API (Port 8001).

Uses Isolation Forest / Local Outlier Factor to predict anomalies.
"""

import time
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Wind Turbine - Classical ML Service",
    description="Microservice 1 serving scikit-learn anomaly detection (Isolation Forest / LOF)",
    version="1.0.0",
)

MODEL_CANDIDATES = [
    Path("models/final_anomaly_detector.joblib"),
    Path("models/isolation_forest.joblib"),
    Path("models/lof_anomaly_detector.joblib"),
]

bundle = None
model = None
features = []
threshold = 0.12


def load_model_bundle():
    """Loads the fitted anomaly detection bundle from disk."""
    global bundle, model, features, threshold
    model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if model_path is None:
        return False
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    threshold = float(bundle.get("threshold", 0.12))
    return True


load_model_bundle()


class SensorDataInput(BaseModel):
    records: List[Dict[str, Any]]


@app.get("/health")
def health():
    """Health check endpoint for Kubernetes liveness & readiness probes."""
    return {
        "status": "healthy",
        "service": "Classical ML Anomaly Detection",
        "port": 8001,
        "model_loaded": model is not None,
        "features_count": len(features),
        "threshold": threshold,
    }


@app.post("/predict")
def predict(payload: SensorDataInput):
    """Predicts anomaly status using the classical ML model."""
    if model is None and not load_model_bundle():
        raise HTTPException(status_code=503, detail="Model bundle not available on disk.")

    if not payload.records:
        return {"predictions": [], "latency_ms": 0.0}

    start_time = time.perf_counter()
    df = pd.DataFrame(payload.records)

    # Ensure required features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required model features: {missing[:5]}",
        )

    # Compute anomaly scores
    model_input = df[features]
    scores = -model.decision_function(model_input)
    predictions = (scores >= threshold).astype(int)

    results = []
    for score, pred in zip(scores, predictions):
        results.append({
            "anomaly_score": round(float(score), 5),
            "is_anomaly": int(pred),
            "status": "ANOMALY ALERT" if pred == 1 else "NORMAL",
        })

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    algo_name = (
        type(model.steps[-1][1]).__name__
        if hasattr(model, "steps")
        else type(model).__name__
    )
    return {
        "service": "Classical ML",
        "algorithm": algo_name,
        "count": len(results),
        "latency_ms": round(latency_ms, 2),
        "predictions": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # nosec B104
