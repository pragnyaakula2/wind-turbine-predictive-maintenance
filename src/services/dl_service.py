"""Microservice 2: Deep Learning LSTM Autoencoder API (Port 8002).

Uses a PyTorch Recurrent Autoencoder to detect anomalies via reconstruction error.
"""

import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Wind Turbine - Deep Learning Service",
    description="Microservice 2 serving PyTorch LSTM Autoencoder",
    version="1.0.0",
)

CORE_SENSORS = [
    "gearbox_oil_temp",
    "gearbox_bearing_temp",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "oil_pressure",
    "particle_count",
]

# Normalization parameters (mean & std approximations for sensor channels)
SENSOR_MEANS = np.array([65.0, 62.0, 1.5, 1.2, 0.9, 4.2, 150.0], dtype=np.float32)
SENSOR_STDS = np.array([12.0, 10.0, 0.8, 0.6, 0.5, 0.8, 60.0], dtype=np.float32)
RECONSTRUCTION_THRESHOLD = 0.085


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for time-series sensor reconstruction."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 16, latent_dim: int = 8):
        super().__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, (h_n, _) = self.encoder_lstm(x)
        latent = torch.relu(self.encoder_fc(h_n[-1]))

        # Repeat latent representation across sequence length
        seq_len = x.size(1)
        repeated = self.decoder_fc(latent).unsqueeze(1).repeat(1, seq_len, 1)

        dec_out, _ = self.decoder_lstm(repeated)
        reconstruction = self.output_fc(dec_out)
        return reconstruction


# Initialize model with consistent seed
torch.manual_seed(42)
dl_model = LSTMAutoencoder(input_dim=len(CORE_SENSORS))
dl_model.eval()


class SensorDataInput(BaseModel):
    records: List[Dict[str, Any]]


@app.get("/health")
def health():
    """Health check endpoint for Kubernetes liveness & readiness probes."""
    return {
        "status": "healthy",
        "service": "Deep Learning LSTM Autoencoder",
        "port": 8002,
        "framework": "PyTorch",
        "architecture": "LSTM Autoencoder (7 -> 16 -> 8 -> 16 -> 7)",
        "reconstruction_threshold": RECONSTRUCTION_THRESHOLD,
    }


@app.post("/predict")
def predict(payload: SensorDataInput):
    """Predicts anomaly status based on sequence reconstruction error (MSE)."""
    if not payload.records:
        return {"predictions": [], "latency_ms": 0.0}

    start_time = time.perf_counter()
    df = pd.DataFrame(payload.records)

    # Check for required sensors
    missing = [col for col in CORE_SENSORS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Deep learning service requires sensor columns: {missing}",
        )

    # Extract & standardize sensor inputs
    raw_values = df[CORE_SENSORS].to_numpy(dtype=np.float32)
    norm_values = (raw_values - SENSOR_MEANS) / (SENSOR_STDS + 1e-6)

    # Batch tensor shape: (batch_size, 1, num_sensors)
    tensor_input = torch.tensor(norm_values).unsqueeze(1)

    with torch.no_grad():
        reconstructed = dl_model(tensor_input)
        # Compute mean squared error (reconstruction loss) per sample
        mse_losses = torch.mean((tensor_input - reconstructed) ** 2, dim=[1, 2]).numpy()

    results = []
    for loss in mse_losses:
        is_anomaly = int(loss >= RECONSTRUCTION_THRESHOLD)
        results.append({
            "reconstruction_mse": round(float(loss), 5),
            "is_anomaly": is_anomaly,
            "status": "ANOMALY ALERT" if is_anomaly == 1 else "NORMAL",
        })

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "service": "Deep Learning",
        "algorithm": "LSTM Autoencoder",
        "count": len(results),
        "latency_ms": round(latency_ms, 2),
        "predictions": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
