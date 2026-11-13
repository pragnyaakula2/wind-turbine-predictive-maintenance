"""Microservice 2: Deep Learning LSTM Autoencoder API (Port 8002).

Uses a PyTorch Recurrent Autoencoder to detect anomalies via reconstruction error.
"""

import time
from pathlib import Path
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

# Model artifact path and fallback parameters
MODEL_WEIGHTS_PATH = Path("models/dl_lstm_autoencoder.pt")

SENSOR_MEANS = np.array([60.87, 67.83, 0.021, 0.0167, 0.0208, 4.28, 918.8], dtype=np.float32)
SENSOR_STDS = np.array([7.34, 7.38, 0.0031, 0.0022, 0.0027, 0.088, 452.2], dtype=np.float32)
RECONSTRUCTION_THRESHOLD = 2.0


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


class MLPAutoencoder(nn.Module):
    """Algorithm 2: Compact Feedforward Dense Autoencoder (7 -> 12 -> 4 -> 12 -> 7)."""

    def __init__(self, input_dim: int = 7):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Initialize model architectures
torch.manual_seed(42)
dl_model = LSTMAutoencoder(input_dim=len(CORE_SENSORS))
mlp_model = MLPAutoencoder(input_dim=len(CORE_SENSORS))

# Load pre-trained weights if available
if MODEL_WEIGHTS_PATH.exists():
    try:
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        dl_model.load_state_dict(checkpoint["model_state_dict"])
        if "means" in checkpoint:
            SENSOR_MEANS = np.array(checkpoint["means"], dtype=np.float32)
        if "stds" in checkpoint:
            SENSOR_STDS = np.array(checkpoint["stds"], dtype=np.float32)
        RECONSTRUCTION_THRESHOLD = float(checkpoint.get("threshold", RECONSTRUCTION_THRESHOLD))
    except Exception as exc:
        print(f"Warning: Failed to load trained DL weights: {exc}")

dl_model.eval()
mlp_model.eval()


class SensorDataInput(BaseModel):
    records: List[Dict[str, Any]]
    algorithm: str = "lstm"  # options: "lstm" (default) or "mlp"


@app.get("/health")
def health():
    """Health check endpoint for Kubernetes liveness & readiness probes."""
    return {
        "status": "healthy",
        "service": "Deep Learning Multi-Algorithm Service",
        "port": 8002,
        "framework": "PyTorch",
        "supported_algorithms": [
            "1. LSTM Autoencoder (Temporal Recurrent)",
            "2. MLP Dense Autoencoder (Feedforward)",
        ],
        "architecture": (
            "LSTM Autoencoder (7 -> 16 -> 8 -> 16 -> 7) + "
            "MLP Autoencoder (7 -> 12 -> 4 -> 12 -> 7)"
        ),
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

    algo_choice = getattr(payload, "algorithm", "lstm").lower()

    if algo_choice == "mlp":
        tensor_2d = torch.tensor(norm_values)
        with torch.no_grad():
            reconstructed = mlp_model(tensor_2d)
            mse_losses = torch.mean((tensor_2d - reconstructed) ** 2, dim=1).numpy()
        algo_name = "Feedforward Dense Autoencoder (MLP)"
    else:
        # Batch tensor shape: (batch_size, 1, num_sensors)
        tensor_input = torch.tensor(norm_values).unsqueeze(1)
        with torch.no_grad():
            reconstructed = dl_model(tensor_input)
            # Compute mean squared error (reconstruction loss) per sample
            mse_losses = torch.mean((tensor_input - reconstructed) ** 2, dim=[1, 2]).numpy()
        algo_name = "LSTM Autoencoder"

    results = []
    for loss in mse_losses:
        is_anomaly = int(loss >= RECONSTRUCTION_THRESHOLD)
        results.append({
            "reconstruction_mse": round(float(loss), 5),
            "is_anomaly": is_anomaly,
            "status": "ANOMALY ALERT" if is_anomaly == 1 else "NORMAL",
            "algorithm": algo_name,
        })

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "service": "Deep Learning",
        "algorithm": algo_name,
        "count": len(results),
        "latency_ms": round(latency_ms, 2),
        "predictions": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)  # nosec B104
