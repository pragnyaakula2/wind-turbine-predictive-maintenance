"""
Unit tests for Wind Turbine Predictive Maintenance Microservices.
Covers:
- Classical ML FastAPI service (Port 8001)
- PyTorch Deep Learning LSTM Autoencoder service (Port 8002)
- PennyLane 4-Qubit Quantum ML VQC service (Port 8003)
"""

import unittest
from fastapi.testclient import TestClient

from src.services.ml_service import app as ml_app, features as ml_features
from src.services.dl_service import app as dl_app, CORE_SENSORS as dl_sensors
from src.services.qml_service import app as qml_app, QML_SENSORS as qml_sensors


class TestMicroservices(unittest.TestCase):
    """Test suite verifying endpoints and inference of all 3 microservices."""

    @classmethod
    def setUpClass(cls):
        """Initialize FastAPI TestClients for all microservices."""
        cls.ml_client = TestClient(ml_app)
        cls.dl_client = TestClient(dl_app)
        cls.qml_client = TestClient(qml_app)

    # ---------------- Classical ML Service Tests (Port 8001) ----------------
    def test_ml_service_health(self):
        """Verify Classical ML health check endpoint returns 200 and healthy status."""
        response = self.ml_client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("port"), 8001)
        self.assertTrue(data.get("model_loaded"))

    def test_ml_service_predict(self):
        """Verify Classical ML inference returns anomaly prediction with valid scores."""
        # Create a sample record with all expected engineered features
        sample_record = {f: 0.1 for f in ml_features}
        response = self.ml_client.post("/predict", json={"records": [sample_record]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("count"), 1)
        self.assertIn("predictions", data)
        self.assertIn(data["predictions"][0]["is_anomaly"], [0, 1])

    def test_ml_service_missing_features(self):
        """Verify Classical ML returns 400 error when required features are missing."""
        invalid_record = {"vibration_x": 0.5}
        response = self.ml_client.post("/predict", json={"records": [invalid_record]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required model features", response.json().get("detail", ""))

    # ---------------- Deep Learning Service Tests (Port 8002) ---------------
    def test_dl_service_health(self):
        """Verify Deep Learning health check endpoint returns 200 and healthy status."""
        response = self.dl_client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("port"), 8002)
        self.assertIn("LSTM Autoencoder", data.get("architecture", ""))

    def test_dl_service_predict(self):
        """Verify Deep Learning autoencoder returns reconstruction MSE and anomaly label."""
        sample_record = {s: 60.0 for s in dl_sensors}
        response = self.dl_client.post("/predict", json={"records": [sample_record]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("count"), 1)
        self.assertIn("reconstruction_mse", data["predictions"][0])
        self.assertIn(data["predictions"][0]["is_anomaly"], [0, 1])

    # ---------------- Quantum ML Service Tests (Port 8003) ------------------
    def test_qml_service_health(self):
        """Verify Quantum ML health check endpoint returns 200 and healthy status."""
        response = self.qml_client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("port"), 8003)
        self.assertEqual(data.get("qubits"), 4)

    def test_qml_service_predict(self):
        """Verify Quantum ML circuit execution returns expectation value and prediction."""
        sample_record = {s: 50.0 for s in qml_sensors}
        response = self.qml_client.post("/predict", json={"records": [sample_record]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("count"), 1)
        self.assertIn("pauli_z_expval", data["predictions"][0])
        self.assertIn("quantum_anomaly_score", data["predictions"][0])
        self.assertIn(data["predictions"][0]["is_anomaly"], [0, 1])


if __name__ == "__main__":
    unittest.main()
