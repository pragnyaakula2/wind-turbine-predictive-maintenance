"""Unit test suite using PyUnit (unittest) for the Wind Turbine Predictive Maintenance system."""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import joblib

from src.validate_data import validate_data, SENSOR_RANGES
from src.feature_engineering import engineer_features, SENSOR_COLUMNS


class TestDataValidation(unittest.TestCase):
    """PyUnit test cases for sensor data validation."""

    def setUp(self):
        """Builds a synthetic dataframe mimicking sensor records."""
        self.sample_data = pd.DataFrame({
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:10:00",
                "2026-01-01 00:20:00",
                "invalid_date",
            ],
            "gearbox_oil_temp": [60.0, 65.0, 70.0, 62.0],
            "gearbox_bearing_temp": [58.0, 62.0, 68.0, 60.0],
            "vibration_x": [1.2, 1.3, 1.4, 1.2],
            "vibration_y": [1.1, 1.2, 1.3, 1.1],
            "vibration_z": [0.9, 1.0, 1.1, 0.9],
            "oil_pressure": [4.5, 4.6, 4.4, 4.5],
            "particle_count": [120, 130, 125, 122],
            "is_anomaly": [0, 0, 0, 0],
        })

    def test_drops_invalid_timestamps(self):
        """Verifies that invalid timestamps are stripped."""
        validated = validate_data(self.sample_data)
        self.assertEqual(len(validated), 3)
        self.assertFalse(validated["timestamp"].isna().any())

    def test_chronological_ordering(self):
        """Verifies that records are sorted chronologically."""
        shuffled = self.sample_data.iloc[::-1].copy()
        validated = validate_data(shuffled)
        self.assertTrue(validated["timestamp"].is_monotonic_increasing)

    def test_sensor_ranges_defined(self):
        """Ensures all expected sensor ranges are present in configuration."""
        expected_sensors = [
            "gearbox_oil_temp",
            "gearbox_bearing_temp",
            "vibration_x",
            "vibration_y",
            "vibration_z",
            "oil_pressure",
            "particle_count",
        ]
        for sensor in expected_sensors:
            self.assertIn(sensor, SENSOR_RANGES)
            low, high = SENSOR_RANGES[sensor]
            self.assertLess(low, high)


class TestFeatureEngineering(unittest.TestCase):
    """PyUnit test cases for feature engineering and rolling statistics."""

    def setUp(self):
        """Generates a 12-period synthetic series (2 hours at 10-min intervals)."""
        dates = pd.date_range("2026-01-01 00:00:00", periods=12, freq="10min")
        self.df = pd.DataFrame({
            "timestamp": dates,
            "gearbox_oil_temp": np.linspace(50, 72, 12),
            "gearbox_bearing_temp": np.linspace(48, 70, 12),
            "vibration_x": np.ones(12) * 1.5,
            "vibration_y": np.ones(12) * 1.2,
            "vibration_z": np.ones(12) * 0.8,
            "oil_pressure": np.ones(12) * 4.0,
            "particle_count": np.ones(12) * 100,
        })

    def test_temperature_differential_feature(self):
        """Checks calculation of oil_bearing_temp_diff."""
        out = engineer_features(self.df, window=6)
        self.assertIn("oil_bearing_temp_diff", out.columns)
        expected = self.df["gearbox_oil_temp"] - self.df["gearbox_bearing_temp"]
        np.testing.assert_allclose(out["oil_bearing_temp_diff"], expected)

    def test_rolling_columns_created(self):
        """Verifies that 1h mean, std, min, and max columns are generated."""
        out = engineer_features(self.df, window=6)
        for col in SENSOR_COLUMNS:
            for suffix in ["_mean_1h", "_std_1h", "_min_1h", "_max_1h"]:
                self.assertIn(f"{col}{suffix}", out.columns)

    def test_rolling_window_null_behavior(self):
        """First (window - 1) rows should be NaN for rolling features."""
        window = 6
        out = engineer_features(self.df, window=window)
        # First 5 rows should have NaN for rolling mean
        self.assertTrue(out["vibration_x_mean_1h"].iloc[:window - 1].isna().all())
        # 6th row and onward should have non-null values
        self.assertFalse(out["vibration_x_mean_1h"].iloc[window - 1:].isna().any())


class TestModelArtifacts(unittest.TestCase):
    """PyUnit test cases for model serialization and inference format."""

    def setUp(self):
        # Look for existing model artifacts
        candidate_paths = [
            Path("models/final_anomaly_detector.joblib"),
            Path("models/isolation_forest.joblib"),
            Path("models/lof_anomaly_detector.joblib"),
        ]
        self.available_model = next((p for p in candidate_paths if p.exists()), None)

    def test_model_bundle_structure(self):
        """Checks that saved model artifact contains model, features, and threshold."""
        if self.available_model is None:
            self.skipTest("No saved model bundle currently in models/")

        bundle = joblib.load(self.available_model)
        self.assertIsInstance(bundle, dict)
        self.assertIn("model", bundle)
        self.assertIn("features", bundle)
        self.assertIn("threshold", bundle)
        self.assertIsInstance(bundle["features"], (list, np.ndarray))
        self.assertIsInstance(float(bundle["threshold"]), float)

    def test_model_inference_dimension(self):
        """Ensures that inference produces finite numerical scores matching input length."""
        if self.available_model is None:
            self.skipTest("No saved model bundle currently in models/")

        bundle = joblib.load(self.available_model)
        model = bundle["model"]
        features = bundle["features"]

        # Synthesize 5 sample input rows matching feature count
        sample_x = pd.DataFrame(
            np.random.normal(loc=1.0, scale=0.5, size=(5, len(features))),
            columns=features,
        )

        scores = -model.decision_function(sample_x)
        self.assertEqual(len(scores), 5)
        self.assertTrue(np.isfinite(scores).all())

        # Test binary classification logic against threshold
        predictions = (scores >= bundle["threshold"]).astype(int)
        self.assertTrue(set(predictions).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
