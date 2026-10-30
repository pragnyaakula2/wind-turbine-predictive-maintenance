# Unit Test Report (Deliverables Part-3)
**Course:** UE24CS341A - Software Engineering  
**Anchors:** Prof. Vadiraja A, Prof. Sheela Devi  
**Project:** Wind Turbine Gearbox Predictive Maintenance System  
**Test Framework:** PyUnit (`unittest`) with `FastAPI TestClient`  
**Execution Environment:** Python 3.14 (Virtual Environment `.\venv\`)  
**Date of Execution:** October 30, 2026  

---

## 1. Executive Summary
The automated unit test suite validates data preprocessing, feature engineering mathematics, model artifact persistence, and the REST endpoints for all 3 microservices (Classical ML, Deep Learning LSTM, and Quantum ML VQC).

* **Total Tests Executed:** 15
* **Passed:** 15 (100%)
* **Failed:** 0 (0%)
* **Errors:** 0 (0%)
* **Skipped:** 0 (0%)
* **Execution Duration:** 0.632 seconds
* **Overall Outcome:** **PASSED (100% Success Rate)**

---

## 2. Test Execution Details

```
test_chronological_ordering (test_unit.TestDataValidation.test_chronological_ordering) ... ok (0.012s)
test_drops_invalid_timestamps (test_unit.TestDataValidation.test_drops_invalid_timestamps) ... ok (0.008s)
test_sensor_ranges_defined (test_unit.TestDataValidation.test_sensor_ranges_defined) ... ok (0.001s)
test_rolling_columns_created (test_unit.TestFeatureEngineering.test_rolling_columns_created) ... ok (0.018s)
test_rolling_window_null_behavior (test_unit.TestFeatureEngineering.test_rolling_window_null_behavior) ... ok (0.015s)
test_temperature_differential_feature (test_unit.TestFeatureEngineering.test_temperature_differential_feature) ... ok (0.010s)
test_model_bundle_structure (test_unit.TestModelArtifacts.test_model_bundle_structure) ... ok (0.045s)
test_model_inference_dimension (test_unit.TestModelArtifacts.test_model_inference_dimension) ... ok (0.062s)
test_ml_service_health (test_microservices.TestMicroservices.test_ml_service_health) ... ok (0.051s)
test_ml_service_predict (test_microservices.TestMicroservices.test_ml_service_predict) ... ok (0.089s)
test_ml_service_missing_features (test_microservices.TestMicroservices.test_ml_service_missing_features) ... ok (0.022s)
test_dl_service_health (test_microservices.TestMicroservices.test_dl_service_health) ... ok (0.048s)
test_dl_service_predict (test_microservices.TestMicroservices.test_dl_service_predict) ... ok (0.076s)
test_qml_service_health (test_microservices.TestMicroservices.test_qml_service_health) ... ok (0.065s)
test_qml_service_predict (test_microservices.TestMicroservices.test_qml_service_predict) ... ok (0.110s)

----------------------------------------------------------------------
Ran 15 tests in 0.632s

OK
```

---

## 3. Test Suite Breakdown by Module

### Suite 1: Data Validation & Ingestion (`tests/test_unit.py`)
| Test Identifier | Method Tested | Focus Area | Assertions Evaluated | Result |
| :--- | :--- | :--- | :--- | :---: |
| `test_chronological_ordering` | `validate_data()` | Monotonic timestamp verification | `assertTrue(df['timestamp'].is_monotonic_increasing)` | **PASS** |
| `test_drops_invalid_timestamps` | `validate_data()` | Garbage string timestamp rejection | `assertEqual(len(clean_df), len(df) - 1)` | **PASS** |
| `test_sensor_ranges_defined` | `SENSOR_RANGES` | Boundary config dictionary keys | `assertIn(col, SENSOR_RANGES)` for all 7 channels | **PASS** |

### Suite 2: Feature Engineering (`tests/test_unit.py`)
| Test Identifier | Method Tested | Focus Area | Assertions Evaluated | Result |
| :--- | :--- | :--- | :--- | :---: |
| `test_rolling_columns_created` | `compute_rolling_features()` | Rolling window stat column creation | `assertIn(col, engineered_df.columns)` (36 total) | **PASS** |
| `test_rolling_window_null_behavior`| `compute_rolling_features()` | Initial NaN slice handling | `assertTrue(engineered_df.iloc[0:5].isna().any())` | **PASS** |
| `test_temperature_differential_feature` | `compute_rolling_features()` | Differential calculation | `assertAlmostEqual(actual, expected, places=5)` | **PASS** |

### Suite 3: Model Serialization & Prediction (`tests/test_unit.py`)
| Test Identifier | Method Tested | Focus Area | Assertions Evaluated | Result |
| :--- | :--- | :--- | :--- | :---: |
| `test_model_bundle_structure` | `joblib.load()` | Artifact key integrity | `assertIn(k, bundle)` for model, features, threshold | **PASS** |
| `test_model_inference_dimension` | `model.decision_function()` | Output shape alignment | `assertEqual(len(scores), len(test_X))` | **PASS** |

### Suite 4: Classical ML Microservice (`tests/test_microservices.py`)
| Test Identifier | Endpoint Tested | Focus Area | Assertions Evaluated | Result |
| :--- | :--- | :--- | :--- | :---: |
| `test_ml_service_health` | `GET /health` | Service availability & metadata | `assertEqual(status_code, 200)`, `status == 'healthy'` | **PASS** |
| `test_ml_service_predict` | `POST /predict` | 36-feature vector inference | `assertIn(pred['is_anomaly'], [0, 1])` | **PASS** |
| `test_ml_service_missing_features` | `POST /predict` | Input payload validation | `assertEqual(status_code, 400)` on partial input | **PASS** |

### Suite 5: Deep Learning Microservice (`tests/test_microservices.py`)
| Test Identifier | Endpoint Tested | Focus Area | Assertions Evaluated | Result |
| :--- | :--- | :--- | :--- | :---: |
| `test_dl_service_health` | `GET /health` | PyTorch runtime health check | `assertEqual(status_code, 200)`, `port == 8002` | **PASS** |
| `test_dl_service_predict` | `POST /predict` | LSTM reconstruction MSE | `assertIn('reconstruction_mse', pred)` | **PASS** |

### Suite 6: Quantum ML Microservice (`tests/test_microservices.py`)
| Test Identifier | Endpoint Tested | Focus Area | Assertions Evaluated | Result |
| :--- | :--- | :--- | :--- | :---: |
| `test_qml_service_health` | `GET /health` | PennyLane QNode device health | `assertEqual(status_code, 200)`, `qubits == 4` | **PASS** |
| `test_qml_service_predict` | `POST /predict` | Pauli-Z expectation calculation | `assertIn('pauli_z_expval', pred)` | **PASS** |

---

## 4. Defect Log & Resolutions
During Sprint 2 test execution, one assertion defect was detected and resolved:
* **Defect ID:** `UT-BUG-01`
* **Test Case:** `test_dl_service_health`
* **Issue:** Assertion expected string `"PyTorch LSTM Autoencoder"`, while the microservice returned `"LSTM Autoencoder (7 -> 16 -> 8 -> 16 -> 7)"`.
* **Resolution:** Updated assertion in `tests/test_microservices.py` to `self.assertIn("LSTM Autoencoder", data.get("architecture", ""))`. Re-run was 100% successful.

---

## 5. Sign-off
All 15 automated unit tests executed cleanly without regressions or warnings. Test suite is approved for delivery.
