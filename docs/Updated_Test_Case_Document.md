# Updated Test Case Document (Deliverables Part-3)
**Course:** UE24CS341A - Software Engineering  
**Anchors:** Prof. Vadiraja A, Prof. Sheela Devi  
**Project:** Wind Turbine Gearbox Predictive Maintenance System  
**System Architecture:** Multi-Tier Microservices (Classical ML, PyTorch DL, PennyLane QML) + Streamlit Frontend + Docker + Kubernetes  
**Document Revision:** 2.0 (Post Sprint-2 Updations)  
**Date:** October 30, 2026  

---

## 1. Introduction & Updation Rationale
During Sprint 1 and Sprint 2, the system evolved from a monolithic batch anomaly detector into a distributed microservices platform featuring:
1. **Classical ML Service (Port 8001):** Isolation Forest / LOF anomaly detection.
2. **Deep Learning Service (Port 8002):** PyTorch LSTM Autoencoder for temporal anomaly reconstruction.
3. **Quantum ML Service (Port 8003):** PennyLane 4-Qubit Variational Quantum Classifier (VQC) with Pauli-Z expectation values.
4. **Streamlit Monitoring Frontend (Port 8501):** Multi-model selection and Triple Model Comparison with Majority Consensus Voting ($\ge 2/3$ agreement).
5. **Containerization & Orchestration:** Docker Compose multi-container networking and Kubernetes multi-pod deployments.

This document updates the original Test Plan to reflect all new architectural components, API contracts, functional requirements (FR), and non-functional requirements (NFR).

---

## 2. Requirements Traceability Matrix (RTM)

| Req ID | Requirement Description | Sprint | Associated Test Case IDs | Test Type | Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **FR-01** | Raw sensor data validation & timestamp ordering | Sprint 1 | UT-01, UT-02, UT-03 | Unit | PASS |
| **FR-02** | 1-Hour rolling statistics & temperature differentials | Sprint 1 | UT-04, UT-05, UT-06 | Unit | PASS |
| **FR-03** | Model bundle structure & score dimensions | Sprint 1 | UT-07, UT-08 | Unit | PASS |
| **FR-04** | Classical ML REST API health & inference | Sprint 2 | UT-09, UT-10, UT-11, SYS-03 | Unit / System | PASS |
| **FR-05** | PyTorch Deep Learning LSTM Autoencoder inference | Sprint 2 | UT-12, UT-13, SYS-04 | Unit / System | PASS |
| **FR-06** | PennyLane Quantum ML 4-Qubit VQC circuit execution | Sprint 2 | UT-14, UT-15, SYS-05 | Unit / System | PASS |
| **FR-07** | Dashboard Triple Model Comparison & Consensus Voting | Sprint 2 | SYS-06, SYS-07 | System | PASS |
| **FR-08** | Time-series filtering & CSV export | Sprint 2 | SYS-02, SYS-08 | System | PASS |
| **NFR-01** | API response latency under 500ms | Sprint 2 | SYS-03, SYS-04, SYS-05 | Performance | PASS |
| **NFR-02** | Fault tolerance & degraded-mode operation | Sprint 2 | SYS-09 | Reliability | PASS |
| **NFR-03** | Code quality, style, & security compliance | Sprints 1-2 | Flake8, Bandit, Pylint | Static QA | PASS |

---

## 3. Detailed Test Case Specifications

### Category A: Automated Unit Test Cases (`tests/test_unit.py` & `tests/test_microservices.py`)

#### UT-01: Chronological Timestamp Sorting
- **Objective:** Verify raw sensor records are ordered monotonically ascending by timestamp.
- **Input:** Unsorted DataFrame sample with inverted timestamps.
- **Expected Result:** `validate_data()` returns chronologically ordered DataFrame (`df['timestamp'].is_monotonic_increasing == True`).
- **Status:** PASS

#### UT-02: Invalid Timestamp Stripping
- **Objective:** Verify unparseable or corrupted timestamp strings are dropped.
- **Input:** Sample record with `timestamp="invalid-timestamp-string"`.
- **Expected Result:** Corrupted row is removed; invalid timestamp counter equals 1.
- **Status:** PASS

#### UT-03: Sensor Boundary Thresholds Presence
- **Objective:** Ensure all 7 critical gearbox physical boundary limits are defined.
- **Expected Result:** `SENSOR_RANGES` contains keys for oil temp, bearing temp, vibration (x, y, z), oil pressure, and particle count.
- **Status:** PASS

#### UT-04: Rolling Feature Generation
- **Objective:** Ensure 1-hour rolling window generates `mean`, `std`, `min`, and `max` columns for each sensor.
- **Expected Result:** DataFrame has 36 engineered columns after calling `compute_rolling_features()`.
- **Status:** PASS

#### UT-05: Rolling Window NaN Behavior
- **Objective:** Verify that the first $(window - 1)$ entries contain NaN prior to filling.
- **Expected Result:** Validated rolling slice behavior matches pandas time-series spec.
- **Status:** PASS

#### UT-06: Oil-Bearing Temperature Differential
- **Objective:** Verify `oil_bearing_temp_diff` correctly computes $(T_{\text{oil}} - T_{\text{bearing}})$.
- **Expected Result:** Mathematical equality within floating-point tolerance $10^{-5}$.
- **Status:** PASS

#### UT-07: Saved Model Artifact Integrity
- **Objective:** Verify serialized bundle contains `model`, `features`, and `threshold`.
- **Expected Result:** `joblib.load()` succeeds and dictionary keys are non-empty.
- **Status:** PASS

#### UT-08: Model Inference Output Dimension
- **Objective:** Verify batch inference output shape matches input sample count.
- **Expected Result:** Output score vector length == input row count.
- **Status:** PASS

#### UT-09: Classical ML Microservice Health Check
- **Objective:** Verify `GET http://localhost:8001/health` returns operational status.
- **Expected Result:** HTTP 200, `status="healthy"`, `port=8001`, `model_loaded=True`.
- **Status:** PASS

#### UT-10: Classical ML Anomaly Scoring Inference
- **Objective:** Verify `POST http://localhost:8001/predict` computes anomaly scores.
- **Input:** Feature dictionary matching the 36 trained features.
- **Expected Result:** HTTP 200, response contains `anomaly_score`, `is_anomaly` $\in \{0, 1\}$.
- **Status:** PASS

#### UT-11: Classical ML Missing Features Guard
- **Objective:** Verify API returns HTTP 400 when required feature keys are absent.
- **Input:** Incomplete dictionary `{"vibration_x": 0.5}`.
- **Expected Result:** HTTP 400 with detail `"Missing required model features"`.
- **Status:** PASS

#### UT-12: Deep Learning Microservice Health Check
- **Objective:** Verify `GET http://localhost:8002/health` returns operational status.
- **Expected Result:** HTTP 200, `status="healthy"`, `architecture="PyTorch LSTM Autoencoder"`.
- **Status:** PASS

#### UT-13: Deep Learning LSTM Reconstruction Inference
- **Objective:** Verify `POST http://localhost:8002/predict` computes reconstruction error.
- **Input:** 7 raw sensor channels `[T_oil, T_bearing, vib_x, vib_y, vib_z, pressure, particles]`.
- **Expected Result:** HTTP 200, returns `reconstruction_mse` and anomaly classification.
- **Status:** PASS

#### UT-14: Quantum ML Microservice Health Check
- **Objective:** Verify `GET http://localhost:8003/health` returns operational status.
- **Expected Result:** HTTP 200, `status="healthy"`, `qubits=4`.
- **Status:** PASS

#### UT-15: Quantum ML Circuit Execution Inference
- **Objective:** Verify `POST http://localhost:8003/predict` executes PennyLane quantum circuit.
- **Input:** 4 normalized sensor angles mapped to qubits.
- **Expected Result:** HTTP 200, returns `pauli_z_expval`, `quantum_anomaly_score`, `is_anomaly`.
- **Status:** PASS

---

### Category B: System & Integration Test Cases

#### SYS-01: Multi-Service Dashboard Health Indicators
- **Preconditions:** All microservices active.
- **Steps:** Load `http://localhost:8501`.
- **Expected:** Header displays 3 green badges: `ML [8001]`, `DL [8002]`, `QML [8003]`.
- **Status:** PASS

#### SYS-02: Dynamic Sensor Range Filtering
- **Steps:** Adjust sidebar window slider from 100 to 1,000 samples.
- **Expected:** Real-time chart updates within 500ms without UI flicker.
- **Status:** PASS

#### SYS-03: Single Model Scoring — Isolation Forest
- **Steps:** Select "Classical ML" in dropdown; review December 2029 failure events.
- **Expected:** Anomaly score spikes above threshold $0.1274$; visual warning triggers.
- **Status:** PASS

#### SYS-04: Single Model Scoring — PyTorch LSTM Autoencoder
- **Steps:** Select "Deep Learning" in dropdown; review reconstruction MSE chart.
- **Expected:** MSE spikes during sensor perturbation bursts, identifying anomalous state.
- **Status:** PASS

#### SYS-05: Single Model Scoring — PennyLane Quantum ML
- **Steps:** Select "Quantum ML" in dropdown; inspect Pauli-Z expectation value.
- **Expected:** Pauli-Z metric maps cleanly to $[0, 1]$ interval; flags anomalies above $0.45$.
- **Status:** PASS

#### SYS-06: Triple Model Comparison View
- **Steps:** Check "Triple Model Comparison" toggle.
- **Expected:** 3 parallel columns display individual model outputs side-by-side.
- **Status:** PASS

#### SYS-07: Ensemble Majority Consensus Voting
- **Steps:** Review consensus banner across normal vs failure timestamps.
- **Expected:** Triggers **CRITICAL ALERT** if and only if $\ge 2$ of 3 models agree on anomaly.
- **Status:** PASS

#### SYS-08: Filtered Data Export
- **Steps:** Click "Download Filtered Dataset as CSV".
- **Expected:** Browser saves `turbine_monitoring_export.csv` with valid records.
- **Status:** PASS

#### SYS-09: Degraded-Mode Backend Resilience
- **Steps:** Stop DL container on port 8002; observe dashboard behavior.
- **Expected:** DL badge turns red; ML and QML continue functioning with warning notification.
- **Status:** PASS

#### SYS-10: High-Frequency UI Stress Test
- **Steps:** Rapidly scrub slider 10 times in 5 seconds.
- **Expected:** Streamlit handles request debounce; no unhandled runtime exceptions.
- **Status:** PASS
