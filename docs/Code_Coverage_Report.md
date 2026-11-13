# Code Coverage Report (Deliverables Part-3)
**Course:** UE24CS341A - Software Engineering  
**Anchors:** Prof. Vadiraja A, Prof. Sheela Devi  
**Project:** Wind Turbine Gearbox Predictive Maintenance System  
**Coverage Tool:** `coverage.py` v7.17.0 (Python 3.14)  
**Execution Command:** `coverage run --branch --source=src -m unittest discover -s tests -p "test_*.py"`  
**Report Generation Date:** October 30, 2026  

---

## 1. Executive Summary
Code coverage measures the percentage of executable Python statements exercised during the automated test suite execution. High coverage ensures that critical error handling, mathematical transformations, and API contract logic have been thoroughly verified.

* **Target Core Production Coverage:** $\ge 75\%$
* **Achieved Core Production Coverage:** **80%**
* **Interactive HTML Report:** Generated in `docs/coverage_html/index.html`

---

## 2. Core Production Modules Coverage Summary

| Module Name | File Location | Statements (Stmts) | Missed (Miss) | Statement Coverage (%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Deep Learning Service** | `src/services/dl_service.py` | 60 | 4 | **90%** | **EXCELLENT** |
| **Quantum ML Service** | `src/services/qml_service.py` | 54 | 4 | **89%** | **EXCELLENT** |
| **Classical ML Service** | `src/services/ml_service.py` | 50 | 5 | **85%** | **EXCELLENT** |
| **Data Validation Pipeline** | `src/validate_data.py` | 34 | 11 | **64%** | **GOOD** |
| **Feature Engineering Pipeline** | `src/feature_engineering.py` | 35 | 12 | **62%** | **GOOD** |
| **Service Package Init** | `src/services/__init__.py` | 0 | 0 | **100%** | **COMPLETE** |
| **TOTAL (Core Pipeline & Services)** | — | **233** | **36** | **80%** | **TARGET EXCEEDED** |

---

## 3. Module-by-Module Coverage Analysis

### 3.1 Deep Learning Service (`src/services/dl_service.py`) — 90% Coverage
- **Covered Logic:**
  - `LSTMAutoencoder` PyTorch neural network class initialization and forward pass.
  - Encoder linear compression (`7 -> 16 -> 8`) and decoder reconstruction (`8 -> 16 -> 7`).
  - `/health` endpoint returning model architecture, parameters, and healthy state.
  - `/predict` endpoint payload validation, feature normalization, PyTorch tensor casting, and MSE reconstruction loss calculation.
- **Uncovered Lines (10%):**
  - Standalone script execution entry point (`if __name__ == "__main__": uvicorn.run(...)`).

### 3.2 Quantum ML Service (`src/services/qml_service.py`) — 89% Coverage
- **Covered Logic:**
  - PennyLane 4-qubit device definition (`default.qubit`) and quantum circuit compilation.
  - `AngleEmbedding` rotation around the Y-axis and CNOT entanglement gates.
  - Pauli-Z expectation value computation (`qml.expval(qml.PauliZ(0))`).
  - `/health` and `/predict` endpoints handling request parsing, feature boundary scaling, and output scoring.
- **Uncovered Lines (11%):**
  - CLI execution block (`if __name__ == "__main__"`).

### 3.3 Classical ML Service (`src/services/ml_service.py`) — 85% Coverage
- **Covered Logic:**
  - Dynamic discovery and loading of saved `.joblib` model bundles.
  - Extraction of fitted model, 36 feature names, and decision threshold.
  - `/health` endpoint metadata generation.
  - Full batch inference on sensor feature dictionaries and anomaly classification.
  - Missing feature detection returning HTTP 400 with missing column names.
- **Uncovered Lines (15%):**
  - Fallback exception path when no model bundle is found on disk and main CLI runner.

### 3.4 Data Validation (`src/validate_data.py`) — 64% Coverage
- **Covered Logic:**
  - Monotonic timestamp ordering verification.
  - Parsing and dropping of corrupted timestamp strings.
  - Boundary limits validation dictionary structure.
- **Uncovered Lines (36%):**
  - Script command-line runner and disk export helper functions.

### 3.5 Feature Engineering (`src/feature_engineering.py`) — 62% Coverage
- **Covered Logic:**
  - 1-hour rolling window computation (`mean`, `std`, `min`, `max`) for 7 sensors.
  - Oil-to-bearing temperature differential feature calculation.
  - Handling of initial window NaN slices.
- **Uncovered Lines (38%):**
  - Standalone batch execution script saving intermediate `.parquet` files to disk.

---

## 4. Test Adequacy & Code Quality Alignment
* **Static Analysis:** Flake8 reports **0 lint violations**; Bandit security scanner reports **0 vulnerabilities**; Pylint rates the core modules at **10.0/10**.
* **Reliability:** All critical decision paths and network contracts across microservices exceed 85% statement coverage.
