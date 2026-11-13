# Wind Turbine Gearbox Predictive Maintenance System (WT-GPMS)

An end-to-end, microservices-based predictive maintenance platform for detecting anomalous operating conditions in wind turbine gearbox SCADA telemetry. The system combines classical machine learning, deep learning, and quantum machine learning microservices behind a consensus-voting engine and an authenticated Streamlit monitoring dashboard backed by MongoDB.

## Key Features

- **Authenticated Security Portal:** Tabbed `Sign In` and `Create Account` flows with salted SHA-256 password hashing stored in MongoDB (`wind_turbine_db`) and an automatic in-memory fallback when MongoDB is unavailable.
- **Multi-Paradigm Microservice Inference (FastAPI):**
  - **Classical ML (`ml-service`, Port `8001`):** `scikit-learn` anomaly detection serving Local Outlier Factor (`lof_anomaly_detector.joblib`, with Isolation Forest / final bundle fallbacks).
  - **Deep Learning (`dl-service`, Port `8002`):** PyTorch reconstruction models supporting **LSTM Autoencoder** (`algorithm="lstm"`, default threshold `2.0`) and **MLP Dense Autoencoder** (`algorithm="mlp"`).
  - **Quantum ML (`qml-service`, Port `8003`):** PennyLane 4-qubit simulator (`default.qubit`) supporting **Variational Quantum Classifier** (`algorithm="vqc"`, threshold `0.52`) and **Quantum State Fidelity Overlap Metric** (`algorithm="fidelity"`, threshold `0.50`).
- **Ensemble Consensus Engine:** Aggregates predictions across the three paradigms via majority voting ($\ge 2/3$ agreement) to eliminate false alarms.
- **Interactive Dashboard (`frontend`, Port `8501`):**
  - **Sidebar Connectivity Telemetry:** Live health indicators for MongoDB (`27017`) and all three FastAPI services (`8001`, `8002`, `8003`).
  - **Tab 1 — Historical SCADA Telemetry & Batch Analysis:** Sensor visualizations, date filtering, triple-model batch comparison, and CSV export.
  - **Tab 2 — Manual Diagnostic & What-If Simulator:** Real-time manual sensor simulation with preset scenarios, an Algorithm Architecture Selector (DL & QML model switching), consensus alerts, and a persistent prediction Audit History table.
- **Data Validation & Temporal Feature Engineering:** Schema validation, timestamp ordering, rolling 1-hour statistics, and thermal gradient features.
- **Containerized & Cloud-Native Deployment:** Docker Compose multi-container orchestration (`turbine-network`) and Kubernetes manifests with Horizontal Pod Autoscaling (`k8s/hpa.yaml`).
- **Automated QA, Security & CI/CD:** GitHub Actions workflow running PyUnit (`unittest`), Flake8, Pylint, Bandit, and Docker image builds, paired with a 15-spec Cypress black-box E2E system test suite.

## Dataset

The project uses historical wind turbine gearbox SCADA sensor data sampled every 10 minutes.

- **Scale:** 262,800 records spanning 2025–2029, including 5,500 labelled anomalous observations across three periods.
- **Target column:** `is_anomaly` (`0` = normal, `1` = anomalous).

### Sensor Measurements

- Gearbox oil temperature
- Gearbox bearing temperature
- Vibration along the X, Y, and Z axes
- Oil pressure
- Particle count

> **Dataset Limitation:** The dataset does not contain turbine identifiers or verified failure/maintenance timestamps. Current models detect labelled anomalous operating states rather than predicting remaining useful life (RUL) across unseen turbines.

## Project Structure

```text
wind-turbine-predictive-maintenance/
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions CI pipeline
├── cypress/
│   ├── e2e/wt_gpms_system.cy.js    # 15 black-box E2E system test specs
│   └── run_cypress.js              # Cross-IDE Cypress launcher wrapper
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── model_accuracy_summary.md   # Paradigm & consensus benchmark summary
│   ├── Code_Coverage_Report.md
│   ├── Branch_Coverage_Report.md
│   ├── Unit_Test_Report.md
│   ├── System_Validation_Report.md
│   └── Updated_Test_Case_Document.md
├── k8s/
│   ├── deployment.yaml             # Frontend Kubernetes Deployment
│   ├── service.yaml                # Frontend Kubernetes Service
│   ├── ml-service.yaml             # Classical ML Deployment & Service
│   ├── dl-service.yaml             # Deep Learning Deployment & Service
│   ├── qml-service.yaml            # Quantum ML Deployment & Service
│   └── hpa.yaml                    # Horizontal Pod Autoscaler
├── models/
│   ├── lof_anomaly_detector.joblib
│   ├── isolation_forest.joblib
│   ├── final_anomaly_detector.joblib
│   └── dl_lstm_autoencoder.pt
├── src/
│   ├── services/
│   │   ├── ml_service.py           # FastAPI Classical ML service (Port 8001)
│   │   ├── dl_service.py           # FastAPI Deep Learning service (Port 8002)
│   │   └── qml_service.py          # FastAPI Quantum ML service (Port 8003)
│   ├── dashboard.py                # Streamlit UI & Consensus Portal (Port 8501)
│   ├── database.py                 # MongoDB client, auth & audit logging
│   ├── validate_data.py            # Dataset schema & integrity validation
│   ├── feature_engineering.py      # Rolling stats & thermal gradient features
│   ├── prepare_dataset.py          # Dataset preprocessing pipeline
│   ├── train_anomaly.py            # Anomaly model training
│   └── evaluate_final_model.py     # Model evaluation scripts
├── tests/
│   ├── test_unit.py                # Database, auth & pipeline unit tests
│   ├── test_microservices.py       # FastAPI ML/DL/QML endpoint tests
│   ├── test_dataset.py
│   └── test_model.py
├── cypress.config.js
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── package.json
├── requirements.txt
└── README.md
```

## Running the Project

### Option 1: Docker Compose (Recommended)

Starts all five containers (`ml-service`, `dl-service`, `qml-service`, `mongodb`, and `frontend`) on the `turbine-network` bridge network:

```bash
docker compose up --build
```

Once running, access:
- **Streamlit Dashboard:** `http://localhost:8501`
- **Classical ML API:** `http://localhost:8001/health`
- **Deep Learning API:** `http://localhost:8002/health`
- **Quantum ML API:** `http://localhost:8003/health`
- **MongoDB:** `mongodb://localhost:27017`

### Option 2: Local Python Environment

#### 1. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

#### 3. (Optional) Validate & prepare data

```bash
python src/validate_data.py
python src/prepare_dataset.py
```

#### 4. Start the microservices and dashboard

Run each service in a separate terminal from the repository root:

```bash
uvicorn src.services.ml_service:app --host 0.0.0.0 --port 8001
uvicorn src.services.dl_service:app --host 0.0.0.0 --port 8002
uvicorn src.services.qml_service:app --host 0.0.0.0 --port 8003
streamlit run src/dashboard.py
```

### Option 3: Kubernetes Deployment

Apply the manifests in `k8s/`:

```bash
kubectl apply -f k8s/
```

## Testing & Quality Assurance

### 1. Unit & Microservice Integration Tests (21 Tests)

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Covers dataset validation, database authentication/registration/audit logging (`tests/test_unit.py`), and FastAPI ML, DL (`lstm` / `mlp`), and QML (`vqc` / `fidelity`) endpoints (`tests/test_microservices.py`).

### 2. Black-Box E2E System Tests (15 Cypress Specs)

With the application running at `http://localhost:8501`:

```bash
npm install
npm run cy:run
```

For interactive mode in the Cypress GUI:

```bash
npm run cy:open
```

> `package.json` invokes `node cypress/run_cypress.js` so Cypress launches cleanly even from IDE terminals that set `ELECTRON_RUN_AS_NODE=1`.

### 3. Linting, Static Analysis & Security Scanning

Matches the CI pipeline in `.github/workflows/ci.yml`:

```bash
flake8 src/ tests/
bandit -c .bandit -r src/
pylint --rcfile=.pylintrc --fail-under=8.0 src/validate_data.py src/feature_engineering.py
```

## Model Benchmark Summary

From [`docs/model_accuracy_summary.md`](docs/model_accuracy_summary.md):

| Model / Paradigm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Classical ML (Isolation Forest / LOF)** | 98.01% | 51.30% | 100.00% | 0.6780 | 1.0000 | High recall; ~5,200 false alarms on normal data |
| **Deep Learning (PyTorch LSTM Autoencoder)** | 99.99% | 99.98% | 100.00% | 0.9999 | 1.0000 | Large MSE separation: Normal (~0.03) vs Anomaly (~79.4) |
| **Quantum ML (PennyLane 4-Qubit VQC)** | 78.71% | 93.69% | 49.11% | 0.6444 | 0.7716 | Conservative high-precision validator |
| **Ensemble Majority Consensus ($\ge 2/3$)** | 80.01% | 100.00% | 49.11% | 0.6587 | — | Eliminates 100% of false alarms |

> Benchmark figures reflect evaluation on the available historical dataset; independent validation on unseen turbines and verified failure logs is required prior to production deployment.

## Technologies Used

- **APIs & UI:** FastAPI, Uvicorn, Streamlit
- **ML / DL / QML:** scikit-learn, PyTorch, PennyLane
- **Data & Experiment Tracking:** pandas, NumPy, joblib, Matplotlib, MLflow, DVC
- **Database:** MongoDB (`pymongo`)
- **Containers & Orchestration:** Docker, Docker Compose, Kubernetes
- **Testing & CI/CD:** Python `unittest`, Cypress (`16.3.0`), GitHub Actions, Flake8, Pylint, Bandit

## License

Add a suitable license before permitting reuse. Check the original dataset provider's license separately; a repository license does not override upstream dataset terms.

## Author

Developed as a Software Engineering & Machine Learning project on wind turbine gearbox anomaly detection and predictive maintenance.
