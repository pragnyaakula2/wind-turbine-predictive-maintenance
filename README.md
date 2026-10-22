# Wind Turbine Gearbox Predictive Maintenance System

A machine learning prototype for detecting anomalous operating conditions in wind turbine gearbox sensor data. The project combines sensor data preprocessing, temporal feature engineering, Local Outlier Factor (LOF) anomaly detection, evaluation tools, and an interactive Streamlit dashboard.

## Features

- **Data validation:** Checks dataset structure, missing values, labels, and timestamps.
- **Feature engineering:** Generates rolling one-hour statistics and temperature-difference features.
- **Anomaly detection:** Uses a saved Local Outlier Factor model to score sensor observations.
- **Model evaluation:** Includes scripts for evaluating predictions, anomaly events, and alert patterns.
- **Interactive dashboard:** Displays sensor trends, prediction tables, date filters, and CSV exports.
- **Automated testing:** Includes tests for dataset validity and model functionality.

## Dataset

The project uses historical wind turbine gearbox sensor data sampled every 10 minutes.

The raw dataset contains 262,800 records spanning 2025–2029, with seven sensor measurements and a binary anomaly label.

### Sensor measurements

- Gearbox oil temperature
- Gearbox bearing temperature
- Vibration along the X, Y, and Z axes
- Oil pressure
- Particle count

The target column, `is_anomaly`, uses `0` for normal observations and `1` for anomalous observations.

**Dataset limitation:** The available dataset does not contain turbine identifiers or verified failure and maintenance records. The current project detects labelled anomalous observations; it does not establish remaining useful life or validated advance prediction of gearbox failures.

Obtain the dataset from its original source and place it at the path expected by the preprocessing scripts. Follow the dataset provider's license and redistribution requirements.

## Project Structure

```text
wind-turbine-predictive-maintenance/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── lof_anomaly_detector.joblib
├── src/
│   ├── validate_data.py
│   ├── check_features.py
│   ├── prepare_dataset.py
│   ├── train_anomaly.py
│   ├── save_lof_model.py
│   ├── predict.py
│   ├── evaluate_final_model.py
│   ├── evaluate_events.py
│   └── dashboard.py
├── tests/
│   ├── test_dataset.py
│   └── test_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

Additional analysis scripts and result files may be included where they are useful and reproducible.

## Requirements

- Python 3
- pip
- Git
- The Python packages listed in `requirements.txt`

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd wind-turbine-predictive-maintenance
```

Create and activate a virtual environment.

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Ensure that the dataset and saved model files are available at the paths expected by the application.

## Running the Project

### 1. Validate the dataset

```bash
python src/validate_data.py
```

### 2. Prepare the processed dataset

```bash
python src/prepare_dataset.py
```

Run preprocessing scripts in the order required by their inputs. Existing processed files may be needed by later stages.

### 3. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard provides sensor visualizations, anomaly predictions, date filtering, an option to display anomaly alerts only, and CSV export functionality.

### 4. Run automated tests

```bash
pytest -q
```

The project includes six automated tests covering dataset validity and basic model functionality.

### 5. Run evaluation scripts

The repository includes scripts for model evaluation and event-level analysis. Run them after confirming that their required datasets and model artifacts are available.

```bash
python src/evaluate_final_model.py
python src/evaluate_events.py
```

## Model

The primary model is Local Outlier Factor (LOF), an unsupervised anomaly detection algorithm that identifies observations whose local density differs from that of their neighbours.

The saved model artifact is expected at:

```text
models/lof_anomaly_detector.joblib
```

The artifact stores the fitted model, expected feature names, and decision threshold.

## Results and Evaluation

The recorded full-dataset evaluation reported approximately:

- Precision: 99.6%
- Recall: 98.9%
- ROC-AUC: 0.9999

These figures are results from the available dataset, not independently verified generalization performance. The evaluation procedure and data overlap must be considered when interpreting them.

The dataset contains 5,500 labelled anomalous records concentrated in three periods. Independent testing on unseen time periods and turbines, using verified failure outcomes, is required before drawing conclusions about real-world predictive maintenance.

## Limitations

- No turbine identifier is available for cross-turbine evaluation.
- Verified failure timestamps and maintenance records are unavailable.
- Anomaly detection does not automatically imply advance failure prediction.
- The current implementation is a prototype, not a production monitoring service.
- Performance on unseen turbines and real operating conditions has not been established.

## Future Work

- Validate on independent operational datasets.
- Evaluate performance across multiple turbines.
- Incorporate confirmed failure and maintenance records.
- Improve alert severity, explainability, and alert management.
- Integrate live sensor streams after validating model reliability.

## Technologies Used

- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- pytest
- joblib

## License

Add a suitable license for the project before permitting reuse. Check the original dataset's license separately; the project license does not override the dataset provider's terms.

## Author

Developed as an academic machine learning project on wind turbine gearbox anomaly detection and predictive maintenance.

