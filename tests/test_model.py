
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

def test_saved_model_exists():
    assert Path("models/lof_anomaly_detector.joblib").exists()


def test_saved_model_can_predict():
    model_path = Path("models/lof_anomaly_detector.joblib")
    data_path = Path("data/processed/training_data.csv")

    bundle = joblib.load(model_path)
    df = pd.read_csv(data_path)

    features = bundle["features"]
    model = bundle["model"]

    sample = df[features].head(10)

    assert not sample.isnull().any().any()

    scores = -model.decision_function(sample)

    assert len(scores) == 10
    assert np.isfinite(scores).all()
