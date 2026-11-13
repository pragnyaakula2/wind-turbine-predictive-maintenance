
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/training_data.csv")


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_required_columns_exist():
    df = pd.read_csv(DATA_PATH, nrows=5)

    required_columns = [
        "timestamp",
        "gearbox_oil_temp",
        "gearbox_bearing_temp",
        "vibration_x",
        "vibration_y",
        "vibration_z",
        "oil_pressure",
        "particle_count",
        "is_anomaly",
    ]

    for column in required_columns:
        assert column in df.columns, f"Missing column: {column}"


def test_anomaly_labels_are_binary():
    df = pd.read_csv(DATA_PATH, usecols=["is_anomaly"])

    assert set(df["is_anomaly"].dropna().unique()).issubset({0, 1})


def test_dataset_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)

    assert not df.isnull().any().any()

