
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/training_data.csv"
LOF_MAX_TRAIN_ROWS = 20000
MIN_RECALL = 0.85
RANDOM_STATE = 42


def choose_threshold(y_true, scores):
    """Choose a threshold using validation data only."""
    candidates = np.unique(
        np.quantile(scores, np.linspace(0, 1, 201))
    )

    best = None

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        if recall >= MIN_RECALL:
            result = (precision, f1, threshold, recall)
            if best is None or result[0] > best[0]:
                best = result

    if best is None:
        raise RuntimeError(
            "Could not find a validation threshold with "
            "at least 85% recall."
        )

    return best[2]


def build_model(name, normal_training_data):
    if name == "Isolation Forest":
        model = make_pipeline(
            StandardScaler(),
            IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
        model.fit(normal_training_data)
        return model

    # LOF is computationally expensive on large datasets.
    # Fit it on a reproducible sample of normal training rows.
    if len(normal_training_data) > LOF_MAX_TRAIN_ROWS:
        normal_training_data = normal_training_data.sample(
            n=LOF_MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
        )

    model = make_pipeline(
        StandardScaler(),
        LocalOutlierFactor(
            n_neighbors=35,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        ),
    )
    model.fit(normal_training_data)
    return model


def anomaly_scores(model, X):
    # Larger scores should mean more anomalous.
    return -model.decision_function(X)


def evaluate_model(name, model, validation, test, features):
    X_val = validation[features]
    y_val = validation["is_anomaly"].astype(int).to_numpy()

    X_test = test[features]
    y_test = test["is_anomaly"].astype(int).to_numpy()

    validation_scores = anomaly_scores(model, X_val)
    threshold = choose_threshold(y_val, validation_scores)

    test_scores = anomaly_scores(model, X_test)
    predictions = (test_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    auc = (
        roc_auc_score(y_test, test_scores)
        if len(np.unique(y_test)) == 2
        else float("nan")
    )

    return {
        "model": name,
        "threshold": threshold,
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": auc,
        "false_alarms": int(fp),
        "missed_anomalies": int(fn),
        "actual_anomalies": int(y_test.sum()),
        "test_records": len(y_test),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Use numeric input features only, excluding the target label.
    features = [
        c for c in df.select_dtypes(include="number").columns
        if c != "is_anomaly"
    ]

    if not features:
        raise ValueError("No numeric sensor features were found.")

    n = len(df)
    print("DATASET")
    print("Records:", n)
    print("Features:", len(features))
    print("Date range:", df["timestamp"].min(), "to",
          df["timestamp"].max())
    print("Anomaly count:", int(df["is_anomaly"].sum()))
    print("\nLOF training is capped at",
          LOF_MAX_TRAIN_ROWS, "normal rows for runtime.")

    # Three expanding-window chronological folds.
    # Each fold uses an earlier training period, followed by
    # validation and then a later, untouched test period.
    folds = [
        (0.50, 0.60, 0.70),
        (0.60, 0.70, 0.80),
        (0.70, 0.80, 0.90),
    ]

    results = []

    for fold_number, (train_end, val_end, test_end) in enumerate(
        folds, start=1
    ):
        train = df.iloc[:int(n * train_end)]
        validation = df.iloc[int(n * train_end):int(n * val_end)]
        test = df.iloc[int(n * val_end):int(n * test_end)]

        normal_train = train.loc[
            train["is_anomaly"] == 0, features
        ]

        print(f"\n{'=' * 55}")
        print(f"FOLD {fold_number}")
        print("Training period:",
              train["timestamp"].min(), "to",
              train["timestamp"].max())
        print("Validation period:",
              validation["timestamp"].min(), "to",
              validation["timestamp"].max())
        print("Test period:",
              test["timestamp"].min(), "to",
              test["timestamp"].max())
        print("Normal training rows:", len(normal_train))
        print("Validation anomalies:",
              int(validation["is_anomaly"].sum()))
        print("Test anomalies:", int(test["is_anomaly"].sum()))

        if validation["is_anomaly"].nunique() < 2:
            print("SKIPPED: validation needs both classes.")
            continue

        if test["is_anomaly"].nunique() < 2:
            print("WARNING: test ROC-AUC cannot be calculated "
                  "unless both classes occur.")

        X_normal = normal_train
        if len(X_normal) < 100:
            print("SKIPPED: too few normal training records.")
            continue

        for name in ["Isolation Forest", "Local Outlier Factor"]:
            print(f"\nTraining {name}...")
            model = build_model(name, X_normal)

            result = evaluate_model(
                name, model, validation, test, features
            )
            result["fold"] = fold_number
            results.append(result)

            print("Validation-selected threshold:",
                  round(result["threshold"], 6))
            print("Test precision:", round(result["precision"], 4))
            print("Test recall:", round(result["recall"], 4))
            print("Test F1:", round(result["f1"], 4))
            print("Test ROC-AUC:", round(result["roc_auc"], 4))
            print("False alarms:", result["false_alarms"])
            print("Missed anomalies:", result["missed_anomalies"])

    if not results:
        raise RuntimeError("No folds completed successfully.")

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        "data/processed/fold_evaluation_results.csv",
        index=False,
    )

    print("\n\nSUMMARY: INDIVIDUAL FOLDS")
    print(
        result_df[
            ["fold", "model", "precision", "recall", "f1",
             "roc_auc", "false_alarms", "missed_anomalies"]
        ].round(4).to_string(index=False)
    )

    print("\nSUMMARY: MEAN ACROSS FOLDS")
    summary = result_df.groupby("model")[
        ["precision", "recall", "f1", "roc_auc",
         "false_alarms", "missed_anomalies"]
    ].mean()
    print(summary.round(4).to_string())

    print("\nSaved results to "
          "data/processed/fold_evaluation_results.csv")


if __name__ == "__main__":
    main()

