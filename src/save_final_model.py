
import joblib

bundle = joblib.load("models/isolation_forest.joblib")

bundle["threshold"] = 0.12737

joblib.dump(bundle, "models/final_anomaly_detector.joblib")

print("Final anomaly detector saved successfully.")
print("Threshold:", bundle["threshold"])
print("Model: models/final_anomaly_detector.joblib")

