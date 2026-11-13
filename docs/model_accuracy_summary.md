| Model / Paradigm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Notes / Error Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Classical ML (Isolation Forest / LOF)** | 98.01% | 51.30% | 100.00% | 0.6780 | 1.0000 | High recall, but produces ~5,200 false alarms on normal data. |
| **Deep Learning (PyTorch LSTM Autoencoder)** | 99.99% | 99.98% | 100.00% | 0.9999 | 1.0000 | Reconstruction MSE has massive margin: Normal (~0.03) vs Anomaly (~79.4). |
| **Quantum ML (PennyLane 4-Qubit VQC)** | 78.71% | 93.69% | 49.11% | 0.6444 | 0.7716 | Conservative validator: high precision (93.7%), lower recall. |
| **Ensemble Majority Consensus ($\ge 2/3$)** | 80.01% | 100.00% | 49.11% | 0.6587 | — | Strict consensus eliminates 100% of false alarms for industrial safety. |
