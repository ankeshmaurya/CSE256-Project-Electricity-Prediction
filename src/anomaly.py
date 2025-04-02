# src/4_anomaly.py
import numpy as np
from pyod.models.auto_encoder import AutoEncoder
import os

def detect_anomalies(data_path, contamination=0.01):
    try:
        data = np.load(data_path)
        X = data['X']
        
        clf = AutoEncoder(contamination=contamination)
        clf.fit(X)
        return clf.decision_scores_
    
    except Exception as e:
        print(f"Anomaly detection failed: {str(e)}")
        raise

if __name__ == "__main__":
    scores = detect_anomalies(os.path.join("..", "data", "processed_data.npz"))
    print(f"Anomaly Scores (last 10): {scores[-10:]}")