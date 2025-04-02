# src/1_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

def preprocess_data():
    try:
        # Define base data directory
        base_dir = r"C:\Users\Ankesh Maurya\Desktop\Programming\Neurogrid_Project"
        data_dir = os.path.join(base_dir, "data")
        
        # Verify input files exist
        pjm_path = os.path.join(data_dir, "synthetic_pjm.csv")
        weather_path = os.path.join(data_dir, "synthetic_weather.csv")
        
        if not os.path.exists(pjm_path):
            raise FileNotFoundError(f"PJM data file not found at {pjm_path}")
        if not os.path.exists(weather_path):
            raise FileNotFoundError(f"Weather data file not found at {weather_path}")
        
        # Load data using absolute paths
        pjm = pd.read_csv(pjm_path, parse_dates=['Datetime'])
        weather = pd.read_csv(weather_path, parse_dates=['DATE'])
        
        # Merge datasets
        merged = pd.merge_asof(
            pjm.sort_values('Datetime'),
            weather.sort_values('DATE'),
            left_on='Datetime',
            right_on='DATE',
            direction='nearest'
        )
        
        # Feature engineering
        merged['hour'] = merged['Datetime'].dt.hour
        merged['day_of_week'] = merged['Datetime'].dt.dayofweek
        merged['month'] = merged['Datetime'].dt.month
        
        # Scaling - Only normalize MW (demand) for the target variable
        scaler = MinMaxScaler()
        merged['demand_norm'] = scaler.fit_transform(merged[['MW']])
        
        # Save scaler for inverse transform later
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save processed data to specific path
        output_path = os.path.join(data_dir, "processed_data.csv")
        merged.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        
        # Prepare data for anomaly detection (scale all features)
        features = merged[['MW', 'TEMP', 'WDSP', 'hour', 'day_of_week', 'month']]
        feature_scaler = MinMaxScaler()
        scaled_features = feature_scaler.fit_transform(features)
        np.savez(
            os.path.join(data_dir, "processed_data.npz"),
            X=scaled_features
        )
        
        print("Preprocessing completed successfully")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data()