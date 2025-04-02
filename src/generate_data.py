# src/0_generate_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import os

def generate_pjm_data(years=3):
    """Generate synthetic PJM energy demand data"""
    # Define the output directory
    base_dir = r"C:\Users\Ankesh Maurya\Desktop\Programming\Neurogrid_Project"
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    dates = pd.date_range(start="2020-01-01", periods=365*24*years, freq='h')
    base_load = 15000  # MW
    seasonal = 3000 * np.sin(2*np.pi*(dates.dayofyear/365))
    daily = 2000 * np.sin(2*np.pi*(dates.hour/24))
    noise = np.random.normal(0, 500, len(dates))
    demand = base_load + seasonal + daily + noise
    
    df = pd.DataFrame({
        'Datetime': dates,
        'MW': np.abs(demand),  # Ensure no negative values
        'Region': 'PJM_Load'
    })
    
    # Save to absolute path
    output_path = os.path.join(data_dir, "synthetic_pjm.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated PJM data -> {output_path}")

def generate_weather_data(years=3):
    """Generate synthetic weather data"""
    # Define the output directory
    base_dir = r"C:\Users\Ankesh Maurya\Desktop\Programming\Neurogrid_Project"
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    dates = pd.date_range(start="2020-01-01", periods=365*24*years, freq='h')
    temp = 10 + 15 * np.sin(2*np.pi*(dates.dayofyear/365)) + np.random.normal(0, 3, len(dates))
    wind = 5 + 10 * np.random.rand(len(dates))
    
    df = pd.DataFrame({
        'DATE': dates,
        'TEMP': temp,  # °C
        'WDSP': wind   # m/s
    })
    
    # Save to absolute path
    output_path = os.path.join(data_dir, "synthetic_weather.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated weather data -> {output_path}")

if __name__ == "__main__":
    generate_pjm_data()
    generate_weather_data()