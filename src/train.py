# src/2_train.py
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from sklearn.model_selection import train_test_split
import pickle
import os
from pathlib import Path

def train_and_export_model():
    try:
        # Load and verify data
        data_path = os.path.join(Path(__file__).parent.parent, "data", "processed_data.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data not found at {data_path}")
            
        data = pd.read_csv(data_path)
        required_columns = ['demand_norm', 'TEMP', 'WDSP', 'hour', 'day_of_week', 'month']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Missing columns in data: {missing}")

        # Prepare features and target
        X = data[['TEMP', 'WDSP', 'hour', 'day_of_week', 'month']].values
        y = data['demand_norm'].values.reshape(-1, 1)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model with explicit input and output names
        input_layer = tf.keras.layers.Input(shape=(X.shape[1],), name="input_layer")
        hidden = tf.keras.layers.Dense(32, activation='relu')(input_layer)
        output = tf.keras.layers.Dense(1, name="output_layer")(hidden)
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
        
        # Create models directory if not exists
        models_dir = os.path.join(Path(__file__).parent.parent, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model.save(os.path.join(models_dir, "neurogrid.keras"))
        
        # Convert to ONNX with explicit output names
        input_signature = [tf.TensorSpec(shape=[None, X.shape[1]], dtype=tf.float32, name="input")]
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            output_path=os.path.join(models_dir, "neurogrid.onnx"),
            opset=13)
        
        # Save test sample
        data_dir = os.path.join(Path(__file__).parent.parent, "data")
        sample_data = {
            'X_test_sample': X_test[:1],
            'y_test_sample': y_test[:1]
        }
        with open(os.path.join(data_dir, "test_sample.pkl"), 'wb') as f:
            pickle.dump(sample_data, f)
            
        print("Model training and export completed successfully")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_export_model()