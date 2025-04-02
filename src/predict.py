# src/3_predict.py
import os
import numpy as np
import onnxruntime as ort
import pickle
from pathlib import Path

class DemandPredictor:
    def __init__(self, model_path, scaler_path=None):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.session = None
        self.scaler = None
        self.input_name = None
        
        self._initialize()

    def _initialize(self):
        """Initialize model and scaler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Initialize ONNX runtime session
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Load scaler if provided
        if self.scaler_path:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def predict(self, input_data):
        """Make prediction from prepared input"""
        # Ensure input is 2D array
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, axis=0)
            
        # Run inference
        onnx_input = {self.input_name: input_data.astype(np.float32)}
        pred = self.session.run(None, onnx_input)[0]
        
        # Inverse transform if scaler available
        if self.scaler is not None:
            pred = self.scaler.inverse_transform(pred)
            
        return pred.flatten()

def main():
    try:
        # Get absolute paths
        base_dir = Path(__file__).parent.parent
        model_path = os.path.join(base_dir, "models", "neurogrid.onnx")
        scaler_path = os.path.join(base_dir, "data", "scaler.pkl")
        test_sample_path = os.path.join(base_dir, "data", "test_sample.pkl")
        
        # Initialize predictor
        predictor = DemandPredictor(
            model_path=model_path,
            scaler_path=scaler_path
        )
        
        # Load test sample
        with open(test_sample_path, 'rb') as f:
            test_data = pickle.load(f)
        
        # Make prediction
        prediction = predictor.predict(test_data['X_test_sample'])
        
        print("\nPrediction Results:")
        print(f"- Input shape: {test_data['X_test_sample'].shape}")
        print(f"- Raw prediction: {prediction}")
        print(f"- Actual value: {test_data['y_test_sample'].flatten()}")
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")

if __name__ == "__main__":
    print("Neurogrid Demand Prediction System")
    main()