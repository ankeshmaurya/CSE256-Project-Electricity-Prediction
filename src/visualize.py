# src/visualize.py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

class VisualizationDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Neurogrid Visualization Dashboard")
        self.root.geometry("1200x800")
        
        # Initialize data attributes
        self.df = pd.DataFrame()
        self.test_data = {}
        self.scaler = None
        self.anomaly_scores = np.array([])
        
        self.setup_ui()
        self.load_data()
        self.update_plots()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_tab("Training Data", 2)
        self.create_tab("Predictions", 1)
        self.create_tab("Anomalies", 1)

    def create_tab(self, name, rows):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=name)
        
        fig, axes = plt.subplots(rows, 1, figsize=(10, 6 if rows == 1 else 8))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store references
        setattr(self, f"{name.lower().replace(' ', '_')}_fig", fig)
        setattr(self, f"{name.lower().replace(' ', '_')}_axes", axes)
        setattr(self, f"{name.lower().replace(' ', '_')}_canvas", canvas)

    def load_data(self):
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data"
        logs_dir = base_dir / "logs"
        
        # 1. Load training data
        try:
            train_path = data_dir / "processed_data.csv"
            if train_path.exists():
                self.df = pd.read_csv(train_path, parse_dates=['Datetime'])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load training data: {str(e)}")

        # 2. Load prediction data
        try:
            test_path = data_dir / "test_sample.pkl"
            scaler_path = data_dir / "scaler.pkl"
            if test_path.exists() and scaler_path.exists():
                with open(test_path, 'rb') as f:
                    self.test_data = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load prediction data: {str(e)}")

        # 3. Load anomaly data
        try:
            anomaly_path = logs_dir / "anomaly_scores.npy"
            if anomaly_path.exists():
                self.anomaly_scores = np.load(anomaly_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load anomaly data: {str(e)}")

    def update_plots(self):
        # Training Data Tab
        if not self.df.empty:
            ax1, ax2 = self.training_data_axes
            ax1.clear()
            ax1.plot(self.df['Datetime'], self.df['MW'])
            ax1.set_title('Energy Demand')
            
            ax2.clear()
            ax2.plot(self.df['Datetime'], self.df['TEMP'], 'r-')
            ax2.plot(self.df['Datetime'], self.df['WDSP'], 'g-')
            ax2.set_title('Weather Data')
            self.training_data_canvas.draw()
        else:
            self.show_placeholder(self.training_data_axes[0], "Run generate_data.py and preprocess.py first")

        # Prediction Tab
        if self.test_data and self.scaler:
            ax = self.predictions_axes
            ax.clear()
            actual = self.scaler.inverse_transform(self.test_data['y_test_sample'])
            predicted = self.scaler.inverse_transform(self.test_data['X_test_sample'])
            ax.bar(['Actual', 'Predicted'], [actual[0][0], predicted[0][0]])
            ax.set_title('Prediction Results')
            self.predictions_canvas.draw()
        else:
            self.show_placeholder(self.predictions_axes, "Run train.py and predict.py first")

        # Anomaly Tab
        if len(self.anomaly_scores) > 0:
            ax = self.anomalies_axes
            ax.clear()
            ax.plot(self.df['Datetime'], self.df['MW'])
            threshold = np.percentile(self.anomaly_scores, 99)
            anomalies = self.df.iloc[np.where(self.anomaly_scores > threshold)]
            ax.scatter(anomalies['Datetime'], anomalies['MW'], color='red')
            ax.set_title('Anomaly Detection')
            self.anomalies_canvas.draw()
        else:
            self.show_placeholder(self.anomalies_axes, "Run anomaly.py first")

    def show_placeholder(self, ax, message):
        ax.clear()
        ax.text(0.5, 0.5, message, 
               ha='center', va='center', 
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

def main():
    try:
        root = tk.Tk()
        VisualizationDashboard(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", str(e))
        raise

if __name__ == "__main__":
    main()