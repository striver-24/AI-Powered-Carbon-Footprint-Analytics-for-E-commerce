import pandas as pd
import numpy as np
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

class CarbonFootprintPredictor:
    def __init__(self):
        self.prophet_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.target_scaler = None

    def prepare_prophet_data(self, data):
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        # Update column names to match current data structure
        prophet_df = df[['order_date', 'total_emissions_kg']].copy()
        prophet_df.columns = ['ds', 'y']
        return prophet_df.dropna()
    
    def train_prophet_model(self, data: pd.DataFrame) -> None:
        """Train Facebook Prophet model for carbon footprint prediction"""
        prophet_data = self.prepare_prophet_data(data)
        
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        self.prophet_model.fit(prophet_data)
    
    def predict_prophet(self, periods: int = 30) -> pd.DataFrame:
        """Generate predictions using Prophet model"""
        if self.prophet_model is None:
            raise ValueError("Prophet model not trained")
        
        future_dates = self.prophet_model.make_future_dataframe(periods=periods)
        forecast = self.prophet_model.predict(future_dates)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def prepare_lstm_data(self, data, lookback=30):
        df = data.copy()
        # Use only target variable for sequence prediction
        target = df['total_emissions_kg'].values.reshape(-1, 1)
        
        # Create separate scaler for target variable
        self.target_scaler = StandardScaler()
        scaled_target = self.target_scaler.fit_transform(target)
        
        # Create time steps using target variable only
        X, y = [], []
        for i in range(len(scaled_target) - lookback - 1):
            X.append(scaled_target[i:(i+lookback)])
            y.append(scaled_target[i+lookback])
        
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, data, lookback=30, epochs=50):
        """Train LSTM model for time series forecasting"""
        X, y = self.prepare_lstm_data(data, lookback)
        
        self.lstm_model = Sequential([
            LSTM(64, activation='relu', input_shape=(lookback, 1)),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        self.lstm_model.fit(X, y, epochs=epochs, verbose=0)
    
    def predict_lstm(self, data: pd.DataFrame, lookback: int = 7, steps: int = 30) -> np.ndarray:
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained")
        
        # Get and scale last sequence using target scaler
        last_sequence = data['total_emissions_kg'].values[-lookback:]
        scaled_sequence = self.target_scaler.transform(last_sequence.reshape(-1, 1))
        
        predictions = []
        current_sequence = scaled_sequence.copy()
        
        for _ in range(steps):
            X = current_sequence.reshape(1, lookback, 1)
            pred = self.lstm_model.predict(X, verbose=0)
            predictions.append(pred[0, 0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        # Inverse transform using target scaler
        predictions = np.array(predictions).reshape(-1, 1)
        return self.target_scaler.inverse_transform(predictions).flatten()
    
    def get_optimization_recommendations(self, data):
        """Get optimization recommendations based on analysis"""
        return {
            'baseline_metrics': {
                'avg_footprint': data['total_emissions_kg'].mean(),
                'total_footprint': data['total_emissions_kg'].sum()
            },
            'recommendations': {
                'best_shipping_method': data.groupby('vehicle_type')['total_emissions_kg'].mean().idxmin(),
                'best_packaging_type': data.groupby('packaging_material')['total_emissions_kg'].mean().idxmin(),
                'potential_reduction_percentage': round(
                    (1 - data['total_emissions_kg'].min() / data['total_emissions_kg'].mean()) * 100, 2
                )
            }
        }