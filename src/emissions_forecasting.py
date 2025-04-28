import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EmissionsForecaster:
    """Class for forecasting future carbon emissions using time-series models"""
    def __init__(self):
        self.data = None
        self.prophet_model = None
        self.arima_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        self.forecast_results = {}
    
    def set_data(self, data: pd.DataFrame, date_column: str = 'order_date', target_column: str = 'total_emissions_kg'):
        """Set the data for forecasting"""
        if date_column not in data.columns or target_column not in data.columns:
            raise ValueError(f"Data must contain {date_column} and {target_column} columns")
        
        # Ensure data is sorted by date
        self.data = data.sort_values(by=date_column).copy()
        self.date_column = date_column
        self.target_column = target_column
        
        # Aggregate data by date if there are multiple entries per date
        self.daily_data = self.data.groupby(date_column)[target_column].sum().reset_index()
        self.daily_data.columns = ['ds', 'y']  # Prophet format
    
    def train_prophet_model(self, seasonality_mode: str = 'multiplicative', 
                           yearly_seasonality: bool = True, 
                           weekly_seasonality: bool = True,
                           daily_seasonality: bool = False):
        """Train a Facebook Prophet model for forecasting"""
        if self.daily_data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        self.prophet_model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        # Add additional regressors if available
        if 'is_holiday' in self.data.columns:
            self.prophet_model.add_regressor('is_holiday')
        
        if 'promotion_active' in self.data.columns:
            self.prophet_model.add_regressor('promotion_active')
        
        # Fit the model
        self.prophet_model.fit(self.daily_data)
        print("Prophet model trained successfully")
    
    def forecast_prophet(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """Generate forecasts using the Prophet model"""
        if self.prophet_model is None:
            raise ValueError("Prophet model not trained. Call train_prophet_model() first.")
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=periods, freq=freq)
        
        # Add additional regressors to future if they were used in training
        for regressor in self.prophet_model.extra_regressors:
            if regressor['name'] == 'is_holiday':
                future['is_holiday'] = 0  # Default value, update with actual holidays
            
            if regressor['name'] == 'promotion_active':
                future['promotion_active'] = 0  # Default value, update with planned promotions
        
        # Generate forecast
        forecast = self.prophet_model.predict(future)
        
        # Store results
        self.forecast_results['prophet'] = forecast
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def train_arima_model(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """Train an ARIMA model for forecasting"""
        if self.daily_data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Prepare time series data
        ts_data = self.daily_data.set_index('ds')['y']
        
        # Fit ARIMA model
        self.arima_model = ARIMA(ts_data, order=order)
        self.arima_fit = self.arima_model.fit()
        print("ARIMA model trained successfully")
    
    def forecast_arima(self, periods: int = 30) -> pd.DataFrame:
        """Generate forecasts using the ARIMA model"""
        if self.arima_fit is None:
            raise ValueError("ARIMA model not trained. Call train_arima_model() first.")
        
        # Generate forecast
        forecast = self.arima_fit.forecast(steps=periods)
        
        # Create forecast dataframe
        last_date = self.daily_data['ds'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast,
            'model': 'ARIMA'
        })
        
        # Store results
        self.forecast_results['arima'] = forecast_df
        
        return forecast_df
    
    def prepare_lstm_data(self, sequence_length: int = 10):
        """Prepare data for LSTM model"""
        if self.daily_data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Scale the data
        values = self.daily_data['y'].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_values) - sequence_length):
            X.append(scaled_values[i:i+sequence_length, 0])
            y.append(scaled_values[i+sequence_length, 0])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train_lstm_model(self, sequence_length: int = 10, epochs: int = 50, batch_size: int = 32):
        """Train an LSTM model for forecasting"""
        if self.daily_data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Prepare data
        X, y = self.prepare_lstm_data(sequence_length)
        
        # Build LSTM model
        self.lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Compile model
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        # Train model
        self.lstm_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        print("LSTM model trained successfully")
        
        # Store sequence length for forecasting
        self.sequence_length = sequence_length
    
    def forecast_lstm(self, periods: int = 30) -> pd.DataFrame:
        """Generate forecasts using the LSTM model"""
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained. Call train_lstm_model() first.")
        
        # Get the last sequence from the data
        values = self.daily_data['y'].values.reshape(-1, 1)
        scaled_values = self.scaler.transform(values)
        last_sequence = scaled_values[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        # Generate forecasts iteratively
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            # Predict next value
            next_value = self.lstm_model.predict(current_sequence)[0, 0]
            forecasts.append(next_value)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_value
        
        # Inverse transform to get actual values
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts)
        
        # Create forecast dataframe
        last_date = self.daily_data['ds'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecasts.flatten(),
            'model': 'LSTM'
        })
        
        # Store results
        self.forecast_results['lstm'] = forecast_df
        
        return forecast_df
    
    def compare_models(self, test_size: int = 30) -> Dict:
        """Compare different forecasting models using holdout data"""
        if self.daily_data is None or len(self.daily_data) <= test_size:
            raise ValueError("Insufficient data for comparison")
        
        # Split data into train and test
        train_data = self.daily_data.iloc[:-test_size]
        test_data = self.daily_data.iloc[-test_size:]
        
        results = {}
        
        # Train and evaluate Prophet
        if self.prophet_model is None:
            temp_prophet = Prophet()
            temp_prophet.fit(train_data)
            prophet_forecast = temp_prophet.predict(pd.DataFrame({'ds': test_data['ds']}))
            prophet_mse = np.mean((prophet_forecast['yhat'].values - test_data['y'].values) ** 2)
            prophet_rmse = np.sqrt(prophet_mse)
            results['prophet'] = {'mse': prophet_mse, 'rmse': prophet_rmse}
        
        # Train and evaluate ARIMA
        try:
            ts_train = train_data.set_index('ds')['y']
            temp_arima = ARIMA(ts_train, order=(5, 1, 0))
            temp_arima_fit = temp_arima.fit()
            arima_forecast = temp_arima_fit.forecast(steps=test_size)
            arima_mse = np.mean((arima_forecast - test_data['y'].values) ** 2)
            arima_rmse = np.sqrt(arima_mse)
            results['arima'] = {'mse': arima_mse, 'rmse': arima_rmse}
        except Exception as e:
            print(f"ARIMA evaluation failed: {e}")
        
        # Train and evaluate LSTM (simplified for comparison)
        try:
            sequence_length = 10
            if len(train_data) > sequence_length + 10:  # Ensure enough data
                # Scale data
                scaler = MinMaxScaler()
                train_values = train_data['y'].values.reshape(-1, 1)
                scaled_train = scaler.fit_transform(train_values)
                
                # Prepare sequences
                X, y = [], []
                for i in range(len(scaled_train) - sequence_length):
                    X.append(scaled_train[i:i+sequence_length, 0])
                    y.append(scaled_train[i+sequence_length, 0])
                
                X = np.array(X).reshape((-1, sequence_length, 1))
                y = np.array(y)
                
                # Simple LSTM model
                temp_lstm = Sequential([
                    LSTM(20, input_shape=(sequence_length, 1)),
                    Dense(1)
                ])
                temp_lstm.compile(optimizer='adam', loss='mse')
                temp_lstm.fit(X, y, epochs=10, batch_size=16, verbose=0)
                
                # Generate forecasts
                lstm_forecasts = []
                last_sequence = scaled_train[-sequence_length:].reshape(1, sequence_length, 1)
                
                for _ in range(test_size):
                    next_pred = temp_lstm.predict(last_sequence, verbose=0)[0, 0]
                    lstm_forecasts.append(next_pred)
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1, 0] = next_pred
                
                # Inverse transform
                lstm_forecasts = np.array(lstm_forecasts).reshape(-1, 1)
                lstm_forecasts = scaler.inverse_transform(lstm_forecasts)
                
                # Calculate metrics
                lstm_mse = np.mean((lstm_forecasts.flatten() - test_data['y'].values) ** 2)
                lstm_rmse = np.sqrt(lstm_mse)
                results['lstm'] = {'mse': lstm_mse, 'rmse': lstm_rmse}
        except Exception as e:
            print(f"LSTM evaluation failed: {e}")
        
        return results
    
    def plot_forecasts(self, include_history: bool = True):
        """Plot forecasts from different models"""
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        if include_history and self.daily_data is not None:
            plt.plot(self.daily_data['ds'], self.daily_data['y'], label='Historical Data', color='black')
        
        # Plot Prophet forecast
        if 'prophet' in self.forecast_results:
            forecast = self.forecast_results['prophet']
            if include_history:
                # Only plot future points
                last_date = self.daily_data['ds'].max()
                future_forecast = forecast[forecast['ds'] > last_date]
                plt.plot(future_forecast['ds'], future_forecast['yhat'], label='Prophet Forecast', color='blue')
                plt.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], 
                                color='blue', alpha=0.2)
            else:
                plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast', color='blue')
                plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                                color='blue', alpha=0.2)
        
        # Plot ARIMA forecast
        if 'arima' in self.forecast_results:
            forecast = self.forecast_results['arima']
            plt.plot(forecast['ds'], forecast['yhat'], label='ARIMA Forecast', color='red')
        
        # Plot LSTM forecast
        if 'lstm' in self.forecast_results:
            forecast = self.forecast_results['lstm']
            plt.plot(forecast['ds'], forecast['yhat'], label='LSTM Forecast', color='green')
        
        plt.title('Carbon Emissions Forecast')
        plt.xlabel('Date')
        plt.ylabel(f'Emissions ({self.target_column})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('emissions_forecast.png')
        plt.close()
    
    def get_seasonal_patterns(self):
        """Analyze seasonal patterns in emissions data"""
        if self.daily_data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Convert to datetime if not already
        self.daily_data['ds'] = pd.to_datetime(self.daily_data['ds'])
        
        # Extract time components
        emissions_by_time = self.daily_data.copy()
        emissions_by_time['year'] = emissions_by_time['ds'].dt.year
        emissions_by_time['month'] = emissions_by_time['ds'].dt.month
        emissions_by_time['day_of_week'] = emissions_by_time['ds'].dt.dayofweek
        emissions_by_time['quarter'] = emissions_by_time['ds'].dt.quarter
        
        # Analyze patterns
        monthly_pattern = emissions_by_time.groupby('month')['y'].mean()
        day_of_week_pattern = emissions_by_time.groupby('day_of_week')['y'].mean()
        quarterly_pattern = emissions_by_time.groupby('quarter')['y'].mean()
        
        # Map day of week to names
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                    4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        day_of_week_pattern.index = [day_names[i] for i in day_of_week_pattern.index]
        
        # Map month to names
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        monthly_pattern.index = [month_names[i] for i in monthly_pattern.index]
        
        return {
            'monthly_pattern': monthly_pattern.to_dict(),
            'day_of_week_pattern': day_of_week_pattern.to_dict(),
            'quarterly_pattern': quarterly_pattern.to_dict()
        }

# Example usage
def main():
    from carbon_footprint_system import CarbonDataIntegrator, CarbonFootprintCalculator
    
    # Initialize the data integrator and load datasets
    data_integrator = CarbonDataIntegrator()
    data_integrator.load_all_datasets()
    
    # Generate synthetic e-commerce data
    ecommerce_data = data_integrator.generate_ecommerce_data(num_records=1000)
    print(f"Generated {len(ecommerce_data)} synthetic e-commerce records")
    
    # Calculate emissions
    calculator = CarbonFootprintCalculator(data_integrator)
    total_emissions = calculator.calculate_total_emissions()
    
    # Initialize forecaster
    forecaster = EmissionsForecaster()
    forecaster.set_data(total_emissions, date_column='order_date', target_column='total_emissions_kg')
    
    # Train and forecast with Prophet
    forecaster.train_prophet_model()
    prophet_forecast = forecaster.forecast_prophet(periods=90)
    print("\nProphet Forecast (next 90 days):")
    print(prophet_forecast.head())
    
    # Train and forecast with LSTM
    forecaster.train_lstm_model(epochs=20)
    lstm_forecast = forecaster.forecast_lstm(periods=90)
    print("\nLSTM Forecast (next 90 days):")
    print(lstm_forecast.head())
    
    # Plot forecasts
    forecaster.plot_forecasts()
    print("\nForecasts plotted and saved to 'emissions_forecast.png'")
    
    # Analyze seasonal patterns
    seasonal_patterns = forecaster.get_seasonal_patterns()
    print("\nSeasonal Patterns:")
    print(f"Monthly: Highest emissions in {max(seasonal_patterns['monthly_pattern'], key=seasonal_patterns['monthly_pattern'].get)}")
    print(f"Day of Week: Highest emissions on {max(seasonal_patterns['day_of_week_pattern'], key=seasonal_patterns['day_of_week_pattern'].get)}")

if __name__ == "__main__":
    main()