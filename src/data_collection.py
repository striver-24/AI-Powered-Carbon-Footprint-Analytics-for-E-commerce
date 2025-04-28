import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from typing import Dict, List

class CarbonDataCollector:
    def __init__(self):
        self.shipping_data = None
        self.carbon_data = None
    
    def load_shipping_data(self, file_path: str) -> None:
        """Load e-commerce shipping data from CSV"""
        try:
            self.shipping_data = pd.read_csv(file_path)
            print(f"Loaded shipping data with {len(self.shipping_data)} records")
        except Exception as e:
            print(f"Error loading shipping data: {e}")
    
    def load_emissions_data(file_path):
        """Load emissions data from XLSX files with improved error handling"""
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            
            if 'DB1 - Vehicle Type' in df.columns:
                # Vehicle emissions data
                df = df.rename(columns={
                    'DB1 - Vehicle Type': 'vehicle_type',
                    'Fuel Type': 'fuel_type',
                    'Total CO₂e (kg/gallon)': 'co2e_per_gallon'
                })
                return df[['vehicle_type', 'fuel_type', 'co2e_per_gallon']]
            elif 'Scope' in df.columns:
                # Scope emissions data
                return df[['ID', 'Scope', 'Level 1', 'Level 2', 'Level 3', 
                         'Column Text', 'UOM', 'GHG/Unit', 'GHG Conversion Factor 2024']]
            else:
                # Generic fallback for other Excel formats
                return df
        except Exception as e:
            print(f"Error loading Excel file {file_path}: {str(e)}")
            return None
    
    def generate_synthetic_data(self, num_records: int = 1000) -> pd.DataFrame:
        """Generate synthetic e-commerce shipping data with more realistic distributions"""
        dates = pd.date_range(start='2023-01-01', periods=num_records)
        
        shipping_methods = ['Ground', 'Air', 'Sea']
        packaging_types = ['Minimal', 'Standard', 'Premium']
        vehicle_types = ['Van', 'Truck', 'Drone']
        
        data = {
            'date': dates,
            'distance_km': np.random.gamma(shape=2, scale=250, size=num_records),  # More realistic distance distribution
            'weight_kg': np.random.lognormal(mean=0.5, sigma=0.5, size=num_records),  # Log-normal for weights
            'shipping_method': np.random.choice(shipping_methods, num_records, p=[0.6, 0.1, 0.3]),  # Weighted probabilities
            'packaging_type': np.random.choice(packaging_types, num_records, p=[0.3, 0.5, 0.2]),
            'vehicle_type': np.random.choice(vehicle_types, num_records, p=[0.7, 0.2, 0.1])
        }
        
        # More realistic carbon footprint calculation
        data['carbon_footprint'] = (
            data['distance_km'] * 0.1 * (data['shipping_method'] == 'Ground') +
            data['distance_km'] * 0.5 * (data['shipping_method'] == 'Air') +
            data['distance_km'] * 0.3 * (data['shipping_method'] == 'Sea') +
            data['weight_kg'] * 0.2 +
            (data['packaging_type'] == 'Premium') * 5 +
            (data['vehicle_type'] == 'Truck') * 10
        )
        
        self.shipping_data = pd.DataFrame(data)
        return self.shipping_data
    
    def fetch_carbon_intensity(self, country: str = 'US') -> Dict:
        """Fetch carbon intensity data from an API (placeholder)"""
        # This would typically use a real API
        # For now, return synthetic data
        return {
            'timestamp': datetime.now().isoformat(),
            'country': country,
            'carbon_intensity': 250.0  # gCO2/kWh
        }
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the collected data"""
        if self.shipping_data is None:
            return {"error": "No data loaded"}
        
        return {
            "total_shipments": len(self.shipping_data),
            "total_carbon_footprint": self.shipping_data['carbon_footprint'].sum(),
            "avg_carbon_per_shipment": self.shipping_data['carbon_footprint'].mean(),
            "shipping_method_distribution": self.shipping_data['shipping_method'].value_counts().to_dict()
        }
    
    class DataCollector:
        def __init__(self):
            self.shipping_data = None
            
        def load_data(self, file_path=None):
            """Load data from specified file or default data directory"""
            if file_path:
                if file_path.endswith('.csv'):
                    self.shipping_data = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    self.shipping_data = pd.read_excel(file_path)
            else:
                # Default data directory path
                data_dir = os.path.join(os.path.dirname(__file__), '../../data')
                if os.path.exists(data_dir):
                    for file in os.listdir(data_dir):
                        if file.endswith('.csv'):
                            self.shipping_data = pd.read_csv(os.path.join(data_dir, file))
                            break
                        elif file.endswith(('.xlsx', '.xls')):
                            self.shipping_data = pd.read_excel(os.path.join(data_dir, file))
                            break
            return self.shipping_data is not None