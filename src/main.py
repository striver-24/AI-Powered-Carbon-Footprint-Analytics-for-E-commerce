import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Import all components
from carbon_footprint_system import CarbonDataIntegrator, CarbonFootprintCalculator, CarbonFootprintOptimizer
from emissions_forecasting import EmissionsForecaster
from data_collection import CarbonDataCollector
from data_analysis import CarbonFootprintAnalyzer
from ml_models import CarbonFootprintPredictor

class CarbonFootprintReductionSystem:
    """Main class that integrates all components of the carbon footprint reduction system"""
    def __init__(self):
        # Initialize all components
        self.data_integrator = CarbonDataIntegrator()
        self.collector = CarbonDataCollector()
        self.analyzer = CarbonFootprintAnalyzer()
        self.predictor = CarbonFootprintPredictor()
        
        # These components will be initialized after data is loaded
        self.calculator = None
        self.optimizer = None
        self.forecaster = None
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.results = {}
    
    def load_data(self, use_enhanced_datasets=True, num_records=1000, file_path=None):
        """Load data from enhanced datasets or from file"""
        if use_enhanced_datasets:
            print("Loading enhanced carbon datasets...")
            success = self.data_integrator.load_all_datasets()
            
            if success:
                print(f"Generating {num_records} synthetic e-commerce records...")
                self.raw_data = self.data_integrator.generate_ecommerce_data(num_records=num_records)
                return True
            else:
                print("Failed to load enhanced carbon datasets")
                return False
        elif file_path:
            try:
                if file_path.endswith('.csv'):
                    self.raw_data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    self.raw_data = pd.read_excel(file_path)
                else:
                    print(f"Unsupported file format: {file_path}")
                    return False
                
                print(f"Loaded data from {file_path} with {len(self.raw_data)} records")
                return True
            except Exception as e:
                print(f"Error loading data from file: {e}")
                return False
        else:
            print("Generating synthetic data...")
            self.raw_data = self.collector.generate_synthetic_data(num_records=num_records)
            return True
    
    def process_data(self):
        """Process the loaded data and calculate emissions"""
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return False
        
        try:
            # Initialize calculator with data integrator
            self.calculator = CarbonFootprintCalculator(self.data_integrator)
            
            # Calculate emissions
            print("Calculating packaging emissions...")
            packaging_emissions = self.calculator.calculate_packaging_emissions()
            
            print("Calculating transportation emissions...")
            transportation_emissions = self.calculator.calculate_transportation_emissions()
            
            print("Calculating waste emissions...")
            waste_emissions = self.calculator.calculate_waste_emissions()
            
            print("Calculating total emissions...")
            total_emissions = self.calculator.calculate_total_emissions()
            
            # Store results
            self.processed_data = total_emissions
            self.results['packaging_emissions'] = packaging_emissions
            self.results['transportation_emissions'] = transportation_emissions
            self.results['waste_emissions'] = waste_emissions
            self.results['total_emissions'] = total_emissions
            
            # Set data for analyzer
            self.analyzer.set_data(total_emissions)
            
            return True
        except Exception as e:
            print(f"Error processing data: {e}")
            return False
    
    def optimize_emissions(self):
        """Generate optimization recommendations"""
        if self.calculator is None or not self.calculator.results:
            print("No processed data available. Please process data first.")
            return False
        
        try:
            # Initialize optimizer
            print("Initializing optimizer...")
            self.optimizer = CarbonFootprintOptimizer(self.calculator)
            
            # Cluster products
            print("Clustering products by emissions profile...")
            self.optimizer.cluster_products()
            
            # Generate recommendations
            print("Generating comprehensive recommendations...")
            self.optimizer.generate_comprehensive_recommendations()
            
            # Store results
            self.results['optimization'] = self.optimizer.recommendations
            
            return True
        except Exception as e:
            print(f"Error optimizing emissions: {e}")
            return False
    
    def forecast_emissions(self, periods=90):
        """Forecast future emissions"""
        if self.processed_data is None:
            print("No processed data available. Please process data first.")
            return False
        
        try:
            # Initialize forecaster
            print("Initializing forecaster...")
            self.forecaster = EmissionsForecaster()
            
            # Set data
            self.forecaster.set_data(
                self.processed_data, 
                date_column='order_date', 
                target_column='total_emissions_kg'
            )
            
            # Train Prophet model
            print("Training Prophet model...")
            self.forecaster.train_prophet_model()
            
            # Generate forecast
            print(f"Forecasting emissions for {periods} days...")
            prophet_forecast = self.forecaster.forecast_prophet(periods=periods)
            
            # Train LSTM model if enough data
            if len(self.forecaster.daily_data) >= 30:  # Need enough data for LSTM
                print("Training LSTM model...")
                self.forecaster.train_lstm_model(epochs=20)
                
                # Generate LSTM forecast
                lstm_forecast = self.forecaster.forecast_lstm(periods=periods)
                self.results['lstm_forecast'] = lstm_forecast
            
            # Store results
            self.results['prophet_forecast'] = prophet_forecast
            
            return True
        except Exception as e:
            print(f"Error forecasting emissions: {e}")
            return False
    
    def run_full_analysis(self, use_enhanced_datasets=True, num_records=1000, file_path=None, forecast_periods=90):
        """Run the complete analysis pipeline"""
        print("Starting carbon footprint reduction analysis...")
        
        # Step 1: Load data
        if not self.load_data(use_enhanced_datasets, num_records, file_path):
            return False
        
        # Step 2: Process data
        if not self.process_data():
            return False
        
        # Step 3: Optimize emissions
        if not self.optimize_emissions():
            return False
        
        # Step 4: Forecast emissions
        if not self.forecast_emissions(periods=forecast_periods):
            return False
        
        print("Analysis complete!")
        return True
    
    def get_summary(self):
        """Get a summary of the analysis results"""
        if not self.results or 'total_emissions' not in self.results:
            return "No analysis results available. Please run analysis first."
        
        total_data = self.results['total_emissions']
        
        summary = {
            'total_orders': len(total_data),
            'total_emissions_kg': total_data['total_emissions_kg'].sum(),
            'avg_emissions_per_order_kg': total_data['total_emissions_kg'].mean(),
            'packaging_emissions_kg': total_data['packaging_emissions_kg'].sum(),
            'transportation_emissions_kg': total_data['transportation_emissions_kg'].sum(),
            'waste_emissions_kg': total_data['waste_emissions_kg'].sum()
        }
        
        if 'impact_category' in total_data.columns:
            high_impact_count = len(total_data[total_data['impact_category'] == 'High Carbon Impact'])
            summary['high_impact_orders_pct'] = (high_impact_count / len(total_data)) * 100
        
        if 'optimization' in self.results and 'comprehensive' in self.results['optimization']:
            summary['top_recommendations'] = self.results['optimization']['comprehensive']['top_recommendations']
        
        return summary

# Example usage
def main():
    # Create the system
    system = CarbonFootprintReductionSystem()
    
    # Run full analysis
    success = system.run_full_analysis(use_enhanced_datasets=True, num_records=1000)
    
    if success:
        # Print summary
        summary = system.get_summary()
        print("\nAnalysis Summary:")
        for key, value in summary.items():
            if isinstance(value, list):
                print(f"\n{key.replace('_', ' ').title()}:")
                for i, item in enumerate(value):
                    print(f"  {i+1}. {item}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()