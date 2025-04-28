import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import only the necessary components
from data_collection import CarbonDataCollector
from data_analysis import CarbonFootprintAnalyzer

print("Carbon Footprint Analysis System - Simplified Runner")
print("==================================================")

# Create a simplified version of the system
class SimplifiedCarbonAnalysis:
    def __init__(self):
        self.collector = CarbonDataCollector()
        self.analyzer = CarbonFootprintAnalyzer()
        self.raw_data = None
        self.results = {}
    
    def load_data(self, num_records=1000):
        """Load synthetic data"""
        print("Generating synthetic data...")
        self.raw_data = self.collector.generate_synthetic_data(num_records=num_records)
        print(f"Generated {len(self.raw_data)} synthetic records")
        return True
    
    def analyze_data(self):
        """Perform basic analysis on the data"""
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return False
        
        try:
            # Set data for analyzer
            self.analyzer.set_data(self.raw_data)
            
            # Perform basic analysis
            print("Analyzing shipping impact...")
            shipping_impact = self.analyzer.analyze_shipping_impact()
            
            print("Identifying high impact factors...")
            high_impact = self.analyzer.identify_high_impact_factors()
            
            print("Clustering shipments...")
            clusters = self.analyzer.cluster_shipments()
            
            # Store results
            self.results['shipping_impact'] = shipping_impact
            self.results['high_impact_factors'] = high_impact
            self.results['clusters'] = clusters
            
            return True
        except Exception as e:
            print(f"Error analyzing data: {e}")
            return False
    
    def run_analysis(self, num_records=1000):
        """Run a simplified analysis pipeline"""
        print("Starting simplified carbon footprint analysis...")
        
        # Step 1: Load data
        if not self.load_data(num_records):
            return False
        
        # Step 2: Analyze data
        if not self.analyze_data():
            return False
        
        print("Analysis complete!")
        return True
    
    def get_summary(self):
        """Get a summary of the analysis results"""
        if not self.results:
            return "No analysis results available. Please run analysis first."
        
        summary = {}
        
        if 'shipping_impact' in self.results:
            summary['shipping_methods'] = len(self.results['shipping_impact'])
            
        if 'clusters' in self.results:
            summary['clusters'] = len(self.results['clusters'])
            
        if 'high_impact_factors' in self.results and 'correlations' in self.results['high_impact_factors']:
            correlations = self.results['high_impact_factors']['correlations']
            summary['top_correlation'] = max(correlations.items(), key=lambda x: abs(x[1]))[0]
        
        return summary

def main():
    # Create the simplified system
    system = SimplifiedCarbonAnalysis()
    
    # Run simplified analysis
    success = system.run_analysis(num_records=500)
    
    if success:
        # Print summary
        summary = system.get_summary()
        print("\nAnalysis Summary:")
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()