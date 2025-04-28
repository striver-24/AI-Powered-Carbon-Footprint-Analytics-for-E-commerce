import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components that don't require tensorflow or prophet
from data_collection import CarbonDataCollector
from data_analysis import CarbonFootprintAnalyzer

print("Carbon Footprint Reduction System - Full Analysis")
print("==================================================")

class ModifiedCarbonFootprintSystem:
    """Modified version of the main carbon footprint reduction system"""
    def __init__(self):
        # Initialize components that don't require tensorflow/prophet
        self.collector = CarbonDataCollector()
        self.analyzer = CarbonFootprintAnalyzer()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.results = {}
    
    def load_data(self, num_records=1000, file_path=None):
        """Load data from file or generate synthetic data"""
        if file_path:
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
            print(f"Generated {len(self.raw_data)} synthetic records")
            return True
    
    def process_data(self):
        """Process the loaded data for analysis"""
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return False
        
        try:
            # Set data for analyzer
            self.analyzer.set_data(self.raw_data)
            self.processed_data = self.raw_data
            
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
            print(f"Error processing data: {e}")
            return False
    
    def analyze_emissions_patterns(self):
        """Analyze patterns in emissions data"""
        if self.processed_data is None:
            print("No processed data available. Please process data first.")
            return False
        
        try:
            # Calculate basic statistics
            emissions_stats = {
                'mean': self.processed_data['carbon_footprint'].mean(),
                'median': self.processed_data['carbon_footprint'].median(),
                'std': self.processed_data['carbon_footprint'].std(),
                'min': self.processed_data['carbon_footprint'].min(),
                'max': self.processed_data['carbon_footprint'].max()
            }
            
            # Identify top contributors
            if 'product_category' in self.processed_data.columns:
                category_impact = self.processed_data.groupby('product_category')['carbon_footprint'].sum()
                top_categories = category_impact.sort_values(ascending=False).head(5)
                emissions_stats['top_categories'] = top_categories.to_dict()
            
            # Store results
            self.results['emissions_stats'] = emissions_stats
            
            print("Emissions patterns analyzed successfully")
            return True
        except Exception as e:
            print(f"Error analyzing emissions patterns: {e}")
            return False
    
    def generate_recommendations(self):
        """Generate recommendations for emissions reduction"""
        if not self.results:
            print("No analysis results available. Please run analysis first.")
            return False
        
        try:
            recommendations = {
                'general': [
                    "Optimize packaging materials to reduce waste",
                    "Consider more efficient shipping methods for high-volume routes",
                    "Implement carbon offset programs for unavoidable emissions"
                ]
            }
            
            # Add shipping-specific recommendations
            if 'shipping_impact' in self.results:
                shipping_methods = self.results['shipping_impact']
                highest_impact = max(shipping_methods.items(), key=lambda x: x[1]['sum'])[0]
                recommendations['shipping'] = [
                    f"Consider alternatives to {highest_impact} shipping which has the highest emissions",
                    "Consolidate shipments to reduce the number of deliveries",
                    "Optimize delivery routes to minimize distance traveled"
                ]
            
            # Add cluster-specific recommendations
            if 'clusters' in self.results:
                clusters = self.results['clusters']
                highest_cluster = clusters['carbon_footprint'].idxmax()
                recommendations['clusters'] = [
                    f"Focus reduction efforts on cluster {highest_cluster} which has the highest emissions",
                    "Develop targeted strategies for each emissions cluster",
                    "Regularly reassess clustering as new data becomes available"
                ]
            
            # Store recommendations
            self.results['recommendations'] = recommendations
            
            print("Recommendations generated successfully")
            return True
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return False
    
    def run_full_analysis(self, num_records=1000, file_path=None):
        """Run the complete analysis pipeline"""
        print("Starting carbon footprint reduction analysis...")
        
        # Step 1: Load data
        if not self.load_data(num_records, file_path):
            return False
        
        # Step 2: Process data
        if not self.process_data():
            return False
        
        # Step 3: Analyze emissions patterns
        if not self.analyze_emissions_patterns():
            return False
        
        # Step 4: Generate recommendations
        if not self.generate_recommendations():
            return False
        
        print("Analysis complete!")
        return True
    
    def get_summary(self):
        """Get a summary of the analysis results"""
        if not self.results:
            return "No analysis results available. Please run analysis first."
        
        summary = {}
        
        # Basic statistics
        if 'emissions_stats' in self.results:
            stats = self.results['emissions_stats']
            summary['average_emissions'] = round(stats['mean'], 2)
            summary['max_emissions'] = round(stats['max'], 2)
            summary['emissions_variability'] = round(stats['std'], 2)
        
        # Shipping impact
        if 'shipping_impact' in self.results:
            shipping = self.results['shipping_impact']
            summary['shipping_methods_analyzed'] = len(shipping)
            highest_method = max(shipping.items(), key=lambda x: x[1]['sum'])[0]
            summary['highest_impact_shipping'] = highest_method
        
        # Clusters
        if 'clusters' in self.results:
            summary['emission_clusters'] = len(self.results['clusters'])
        
        # Recommendations
        if 'recommendations' in self.results:
            all_recs = []
            for rec_type, recs in self.results['recommendations'].items():
                if isinstance(recs, list):
                    all_recs.extend(recs)
            summary['top_recommendations'] = all_recs[:5]
        
        return summary

def main():
    # Create the system
    system = ModifiedCarbonFootprintSystem()
    
    # Run full analysis
    success = system.run_full_analysis(num_records=1000)
    
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