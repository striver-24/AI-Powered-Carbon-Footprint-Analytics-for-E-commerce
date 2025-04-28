import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class CarbonDataIntegrator:
    """Class for loading and integrating multiple carbon footprint datasets"""
    def __init__(self):
        self.datasets = {}
        self.vehicle_emissions = None
        self.emissions_data = None
        self.packaging_materials = None
        self.waste_emissions = None
        self.ecommerce_data = None
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    def load_all_datasets(self) -> bool:
        """Load all available datasets from the data directory"""
        try:
            # Load vehicle emissions data
            self.vehicle_emissions = self.load_dataset('DB1_vehicle_emissions.csv')
            print(f"Loaded vehicle emissions data with {len(self.vehicle_emissions)} records")
            
            # Load general emissions data
            self.emissions_data = self.load_dataset('DB2_Emissions_Data.csv')
            print(f"Loaded emissions data with {len(self.emissions_data)} records")
            
            # Load packaging materials emissions data
            self.packaging_materials = self.load_dataset('DB3_Pack_Material_GHG.csv')
            print(f"Loaded packaging materials data with {len(self.packaging_materials)} records")
            
            # Load waste emissions data
            self.waste_emissions = self.load_dataset('DB4_WasteEmissions_PacMat.csv')
            print(f"Loaded waste emissions data with {len(self.waste_emissions)} records")
            
            return True
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return False
    
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load a specific dataset from the data directory"""
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def generate_ecommerce_data(self, num_records: int = 1000) -> pd.DataFrame:
        """Generate synthetic e-commerce transaction data"""
        # Create date range for the past year
        dates = pd.date_range(start='2023-01-01', periods=num_records)
        
        # Sample vehicle types from the vehicle emissions dataset
        if self.vehicle_emissions is not None:
            vehicle_types = self.vehicle_emissions['Vehicle Type'].unique()
        else:
            vehicle_types = ['Passenger Cars', 'Light-Duty Trucks', 'Heavy-Duty Vehicles']
        
        # Sample packaging materials from the packaging dataset
        if self.packaging_materials is not None:
            packaging_materials = self.packaging_materials[self.packaging_materials['Level 2'] == 'Packaging']['Level 3'].unique()
            if len(packaging_materials) == 0:
                packaging_materials = ['Cardboard', 'Plastic', 'Paper', 'Biodegradable']
        else:
            packaging_materials = ['Cardboard', 'Plastic', 'Paper', 'Biodegradable']
        
        # Generate synthetic data
        data = {
            'order_date': dates,
            'product_id': np.random.randint(1000, 9999, size=num_records),
            'product_weight_kg': np.random.lognormal(mean=1.0, sigma=0.8, size=num_records),
            'product_volume_m3': np.random.lognormal(mean=-2.0, sigma=0.5, size=num_records),
            'shipping_distance_km': np.random.gamma(shape=2, scale=250, size=num_records),
            'vehicle_type': np.random.choice(vehicle_types, size=num_records),
            'packaging_material': np.random.choice(packaging_materials, size=num_records),
            'customer_location': np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_records, p=[0.6, 0.3, 0.1]),
            'delivery_priority': np.random.choice(['Standard', 'Express', 'Next Day'], size=num_records, p=[0.7, 0.2, 0.1])
        }
        
        self.ecommerce_data = pd.DataFrame(data)
        return self.ecommerce_data

class CarbonFootprintCalculator:
    """Class for calculating carbon footprint from e-commerce operations"""
    def __init__(self, data_integrator: CarbonDataIntegrator):
        self.data_integrator = data_integrator
        self.results = {}
    
    def calculate_packaging_emissions(self) -> pd.DataFrame:
        """Calculate emissions from packaging materials"""
        if self.data_integrator.ecommerce_data is None or self.data_integrator.packaging_materials is None:
            raise ValueError("E-commerce data or packaging materials data not loaded")
        
        # Create a mapping of packaging materials to emission factors
        packaging_emissions = {}
        packaging_df = self.data_integrator.packaging_materials
        
        # Filter for packaging materials
        packaging_df = packaging_df[packaging_df['Level 2'] == 'Packaging']
        
        # Create mapping from material type to emission factor
        for _, row in packaging_df.iterrows():
            material = row['Level 3']
            if pd.notna(row['GHG Conversion Factor 2024']):
                packaging_emissions[material] = row['GHG Conversion Factor 2024']
        
        # Default values for materials not found in the dataset
        default_emissions = {
            'Cardboard': 1.038,  # kg CO2e per kg
            'Plastic': 2.73,    # kg CO2e per kg
            'Paper': 0.95,      # kg CO2e per kg
            'Biodegradable': 0.65  # kg CO2e per kg
        }
        
        # Calculate packaging weight based on product weight (simplified model)
        ecommerce_df = self.data_integrator.ecommerce_data.copy()
        ecommerce_df['packaging_weight_kg'] = ecommerce_df['product_weight_kg'] * 0.1  # Assume packaging is 10% of product weight
        
        # Calculate packaging emissions
        def get_emission_factor(material):
            if material in packaging_emissions:
                return packaging_emissions[material]
            elif material in default_emissions:
                return default_emissions[material]
            else:
                return default_emissions['Cardboard']  # Default fallback
        
        ecommerce_df['packaging_emission_factor'] = ecommerce_df['packaging_material'].apply(get_emission_factor)
        ecommerce_df['packaging_emissions_kg'] = ecommerce_df['packaging_weight_kg'] * ecommerce_df['packaging_emission_factor']
        
        self.results['packaging_emissions'] = ecommerce_df[['order_date', 'product_id', 'packaging_material', 
                                                          'packaging_weight_kg', 'packaging_emission_factor', 
                                                          'packaging_emissions_kg']]
        return self.results['packaging_emissions']
    
    def calculate_transportation_emissions(self) -> pd.DataFrame:
        """Calculate emissions from transportation"""
        if self.data_integrator.ecommerce_data is None:
            raise ValueError("E-commerce data not loaded")
        
        # Create a mapping of vehicle types to emission factors
        vehicle_emissions = {}
        
        # Use DB1 for vehicle emission factors
        if self.data_integrator.vehicle_emissions is not None:
            for _, row in self.data_integrator.vehicle_emissions.iterrows():
                vehicle_type = row['Vehicle Type']
                emission_factor = row['Total CO₂e (kg/gallon)']
                vehicle_emissions[vehicle_type] = emission_factor
        
        # Use DB2 for more detailed emission factors by distance
        distance_emissions = {}
        if self.data_integrator.emissions_data is not None:
            delivery_vehicles = self.data_integrator.emissions_data[
                (self.data_integrator.emissions_data['Level 1'] == 'Delivery vehicles') &
                (self.data_integrator.emissions_data['UOM'] == 'km')
            ]
            
            for _, row in delivery_vehicles.iterrows():
                vehicle_type = f"{row['Level 2']} - {row['Level 3']}"
                if pd.notna(row['GHG Conversion Factor 2024']):
                    distance_emissions[vehicle_type] = row['GHG Conversion Factor 2024']
        
        # Default emission factors by distance (kg CO2e per km)
        default_distance_emissions = {
            'Passenger Cars': 0.17,
            'Light-Duty Trucks': 0.25,
            'Heavy-Duty Vehicles': 0.85
        }
        
        # Calculate transportation emissions
        ecommerce_df = self.data_integrator.ecommerce_data.copy()
        
        # Function to get emission factor by distance
        def get_distance_emission_factor(vehicle_type):
            # Try to find in DB2 first
            for key, value in distance_emissions.items():
                if vehicle_type in key:
                    return value
            
            # Fall back to default values
            if vehicle_type in default_distance_emissions:
                return default_distance_emissions[vehicle_type]
            else:
                return default_distance_emissions['Passenger Cars']  # Default fallback
        
        ecommerce_df['distance_emission_factor'] = ecommerce_df['vehicle_type'].apply(get_distance_emission_factor)
        ecommerce_df['transportation_emissions_kg'] = ecommerce_df['shipping_distance_km'] * ecommerce_df['distance_emission_factor']
        
        # Adjust for delivery priority
        priority_factors = {
            'Standard': 1.0,
            'Express': 1.2,
            'Next Day': 1.5
        }
        ecommerce_df['priority_factor'] = ecommerce_df['delivery_priority'].map(priority_factors)
        ecommerce_df['transportation_emissions_kg'] *= ecommerce_df['priority_factor']
        
        self.results['transportation_emissions'] = ecommerce_df[['order_date', 'product_id', 'vehicle_type', 
                                                              'shipping_distance_km', 'delivery_priority',
                                                              'distance_emission_factor', 'priority_factor',
                                                              'transportation_emissions_kg']]
        return self.results['transportation_emissions']
    
    def calculate_waste_emissions(self) -> pd.DataFrame:
        """Calculate emissions from waste management of packaging"""
        if self.data_integrator.ecommerce_data is None or self.data_integrator.waste_emissions is None:
            raise ValueError("E-commerce data or waste emissions data not loaded")
        
        # Create a mapping of waste disposal methods to emission factors
        waste_emissions = {}
        waste_df = self.data_integrator.waste_emissions
        
        # Filter for packaging waste
        packaging_waste = waste_df[waste_df['Level 2'] == 'Packaging']
        
        # Create mapping from disposal method to emission factor
        for _, row in packaging_waste.iterrows():
            material = row['Level 3']
            disposal_method = row['Level 4']
            if pd.notna(row['GHG Conversion Factor 2024']):
                key = f"{material}_{disposal_method}"
                waste_emissions[key] = row['GHG Conversion Factor 2024']
        
        # Default values for waste disposal methods (kg CO2e per kg)
        default_waste_emissions = {
            'Landfill': 0.99,
            'Recycling': 0.21,
            'Incineration': 0.58
        }
        
        # Assign waste disposal methods based on packaging material
        ecommerce_df = self.data_integrator.ecommerce_data.copy()
        
        # Simplified model: assign disposal methods based on material type
        disposal_methods = {
            'Cardboard': 'Recycling',
            'Plastic': 'Landfill',
            'Paper': 'Recycling',
            'Biodegradable': 'Composting'
        }
        
        ecommerce_df['disposal_method'] = ecommerce_df['packaging_material'].map(disposal_methods)
        ecommerce_df['packaging_weight_kg'] = ecommerce_df['product_weight_kg'] * 0.1  # Assume packaging is 10% of product weight
        
        # Calculate waste emissions
        def get_waste_emission_factor(material, disposal):
            key = f"{material}_{disposal}"
            if key in waste_emissions:
                return waste_emissions[key]
            elif disposal in default_waste_emissions:
                return default_waste_emissions[disposal]
            else:
                return default_waste_emissions['Landfill']  # Default fallback
        
        ecommerce_df['waste_emission_factor'] = ecommerce_df.apply(
            lambda row: get_waste_emission_factor(row['packaging_material'], row['disposal_method']), axis=1
        )
        ecommerce_df['waste_emissions_kg'] = ecommerce_df['packaging_weight_kg'] * ecommerce_df['waste_emission_factor']
        
        self.results['waste_emissions'] = ecommerce_df[['order_date', 'product_id', 'packaging_material', 
                                                      'disposal_method', 'packaging_weight_kg',
                                                      'waste_emission_factor', 'waste_emissions_kg']]
        return self.results['waste_emissions']
    
    def calculate_total_emissions(self) -> pd.DataFrame:
        """Calculate total emissions by combining all sources"""
        # Ensure all emission calculations have been performed
        if 'packaging_emissions' not in self.results:
            self.calculate_packaging_emissions()
        
        if 'transportation_emissions' not in self.results:
            self.calculate_transportation_emissions()
        
        if 'waste_emissions' not in self.results:
            self.calculate_waste_emissions()
        
        # Merge all emission results
        ecommerce_df = self.data_integrator.ecommerce_data.copy()
        
        # Add packaging emissions
        packaging_emissions = self.results['packaging_emissions'][['product_id', 'packaging_emissions_kg']]
        ecommerce_df = ecommerce_df.merge(packaging_emissions, on='product_id', how='left')
        
        # Add transportation emissions
        transportation_emissions = self.results['transportation_emissions'][['product_id', 'transportation_emissions_kg']]
        ecommerce_df = ecommerce_df.merge(transportation_emissions, on='product_id', how='left')
        
        # Add waste emissions
        waste_emissions = self.results['waste_emissions'][['product_id', 'waste_emissions_kg']]
        ecommerce_df = ecommerce_df.merge(waste_emissions, on='product_id', how='left')
        
        # Calculate total emissions
        ecommerce_df['total_emissions_kg'] = (
            ecommerce_df['packaging_emissions_kg'] + 
            ecommerce_df['transportation_emissions_kg'] + 
            ecommerce_df['waste_emissions_kg']
        )
        
        self.results['total_emissions'] = ecommerce_df
        return self.results['total_emissions']

class CarbonFootprintOptimizer:
    """Class for optimizing carbon footprint using ML techniques"""
    def __init__(self, calculator: CarbonFootprintCalculator):
        self.calculator = calculator
        self.data = None
        self.clusters = None
        self.scaler = StandardScaler()
        self.recommendations = {}
    
    def prepare_data(self):
        """Prepare data for optimization"""
        if 'total_emissions' not in self.calculator.results:
            self.calculator.calculate_total_emissions()
        
        self.data = self.calculator.results['total_emissions']
    
    def cluster_products(self, n_clusters: int = 3) -> pd.DataFrame:
        """Cluster products based on their carbon footprint"""
        if self.data is None:
            self.prepare_data()
        
        # Select features for clustering
        features = ['product_weight_kg', 'shipping_distance_km', 'packaging_emissions_kg', 
                   'transportation_emissions_kg', 'waste_emissions_kg', 'total_emissions_kg']
        
        # Handle missing values
        X = self.data[features].fillna(0)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Get cluster characteristics
        self.clusters = self.data.groupby('cluster')[features].mean()
        
        # Label clusters based on emissions
        cluster_labels = {}
        for i in range(n_clusters):
            if self.clusters.loc[i, 'total_emissions_kg'] <= self.clusters['total_emissions_kg'].quantile(0.33):
                cluster_labels[i] = 'Low Carbon Impact'
            elif self.clusters.loc[i, 'total_emissions_kg'] <= self.clusters['total_emissions_kg'].quantile(0.67):
                cluster_labels[i] = 'Medium Carbon Impact'
            else:
                cluster_labels[i] = 'High Carbon Impact'
        
        self.data['impact_category'] = self.data['cluster'].map(cluster_labels)
        
        return self.clusters
    
    def generate_packaging_recommendations(self) -> Dict:
        """Generate recommendations for packaging optimization"""
        if self.data is None:
            self.prepare_data()
        
        # Analyze packaging emissions by material
        packaging_impact = self.data.groupby('packaging_material')['packaging_emissions_kg'].mean().sort_values()
        
        # Identify best and worst packaging materials
        best_packaging = packaging_impact.index[0]
        worst_packaging = packaging_impact.index[-1]
        
        # Calculate potential savings
        potential_savings = packaging_impact[worst_packaging] - packaging_impact[best_packaging]
        percentage_reduction = (potential_savings / packaging_impact[worst_packaging]) * 100
        
        # Generate recommendations
        recommendations = {
            'best_packaging_material': best_packaging,
            'worst_packaging_material': worst_packaging,
            'potential_savings_kg_per_package': round(potential_savings, 3),
            'percentage_reduction': round(percentage_reduction, 2),
            'recommendation': f"Switch from {worst_packaging} to {best_packaging} packaging to reduce emissions by {round(percentage_reduction, 2)}%"
        }
        
        self.recommendations['packaging'] = recommendations
        return recommendations
    
    def generate_transportation_recommendations(self) -> Dict:
        """Generate recommendations for transportation optimization"""
        if self.data is None:
            self.prepare_data()
        
        # Analyze transportation emissions by vehicle type
        transport_impact = self.data.groupby('vehicle_type')['transportation_emissions_kg'].mean().sort_values()
        
        # Identify best and worst vehicle types
        best_vehicle = transport_impact.index[0]
        worst_vehicle = transport_impact.index[-1]
        
        # Calculate potential savings
        potential_savings = transport_impact[worst_vehicle] - transport_impact[best_vehicle]
        percentage_reduction = (potential_savings / transport_impact[worst_vehicle]) * 100
        
        # Analyze impact of delivery priority
        priority_impact = self.data.groupby('delivery_priority')['transportation_emissions_kg'].mean().sort_values()
        
        # Generate recommendations
        recommendations = {
            'best_vehicle_type': best_vehicle,
            'worst_vehicle_type': worst_vehicle,
            'potential_vehicle_savings_kg': round(potential_savings, 3),
            'vehicle_percentage_reduction': round(percentage_reduction, 2),
            'delivery_priority_impact': priority_impact.to_dict(),
            'recommendation': f"Use {best_vehicle} instead of {worst_vehicle} when possible to reduce emissions by {round(percentage_reduction, 2)}%"
        }
        
        self.recommendations['transportation'] = recommendations
        return recommendations
    
    def generate_waste_recommendations(self):
        """Generate recommendations for waste management optimization."""
        if not hasattr(self, 'data') or self.data is None:
            self.prepare_data()
        
        # Check if disposal_method column exists
        if 'disposal_method' not in self.data.columns:
            # Handle missing column - create a dictionary with default values
            self.recommendations['waste'] = {
                'recommendation': "Data for waste disposal methods is missing. Consider collecting data on disposal methods to optimize waste management.",
                'potential_savings_kg': 0.0,  # Default value to avoid TypeError
                'percentage_reduction': 0.0,
                'best_disposal_method': "Unknown",
                'worst_disposal_method': "Unknown"
            }
            return self.recommendations['waste']
            
        # Analyze waste emissions by disposal method
        waste_impact = self.data.groupby('disposal_method')['waste_emissions_kg'].mean().sort_values()
        
        # Identify best and worst disposal methods
        best_disposal = waste_impact.index[0]
        worst_disposal = waste_impact.index[-1]
        
        # Calculate potential savings
        potential_savings = waste_impact[worst_disposal] - waste_impact[best_disposal]
        percentage_reduction = (potential_savings / waste_impact[worst_disposal]) * 100
        
        # Generate recommendations as a dictionary (not a list)
        self.recommendations['waste'] = {
            'best_disposal_method': best_disposal,
            'worst_disposal_method': worst_disposal,
            'potential_savings_kg': round(potential_savings, 3),
            'percentage_reduction': round(percentage_reduction, 2),
            'recommendation': f"Switch from {worst_disposal} to {best_disposal} disposal methods where possible to reduce emissions by {round(percentage_reduction, 2)}%"
        }
        
        return self.recommendations['waste']
    
    def generate_comprehensive_recommendations(self) -> Dict:
        """Generate comprehensive recommendations for carbon footprint reduction"""
        # Ensure all individual recommendations have been generated
        if 'packaging' not in self.recommendations:
            self.generate_packaging_recommendations()
        
        if 'transportation' not in self.recommendations:
            self.generate_transportation_recommendations()
        
        if 'waste' not in self.recommendations:
            self.generate_waste_recommendations()
        
        # Identify high-impact products
        if 'cluster' not in self.data.columns:
            self.cluster_products()
        
        high_impact_products = self.data[self.data['impact_category'] == 'High Carbon Impact']
        
        # Calculate total potential savings
        total_potential_savings = (
            self.recommendations['packaging']['potential_savings_kg_per_package'] +
            self.recommendations['transportation']['potential_vehicle_savings_kg'] +
            self.recommendations['waste']['potential_savings_kg']
        )
        
        # Generate comprehensive recommendations
        comprehensive_recommendations = {
            'total_potential_savings_kg_per_package': round(total_potential_savings, 3),
            'high_impact_product_count': len(high_impact_products),
            'high_impact_percentage': round(len(high_impact_products) / len(self.data) * 100, 2),
            'top_recommendations': [
                self.recommendations['packaging']['recommendation'],
                self.recommendations['transportation']['recommendation'],
                self.recommendations['waste']['recommendation']
            ],
            'priority_actions': [
                f"Focus on optimizing {len(high_impact_products)} high-impact products that represent {round(len(high_impact_products) / len(self.data) * 100, 2)}% of your inventory",
                f"Implement all recommendations to achieve a potential {round(total_potential_savings, 2)} kg CO2e reduction per package"
            ]
        }
        
        self.recommendations['comprehensive'] = comprehensive_recommendations
        return comprehensive_recommendations

# Example usage
def main():
    # Initialize the data integrator and load datasets
    data_integrator = CarbonDataIntegrator()
    data_integrator.load_all_datasets()
    
    # Generate synthetic e-commerce data
    ecommerce_data = data_integrator.generate_ecommerce_data(num_records=1000)
    print(f"Generated {len(ecommerce_data)} synthetic e-commerce records")
    
    # Initialize the carbon footprint calculator
    calculator = CarbonFootprintCalculator(data_integrator)
    
    # Calculate emissions from different sources
    packaging_emissions = calculator.calculate_packaging_emissions()
    print(f"Calculated packaging emissions for {len(packaging_emissions)} records")
    
    transportation_emissions = calculator.calculate_transportation_emissions()
    print(f"Calculated transportation emissions for {len(transportation_emissions)} records")
    
    waste_emissions = calculator.calculate_waste_emissions()
    print(f"Calculated waste emissions for {len(waste_emissions)} records")
    
    total_emissions = calculator.calculate_total_emissions()
    print(f"Calculated total emissions for {len(total_emissions)} records")
    
    # Initialize the optimizer
    optimizer = CarbonFootprintOptimizer(calculator)
    
    # Cluster products by carbon impact
    clusters = optimizer.cluster_products(n_clusters=3)
    print("\nProduct clusters by carbon impact:")
    print(clusters)
    
    # Generate recommendations
    packaging_recommendations = optimizer.generate_packaging_recommendations()
    print("\nPackaging recommendations:")
    print(packaging_recommendations['recommendation'])
    
    transportation_recommendations = optimizer.generate_transportation_recommendations()
    print("\nTransportation recommendations:")
    print(transportation_recommendations['recommendation'])
    
    waste_recommendations = optimizer.generate_waste_recommendations()
    print("\nWaste management recommendations:")
    print(waste_recommendations['recommendation'])
    
    comprehensive_recommendations = optimizer.generate_comprehensive_recommendations()
    print("\nComprehensive recommendations:")
    for action in comprehensive_recommendations['priority_actions']:
        print(f"- {action}")

if __name__ == "__main__":
    main()