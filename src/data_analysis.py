import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CarbonFootprintAnalyzer:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self.scaler = StandardScaler()
        self._cache = {}  # Add caching dictionary
        
    def clear_cache(self):
        """Clear the analysis cache"""
        self._cache = {}
        
    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data for analysis"""
        self.data = data
    
    def analyze_shipping_impact(self):
        """Analyze carbon impact by shipping method"""
        # Use correct column name from current data structure
        impact_by_method = self.data.groupby('vehicle_type')['total_emissions_kg'].agg([
            ('total_emissions', 'sum'),
            ('average_emissions', 'mean'),
            ('shipment_count', 'count')
        ]).reset_index()
        
        result = impact_by_method.to_dict('index')
        self._cache[cache_key] = result
        return result
    
    def cluster_shipments(self, n_clusters: int = 3) -> pd.DataFrame:
        """Cluster shipments based on carbon impact"""
        # Update features to match calculator output
        features = ['shipping_distance_km', 'product_weight_kg', 'total_emissions_kg']
        X = self.data[features]
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Get cluster characteristics
        cluster_stats = self.data.groupby('cluster')[features].mean()
        return cluster_stats
    
    def identify_high_impact_factors(self) -> Dict:
        """Identify factors contributing to high carbon footprint"""
        # Use actual column names from your dataset
        correlations = self.data[['shipping_distance_km', 'product_weight_kg', 'total_emissions_kg']].corr()
        
        # Calculate average footprint by packaging material
        packaging_impact = self.data.groupby('packaging_material')['total_emissions_kg'].mean()
        
        return {
            "correlations": correlations['total_emissions_kg'].to_dict(),
            "packaging_impact": packaging_impact.to_dict()
        }
    
    def cluster_shipments(self, n_clusters: int = 3) -> pd.DataFrame:
        """Cluster shipments based on carbon impact"""
        features = ['distance_km', 'weight_kg', 'carbon_footprint']
        X = self.data[features]
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Get cluster characteristics
        cluster_stats = self.data.groupby('cluster')[features].mean()
        return cluster_stats
    
    def plot_carbon_trends(self) -> None:
        """Plot carbon footprint trends over time"""
        plt.figure(figsize=(12, 6))
        
        # Time series plot
        sns.lineplot(data=self.data, x='date', y='carbon_footprint')
        plt.title('Carbon Footprint Trends Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('carbon_trends.png')
        plt.close()
    
    def plot_shipping_distribution(self) -> None:
        """Plot distribution of carbon footprint by shipping method"""
        plt.figure(figsize=(10, 6))
        
        sns.boxplot(data=self.data, x='shipping_method', y='carbon_footprint')
        plt.title('Carbon Footprint Distribution by Shipping Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('shipping_distribution.png')
        plt.close()
    
    def analyze_vehicle_impact(self, shipping_data):
        """Calculate carbon impact by vehicle type"""
        if not hasattr(self, 'emissions_df'):
            # Load emissions data if not already loaded
            self.emissions_df = pd.DataFrame({
                'vehicle_type': ['Van', 'Truck', 'Drone'],
                'co2e_per_gallon': [0.15356, 0.21542, 0.04231]
            })
        
        # Check if required columns exist
        if 'vehicle_type' not in shipping_data.columns or 'distance' not in shipping_data.columns:
            return {'error': 'Missing required columns (vehicle_type and/or distance)'}
        
        try:
            merged = shipping_data.merge(self.emissions_df, on='vehicle_type', how='left')
            merged['total_emissions'] = merged['distance'] * merged['co2e_per_gallon']
            return merged.groupby('vehicle_type')['total_emissions'].sum().to_dict()
        except Exception as e:
            return {'error': f'Error processing vehicle impact: {str(e)}'}
    
    def get_summary_stats(self):
        """Calculate and return summary statistics with enhanced validation"""
        if self.data is None or 'total_emissions_kg' not in self.data.columns:
            return {}
            
        return {
            'total_shipments': len(self.data),
            'total_carbon_footprint': self.data['total_emissions_kg'].sum(),
            'avg_carbon_per_shipment': self.data['total_emissions_kg'].mean(),
            'avg_weight_kg': self.data.get('product_weight_kg', 0).mean(),
            'avg_distance_km': self.data.get('shipping_distance_km', 0).mean(),
            'best_packaging_type': self.data.groupby('packaging_material')['total_emissions_kg'].mean().idxmin(),
            'transport_efficiency_score': (1 - (self.data['total_emissions_kg'] / 
                                             self.data['shipping_distance_km']).mean()) * 100
        }