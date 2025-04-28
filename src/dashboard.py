import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta

# Import original modules
from data_collection import CarbonDataCollector
from data_analysis import CarbonFootprintAnalyzer
from ml_models import CarbonFootprintPredictor

# Import new modules
from carbon_footprint_system import CarbonDataIntegrator, CarbonFootprintCalculator, CarbonFootprintOptimizer
from emissions_forecasting import EmissionsForecaster

st.set_page_config(page_title="Carbon Footprint Analytics", layout="wide")

class CarbonFootprintDashboard:
    def __init__(self):
        # Original components
        self.collector = CarbonDataCollector()
        self.analyzer = CarbonFootprintAnalyzer()
        self.predictor = CarbonFootprintPredictor()
        self.data = None
        
        # New components
        self.data_integrator = CarbonDataIntegrator()
        self.calculator = None
        self.optimizer = None
        self.forecaster = None
        
        # Initialize session state
        if 'enhanced_data_loaded' not in st.session_state:
            st.session_state.enhanced_data_loaded = False

    def load_data(self):
        """Load data from file upload or generate synthetic data"""
        data_source = st.sidebar.radio(
            "Select data source:",
            ["Upload real data", "Use enhanced carbon datasets", "Generate synthetic data"]
        )
        
        if data_source == "Upload real data":
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV/Excel file", 
                type=["csv", "xlsx"],
                help="Upload your shipping data file"
            )
            
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    self.data = pd.read_csv(uploaded_file)
                else:
                    self.data = pd.read_excel(uploaded_file)
                self.analyzer.set_data(self.data)
                return True
            
            st.warning("Please upload a data file to begin analysis")
            return False
            
        elif data_source == "Use enhanced carbon datasets":
            with st.spinner("Loading carbon footprint datasets..."):
                # Load all datasets
                success = self.data_integrator.load_all_datasets()
                
                if success:
                    # Generate synthetic e-commerce data
                    num_records = st.sidebar.slider("Number of records", 100, 5000, 1000, 100)
                    self.data_integrator.generate_ecommerce_data(num_records=num_records)
                    
                    # Initialize calculator
                    self.calculator = CarbonFootprintCalculator(self.data_integrator)
                    
                    # Calculate emissions
                    self.calculator.calculate_packaging_emissions()
                    self.calculator.calculate_transportation_emissions()
                    self.calculator.calculate_waste_emissions()
                    total_emissions = self.calculator.calculate_total_emissions()
                    
                    # Initialize optimizer
                    self.optimizer = CarbonFootprintOptimizer(self.calculator)
                    self.optimizer.cluster_products()
                    self.optimizer.generate_comprehensive_recommendations()
                    
                    # Initialize forecaster
                    self.forecaster = EmissionsForecaster()
                    self.forecaster.set_data(total_emissions, date_column='order_date', target_column='total_emissions_kg')
                    
                    # Set session state
                    st.session_state.enhanced_data_loaded = True
                    
                    # Also set the original data for backward compatibility
                    self.data = total_emissions
                    self.analyzer.set_data(self.data)
                    
                    return True
                else:
                    st.error("Failed to load carbon footprint datasets")
                    return False
        else:  # Generate synthetic data
            num_records = st.sidebar.slider("Number of synthetic records", 100, 5000, 1000, 100)
            self.data = self.collector.generate_synthetic_data(num_records=num_records)
            self.analyzer.set_data(self.data)
            return True

    def generate_sample_data(self):
        """Generate synthetic sample data for demonstration"""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create date range
        dates = [datetime.now() - timedelta(days=x) for x in range(30)]
        
        # Generate synthetic data
        data = {
            'order_date': dates * 10,
            'product_id': np.random.randint(1000, 9999, 300),
            'product_weight_kg': np.random.uniform(0.5, 20.0, 300),
            'shipping_distance_km': np.random.randint(5, 500, 300),
            'vehicle_type': np.random.choice(['Van', 'Truck', 'Drone'], 300),
            'packaging_material': np.random.choice(['Cardboard', 'Plastic', 'Biodegradable'], 300),
            'total_emissions_kg': np.random.uniform(0.5, 15.0, 300)
        }
        
        self.data = pd.DataFrame(data)
        st.success("Generated synthetic sample data with 300 records")

    def show_overview(self):
        """Display high-level metrics and summary statistics."""
        st.header("Overview Metrics")
        
        # Get or calculate summary statistics
        summary = self.analyzer.get_summary_stats()
        
        # Add data validation
        required_metrics = ['total_shipments', 'total_carbon_footprint', 'avg_carbon_per_shipment']
        if not all(m in summary for m in required_metrics):
            st.warning("Missing critical metrics - regenerating sample data")
            self.data = self.collector.generate_synthetic_data(1000)
            self.analyzer.set_data(self.data)
            summary = self.analyzer.get_summary_stats()
        
        # Create two columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Add fallback values for missing metrics
        with col1:
            st.metric("Total Shipments", f"{summary.get('total_shipments', 0):,}")
            st.metric("Average Weight", f"{summary.get('avg_weight_kg', 0):.2f} kg")
        
        with col2:
            st.metric("Total Emissions", f"{summary.get('total_emissions_kg', 0):,.0f} kg")
            st.metric("Avg Distance", f"{summary.get('avg_distance_km', 0):,.0f} km")
        
        with col3:
            st.metric("Packaging Impact", f"{summary.get('best_packaging_type', 'N/A')}")
            st.metric("Transport Efficiency", f"{summary.get('transport_efficiency_score', 0):.1f}%")
        
        # Add warning for missing data
        if 'total_shipments' not in summary:
            st.warning("Shipment data not found - using synthetic data for demonstration")
            self.generate_sample_data()
    
        with col2:
            st.metric("Total Carbon Footprint", f"{summary.get('total_carbon_footprint', 0):,.2f} kg CO2")
        with col3:
            st.metric("Avg Carbon per Shipment", f"{summary.get('avg_carbon_per_shipment', 0):.2f} kg CO2")

    def show_shipping_analysis(self):
        """Display shipping method analysis"""
        st.header("Shipping Method Analysis")
        
        impact = self.analyzer.analyze_shipping_impact()
        
        # Transform data from wide to long format
        df = pd.DataFrame.from_dict(impact, orient='index').reset_index()
        df = df.melt(id_vars='index', 
                    var_name='shipping_method',
                    value_name='carbon_impact')
        
        # Create visualization
        fig = px.bar(
            df,
            x='shipping_method',
            y='carbon_impact',
            title='Carbon Impact by Shipping Method',
            labels={'carbon_impact': 'Carbon Footprint (kg CO2)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_packaging_analysis(self):
        """Display packaging type analysis"""
        st.header("Packaging Type Analysis")
        
        impact = self.analyzer.identify_high_impact_factors()
        df = pd.DataFrame.from_dict(impact['packaging_impact'], orient='index').reset_index()
        df.columns = ['packaging_type', 'carbon_impact']
        
        fig = px.bar(
            df,
            x='packaging_type',
            y='carbon_impact',
            title='Carbon Impact by Packaging Type',
            labels={'carbon_impact': 'Carbon Footprint (kg CO2)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_trends(self):
        """Display carbon footprint trends"""
        st.header("Carbon Footprint Trends")
        
        fig = px.line(
            self.data,
            x='order_date',  # Changed from 'date'
            y='total_emissions_kg',  # Changed from 'carbon_footprint'
            title='Carbon Footprint Over Time'
        )
        st.plotly_chart(fig)

    def show_predictions(self):
        """Display carbon footprint predictions"""
        st.header("Carbon Footprint Predictions")
        
        prediction_type = st.radio(
            "Select Prediction Model",
            ["Prophet", "LSTM"]
        )
        
        periods = st.slider("Prediction Periods (Days)", 7, 90, 30)
        
        if prediction_type == "Prophet":
            self.predictor.train_prophet_model(self.data)
            forecast = self.predictor.predict_prophet(periods)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.data['order_date'],  # Changed from 'date'
                y=self.data['total_emissions_kg'],  # Changed from 'carbon_footprint'
                name='Historical'
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name='Forecast'
            ))
            fig.update_layout(title='Prophet Forecast')
            st.plotly_chart(fig)
        # In the LSTM prediction section:
        else:
            self.predictor.train_lstm_model(self.data)
            predictions = self.predictor.predict_lstm(self.data)
            
            future_dates = pd.date_range(
                start=self.data['order_date'].max(),
                periods=len(predictions)+1
            )[1:]
            
            # Initialize figure before adding traces
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.data['order_date'],
                y=self.data['total_emissions_kg'],
                name='Historical'
            ))
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                name='Forecast'
            ))
            fig.update_layout(title='LSTM Forecast')
            st.plotly_chart(fig)

    def show_recommendations(self):
        """Display optimization recommendations"""
        st.header("Optimization Recommendations")
        
        recommendations = self.predictor.get_optimization_recommendations(self.data)
        
        st.subheader("Current Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Average Carbon Footprint",
                f"{recommendations['baseline_metrics']['avg_footprint']:.2f} kg CO2"
            )
        with col2:
            st.metric(
                "Total Carbon Footprint",
                f"{recommendations['baseline_metrics']['total_footprint']:.2f} kg CO2"
            )
        
        st.subheader("Recommendations")
        st.write(f"🚚 Best Shipping Method: {recommendations['recommendations']['best_shipping_method']}")
        st.write(f"📦 Best Packaging Type: {recommendations['recommendations']['best_packaging_type']}")
        st.write(f"📉 Potential Reduction: {recommendations['recommendations']['potential_reduction_percentage']}%")
        
        # Remove or comment out the scope analysis section until implemented
        # st.subheader("Scope-Specific Recommendations")
        # scope_data = self.analyzer.get_scope_analysis()
        # 
        # for scope, details in scope_data.items():
        #     with st.expander(f"Scope {scope[-1]} Recommendations"):
        #         st.write(f"**Highest Impact Category:** {details['highest_impact']}")
        #         st.write(f"**Potential Reduction:** {details['reduction_potential']}%")
        #         st.write(f"**Recommended Action:** {details['recommendation']}")

    def show_correlations(self):
        """Display feature correlations"""
        st.header("Feature Correlations")
        
        impact = self.analyzer.identify_high_impact_factors()
        corr_df = pd.DataFrame.from_dict(impact['correlations'], orient='index').reset_index()
        corr_df.columns = ['feature', 'correlation']
        
        fig = px.bar(
            corr_df,
            x='feature',
            y='correlation',
            title='Feature Correlation with Carbon Footprint',
            color='correlation',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_vehicle_analysis(self):
        """Display vehicle type analysis"""
        st.header("Vehicle Emissions Analysis")
        
        impact = self.analyzer.analyze_vehicle_impact(self.data)  # Pass self.data as shipping_data
        df = pd.DataFrame.from_dict(impact, orient='index').reset_index()
        df.columns = ['vehicle_type', 'carbon_impact']
        
        fig = px.bar(
            df,
            x='vehicle_type',
            y='carbon_impact',
            title='Carbon Impact by Vehicle Type',
            labels={'carbon_impact': 'Carbon Footprint (kg CO2)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_enhanced_analysis(self):
        from enhanced_dashboard import show_enhanced_analysis
        show_enhanced_analysis(self)
    
    def run(self):
        st.title("E-commerce Carbon Footprint Analytics")
        
        if self.load_data():
            self.show_overview()
            
            # Create tabs for different analyses
            tabs = st.tabs(["Impact Analysis", "Trends & Predictions", "Enhanced Analysis"])
            
            with tabs[0]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    self.show_shipping_analysis()
                with col2:
                    self.show_packaging_analysis()
                with col3:
                    self.show_vehicle_analysis()
                
                st.subheader("Feature Analysis")
                self.show_correlations()
                
                self.show_recommendations()
            
            with tabs[1]:
                col1, col2 = st.columns(2)
                with col1:
                    self.show_trends()
                with col2:
                    self.show_predictions()
                    # Remove the duplicate plot addition
                    # --- REMOVE THESE LINES ---
                    # fig.add_trace(go.Scatter(
                    #     x=self.data['order_date'],
                    #     y=self.data['total_emissions_kg'], 
                    #     name='Historical'
                    # ))
            
            with tabs[2]:
                self.show_enhanced_analysis()
        else:
            st.warning("Please load or generate data to view the analytics.")

if __name__ == "__main__":
    dashboard = CarbonFootprintDashboard()
    dashboard.run()