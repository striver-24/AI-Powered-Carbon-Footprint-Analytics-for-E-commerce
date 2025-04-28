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

def show_enhanced_analysis(dashboard):
    """Show enhanced analysis using the new carbon footprint system"""
    
    # Overview metrics
    total_data = dashboard.calculator.results['total_emissions']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Orders", 
            value=f"{len(total_data):,}"
        )
    
    with col2:
        total_emissions = total_data['total_emissions_kg'].sum()
        st.metric(
            label="Total Carbon Emissions", 
            value=f"{total_emissions:.2f} kg CO₂e"
        )
    
    with col3:
        avg_emissions = total_data['total_emissions_kg'].mean()
        st.metric(
            label="Avg. Emissions per Order", 
            value=f"{avg_emissions:.2f} kg CO₂e"
        )
    
    with col4:
        if 'impact_category' in total_data.columns:
            high_impact_count = len(total_data[total_data['impact_category'] == 'High Carbon Impact'])
            high_impact_pct = (high_impact_count / len(total_data)) * 100
            st.metric(
                label="High Impact Orders", 
                value=f"{high_impact_pct:.1f}%"
            )
    
    # Emissions breakdown
    st.subheader("Emissions Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Emissions by source
        emissions_by_source = pd.DataFrame({
            'Source': ['Packaging', 'Transportation', 'Waste'],
            'Emissions (kg CO₂e)': [
                total_data['packaging_emissions_kg'].sum(),
                total_data['transportation_emissions_kg'].sum(),
                total_data['waste_emissions_kg'].sum()
            ]
        })
        
        fig = px.pie(
            emissions_by_source, 
            values='Emissions (kg CO₂e)', 
            names='Source',
            title='Emissions by Source'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Emissions by vehicle type
        if 'vehicle_type' in total_data.columns:
            transport_emissions = total_data.groupby('vehicle_type')['transportation_emissions_kg'].mean().reset_index()
            transport_emissions = transport_emissions.sort_values('transportation_emissions_kg')
            
            fig = px.bar(
                transport_emissions,
                x='vehicle_type',
                y='transportation_emissions_kg',
                title='Average Emissions by Vehicle Type',
                labels={'vehicle_type': 'Vehicle Type', 'transportation_emissions_kg': 'Avg. Emissions (kg CO₂e)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Packaging analysis
    st.subheader("Packaging Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Emissions by packaging material
        if 'packaging_material' in total_data.columns:
            packaging_emissions = total_data.groupby('packaging_material')['packaging_emissions_kg'].mean().reset_index()
            packaging_emissions = packaging_emissions.sort_values('packaging_emissions_kg')
            
            fig = px.bar(
                packaging_emissions,
                x='packaging_material',
                y='packaging_emissions_kg',
                title='Average Emissions by Packaging Material',
                labels={'packaging_material': 'Packaging Material', 'packaging_emissions_kg': 'Avg. Emissions (kg CO₂e)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Waste emissions by disposal method
        if 'disposal_method' in total_data.columns:
            waste_emissions = total_data.groupby('disposal_method')['waste_emissions_kg'].mean().reset_index()
            waste_emissions = waste_emissions.sort_values('waste_emissions_kg')
            
            fig = px.bar(
                waste_emissions,
                x='disposal_method',
                y='waste_emissions_kg',
                title='Average Emissions by Disposal Method',
                labels={'disposal_method': 'Disposal Method', 'waste_emissions_kg': 'Avg. Emissions (kg CO₂e)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Optimization recommendations
    st.subheader("Optimization Recommendations")
    
    if dashboard.optimizer and hasattr(dashboard.optimizer, 'recommendations'):
        if 'comprehensive' in dashboard.optimizer.recommendations:
            comprehensive_rec = dashboard.optimizer.recommendations['comprehensive']
            
            # Top recommendations
            for i, recommendation in enumerate(comprehensive_rec['top_recommendations']):
                st.info(f"**Recommendation {i+1}:** {recommendation}")
            
            # Priority actions
            for i, action in enumerate(comprehensive_rec['priority_actions']):
                st.warning(f"**Action {i+1}:** {action}")
    
    # Forecasting
    st.subheader("Emissions Forecasting")
    
    if dashboard.forecaster:
        forecast_periods = st.slider("Forecast periods (days)", 30, 365, 90, 30)
        forecast_model = st.selectbox("Forecast model", ["Prophet", "LSTM"])
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                if forecast_model == "Prophet":
                    try:
                        dashboard.forecaster.train_prophet_model()
                        forecast = dashboard.forecaster.forecast_prophet(periods=forecast_periods)
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=dashboard.forecaster.daily_data['ds'],
                            y=dashboard.forecaster.daily_data['y'],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='black')
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat'],
                            mode='lines',
                            name='Prophet Forecast',
                            line=dict(color='blue')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0, 0, 255, 0.1)',
                            line=dict(color='rgba(255, 255, 255, 0)'),
                            name='95% Confidence Interval'
                        ))
                        
                        fig.update_layout(
                            title=f'Carbon Emissions Forecast ({forecast_periods} days)',
                            xaxis_title='Date',
                            yaxis_title='Emissions (kg CO₂e)',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating Prophet forecast: {e}")
                else:  # LSTM
                    try:
                        dashboard.forecaster.train_lstm_model(epochs=20)
                        forecast = dashboard.forecaster.forecast_lstm(periods=forecast_periods)
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=dashboard.forecaster.daily_data['ds'],
                            y=dashboard.forecaster.daily_data['y'],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='black')
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat'],
                            mode='lines',
                            name='LSTM Forecast',
                            line=dict(color='green')
                        ))
                        
                        fig.update_layout(
                            title=f'Carbon Emissions Forecast ({forecast_periods} days)',
                            xaxis_title='Date',
                            yaxis_title='Emissions (kg CO₂e)',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating LSTM forecast: {e}")

def integrate_enhanced_dashboard():
    """Instructions for integrating the enhanced dashboard functionality into the main dashboard.py file
    
    This function provides instructions on how to integrate the enhanced dashboard functionality
    into the main CarbonFootprintDashboard class. It is not meant to be called directly.
    
    Steps to integrate:
    1. Import the show_enhanced_analysis function in dashboard.py:
       from enhanced_dashboard import show_enhanced_analysis
       
    2. Add the show_enhanced_analysis method to the CarbonFootprintDashboard class:
       def show_enhanced_analysis(self):
           '''Show enhanced analysis using the new carbon footprint system'''
           from enhanced_dashboard import show_enhanced_analysis
           show_enhanced_analysis(self)
           
    3. Update the run method to include the enhanced analysis tab:
       - Add "Enhanced Analysis" to the tabs list
       - Add a new tab section that calls self.show_enhanced_analysis()
    
    Example implementation for the run method:
    def run(self):
        '''Run the dashboard'''
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
            
            with tabs[2]:
                self.show_enhanced_analysis()
        else:
            st.warning("Please load or generate data to view the analytics.")
    """
    pass