# AI-Powered Carbon Footprint Analytics for E-commerce

## Overview
This project implements an AI-powered system for analyzing and optimizing carbon footprint in e-commerce operations. It provides data-driven insights and predictions to help businesses reduce their environmental impact while maintaining operational efficiency.

## Features
- **Data Generation**: Synthetic data generation for e-commerce shipping and carbon footprint analysis
- **Advanced Analytics**: Comprehensive data analysis and visualization of carbon impact factors
- **Machine Learning Models**:
  - Time Series Forecasting: Facebook Prophet for seasonal pattern analysis
  - Deep Learning: LSTM neural networks for complex pattern recognition
- **Dashboard**: Interactive Streamlit interface for real-time analytics
- **Optimization**: Actionable recommendations for reducing carbon emissions

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-repo/project.git
cd project
```
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit dashboard:

```bash
cd src
streamlit run dashboard.py
```

2. Use the dashboard to:
   - Generate or upload shipping data
   - View carbon footprint analytics
   - Get predictions and optimization recommendations

## Project Structure

```
.
├── README.md
├── requirements.txt
└── src/
    ├── data_collection.py    # Data collection and synthetic data generation
    ├── data_analysis.py      # Data analysis and visualization
    ├── ml_models.py          # Machine learning models for prediction
    └── dashboard.py          # Streamlit dashboard implementation
```

## Data Analysis

The system analyzes various factors affecting carbon footprint:
- Shipping methods (Ground, Air, Sea)
- Package weights and distances
- Packaging types and materials
- Temporal patterns and trends

## Machine Learning Models

1. Prophet Model:
   - Time series forecasting
   - Captures seasonal patterns
   - Handles missing data

2. LSTM Model:
   - Deep learning for complex patterns
   - Multi-factor analysis
   - Robust to noise

## Dashboard Features

- Real-time carbon footprint metrics
- Interactive visualizations
- Predictive analytics
- Optimization recommendations
- Data upload and synthetic data generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.