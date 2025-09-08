# Electric Consumption Prediction

## Team Members
- **Student 1**: Pham Ngoc Hieu - 2430904  
- **Student 2**: Nguyen Nhat Phap - 2430911  
- **Instructor**: Ph.D Bui Ha Duc - HCMUTE  

---

## Project Overview
This project focuses on forecasting **household energy consumption** using the **London Smart Meters dataset**.  
We evaluate two state-of-the-art deep learning architectures:  

- **LSTM (Long Short-Term Memory)**: Specialized recurrent neural network for sequential data.  
- **TSMixer**: Modern deep learning model designed for capturing complex temporal dependencies.  

The goal is to identify which architecture is more effective for residential energy consumption forecasting.

---

## Features
- **Deep Learning Models**: LSTM and TSMixer implemented with Darts (PyTorch backend).  
- **Data Processing**: Automated pipeline for missing value imputation, feature engineering, and scaling.  
- **Exploratory Data Analysis**: Visualization of patterns, seasonality, and correlations in energy data.  
- **Performance Evaluation**: Metrics include MSE, RMSE, MAE, and Forecast Bias.  
- **Scalable Architecture**: Can handle multiple households with diverse consumption behaviors.  

---

## Installation
```bash
# Clone this repository
git clone https://github.com/HienNguyen2311/london-energy-forecasting.git
cd london_smart_meters

# Install dependencies
pip install pandas numpy scikit-learn plotly missingno tqdm
pip install darts torch
pip install jupyter ipywidgets
