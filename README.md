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
git clone https://github.com/HieuPhamUTE/Electric_consumption_prediction

# Install dependencies
pip install pandas numpy scikit-learn plotly missingno tqdm
pip install darts torch
pip install jupyter ipywidgets

---

## Data Sources
The London Smart Meters dataset from the UK:  
🔗 https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london  

- 5,567 London households  
- Half-hourly energy readings (Nov 2011 – Feb 2014)  
- Household metadata (ACORN groups)  
- Weather data (temperature, humidity, etc.)  
- UK bank holiday information  

---

## Technologies Used
- **Python**, **Pandas**, **NumPy**, **Scikit-learn**  
- **PyTorch & Darts** for deep learning forecasting  
- **Matplotlib & Plotly** for visualization  

---

## Key Findings
- **TSMixer Superiority**: Best accuracy (MSE ≈ 0.0008), but exhibited higher systematic bias.  
- **LSTM Robustness**: Provided more stable forecasts across households.  
- **Insights**: Household diversity and extreme events significantly affect prediction accuracy.  

