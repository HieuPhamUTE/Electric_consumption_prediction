Electric_consumption_prediction
Team Members
Student 1 - Pham Ngoc Hieu - 2430904
Student 2 - Nguyen Nhat Phap - 2430911
Instructor: Ph.D Bui Ha Duc - HCMUTE

This project focuses on forecasting household energy consumption using the London Smart Meters dataset. The research evaluates two state-of-the-art deep learning architectures — LSTM and TSMixer — to determine the most effective approach for residential energy consumption prediction.

Features

Deep Learning Models:

LSTM (Long Short-Term Memory) for capturing sequential dependencies in time series.

TSMixer for modeling complex temporal structures with high accuracy.

Data Processing: Automated data preparation pipeline with missing value imputation and feature engineering.

Exploratory Data Analysis: Visualization of energy consumption patterns, seasonality, and correlations.

Performance Evaluation: Metrics include MSE, RMSE, MAE, and Forecast Bias for model comparison.

Scalable Architecture: Designed to handle multiple households with diverse consumption patterns.

Installation
# Clone this repository
git clone <https://github.com/HieuPhamUTE/Electric_consumption_prediction>

# Create a virtual environment and install dependencies
pip install pandas numpy scikit-learn plotly missingno tqdm
pip install darts torch
pip install jupyter ipywidgets


London Smart Meters dataset from Kaggle: https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london

Usage

Run the data preparation pipeline: jupyter notebook data_preparation.ipynb

Explore dataset characteristics: jupyter notebook eda.ipynb

Train and evaluate models:

dl_lstm.ipynb for LSTM

dl_tsmixer.ipynb for TSMixer

Data Sources

The London Smart Meters dataset from the UK Power Networks-led Low Carbon London project, which contains:

Energy consumption readings for 5,567 London households

Half-hourly measurements from November 2011 to February 2014

Household metadata including socio-demographic classifications (ACORN groups)

Weather data enrichment with temperature, humidity, and other meteorological variables

UK bank holiday information

Technologies Used

Python, Pandas, NumPy, Scikit-learn

PyTorch & Darts (deep learning forecasting)

Plotly, Matplotlib (visualizations)

Project Report

The project report provides detailed analysis of methodology, experimental setup, and results. Key findings include:

TSMixer Superiority: Achieved the best accuracy metrics (MSE: 0.0008) but showed higher systematic bias.

LSTM Robustness: Provided stable forecasts with lower variance across households.

Practical Insights: Different household consumption patterns and extreme events impact model performance differently.
