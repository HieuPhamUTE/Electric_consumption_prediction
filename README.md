⚡ Electric Consumption Prediction
👥 Team Members

Pham Ngoc Hieu – 2430904

Nguyen Nhat Phap – 2430911

Instructor: Ph.D. Bui Ha Duc – HCMUTE

📌 Project Overview

This project aims to forecast household electricity consumption using the London Smart Meters dataset.
We benchmark two advanced deep learning models:

LSTM (Long Short-Term Memory) – A recurrent neural network well-suited for sequential data.

TSMixer – A recent architecture designed to capture complex temporal dependencies.

The objective is to determine which model provides more accurate and reliable predictions for household energy usage.

🔑 Features

Deep Learning Models: LSTM and TSMixer implemented via Darts (PyTorch backend).

Data Processing: Automated pipeline including missing value imputation, feature engineering, and scaling.

Exploratory Data Analysis: Identification of consumption patterns, seasonality, and correlations.

Evaluation Metrics: MSE, RMSE, MAE, and Forecast Bias.

Scalable Design: Adaptable for large-scale multi-household energy forecasting.

⚙️ Installation
# Clone the repository
git clone https://github.com/HieuPhamUTE/Electric_consumption_prediction

# Install dependencies
pip install pandas numpy scikit-learn plotly missingno tqdm
pip install darts torch
pip install jupyter ipywidgets

📂 Data Sources

We use the London Smart Meters dataset:
🔗 https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london

5,567 households in London

Half-hourly electricity readings (Nov 2011 – Feb 2014)

Household metadata (ACORN groups)

Weather data (temperature, humidity, etc.)

UK public holidays

🛠️ Technologies

Programming & Data: Python, Pandas, NumPy, Scikit-learn

Deep Learning: PyTorch, Darts

Visualization: Matplotlib, Plotly

📊 Key Findings

TSMixer achieved the best accuracy (MSE ≈ 0.0008), though results showed higher systematic bias.

LSTM produced more stable predictions across diverse households.

Insights: Household heterogeneity and extreme weather/events significantly impact forecasting performance.
