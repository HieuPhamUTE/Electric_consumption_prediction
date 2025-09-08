# ⚡ Electric Consumption Prediction

## 👥 Team Members
- **Pham Ngoc Hieu** – 2430904  
- **Nguyen Nhat Phap** – 2430911  
- **Instructor**: Ph.D. Bui Ha Duc – HCMUTE  

---

## 📌 Project Overview
This project focuses on forecasting **household electricity consumption** using the **London Smart Meters dataset**.  
We benchmark two advanced deep learning architectures:  

- **LSTM (Long Short-Term Memory)** – Specialized recurrent neural network for sequential data.  
- **TSMixer** – A modern architecture for capturing complex temporal dependencies.  

### 🔑 Features
- **Deep Learning Models**: LSTM and TSMixer implemented via **Darts (PyTorch backend)**  
- **Data Processing**: Automated pipeline for missing value imputation, feature engineering, and scaling  
- **Exploratory Data Analysis**: Detect patterns, seasonality, and correlations in energy data  
- **Evaluation Metrics**: MSE, RMSE, MAE, and Forecast Bias  
- **Scalable Design**: Works across multiple households with diverse consumption behaviors  

### 📂 Data Sources
Dataset: **London Smart Meters**  
🔗 https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london  

- 5,567 London households  
- Half-hourly readings (Nov 2011 – Feb 2014)  
- Household metadata (ACORN groups)  
- Weather variables (temperature, humidity, etc.)  
- UK public holidays  

### 🛠️ Technologies
- **Python**, **Pandas**, **NumPy**, **Scikit-learn**  
- **PyTorch & Darts** (deep learning forecasting)  
- **Matplotlib & Plotly** (visualization)  

### 📊 Key Findings
- **TSMixer**: Best accuracy (**MSE ≈ 0.0008**) but introduced higher systematic bias  
- **LSTM**: Provided more stable forecasts across households  
- **Insights**: Household diversity and extreme events strongly affect forecasting performance  

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/HieuPhamUTE/Electric_consumption_prediction

# Install dependencies
pip install pandas numpy scikit-learn plotly missingno tqdm
pip install darts torch
pip install jupyter ipywidgets

Result prediction between LSTM and TSMixer
<img width="2993" height="1763" alt="tsmixer_matplotlib" src="https://github.com/user-attachments/assets/d940623a-82b2-42f3-bedf-d90f76d3fe0b" />
<img width="2993" height="1763" alt="lstm_matplotlib" src="https://github.com/user-attachments/assets/f18da8d9-b34f-4dd2-b2d2-ae7bd8a8436f" />


