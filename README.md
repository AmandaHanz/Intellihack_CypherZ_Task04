# 📈 **Stock Price Prediction Project**  

---

## 📝 **Project Overview**  

The **Stock Price Prediction Project** is designed to create a machine learning model that predicts a stock's closing price **5 trading days** into the future. The project uses techniques like **data preprocessing**, **exploratory data analysis (EDA)**, **feature engineering**, **model selection**, and **evaluation** to build a practical and accurate predictive model.

---

## 🚀 **Approach**  

### 1. **Data Preprocessing:**  
- Handled missing values with **median** and **mode imputation**.  
- Removed outliers using the **IQR method**.  
- Ensured data integrity by correcting negative values.  
- Scaled features using **MinMaxScaler**.  
- Reduced dimensionality with **Principal Component Analysis (PCA)**.  

### 2. **Exploratory Data Analysis (EDA):**  
- Visualized stock price trends with **line plots** and **moving averages**.  
- Analyzed data distributions using **histograms** and **boxplots**.  
- Evaluated feature correlations with a **heatmap**.  
- Detected market trends and anomalies with **advanced visualizations**.  

### 3. **Feature Engineering:**  
- Created new features such as **Daily Return**, **Volatility**, **Price Range**, and **Day of the Week**.  
- Applied **rolling window statistics** to capture market trends.  

### 4. **Model Selection:**  
- Tested four models: **RandomForest**, **XGBoost**, **LightGBM**, **Linear Regression**.  
- Selected **LightGBM** for its **high accuracy** and **adaptability** to time series data.  

### 5. **Model Evaluation:**  
- Achieved **RMSE: 1.15**, **R² Score: 0.88**, and **Directional Accuracy: 73.1%**.  
- Simulated trading performance using **predicted price directions**.  

---

## 🔑 **Key Findings:**  

- **LightGBM** performed best in both prediction accuracy and trading simulation.  
- **Feature engineering** significantly improved model performance.  
- Key predictive features: **Daily Return**, **Volatility**, **Moving Averages**.  
- The model effectively adapted to **market trends** and **seasonal patterns**.  

---

## 🛠️ **Instructions to Reproduce Results:**  

### 1. **Setup Environment:**  
- Recommended: **Python 3.8+**  
- Install required libraries:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

### 2. **Prepare Dataset:**  
- Ensure the dataset is saved as **stock_data.csv** in the **/data** directory.  
- Required columns: `['Date', 'Open', 'High', 'Low', 'Close', 'Volume']`.  

### 3. **Run the Jupyter Notebook:**  
- Open and execute the **Stock_Price_Prediction.ipynb** file step-by-step.  
- Run all cells sequentially to avoid errors.  

### 4. **Generate Predictions:**  
- The notebook will generate a **stock_predictions.csv** file.  
- The output will be saved in the **/output** directory.  

### 5. **Optional: Model Retraining:**  
- To retrain with new data, update the dataset and rerun the notebook.  
- Use **GridSearchCV** for **hyperparameter tuning** to improve the model.  

---

## 💡 **Additional Recommendations:**  

- Integrate **macroeconomic data** and **sentiment analysis** to enhance predictions.  
- Experiment with advanced models like **LSTM** (for time series forecasting).  
- Develop a **live trading simulation** to test the model's real-world performance.  

---

## 🎯 **Conclusion:**  

The **Stock Price Prediction Project** demonstrates how machine learning can provide valuable insights for trading strategies. The **LightGBM model**'s strong performance in both accuracy and trading simulation indicates its potential for practical financial applications. This project sets the foundation for future improvements, including the integration of external data sources and the development of real-time trading tools.
