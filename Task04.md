# README: Stock Price Prediction Project

## Project Overview
The Stock Price Prediction Project aims to develop a robust machine learning model capable of predicting a stock's closing price 5 trading days into the future. This project combines advanced data preprocessing, comprehensive exploratory data analysis (EDA), feature engineering, model selection, and evaluation to create a predictive model with real-world trading applicability.

---

## Approach
1. **Data Preprocessing:**
   - Handled missing values using median and mode imputation techniques.
   - Capped outliers using the Interquartile Range (IQR) method.
   - Replaced negative values in financial data to maintain data integrity.
   - Standardized and scaled features using MinMaxScaler.
   - Performed dimensionality reduction using PCA (Principal Component Analysis).

2. **Exploratory Data Analysis (EDA):**
   - Visualized historical stock price trends using line plots and moving averages.
   - Analyzed feature distributions with histograms and boxplots.
   - Evaluated feature correlations using a heatmap to guide feature selection.
   - Detected anomalies and seasonal trends with advanced visualization techniques.

3. **Feature Engineering:**
   - Generated new features including Daily Return, Volatility, Price Range, and Day of the Week.
   - Implemented rolling window statistics to capture market trends.

4. **Model Selection:**
   - Tested four models: RandomForest, XGBoost, LightGBM, and Linear Regression.
   - Evaluated models using RMSE, MAE, R² Score, and Directional Accuracy.
   - Selected **LightGBM** as the final model due to its superior performance and adaptability to time series data.

5. **Model Evaluation:**
   - Achieved the best performance with **LightGBM**, showing the lowest RMSE (1.15) and highest R² Score (0.88).
   - Directional accuracy of 73.1%, indicating strong predictive power for trading strategies.

6. **Simulated Trading Performance:**
   - Evaluated the model's practical value by simulating trading decisions based on predicted price directions.
   - Measured directional accuracy as a critical metric for real-world trading applicability.

---

## Key Findings
- **LightGBM** outperformed other models in both predictive accuracy and trading performance metrics.
- The integration of advanced feature engineering significantly improved model accuracy.
- Identified key predictive features such as Daily Return, Volatility, and Moving Averages.
- The model showed resilience in identifying market trends and adapting to seasonal patterns.

---

## Instructions to Reproduce Results
### 1. Environment Setup
- Python 3.8+ is recommended.
- Required Libraries:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

### 2. Dataset Preparation
- Ensure the historical stock price dataset is in the `/data` directory as `stock_data.csv`.
- The dataset should contain columns: `['Date', 'Open', 'High', 'Low', 'Close', 'Volume']`.

### 3. Run the Jupyter Notebook
- Execute the notebook `Stock_Price_Prediction.ipynb` step-by-step.
- Ensure all cells are run sequentially to avoid errors in variable definitions.

### 4. Generate Predictions
- The notebook will output a CSV file named `stock_predictions.csv` containing predicted and actual stock prices.
- The output will be saved in the `/output` directory.

### 5. Model Retraining (Optional)
- To retrain the model with updated data, replace the dataset and rerun the notebook.
- Hyperparameter tuning can be implemented using GridSearchCV for model improvement.

---

## Additional Recommendations
- Extend the model by integrating external macroeconomic data and sentiment analysis.
- Experiment with advanced models such as LSTM (Long Short-Term Memory) networks for time series forecasting.
- Implement a live trading simulation using the model predictions to validate real-world performance.



