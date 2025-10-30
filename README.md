# Bank Loan Portfolio Forecasting System

A professional, interactive dashboard for time series forecasting and ML-based default risk prediction on bank loan data.

## Features
- Upload & clean your loan portfolio dataset (or use included sample)
- View summary stats, heatmaps, trends, and feature distributions
- Forecast future portfolio metrics with auto-configured ARIMA models
- Predict loan default risks with logistic regression (selectable features)
- Integrated dashboard: Portfolio KPIs, default risk factors, and chart downloads
- Export reports/results as CSV or PDF

## Setup
1. Clone/download the repository.
2. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```
4. [Optional] Replace `loan_data.csv` with your own data.

## Usage
- Use the sidebar to navigate between modules.
- Upload your CSV file or use the sample dataset.
- Explore EDA, run forecasts, and view model insights and KPIs.
- Export/download reports as needed.

## Tech Stack
- Streamlit, Plotly, Altair, scikit-learn, statsmodels, pandas, numpy, seaborn, fpdf

## Contact/Support
For queries or feedback, please contact [your-email-here].


