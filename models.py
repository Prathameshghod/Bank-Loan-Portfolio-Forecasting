import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

"""
ML functions: ARIMA/pmdarima forecasting, Logistic Regression Classification, model diagnostics.
"""

# ARIMA Forecast: Try pmdarima (auto_arima) fallback to statsmodels ARIMA
try:
    from pmdarima import auto_arima
    HAVE_PM = True
except ImportError:
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_PM = False
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def arima_forecast(df, target_col, n_periods=6):
    """Forecast loan trends using ARIMA or auto_arima. Returns historical TS, forecast, fitted model, residuals, stationarity."""
    series = df.copy()
    series['Date_of_Issue'] = pd.to_datetime(series['Date_of_Issue'])
    ts = series.groupby(pd.Grouper(key='Date_of_Issue', freq='M'))[target_col].sum()
    ts = ts.asfreq('M').fillna(0)
    result = adfuller(ts)
    is_stationary = result[1] < 0.05
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if HAVE_PM:
            model = auto_arima(ts, seasonal=False, stepwise=True, suppress_warnings=True)
            forecast = model.predict(n_periods=n_periods, return_conf_int=True)
            pred, conf = forecast
            forecast_df = pd.DataFrame({
                'Forecast': pred,
                'Lower CI': conf[:, 0],
                'Upper CI': conf[:, 1]
            })
        else:
            smodel = ARIMA(ts, order=(1,1,1))
            fit = smodel.fit()
            forecast = fit.get_forecast(steps=n_periods)
            forecast_df = pd.DataFrame({
                'Forecast': forecast.predicted_mean,
                'Lower CI': forecast.conf_int().iloc[:,0],
                'Upper CI': forecast.conf_int().iloc[:,1]
            })
            pred = forecast.predicted_mean
            conf = forecast.conf_int()
    forecast_df['Date'] = pd.date_range(ts.index[-1]+pd.offsets.MonthBegin(1), periods=n_periods, freq='M')
    resid = ts - ts.mean() if not HAVE_PM else pd.Series(np.zeros_like(ts))
    return ts, forecast_df, None, resid, is_stationary

def arima_diagnostics(ts, resid):
    """Return matplotlib diagnostic plots for time series and residuals."""
    fig, axs = plt.subplots(2,2, figsize=(12,8))
    ts.plot(ax=axs[0,0]); axs[0,0].set_title('Time Series')
    axs[0,1].hist(resid, bins=20, color='#0066CC'); axs[0,1].set_title('Residuals')
    plot_acf(resid, ax=axs[1,0])
    plot_pacf(resid, ax=axs[1,1])
    plt.tight_layout()
    return fig

def logistic_regression_classification(df, features, target, test_size=0.25, random_state=42):
    """Train/test a logistic regression default prediction model with selected features.
    - One-hot encode categoricals
    - Standardize features for stable coefficients
    - Use class_weight='balanced' for imbalanced targets
    Returns model and evaluation artifacts plus coefficients and odds ratios.
    """
    X = df[features]
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    rocauc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    coef = model.coef_[0]
    odds_ratio = np.exp(coef)
    return model, cm, report, rocauc, fpr, tpr, coef, X_train.columns, odds_ratio
