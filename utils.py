import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from sklearn.model_selection import train_test_split

"""
Utility functions for data loading, preprocessing, summary/statistics, and visualization.
"""

def load_data(uploaded_file=None):
    """Loads loan data from uploaded file or bundled sample CSV."""
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv('loan_data.csv')
    except Exception as e:
        return pd.DataFrame(), f"Error loading data: {e}"
    return df, None

def clean_data(df):
    """Fill missing values and clip outliers for numerical columns."""
    df = df.copy()
    # Numeric missing values
    for col in df.select_dtypes(include=['float','int']).columns:
        df[col] = df[col].fillna(df[col].median())
    # Categorical missing values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')
    # Clip outlier values to [P1,P99]
    for col in ['Loan_Amount','Interest_Rate','Tenure','Credit_Score','Income']:
        if col in df:
            df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    return df

def encode_categoricals(df, drop=[]):
    """One-hot encode categorical columns except for the columns in drop."""
    cat_cols = df.select_dtypes(include='object').columns.difference(drop)
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

def get_summary_stats(df):
    """Return basic descriptive statistics for numerical columns."""
    return df.describe().T

def plot_corr_heatmap(df):
    """Plot a correlation heatmap using seaborn. Returns matplotlib figure."""
    fig, ax = plt.subplots(figsize=(8,5))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='Blues', ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    return fig

def plot_distribution(df, col):
    """Plot a feature distribution using Plotly, fallback to Altair if fails."""
    try:
        fig = px.histogram(df, x=col, nbins=25, title=f'{col} Distribution', color_discrete_sequence=['#273c75'])
    except Exception:
        base = alt.Chart(df).mark_bar().encode(x=col, y='count()', tooltip=[col])
        fig = base
    return fig

def plot_time_trend(df, col, date_col='Date_of_Issue'):
    temp = df.copy()
    try:
        temp[date_col] = pd.to_datetime(temp[date_col])
        temp = temp.groupby(pd.Grouper(key=date_col, freq='M')).agg({col: 'sum'}).reset_index()
        fig = px.line(temp, x=date_col, y=col, title=f'Time Trend of {col}', markers=True, color_discrete_sequence=['#273c75'])
    except Exception:
        fig = None
    return fig

def kpi_cards(df):
    """Quick compute of main loan KPIs for dashboard cards."""
    total_loans = df['Loan_Amount'].sum() if 'Loan_Amount' in df else 0
    try:
        loan_growth = (df['Loan_Amount'].iloc[-1] - df['Loan_Amount'].iloc[0]) / df['Loan_Amount'].iloc[0] * 100
    except:
        loan_growth = 0
    avg_interest = df['Interest_Rate'].mean() if 'Interest_Rate' in df else 0
    default_ratio = df['Loan_Status'].mean()*100 if 'Loan_Status' in df else 0
    return total_loans, loan_growth, avg_interest, default_ratio
