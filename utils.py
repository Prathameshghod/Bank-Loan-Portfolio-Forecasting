import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
Utility functions for enhanced, professional visualization and analytics.
"""

PLOTLY_DARK = dict(
    template="plotly_dark",
    marker=dict(color="#2980b9"),
    line=dict(color="#2980b9", width=4),
    font=dict(family="Segoe UI, Arial", size=15)
)

PLOTLY_BAR = dict(color="#40739e")

# --- DATA LOAD/CLEAN UNCHANGED ---
def load_data(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv('loan_data.csv')
    except Exception as e:
        return pd.DataFrame(), f"Error loading data: {e}"
    return df, None

def clean_data(df):
    df = df.copy()
    for col in df.select_dtypes(include=['float','int']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')
    for col in ['Loan_Amount','Interest_Rate','Tenure','Credit_Score','Income']:
        if col in df:
            df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    return df

def encode_categoricals(df, drop=[]):
    cat_cols = df.select_dtypes(include='object').columns.difference(drop)
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

def get_summary_stats(df):
    return df.describe().T

def plot_corr_heatmap(df):
    # Heatmap stays with seaborn/matplotlib for now
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='Blues', ax=ax, fmt=".2f", linewidths=0.5)
    ax.set_title('Correlation Heatmap', pad=11, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig

def plot_distribution(df, col):
    fig = px.histogram(
        df, x=col, nbins=12,
        title=f'{col} Distribution',
        color_discrete_sequence=[PLOTLY_BAR['color']],
        template="plotly_dark",
        opacity=0.8
    )
    fig.update_layout(
        bargap=0.15,
        plot_bgcolor="#191c24",
        paper_bgcolor="#191c24",
        font=dict(color="#ecf0f1", size=15),
        title=dict(x=0.5, font_size=18),
        xaxis_title=col,
        yaxis_title='Count',
        margin=dict(l=20, r=20, t=60, b=30),
        height=260
    )
    fig.update_traces(marker_color='#40739e', marker_line_color='white', marker_line_width=1)
    return fig

def plot_time_trend(df, col, date_col='Date_of_Issue'):
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])
    temp = temp.groupby(pd.Grouper(key=date_col, freq='M')).agg({col: 'sum'}).reset_index()
    fig = px.line(
        temp, x=date_col, y=col, title=f'{col} Trend Over Time',
        markers=True,
        template="plotly_dark",
        color_discrete_sequence=[PLOTLY_BAR['color']]
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=col,
        plot_bgcolor="#191c24",
        paper_bgcolor="#191c24",
        font=dict(color="#ecf0f1", size=15),
        title=dict(x=0.5, font_size=18),
        margin=dict(l=25, r=25, t=55, b=35),
        height=300,
        showlegend=False
    )
    fig.update_traces(line_width=4, marker_size=8)
    return fig

def kpi_cards(df):
    total_loans = df['Loan_Amount'].sum() if 'Loan_Amount' in df else 0
    try:
        loan_growth = (df['Loan_Amount'].iloc[-1] - df['Loan_Amount'].iloc[0]) / df['Loan_Amount'].iloc[0] * 100
    except:
        loan_growth = 0
    avg_interest = df['Interest_Rate'].mean() if 'Interest_Rate' in df else 0
    default_ratio = df['Loan_Status'].mean()*100 if 'Loan_Status' in df else 0
    return total_loans, loan_growth, avg_interest, default_ratio

def explain_time_series(df, x_col, y_col, title, percent=False, currency=False):
	"""Generate a short, simple explanation for a time series chart, with its business use."""
	try:
		d = df[[x_col, y_col]].dropna().copy()
		d[x_col] = pd.to_datetime(d[x_col])
		d = d.sort_values(x_col)
		start, end = d[x_col].min(), d[x_col].max()
		start_s, end_s = start.strftime('%b %Y'), end.strftime('%b %Y')
		y0, yN = float(d.iloc[0][y_col]), float(d.iloc[-1][y_col])
		ymin_row = d.loc[d[y_col].idxmin()]
		ymax_row = d.loc[d[y_col].idxmax()]
		ymin_v, ymax_v = float(ymin_row[y_col]), float(ymax_row[y_col])
		ymin_t, ymax_t = ymin_row[x_col].strftime('%b %Y'), ymax_row[x_col].strftime('%b %Y')
		trend = 'increasing' if yN > y0 else ('decreasing' if yN < y0 else 'stable')
		def fmt(v):
			if percent:
				return f"{v*100:.1f}%" if v <= 1.0 else f"{v:.1f}%"
			if currency:
				return f"₹{v:,.0f}"
			return f"{v:,.2f}"
		use_line = f"- Use: Track {y_col} over time to spot growth/decline, seasonality, and anomalies for planning and early alerts."
		text = (
			f"- Title: {title}\n"
			f"- X-axis: Months ({start_s} → {end_s})\n"
			f"- Y-axis: {y_col}{' (%)' if percent else ''}\n"
			f"- Change: starts at {fmt(y0)}, ends at {fmt(yN)} — trend is {trend}.\n"
			f"- Peak: {fmt(ymax_v)} in {ymax_t}; Lowest: {fmt(ymin_v)} in {ymin_t}.\n"
			f"{use_line}"
		)
	except Exception:
		text = f"- Title: {title}\n- Use: Time-series view to monitor trend and seasonality."
	return text

def explain_histogram(series, title):
	s = pd.Series(series).dropna()
	if s.empty:
		return f"- Title: {title}\n- Use: Understand distribution; no data available."
	mean_v, med_v = s.mean(), s.median()
	p25, p75 = s.quantile(0.25), s.quantile(0.75)
	return (
		f"- Title: {title}\n"
		f"- Center: mean ≈ {mean_v:,.2f}, median ≈ {med_v:,.2f}\n"
		f"- Spread: 25th–75th percentile ≈ {p25:,.2f}–{p75:,.2f}\n"
		f"- Use: Shows shape and typical range, helps find outliers and set risk thresholds."
	)
