import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from utils import load_data, clean_data, get_summary_stats, plot_corr_heatmap, plot_distribution, plot_time_trend, kpi_cards, encode_categoricals
from models import arima_forecast, arima_diagnostics, logistic_regression_classification
from reporting import export_csv, export_pdf
import base64

# --- THEME/STYLE ---
st.set_page_config(page_title='Bank Loan Portfolio Forecasting System', page_icon='🏦', layout='wide')

# APP STYLE
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .stApp {background-color: #1a1c23;}
    .css-1d391kg, .css-1v0mbdj { background: #273359; }
    .metric { color: #002d4a;}
    </style>
    """, unsafe_allow_html=True)

# SIDEBAR: Navigation
with st.sidebar:
    st.title('Navigation')
    menu = st.radio('', ['Home', 'EDA', 'ARIMA Forecasting', 'Default Prediction', 'Combined Dashboard'], key='nav_radio')
    st.markdown('---')
    st.markdown(
      '<span style="font-size:13px">Upload CSV Dataset</span>',
      unsafe_allow_html=True
    )
    upload_file = st.file_uploader(' ', type=['csv'])
    st.markdown('---')

# DATA LOADING
with st.spinner('Loading and cleaning data...'):
    df, loaderr = load_data(upload_file)
    if loaderr or df is None or df.empty:
        df = pd.read_csv('loan_data.csv')
        st.warning(loaderr or 'Loaded bundled sample dataset.')
    df = clean_data(df)
# --- HOME PAGE ---
if menu == 'Home':
    st.title('🏦 Bank Loan Portfolio Forecasting System')
    c1, c2 = st.columns([2,3])
    with c1:
        st.subheader('Advanced Forecasting & Insights Tool')
        st.write('This dashboard lets you analyze, forecast, and predict key financial indicators for your loan portfolio using interactive EDA, ARIMA-based time series forecasting and machine learning defaults prediction.')
        st.write('Navigate using the sidebar.')
        st.markdown('---')
        st.success('''**Key modules:**\n- EDA: Data analysis\n- ARIMA Forecast: Loan trends\n- Default ML: Default Risk\n- Dashboard: Portfolio KPIs and exports''')
    with c2:
        st.image('https://img.freepik.com/free-vector/bank-loan-concept-illustration_114360-13212.jpg', width=450)

# --- EDA ---
if menu == 'EDA':
    st.title('🔍 Exploratory Data Analysis')
    st.divider()
    st.write('### Data Summary Statistics')
    st.dataframe(get_summary_stats(df))
    st.divider()
    st.write('### Correlation Heatmap')
    st.pyplot(plot_corr_heatmap(df))
    col_choice = st.multiselect('Select columns for distribution:', [x for x in ['Loan_Amount','Interest_Rate','Income','Credit_Score'] if x in df], default=['Loan_Amount','Credit_Score'])
    for col in col_choice:
        st.plotly_chart(plot_distribution(df, col),use_container_width=True)
    st.divider()
    st.write('### Loan Amount Trend')
    st.plotly_chart(plot_time_trend(df, 'Loan_Amount'), use_container_width=True)
    st.write('### NPA/Default Monthly Trend')
    try:
        npa_trend = df.groupby(pd.to_datetime(df['Date_of_Issue']).dt.to_period('M'))['Loan_Status'].mean()
        st.line_chart(npa_trend)
    except: st.warning("Can't plot NPA trend")

# --- ARIMA FORECAST ---
if menu == 'ARIMA Forecasting':
    st.title('📈 ARIMA Loan Portfolio Forecasting')
    st.divider()
    st.write('Forecast disbursed loan trends using ARIMA (auto-configured if supported).')
    tcols = [c for c in ['Loan_Amount', 'Interest_Rate', 'Credit_Score'] if c in df]
    if not tcols:
        st.error('Required columns not in data!')
    else:
        target_col = st.selectbox('Target Variable', tcols)
        periods = st.slider('Forecast Periods (months)',3,24,6)
        ts, forecast_df, _, resid, is_stationary = arima_forecast(df, target_col, periods)
        st.caption(f"Data Stationary: {'Yes' if is_stationary else 'No'}")
        diag = arima_diagnostics(ts, resid)
        st.pyplot(diag)
        st.dataframe(forecast_df)
        # Plot actual vs forecast
        fig = px.line(x=list(ts.index)+list(forecast_df['Date']), y=list(ts.values)+list(forecast_df['Forecast']), labels={'x':'Date','y':target_col}, title=f'{target_col} Actual & Forecast')
        st.plotly_chart(fig, use_container_width=True)
        st.download_button('Download Forecast (CSV)',forecast_df.to_csv(index=False),file_name='forecast.csv')

# --- DEFAULT PREDICTION ---
if menu == 'Default Prediction':
    st.title('🧠 Loan Default ML Prediction')
    st.write('Trains a logistic regression model for default risk.')
    ml_cols = [x for x in df.columns if x not in ['Customer_ID', 'Loan_Status', 'Date_of_Issue', 'Region']]
    if 'Loan_Status' not in df:
        st.error('No Loan_Status (target) column in data!')
    else:
        sel_features = st.multiselect('Select Features', ml_cols, default=['Loan_Amount','Interest_Rate','Credit_Score','Income'])
        if sel_features:
            with st.spinner('Training ML model and calculating metrics...'):
                model, cm, report, rocauc, fpr, tpr, importance, feat_names = logistic_regression_classification(df, sel_features, 'Loan_Status')
                st.write('#### Confusion Matrix')
                st.write(cm)
                st.write('#### Classification Report')
                st.json(report)
                st.metric(label="ROC-AUC", value=f"{rocauc:.3f}")
                st.line_chart(pd.DataFrame({'FPR':fpr, 'TPR':tpr}))
                st.write('#### Feature Importance')
                imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importance})
                st.bar_chart(imp_df.set_index('Feature'))
# --- COMBINED DASHBOARD ---
if menu == 'Combined Dashboard':
    st.title('📊 Loan Portfolio Overview Dashboard')
    k1, k2, k3, k4 = st.columns(4)
    tl, lg, ai, dr = kpi_cards(df)
    k1.metric('Total Disbursed', f"₹{tl:,.0f}")
    k2.metric('Loan Growth (%)', f"{lg:.2f}")
    k3.metric('Avg Interest Rate', f"{ai:.2f}%")
    k4.metric('Default Ratio (%)', f"{dr:.2f}")
    st.divider()
    st.write('### Portfolio Loan Amount Forecast')
    target_col = 'Loan_Amount' if 'Loan_Amount' in df else df.columns[df.select_dtypes(include=['float','int']).columns[0]]
    ts, forecast_df, _, _, _ = arima_forecast(df, target_col)
    st.plotly_chart(px.line(x=list(ts.index)+list(forecast_df['Date']), y=list(ts.values)+list(forecast_df['Forecast']), labels={'x': 'Date', 'y': target_col}, title='Actual vs Forecast', color_discrete_sequence=['#40739e']), use_container_width=True)
    st.write('### Default Feature Importance')
    try:
        _,_,_,_,_,_,importance,feat_names = logistic_regression_classification(df, ['Loan_Amount','Credit_Score','Interest_Rate','Income'], 'Loan_Status')
        st.bar_chart(pd.DataFrame({'Feature': feat_names, 'Importance': importance}).set_index('Feature'))
    except: st.warning('Default ML could not run.')
    st.write('### NPA/Default Monthly Trend')
    try:
        npa_trend = df.groupby(pd.to_datetime(df['Date_of_Issue']).dt.to_period('M'))['Loan_Status'].mean()
        st.area_chart(npa_trend)
    except: pass
    st.divider()
    st.write('### Download Portfolio KPIs')
    st.download_button('Download KPIs (CSV)', data=pd.DataFrame({'Total Loans':[tl], 'Loan Growth':[lg], 'Avg Interest Rate':[ai], 'Default Ratio':[dr]}).to_csv(index=False), file_name='kpis.csv')
    sum_dict = {"Total Loans": f"₹{tl:,.0f}", "Loan Growth (%)": f"{lg:.2f}", "Avg Interest Rate": f"{ai:.2f}%", "Default Ratio (%)": f"{dr:.2f}"}
    if st.button('Download KPI PDF'):
        fname = export_pdf(sum_dict)
        with open(fname, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{fname}">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
# FOOTER
st.markdown("""
---
<center><b>Bank Loan Portfolio Forecasting System</b></center>
<style>footer {visibility: hidden;}</style>
""", unsafe_allow_html=True)
