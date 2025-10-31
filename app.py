import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, clean_data, get_summary_stats, plot_corr_heatmap, plot_distribution, plot_time_trend, kpi_cards, encode_categoricals, explain_time_series, explain_histogram
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
    # no image

# --- EDA ---
if menu == 'EDA':
    st.title('🔍 Exploratory Data Analysis')
    st.divider()
    st.write('### Data Summary Statistics')
    st.dataframe(get_summary_stats(df))
    st.divider()
    st.write('### Correlation Heatmap')
    st.pyplot(plot_corr_heatmap(df))
    with st.expander('Explain this heatmap'):
        st.markdown('- Title: Correlation Heatmap\n- X/Y axes: Numeric variables\n- Darker/lighter colors show stronger/weaker correlation; values near ±1 indicate strong relationships.\n- Use: Identify which features move together to avoid multicollinearity and to select meaningful predictors for models.')
    col_choice = st.multiselect('Select columns for distribution:', [x for x in ['Loan_Amount','Interest_Rate','Income','Credit_Score'] if x in df], default=['Loan_Amount','Credit_Score'])
    for col in col_choice:
        st.plotly_chart(plot_distribution(df, col),use_container_width=True)
        with st.expander('Explain this chart'):
            st.markdown(explain_histogram(df[col], f'{col} Distribution'))
    st.divider()
    st.write('### Loan Amount Trend')
    st.plotly_chart(plot_time_trend(df, 'Loan_Amount'), use_container_width=True)
    # Explanation for time trend
    temp = df.copy()
    temp['Date_of_Issue'] = pd.to_datetime(temp['Date_of_Issue'])
    temp = temp.groupby(pd.Grouper(key='Date_of_Issue', freq='M')).agg({'Loan_Amount': 'sum'}).reset_index()
    with st.expander('Explain this chart'):
        st.markdown(explain_time_series(temp, 'Date_of_Issue', 'Loan_Amount', 'Loan Amount Trend', currency=True))
    st.write('### NPA/Default Monthly Trend')
    try:
        npa = df.copy()
        npa['Date_of_Issue'] = pd.to_datetime(npa['Date_of_Issue'])
        npa_trend = npa.groupby(pd.Grouper(key='Date_of_Issue', freq='M'))['Loan_Status'].mean().reset_index()
        npa_trend['Default %'] = npa_trend['Loan_Status'] * 100
        fig_npa = px.area(npa_trend, x='Date_of_Issue', y='Default %', template='plotly_dark', title='Default Ratio (%) by Month', color_discrete_sequence=['#6ab0de'])
        fig_npa.update_traces(mode='lines+markers', line_width=3)
        fig_npa.update_layout(yaxis_title='Default (%)', xaxis_title='Date', height=320, title=dict(x=0.5))
        st.plotly_chart(fig_npa, use_container_width=True)
        with st.expander('Explain this chart'):
            st.markdown(explain_time_series(npa_trend.rename(columns={'Default %':'Default_Percent'}), 'Date_of_Issue', 'Default_Percent', 'NPA/Default Monthly Trend', percent=True))
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
        # Fixed heading instead of dropdown
        target_col = st.selectbox('Target Variable', tcols)
        periods = st.slider('Forecast Periods (months)',3,24,6)
        ts, forecast_df, _, resid, is_stationary = arima_forecast(df, target_col, periods)
        st.caption(f"Data Stationary: {'Yes' if is_stationary else 'No'}")
        diag = arima_diagnostics(ts, resid)
        st.pyplot(diag)
        st.dataframe(forecast_df)
        # Enhanced Actual vs Forecast with CI band
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines+markers', name='Actual', line=dict(color='#4da3ff', width=3)))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='#f5b041', width=3, dash='dash')))
        # Confidence interval band
        fig.add_traces([
            go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper CI'], line=dict(width=0), showlegend=False, hoverinfo='skip'),
            go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower CI'], fill='tonexty', line=dict(width=0), name='95% CI', hoverinfo='skip', fillcolor='rgba(245,176,65,0.2)')
        ])
        fig.update_layout(template='plotly_dark', title=f'{target_col}: Actual vs Forecast', title_x=0.5, xaxis_title='Date', yaxis_title=target_col, height=420)
        if target_col == 'Loan_Amount':
            fig.update_yaxes(tickprefix='₹', separatethousands=True)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander('Explain this chart'):
            hist_df = pd.DataFrame({'Date': ts.index, 'Value': ts.values})
            st.markdown(explain_time_series(hist_df, 'Date', 'Value', f'{target_col} (historical trend)', currency=(target_col=='Loan_Amount')))
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
                model, cm, report, rocauc, fpr, tpr, coef, feat_names, odds = logistic_regression_classification(df, sel_features, 'Loan_Status')
                st.write('#### Confusion Matrix')
                st.write(cm)
                st.write('#### Classification Report')
                st.json(report)
                st.metric(label="ROC-AUC", value=f"{rocauc:.3f}")
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='#4da3ff', width=3)))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(color='#aaaaaa', dash='dash')))
                roc_fig.update_layout(template='plotly_dark', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=340, title='ROC Curve', title_x=0.5)
                st.plotly_chart(roc_fig, use_container_width=True)
                with st.expander('Explain this ROC curve'):
                    st.markdown('- Title: ROC Curve shows model discrimination power.\n- X-axis: False Positive Rate; Y-axis: True Positive Rate.\n- Higher curve above diagonal means better model; AUC near 1.0 is strong.\n- Use: Decide threshold and compare models for default prediction.')
                st.write('#### Feature Importance')
                imp_df = pd.DataFrame({'Feature': feat_names, 'Coefficient': coef, 'AbsCoef': np.abs(coef), 'OddsRatio': odds})
                imp_df = imp_df.sort_values('AbsCoef', ascending=True)
                colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in imp_df['Coefficient']]
                bar = go.Figure(go.Bar(x=imp_df['AbsCoef'], y=imp_df['Feature'], orientation='h', marker_color=colors, text=[f"coef={c:.3f}\nOR={o:.2f}" for c,o in zip(imp_df['Coefficient'], imp_df['OddsRatio'])], textposition='outside'))
                bar.update_layout(template='plotly_dark', height=420, title='Standardized Coefficients (|coef|), color by sign', title_x=0.5, xaxis_title='|Coefficient|', yaxis_title='Feature', margin=dict(l=120, r=20, t=60, b=40))
                st.plotly_chart(bar, use_container_width=True)
                with st.expander('Explain this importance chart'):
                    st.markdown('- Title: Feature Importance (Logistic coefficients).\n- Bars show size of effect; green raises default odds, red lowers.\n- Hover shows coefficient and odds ratio (exp(coef)).\n- Use: Identify drivers of default risk and guide credit policy or feature selection.')
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
    fig_dash = go.Figure()
    fig_dash.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines+markers', name='Actual', line=dict(color='#4da3ff', width=3)))
    fig_dash.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='#f5b041', width=3, dash='dash')))
    fig_dash.add_traces([
        go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper CI'], line=dict(width=0), showlegend=False, hoverinfo='skip'),
        go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower CI'], fill='tonexty', line=dict(width=0), name='95% CI', hoverinfo='skip', fillcolor='rgba(245,176,65,0.2)')
    ])
    fig_dash.update_layout(template='plotly_dark', title='Actual vs Forecast', title_x=0.5, xaxis_title='Date', yaxis_title=target_col, height=420)
    fig_dash.update_yaxes(tickprefix='₹', separatethousands=True)
    st.plotly_chart(fig_dash, use_container_width=True)
    with st.expander('Explain this chart'):
        hist_df = pd.DataFrame({'Date': ts.index, 'Value': ts.values})
        st.markdown(explain_time_series(hist_df, 'Date', 'Value', f'{target_col} (historical trend)', currency=(target_col=='Loan_Amount')))
    st.write('### Default Feature Importance')
    try:
        _,_,_,_,_,_,coef,feat_names,odds = logistic_regression_classification(df, ['Loan_Amount','Credit_Score','Interest_Rate','Income'], 'Loan_Status')
        imp_df = pd.DataFrame({'Feature': feat_names, 'Coefficient': coef, 'AbsCoef': np.abs(coef), 'OddsRatio': odds}).sort_values('AbsCoef', ascending=True)
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in imp_df['Coefficient']]
        st.plotly_chart(go.Figure(go.Bar(x=imp_df['AbsCoef'], y=imp_df['Feature'], orientation='h', marker_color=colors, text=[f"coef={c:.3f}\nOR={o:.2f}" for c,o in zip(imp_df['Coefficient'], imp_df['OddsRatio'])], textposition='outside')).update_layout(template='plotly_dark', height=360, title='Standardized Coefficients (|coef|), color by sign', title_x=0.5, xaxis_title='|Coefficient|', yaxis_title='Feature', margin=dict(l=120, r=20, t=60, b=40)), use_container_width=True)
        with st.expander('Explain this importance chart'):
            st.markdown('- Title: Feature Importance (Logistic coefficients).\n- Bars show size of effect; green raises default odds, red lowers.\n- Hover shows coefficient and odds ratio (exp(coef)).\n- Use: Identify drivers of default risk and guide credit policy or feature selection.')
    except: st.warning('Default ML could not run.')
    st.write('### NPA/Default Monthly Trend')
    try:
        npa = df.copy()
        npa['Date_of_Issue'] = pd.to_datetime(npa['Date_of_Issue'])
        npa_trend = npa.groupby(pd.Grouper(key='Date_of_Issue', freq='M'))['Loan_Status'].mean().reset_index()
        npa_trend['Default %'] = npa_trend['Loan_Status'] * 100
        fig_npa2 = px.area(npa_trend, x='Date_of_Issue', y='Default %', template='plotly_dark', color_discrete_sequence=['#6ab0de'])
        fig_npa2.update_traces(mode='lines+markers', line_width=3)
        fig_npa2.update_layout(yaxis_title='Default (%)', xaxis_title='Date', height=320, title='Default Ratio (%) by Month', title_x=0.5)
        st.plotly_chart(fig_npa2, use_container_width=True)
        with st.expander('Explain this chart'):
            st.markdown(explain_time_series(npa_trend.rename(columns={'Default %':'Default_Percent'}), 'Date_of_Issue', 'Default_Percent', 'NPA/Default Monthly Trend', percent=True))
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
