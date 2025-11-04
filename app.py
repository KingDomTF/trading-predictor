import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import sqlite3
import yfinance as yf
from fredapi import Fred
import requests
from ta import add_all_ta_features
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

# Config
ASSET_CLASSES = {
    'Precious Metals': ['GC=F', 'SI=F', 'PL=F', 'PA=F'],
    'Cryptocurrencies': ['BTC-USD', 'ETH-USD'],  # Add top 10 dynamically if needed
    'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'NZDUSD=X', 'USDCAD=X'],
    'Indices': ['^GSPC', '^IXIC', '^DJI', '^VIX']
}

API_KEYS = {
    'FRED': 'your_fred_api_key_here',  # Replace with your actual key
    # Add others as needed
}

fred = Fred(api_key=API_KEYS['FRED'])

# Functions from fetch_data.py
def fetch_historical_data(asset):
    return yf.download(asset, period='10y', progress=False)

def fetch_macro_data():
    data = {
        'Federal Funds Rate': fred.get_series('FEDFUNDS'),
        '10Y Treasury': fred.get_series('DGS10'),
        'Inflation CPI': fred.get_series('CPIAUCSL'),
        'Unemployment': fred.get_series('UNRATE'),
        'M2 Supply': fred.get_series('M2SL'),
        'GDP Growth': fred.get_series('A191RL1Q225SBEA')
    }
    return pd.DataFrame(data).iloc[-1]

def fetch_fear_greed():
    url = 'https://api.alternative.me/fng/?limit=1'
    response = requests.get(url)
    return response.json()['data'][0]

# Functions from preprocessor.py
def preprocess_data(df):
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

def calculate_technical_indicators(df):
    df = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
    return df

# Functions from macro_factors.py
def integrate_macro_factors(tech_data, macro_data):
    for asset, df in tech_data.items():
        for key, value in macro_data.items():
            df[key] = value  # Broadcast latest value; in production, align dates
    return tech_data

# Functions from seasonality.py
def analyze_seasonality(df):
    decomposition = sm.tsa.seasonal_decompose(df['Close'], model='additive', period=365)
    df['seasonal'] = decomposition.seasonal
    return df

# Functions from arima_model.py
def train_arima(series):
    model = ARIMA(series, order=(5,1,0))
    return model.fit()

def predict_arima(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

# Functions from ensemble_model.py
def create_features(df):
    df['lag1'] = df['Close'].shift(1)
    df['lag7'] = df['Close'].shift(7)
    df = df.dropna()
    return df

def train_xgboost(df):
    df = create_features(df)
    X = df.drop('Close', axis=1)
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, last_data, steps):
    predictions = []
    current = last_data.copy()
    for _ in range(steps):
        pred = model.predict(current)[0]
        predictions.append(pred)
        current['lag1'] = pred
        # Simplified lag7 update
        if 'lag7' in current.columns:
            current['lag7'] = current['lag1'].shift(6, fill_value=pred)[-1]
    return np.array(predictions)

# Functions from lstm_model.py
def train_lstm(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1,1))
    X = scaled[:-1].reshape(-1, 1, 1)
    y = scaled[1:]
    model = Sequential()
    model.add(LSTM(50, input_shape=(1,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)
    return model, scaler

def predict_lstm(model, scaler, series, steps):
    predictions = []
    current = scaler.transform([[series[-1]]])
    for _ in range(steps):
        pred = model.predict(current.reshape(1,1,1), verbose=0)[0][0]
        predictions.append(pred)
        current = np.array([[pred]])
    return scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

# Functions from prophet_model.py
def train_prophet(df):
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model

def predict_prophet(model, steps):
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast.iloc[-steps:]

# Functions from helpers.py
def calculate_risk_metrics(returns):
    metrics = {
        'Sharpe Ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
        'Max Drawdown': np.min(returns.cumsum() - returns.cumsum().cummax()),
        'VaR 95%': np.percentile(returns, 5)
    }
    return metrics

def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean()) * 252

def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

def minimize_volatility(weights, returns):
    return portfolio_volatility(weights, returns)

def optimize_portfolio(returns_df):
    num_assets = returns_df.shape[1]
    args = (returns_df,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(minimize_volatility, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns_df.columns)

# Main App Code
# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database setup
conn = sqlite3.connect('financial_data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS historical_data
             (asset TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)''')
conn.commit()

# Disclaimer
st.sidebar.warning("This application is for educational purposes only and does not constitute financial advice. Use at your own risk.")

# Sidebar for selections
st.sidebar.title("Asset Selection")
selected_assets = st.sidebar.multiselect("Select Assets", sum(ASSET_CLASSES.values(), []), default=['GC=F', 'BTC-USD', 'EURUSD=X', '^GSPC'])
time_horizon = st.sidebar.selectbox("Prediction Horizon", ["1 Day", "7 Days", "30 Days", "90 Days"])
horizon_days = {"1 Day": 1, "7 Days": 7, "30 Days": 30, "90 Days": 90}[time_horizon]

# Main Dashboard
st.title("Financial Predictive Analytics Dashboard")

# Step 1: Data Collection
@st.cache_data(ttl=3600)
def load_data(assets):
    data = {}
    for asset in assets:
        try:
            df = fetch_historical_data(asset)
            df['asset'] = asset
            df['date'] = df.index.astype(str)
            df.to_sql('historical_data', conn, if_exists='append', index=False)
            data[asset] = df
        except Exception as e:
            logging.error(f"Error fetching data for {asset}: {e}")
    return data

data = load_data(selected_assets)

# Real-time Macro Data
macro_data = fetch_macro_data()
fear_greed = fetch_fear_greed()

# Display Real-time Prices
st.subheader("Real-time Prices")
for asset, df in data.items():
    latest = df.iloc[-1]
    change = (latest['Close'] - latest['Open']) / latest['Open'] * 100
    st.write(f"{asset}: ${latest['Close']:.2f} ({change:.2f}%)")

# Preprocess and Feature Engineering
processed_data = {asset: preprocess_data(df) for asset, df in data.items()}
tech_data = {asset: calculate_technical_indicators(df) for asset, df in processed_data.items()}
integrated_data = integrate_macro_factors(tech_data, macro_data)
seasonal_data = {asset: analyze_seasonality(df) for asset, df in integrated_data.items()}

# Model Training and Predictions
models = ['ARIMA', 'XGBoost', 'LSTM', 'Prophet']
predictions = {}
metrics = {}
for asset, df in seasonal_data.items():
    df['Date'] = df.index  # For Prophet
    train_df = df.iloc[:-int(0.15*len(df))]
    test_df = df.iloc[-int(0.15*len(df)):]
    
    # ARIMA
    arima_model = train_arima(train_df['Close'])
    arima_pred = predict_arima(arima_model, horizon_days)
    predictions[f'{asset}_ARIMA'] = arima_pred
    
    # XGBoost
    xgb_model = train_xgboost(train_df)
    last_data = train_df.iloc[-1:].drop('Close', axis=1)
    xgb_pred = predict_xgboost(xgb_model, last_data, horizon_days)
    predictions[f'{asset}_XGBoost'] = xgb_pred
    
    # LSTM
    lstm_model, scaler = train_lstm(train_df['Close'].values)
    lstm_pred = predict_lstm(lstm_model, scaler, train_df['Close'].values, horizon_days)
    predictions[f'{asset}_LSTM'] = lstm_pred
    
    # Prophet
    prophet_model = train_prophet(train_df)
    prophet_pred = predict_prophet(prophet_model, horizon_days)
    predictions[f'{asset}_Prophet'] = prophet_pred['yhat'].values
    
    # Metrics (using test set for example, adjust lengths)
    test_len = len(test_df)
    for model_name, pred in zip(models, [arima_pred[:test_len], xgb_pred[:test_len], lstm_pred[:test_len], prophet_pred['yhat'][:test_len]]):
        if len(pred) < test_len:
            pred = np.pad(pred, (0, test_len - len(pred)), 'constant', constant_values=np.nan)
        rmse = np.sqrt(mean_squared_error(test_df['Close'], pred, squared=False))
        mae = mean_absolute_error(test_df['Close'], pred)
        metrics[f'{asset}_{model_name}'] = {'RMSE': rmse, 'MAE': mae}

# Visualization
st.subheader("Forecast Charts")
for asset in selected_assets:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data[asset].index, open=data[asset]['Open'], high=data[asset]['High'], low=data[asset]['Low'], close=data[asset]['Close'], name=asset))
    for model in models:
        future_dates = pd.date_range(start=data[asset].index[-1] + timedelta(days=1), periods=horizon_days)
        fig.add_trace(go.Scatter(x=future_dates, y=predictions[f'{asset}_{model}'], name=f'{model} Forecast'))
    st.plotly_chart(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
closes = pd.DataFrame({asset: df['Close'] for asset, df in data.items()})
corr = closes.corr()
fig_corr = px.imshow(corr, text_auto=True)
st.plotly_chart(fig_corr)

# Macro Dashboard
st.subheader("Macro Indicators")
st.write(macro_data)
# Note: st.gauge is not a standard Streamlit function; using st.metric instead
st.metric("Fear & Greed Index", fear_greed['value'], delta=None)

# Risk Management
st.subheader("Risk Metrics")
portfolio_returns = closes.pct_change().dropna()
risk_metrics = calculate_risk_metrics(portfolio_returns.mean(axis=1))
st.write(risk_metrics)

# Portfolio Optimizer
st.subheader("Portfolio Optimization")
optimized_weights = optimize_portfolio(closes.pct_change().dropna())
st.write(optimized_weights)

# Backtesting Metrics
st.subheader("Backtesting Metrics")
st.write(metrics)

# Close DB
conn.close()
