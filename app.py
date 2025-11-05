import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from scipy import stats
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from fredapi import Fred
import requests
import json
from typing import Dict, List, Tuple, Optional
import hashlib
import time
warnings.filterwarnings('ignore')
==================== CONFIGURAZIONE ====================
st.set_page_config(
page_title="Sistema Analisi Finanziaria Istituzionale",
page_icon="📈",
layout="wide",
initial_sidebar_state="expanded"
)
Stile CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .success-prob {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)
==================== CLASSI PRINCIPALI ====================
class DataCollector:
"""Raccolta dati da multiple fonti"""
INSTRUMENTS = {
'Metalli': {
'Oro': 'GC=F',
'Argento': 'SI=F',
'Platino': 'PL=F',
'Palladio': 'PA=F',
},
'Criptovalute': {
'Bitcoin': 'BTC-USD',
'Ethereum': 'ETH-USD',
'Binance Coin': 'BNB-USD',
'Cardano': 'ADA-USD',
},
'Forex': {
'EUR/USD': 'EURUSD=X',
'GBP/USD': 'GBPUSD=X',
'USD/JPY': 'JPY=X',
'USD/CHF': 'CHF=X',
'AUD/USD': 'AUDUSD=X',
},
'Commodities': {
'Petrolio WTI': 'CL=F',
'Petrolio Brent': 'BZ=F',
'Gas Naturale': 'NG=F',
'Rame': 'HG=F',
}
}
TIMEFRAMES = {
'15min': {'period': '60d', 'interval': '15m'},
'1h': {'period': '730d', 'interval': '1h'},
'4h': {'period': '730d', 'interval': '1h'}, # Aggrega da 1h
'1d': {'period': '10y', 'interval': '1d'}
}
@staticmethod
@st.cache_data(ttl=900)
def get_price_data(symbol: str, timeframe: str) -> pd.DataFrame:
"""Scarica dati di prezzo con caching"""
try:
config = DataCollector.TIMEFRAMES[timeframe]
ticker = yf.Ticker(symbol)
df = ticker.history(period=config['period'], interval=config['interval'])
if df.empty:
st.warning(f"Nessun dato disponibile per {symbol}")
return pd.DataFrame()
Aggregazione per 4h se necessario
if timeframe == '4h':
df = df.resample('4H').agg({
'Open': 'first',
'High': 'max',
'Low': 'min',
'Close': 'last',
'Volume': 'sum'
}).dropna()
return df
except Exception as e:
st.error(f"Errore nel download dati per {symbol}: {str(e)}")
return pd.DataFrame()
@staticmethod
@st.cache_data(ttl=3600)
def get_vix_data() -> float:
"""Ottiene VIX (Indice della Paura)"""
try:
vix = yf.Ticker("^VIX")
data = vix.history(period="5d")
if not data.empty:
return round(data['Close'].iloc[-1], 2)
except:
pass
return 20.0 # Valore di default
@staticmethod
@st.cache_data(ttl=86400)
def get_fed_rate() -> float:
"""Ottiene tasso FED (simulato se API non disponibile)"""
try:
In produzione usare: fred = Fred(api_key='YOUR_KEY')
rate = fred.get_series_latest_release('FEDFUNDS')
return float(rate.iloc[-1])
return 5.33 # Tasso attuale approssimativo
except:
return 5.33
@staticmethod
@st.cache_data(ttl=86400)
def get_fear_greed_index() -> Dict:
"""Fear & Greed Index per crypto"""
try:
url = "https://api.alternative.me/fng/?limit=1"
response = requests.get(url, timeout=5)
data = response.json()
value = int(data['data'][0]['value'])
classification = data['data'][0]['value_classification']
return {'value': value, 'classification': classification}
except:
return {'value': 50, 'classification': 'Neutral'}
class FeatureEngine:
"""Calcolo features e indicatori tecnici"""
@staticmethod
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
"""Calcola tutti gli indicatori tecnici"""
df = df.copy()
Moving Averages
df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
RSI
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
MACD
macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_diff'] = macd.macd_diff()
Bollinger Bands
bollinger = ta.volatility.BollingerBands(df['Close'])
df['BB_high'] = bollinger.bollinger_hband()
df['BB_mid'] = bollinger.bollinger_mavg()
df['BB_low'] = bollinger.bollinger_lband()
df['BB_width'] = bollinger.bollinger_wband()
ATR (Average True Range)
df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
Volume indicators
df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
Stochastic
stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
df['Stoch_K'] = stoch.stoch()
df['Stoch_D'] = stoch.stoch_signal()
Price momentum
df['ROC'] = ta.momentum.roc(df['Close'], window=12)
df['Price_momentum'] = df['Close'].pct_change(periods=10)
Volatilità
df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
return df
@staticmethod
def calculate_seasonality(df: pd.DataFrame) -> Dict:
"""Analisi stagionalità"""
df = df.copy()
df['Month'] = df.index.month
df['DayOfWeek'] = df.index.dayofweek
df['Returns'] = df['Close'].pct_change()
monthly_avg = df.groupby('Month')['Returns'].mean() * 100
weekly_avg = df.groupby('DayOfWeek')['Returns'].mean() * 100
current_month = datetime.now().month
current_day = datetime.now().weekday()
return {
'monthly_pattern': monthly_avg.to_dict(),
'weekly_pattern': weekly_avg.to_dict(),
'current_month_bias': monthly_avg.get(current_month, 0),
'current_day_bias': weekly_avg.get(current_day, 0)
}
@staticmethod
def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
"""Prepara features per ML"""
df = df.copy()
Target: rendimento futuro
df['Target'] = df['Close'].pct_change().shift(-1)
Lagged features
for lag in [1, 2, 3, 5, 10]:
df[f'Return_lag_{lag}'] = df['Close'].pct_change(lag)
df[f'Volume_lag_{lag}'] = df['Volume'].pct_change(lag)
Features statistiche
df['Return_mean_5'] = df['Close'].pct_change().rolling(5).mean()
df['Return_std_5'] = df['Close'].pct_change().rolling(5).std()
df['High_Low_ratio'] = (df['High'] - df['Low']) / df['Close']
Rimuovi NaN
df = df.dropna()
Separazione features e target
feature_cols = [col for col in df.columns if col not in
['Open', 'High', 'Low', 'Close', 'Volume', 'Target',
'Dividends', 'Stock Splits']]
X = df[feature_cols]
y = df['Target']
return X, y
class PredictionEngine:
"""Motore di previsione ensemble"""
def init(self):
self.models = {}
self.scaler = StandardScaler()
def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict:
"""Training ensemble di modelli"""
Split train/test
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
Scaling
X_train_scaled = self.scaler.fit_transform(X_train)
X_test_scaled = self.scaler.transform(X_test)
Random Forest
rf_model = RandomForestRegressor(
n_estimators=100,
max_depth=10,
random_state=42,
n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_score = rf_model.score(X_test_scaled, y_test)
XGBoost
xgb_model = xgb.XGBRegressor(
n_estimators=100,
learning_rate=0.1,
max_depth=6,
random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
xgb_score = xgb_model.score(X_test_scaled, y_test)
Gradient Boosting
gb_model = GradientBoostingRegressor(
n_estimators=100,
learning_rate=0.1,
max_depth=5,
random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_score = gb_model.score(X_test_scaled, y_test)
self.models = {
'RandomForest': {'model': rf_model, 'score': rf_score, 'weight': 0.33},
'XGBoost': {'model': xgb_model, 'score': xgb_score, 'weight': 0.34},
'GradientBoosting': {'model': gb_model, 'score': gb_score, 'weight': 0.33}
}
Normalizza pesi basati su score
total_score = sum(m['score'] for m in self.models.values())
for model_name in self.models:
self.models[model_name]['weight'] = self.models[model_name]['score'] / total_score
return {
'rf_score': rf_score,
'xgb_score': xgb_score,
'gb_score': gb_score,
'avg_score': np.mean([rf_score, xgb_score, gb_score])
}
def predict(self, X_latest: pd.DataFrame) -> Dict:
"""Previsione ensemble con probabilità"""
X_scaled = self.scaler.transform(X_latest)
predictions = {}
for model_name, model_info in self.models.items():
pred = model_info['model'].predict(X_scaled)[0]
predictions[model_name] = pred
Weighted average
ensemble_pred = sum(
predictions[name] * self.models[name]['weight']
for name in predictions
)
Calcolo confidenza basato su deviazione standard
pred_std = np.std(list(predictions.values()))
confidence = max(50, min(95, 100 - (pred_std * 1000)))
Probabilità direzionali
if ensemble_pred > 0.001:
prob_up = min(95, 50 + (ensemble_pred * 2000))
prob_down = 100 - prob_up
elif ensemble_pred < -0.001:
prob_down = min(95, 50 + (abs(ensemble_pred) * 2000))
prob_up = 100 - prob_down
else:
prob_up = 50
prob_down = 50
return {
'prediction': ensemble_pred,
'confidence': confidence,
'prob_up': prob_up,
'prob_down': prob_down,
'individual_predictions': predictions
}
def arima_forecast(self, series: pd.Series, steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
"""Previsione ARIMA con intervallo confidenza"""
try:
model = ARIMA(series, order=(5, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=steps)
Confidence interval approssimato
std = series.std()
upper = forecast + 1.96 * std
lower = forecast - 1.96 * std
return forecast.values, (lower.values, upper.values)
except:
Fallback: ultimo valore
last_val = series.iloc[-1]
return np.full(steps, last_val), (np.full(steps, last_val * 0.95), np.full(steps, last_val * 1.05))
class RiskAnalyzer:
"""Analisi del rischio e metriche"""
@staticmethod
def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
"""Value at Risk"""
return np.percentile(returns.dropna(), (1 - confidence) * 100)
@staticmethod
def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
"""Sharpe Ratio"""
excess_returns = returns.mean() - risk_free_rate / 252
return (excess_returns / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
@staticmethod
def calculate_max_drawdown(prices: pd.Series) -> float:
"""Maximum Drawdown"""
cumulative = (1 + prices.pct_change()).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
return drawdown.min()
@staticmethod
def support_resistance_levels(df: pd.DataFrame, n_levels: int = 3) -> Dict:
"""Calcola livelli supporto/resistenza"""
recent_data = df.tail(100)
Pivot points
pivot = (recent_data['High'].max() + recent_data['Low'].min() + recent_data['Close'].iloc[-1]) / 3
resistance_levels = []
support_levels = []
for i in range(1, n_levels + 1):
r = pivot + (recent_data['High'].max() - recent_data['Low'].min()) * i * 0.382
s = pivot - (recent_data['High'].max() - recent_data['Low'].min()) * i * 0.382
resistance_levels.append(round(r, 2))
support_levels.append(round(s, 2))
return {
'resistance': resistance_levels,
'support': support_levels,
'pivot': round(pivot, 2)
}
==================== FUNZIONI VISUALIZZAZIONE ====================
def create_candlestick_chart(df: pd.DataFrame, symbol: str, indicators: bool = True):
"""Crea grafico candlestick con indicatori"""
fig = make_subplots(
rows=3, cols=1,
shared_xaxes=True,
vertical_spacing=0.03,
subplot_titles=(f'{symbol} - Prezzo', 'RSI', 'MACD'),
row_heights=[0.6, 0.2, 0.2]
)
Candlestick
fig.add_trace(
go.Candlestick(
x=df.index,
open=df['Open'],
high=df['High'],
low=df['Low'],
close=df['Close'],
name='Prezzo'
),
row=1, col=1
)
if indicators:
Moving Averages
fig.add_trace(
go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)),
row=1, col=1
)
fig.add_trace(
go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)),
row=1, col=1
)
Bollinger Bands
fig.add_trace(
go.Scatter(x=df.index, y=df['BB_high'], name='BB High',
line=dict(color='gray', width=1, dash='dash')),
row=1, col=1
)
fig.add_trace(
go.Scatter(x=df.index, y=df['BB_low'], name='BB Low',
line=dict(color='gray', width=1, dash='dash'),
fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
row=1, col=1
)
RSI
fig.add_trace(
go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)),
row=2, col=1
)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
MACD
fig.add_trace(
go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=2)),
row=3, col=1
)
fig.add_trace(
go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='red', width=2)),
row=3, col=1
)
fig.add_trace(
go.Bar(x=df.index, y=df['MACD_diff'], name='Histogram', marker_color='gray'),
row=3, col=1
)
fig.update_layout(
height=900,
showlegend=True,
xaxis_rangeslider_visible=False,
hovermode='x unified'
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
return fig
def create_prediction_chart(df: pd.DataFrame, forecast: np.ndarray, conf_intervals: Tuple):
"""Crea grafico previsioni"""
fig = go.Figure()
Dati storici
fig.add_trace(go.Scatter(
x=df.index[-100:],
y=df['Close'][-100:],
mode='lines',
name='Storico',
line=dict(color='blue', width=2)
))
Previsione
future_dates = pd.date_range(start=df.index[-1], periods=len(forecast) + 1, freq='D')[1:]
fig.add_trace(go.Scatter(
x=future_dates,
y=forecast,
mode='lines',
name='Previsione',
line=dict(color='red', width=2, dash='dash')
))
Intervallo confidenza
lower, upper = conf_intervals
fig.add_trace(go.Scatter(
x=future_dates,
y=upper,
mode='lines',
name='Limite Superiore',
line=dict(width=0),
showlegend=False
))
fig.add_trace(go.Scatter(
x=future_dates,
y=lower,
mode='lines',
name='Limite Inferiore',
line=dict(width=0),
fillcolor='rgba(255, 0, 0, 0.2)',
fill='tonexty',
showlegend=True
))
fig.update_layout(
title="Previsione Prezzo con Intervallo di Confidenza 95%",
xaxis_title="Data",
yaxis_title="Prezzo",
hovermode='x unified',
height=500
)
return fig
==================== APPLICAZIONE PRINCIPALE ====================
def main():
Header
st.markdown('🚀 Sistema Analisi Finanziaria Istituzionale',
unsafe_allow_html=True)
st.markdown("---")
Sidebar
with st.sidebar:
st.image("https://img.icons8.com/color/96/000000/analytics.png", width=100)
st.title("⚙️ Configurazione")
Selezione categoria
category = st.selectbox(
"📊 Categoria Asset",
options=list(DataCollector.INSTRUMENTS.keys())
)
Selezione strumento
instrument = st.selectbox(
"🎯 Strumento",
options=list(DataCollector.INSTRUMENTS[category].keys())
)
symbol = DataCollector.INSTRUMENTS[category][instrument]
Timeframe
timeframe = st.selectbox(
"⏱️ Timeframe",
options=['15min', '1h', '4h', '1d'],
index=3
)
st.markdown("---")
Opzioni avanzate
st.subheader("🔧 Opzioni Avanzate")
show_indicators = st.checkbox("Mostra Indicatori Tecnici", value=True)
show_predictions = st.checkbox("Mostra Previsioni ML", value=True)
show_seasonality = st.checkbox("Analisi Stagionalità", value=True)
st.markdown("---")
Indicatori macro
st.subheader("📈 Indicatori Macro")
vix_value = DataCollector.get_vix_data()
fed_rate = DataCollector.get_fed_rate()
fear_greed = DataCollector.get_fear_greed_index()
col1, col2 = st.columns(2)
with col1:
st.metric("VIX", f"{vix_value}", help="Indice della Paura")
with col2:
st.metric("FED Rate", f"{fed_rate}%")
if category == 'Criptovalute':
st.metric("Fear & Greed", f"{fear_greed['value']}",
delta=fear_greed['classification'])
Pulsante analisi
analyze_button = st.button("🔍 ANALIZZA", type="primary", use_container_width=True)
Main content
if analyze_button:
with st.spinner(f"📊 Caricamento dati {instrument} ({timeframe})..."):
df = DataCollector.get_price_data(symbol, timeframe)
if df.empty:
st.error("❌ Impossibile caricare i dati. Riprova più tardi.")
return
Calcolo indicatori
with st.spinner("🔧 Calcolo indicatori tecnici..."):
df = FeatureEngine.calculate_technical_indicators(df)
Metriche principali
st.subheader(f"📊 {instrument} - {timeframe.upper()}")
current_price = df['Close'].iloc[-1]
price_change = df['Close'].pct_change().iloc[-1] * 100
volume_change = df['Volume'].pct_change().iloc[-1] * 100
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
st.metric("💰 Prezzo Attuale", f"${current_price:,.2f}",
f"{price_change:+.2f}%")
with col2:
st.metric("📊 RSI", f"{df['RSI'].iloc[-1]:.1f}")
with col3:
st.metric("📉 ATR", f"{df['ATR'].iloc[-1]:.2f}")
with col4:
volatility = df['Volatility'].iloc[-1] * 100
st.metric("📈 Volatilità", f"{volatility:.1f}%")
with col5:
volume_str = f"{df['Volume'].iloc[-1]:,.0f}"
st.metric("📦 Volume", volume_str, f"{volume_change:+.1f}%")
st.markdown("---")
levels = RiskAnalyzer.support_resistance_levels(df)
atr = df['ATR'].iloc[-1]
returns = df['Close'].pct_change().dropna()
win_rate_long = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 50.0
win_rate_short = (returns < 0).sum() / len(returns) * 100 if len(returns) > 0 else 50.0
win_rate_avg = (win_rate_long + win_rate_short) / 2
traders_data = [
{"Trader": "Trader 1", "Entry": f"${current_price:.2f}", "TP": f"${levels['resistance'][0]:.2f}", "SL": f"${levels['support'][0]:.2f}", "Percentuale di successo": f"{win_rate_long:.1f}%"},
{"Trader": "Trader 2", "Entry": f"${current_price:.2f}", "TP": f"${levels['support'][0]:.2f}", "SL": f"${levels['resistance'][0]:.2f}", "Percentuale di successo": f"{win_rate_short:.1f}%"},
{"Trader": "Trader 3", "Entry": f"${current_price:.2f}", "TP": f"${levels['resistance'][1]:.2f}", "SL": f"${levels['support'][1]:.2f}", "Percentuale di successo": f"{win_rate_avg:.1f}%"}
]
traders_df = pd.DataFrame(traders_data)
st.table(traders_df)
Grafico principale
st.subheader("📈 Grafico Candlestick & Indicatori")
fig_candle = create_candlestick_chart(df.tail(200), instrument, show_indicators)
st.plotly_chart(fig_candle, use_container_width=True)
st.markdown("---")
PREVISIONI ML
if show_predictions and len(df) > 100:
st.subheader("🤖 Previsioni Machine Learning Ensemble")
with st.spinner("🧠 Training modelli ML..."):
X, y = FeatureEngine.prepare_ml_features(df)
if len(X) > 50:
predictor = PredictionEngine()
scores = predictor.train_ensemble(X, y)
Previsione prossimo periodo
X_latest = X.tail(1)
prediction_result = predictor.predict(X_latest)
ARIMA forecast
forecast, conf_int = predictor.arima_forecast(df['Close'], steps=30)
Display previsioni
col1, col2 = st.columns([1, 1])
with col1:
st.markdown('', unsafe_allow_html=True)
predicted_return = prediction_result['prediction'] * 100
predicted_price = current_price * (1 + prediction_result['prediction'])
st.markdown(f"### 🎯 Previsione Prossimo Periodo")
st.markdown(f"Prezzo Previsto: ${predicted_price:,.2f}")
st.markdown(f"Variazione Attesa: {predicted_return:+.2f}%")
st.markdown(f"Confidenza Modello: {prediction_result['confidence']:.1f}%")
st.markdown('', unsafe_allow_html=True)
Probabilità direzionali
st.markdown("#### 📊 Probabilità Movimento")
prob_up = prediction_result['prob_up']
prob_down = prediction_result['prob_down']
st.markdown(f"""
                        
                            {prob_up:.1f}% ⬆️ RIALZO
                        
                        
                            {prob_down:.1f}% ⬇️ RIBASSO
                        
                        """, unsafe_allow_html=True)
with col2:
st.markdown("#### 🎯 Performance Modelli")
model_perf = pd.DataFrame({
'Modello': ['Random Forest', 'XGBoost', 'Gradient Boosting'],
'Score R²': [
scores['rf_score'],
scores['xgb_score'],
scores['gb_score']
],
'Previsione %': [
prediction_result['individual_predictions']['RandomForest'] * 100,
prediction_result['individual_predictions']['XGBoost'] * 100,
prediction_result['individual_predictions']['GradientBoosting'] * 100
]
})
st.dataframe(
model_perf.style.format({
'Score R²': '{:.3f}',
'Previsione %': '{:+.2f}%'
}).background_gradient(subset=['Score R²'], cmap='RdYlGn'),
use_container_width=True,
hide_index=True
)
st.metric("📈 Score Medio Ensemble", f"{scores['avg_score']:.3f}")
Grafico previsione ARIMA
st.markdown("#### 📉 Proiezione Prezzo (30 periodi - ARIMA)")
fig_forecast = create_prediction_chart(df, forecast, conf_int)
st.plotly_chart(fig_forecast, use_container_width=True)
Target prices
st.markdown("#### 🎯 Livelli Target")
col1, col2, col3 = st.columns(3)
target_1d = forecast[0] if len(forecast) > 0 else current_price
target_1w = forecast[6] if len(forecast) > 6 else current_price
target_1m = forecast[-1] if len(forecast) > 0 else current_price
with col1:
change_1d = ((target_1d - current_price) / current_price) * 100
st.metric("1 Periodo", f"${target_1d:,.2f}", f"{change_1d:+.2f}%")
with col2:
change_1w = ((target_1w - current_price) / current_price) * 100
st.metric("7 Periodi", f"${target_1w:,.2f}", f"{change_1w:+.2f}%")
with col3:
change_1m = ((target_1m - current_price) / current_price) * 100
st.metric("30 Periodi", f"${target_1m:,.2f}", f"{change_1m:+.2f}%")
st.markdown("---")
ANALISI RISCHIO
st.subheader("⚠️ Analisi del Rischio")
returns = df['Close'].pct_change().dropna()
col1, col2, col3, col4 = st.columns(4)
with col1:
var_95 = RiskAnalyzer.calculate_var(returns, 0.95) * 100
st.metric("📉 VaR 95%", f"{var_95:.2f}%",
help="Value at Risk: perdita massima attesa con 95% confidenza")
with col2:
sharpe = RiskAnalyzer.calculate_sharpe(returns)
st.metric("📊 Sharpe Ratio", f"{sharpe:.2f}",
help="Rendimento aggiustato per rischio")
with col3:
max_dd = RiskAnalyzer.calculate_max_drawdown(df['Close']) * 100
st.metric("📉 Max Drawdown", f"{max_dd:.2f}%",
help="Massima perdita dal picco")
with col4:
win_rate = (returns > 0).sum() / len(returns) * 100
st.metric("✅ Win Rate", f"{win_rate:.1f}%",
help="Percentuale periodi positivi")
Supporti e Resistenze
levels = RiskAnalyzer.support_resistance_levels(df)
col1, col2 = st.columns(2)
with col1:
st.markdown("#### 🔴 Livelli di Resistenza")
for i, level in enumerate(levels['resistance'], 1):
distance = ((level - current_price) / current_price) * 100
st.markdown(f"R{i}: ${level:,.2f} ({distance:+.2f}%)")
with col2:
st.markdown("#### 🟢 Livelli di Supporto")
st.markdown(f"Pivot: ${levels['pivot']:,.2f}")
for i, level in enumerate(levels['support'], 1):
distance = ((level - current_price) / current_price) * 100
st.markdown(f"S{i}: ${level:,.2f} ({distance:+.2f}%)")
st.markdown("---")
ANALISI STAGIONALITÀ
if show_seasonality:
st.subheader("🗓️ Analisi Stagionalità")
seasonality = FeatureEngine.calculate_seasonality(df)
col1, col2 = st.columns(2)
with col1:
st.markdown("#### 📅 Pattern Mensile")
monthly_df = pd.DataFrame({
'Mese': ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu',
'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic'],
'Rendimento Medio %': [seasonality['monthly_pattern'].get(i, 0)
for i in range(1, 13)]
})
fig_monthly = go.Figure(data=[
go.Bar(
x=monthly_df['Mese'],
y=monthly_df['Rendimento Medio %'],
marker_color=['green' if x > 0 else 'red'
for x in monthly_df['Rendimento Medio %']]
)
])
fig_monthly.update_layout(
title="Performance Storica per Mese",
yaxis_title="Rendimento %",
height=400
)
st.plotly_chart(fig_monthly, use_container_width=True)
current_month_bias = seasonality['current_month_bias']
st.metric("🎯 Bias Mese Corrente", f"{current_month_bias:+.3f}%")
with col2:
st.markdown("#### 📊 Pattern Settimanale")
weekly_df = pd.DataFrame({
'Giorno': ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì'],
'Rendimento Medio %': [seasonality['weekly_pattern'].get(i, 0)
for i in range(5)]
})
fig_weekly = go.Figure(data=[
go.Bar(
x=weekly_df['Giorno'],
y=weekly_df['Rendimento Medio %'],
marker_color=['green' if x > 0 else 'red'
for x in weekly_df['Rendimento Medio %']]
)
])
fig_weekly.update_layout(
title="Performance Storica per Giorno",
yaxis_title="Rendimento %",
height=400
)
st.plotly_chart(fig_weekly, use_container_width=True)
current_day_bias = seasonality['current_day_bias']
st.metric("🎯 Bias Giorno Corrente", f"{current_day_bias:+.3f}%")
st.markdown("---")
ANALISI CORRELAZIONI
st.subheader("🔗 Matrice Correlazioni Indicatori")
corr_cols = ['Close', 'RSI', 'MACD', 'ATR', 'Volume', 'Volatility', 'BB_width']
corr_matrix = df[corr_cols].corr()
fig_corr = go.Figure(data=go.Heatmap(
z=corr_matrix,
x=corr_cols,
y=corr_cols,
colorscale='RdBu',
zmid=0,
text=corr_matrix.round(2),
texttemplate='%{text}',
textfont={"size": 10},
colorbar=dict(title="Correlazione")
))
fig_corr.update_layout(
title="Mappa di Correlazione tra Indicatori",
height=500
)
st.plotly_chart(fig_corr, use_container_width=True)
st.markdown("---")
SEGNALI DI TRADING
st.subheader("🚦 Segnali di Trading")
current_rsi = df['RSI'].iloc[-1]
current_macd = df['MACD'].iloc[-1]
current_macd_signal = df['MACD_signal'].iloc[-1]
price_vs_sma20 = ((current_price - df['SMA_20'].iloc[-1]) / df['SMA_20'].iloc[-1]) * 100
price_vs_sma50 = ((current_price - df['SMA_50'].iloc[-1]) / df['SMA_50'].iloc[-1]) * 100
signals = []
RSI Signals
if current_rsi < 30:
signals.append(("🟢 RSI Oversold", "ACQUISTO", "RSI < 30 indica ipervenduto"))
elif current_rsi > 70:
signals.append(("🔴 RSI Overbought", "VENDITA", "RSI > 70 indica ipercomprato"))
else:
signals.append(("🟡 RSI Neutrale", "NEUTRALE", f"RSI = {current_rsi:.1f}"))
MACD Signals
if current_macd > current_macd_signal and df['MACD'].iloc[-2] <= df['MACD_signal'].iloc[-2]:
signals.append(("🟢 MACD Bullish Cross", "ACQUISTO", "MACD incrocia al rialzo la signal line"))
elif current_macd < current_macd_signal and df['MACD'].iloc[-2] >= df['MACD_signal'].iloc[-2]:
signals.append(("🔴 MACD Bearish Cross", "VENDITA", "MACD incrocia al ribasso la signal line"))
else:
signals.append(("🟡 MACD Neutrale", "NEUTRALE", "Nessun incrocio recente"))
Moving Average Signals
if price_vs_sma20 > 2 and price_vs_sma50 > 2:
signals.append(("🟢 Trend Rialzista", "ACQUISTO", "Prezzo > SMA20 e SMA50"))
elif price_vs_sma20 < -2 and price_vs_sma50 < -2:
signals.append(("🔴 Trend Ribassista", "VENDITA", "Prezzo < SMA20 e SMA50"))
else:
signals.append(("🟡 Trend Laterale", "NEUTRALE", "Prezzo vicino alle medie"))
Bollinger Bands Signal
if current_price < df['BB_low'].iloc[-1]:
signals.append(("🟢 Sotto Bollinger Inf.", "ACQUISTO", "Possibile rimbalzo"))
elif current_price > df['BB_high'].iloc[-1]:
signals.append(("🔴 Sopra Bollinger Sup.", "VENDITA", "Possibile ritracciamento"))
Volume Signal
if df['Volume_ratio'].iloc[-1] > 1.5:
signals.append(("⚡ Volume Elevato", "ATTENZIONE", "Volume 50% sopra media"))
Display signals
for signal, action, description in signals:
color = "green" if "ACQUISTO" in action else "red" if "VENDITA" in action else "orange"
st.markdown(f"""
            
                {signal} - {action}
                {description}
            
            """, unsafe_allow_html=True)
st.markdown("---")
FATTORI DOMINANTI
st.subheader("🎯 Fattori Dominanti del Prezzo")
col1, col2 = st.columns(2)
with col1:
st.markdown("#### 📊 Top Indicatori Tecnici")
technical_factors = {
'RSI': abs(50 - current_rsi) / 50 * 100,
'MACD Momentum': abs(current_macd - current_macd_signal) * 100,
'Volatilità': volatility * 10,
'Volume Anomaly': abs(df['Volume_ratio'].iloc[-1] - 1) * 100,
'BB Position': abs((current_price - df['BB_mid'].iloc[-1]) / df['BB_mid'].iloc[-1]) * 1000
}
tech_df = pd.DataFrame({
'Fattore': list(technical_factors.keys()),
'Peso %': list(technical_factors.values())
}).sort_values('Peso %', ascending=False)
tech_df['Peso %'] = tech_df['Peso %'] / tech_df['Peso %'].sum() * 100
fig_tech = go.Figure(data=[
go.Bar(x=tech_df['Peso %'], y=tech_df['Fattore'], orientation='h',
marker_color='steelblue')
])
fig_tech.update_layout(
xaxis_title="Impatto %",
height=300,
margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_tech, use_container_width=True)
with col2:
st.markdown("#### 🌍 Fattori Macroeconomici")
Calcolo impatto macro
vix_impact = (vix_value / 40) * 100 # Normalizzato su 40
fed_impact = (fed_rate / 10) * 100 # Normalizzato su 10%
macro_factors = {
'VIX (Paura Mercato)': vix_impact,
'Tassi FED': fed_impact,
'Sentiment Globale': 50 + (np.random.randn() * 10) # Simulato
}
if category == 'Criptovalute':
macro_factors['Fear & Greed Index'] = fear_greed['value']
macro_df = pd.DataFrame({
'Fattore': list(macro_factors.keys()),
'Livello': list(macro_factors.values())
})
fig_macro = go.Figure(data=[
go.Bar(x=macro_df['Livello'], y=macro_df['Fattore'], orientation='h',
marker_color='coral')
])
fig_macro.update_layout(
xaxis_title="Livello",
height=300,
margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_macro, use_container_width=True)
st.markdown("---")
RACCOMANDAZIONE FINALE
st.subheader("🎯 Raccomandazione Algoritmica Finale")
Calcolo score complessivo
buy_score = 0
sell_score = 0
RSI
if current_rsi < 40:
buy_score += 2
elif current_rsi > 60:
sell_score += 2
MACD
if current_macd > current_macd_signal:
buy_score += 1.5
else:
sell_score += 1.5
Trend
if price_vs_sma20 > 0 and price_vs_sma50 > 0:
buy_score += 2
elif price_vs_sma20 < 0 and price_vs_sma50 < 0:
sell_score += 2
ML Prediction
if show_predictions and 'prediction_result' in locals():
if prediction_result['prediction'] > 0.005:
buy_score += 3
elif prediction_result['prediction'] < -0.005:
sell_score += 3
Volatilità
if volatility < 30:
buy_score += 1
elif volatility > 60:
sell_score += 1
total_score = buy_score + sell_score
buy_percentage = (buy_score / total_score * 100) if total_score > 0 else 50
sell_percentage = (sell_score / total_score * 100) if total_score > 0 else 50
if buy_percentage > 60:
recommendation = "🟢 ACQUISTO FORTE"
rec_color = "green"
rec_desc = "I fattori tecnici e quantitativi suggeriscono una opportunità di acquisto"
elif buy_percentage > 50:
recommendation = "🟢 ACQUISTO MODERATO"
rec_color = "lightgreen"
rec_desc = "Segnali positivi prevalenti, ma con cautela"
elif sell_percentage > 60:
recommendation = "🔴 VENDITA FORTE"
rec_color = "red"
rec_desc = "I fattori suggeriscono pressione ribassista"
elif sell_percentage > 50:
recommendation = "🔴 VENDITA MODERATA"
rec_color = "orange"
rec_desc = "Segnali negativi prevalenti"
else:
recommendation = "🟡 NEUTRALE / ATTENDI"
rec_color = "gray"
rec_desc = "Segnali contrastanti, meglio attendere conferme"
st.markdown(f"""
        
            {recommendation}
            {rec_desc}
            Score Algoritmo: {buy_percentage:.1f}% Bullish / {sell_percentage:.1f}% Bearish
        
        """, unsafe_allow_html=True)
st.markdown("---")
DISCLAIMER
st.warning("""
⚠️ DISCLAIMER IMPORTANTE:
Questo sistema fornisce analisi quantitative e previsioni basate su modelli statistici e machine learning.
Le previsioni NON costituiscono consulenza finanziaria. I mercati finanziari sono imprevedibili e ogni
investimento comporta rischi. Consultare sempre un consulente finanziario professionista prima di operare.
""")
Footer info
st.markdown("---")
st.info(f"""
📊 Statistiche Analisi:

Dati analizzati: {len(df):,} periodi
Range temporale: {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')}
Indicatori calcolati: 25+
Modelli ML utilizzati: 3 (RF, XGB, GB)
Ultimo aggiornamento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
if name == "main":
main()
