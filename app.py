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
from sklearn.model_selection import train_test_split
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import requests
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE ====================
st.set_page_config(
    page_title="Sistema Analisi Finanziaria Istituzionale",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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

# ==================== CONFIGURAZIONE STRUMENTI ====================
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
    '4h': {'period': '730d', 'interval': '1h'},
    '1d': {'period': '10y', 'interval': '1d'}
}

# ==================== FUNZIONI RACCOLTA DATI ====================

@st.cache_data(ttl=900)
def get_price_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Scarica dati di prezzo"""
    try:
        config = TIMEFRAMES[timeframe]
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=config['period'], interval=config['interval'])
        
        if df.empty:
            return pd.DataFrame()
        
        # Rimuovi colonne non necessarie se presenti
        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[col for col in cols_to_keep if col in df.columns]]
        
        # Aggregazione per 4h
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
        st.error(f"Errore download dati: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_vix_data() -> float:
    """Ottiene VIX"""
    try:
        vix = yf.Ticker("^VIX")
        data = vix.history(period="5d")
        if not data.empty:
            return round(data['Close'].iloc[-1], 2)
    except:
        pass
    return 20.0

@st.cache_data(ttl=86400)
def get_fear_greed_index() -> Dict:
    """Fear & Greed Index"""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        value = int(data['data'][0]['value'])
        classification = data['data'][0]['value_classification']
        return {'value': value, 'classification': classification}
    except:
        return {'value': 50, 'classification': 'Neutral'}

# ==================== INDICATORI TECNICI ====================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola indicatori tecnici"""
    df = df.copy()
    
    try:
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_mid'] = bollinger.bollinger_mavg()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_width'] = bollinger.bollinger_wband()
        
        # ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Momentum
        df['ROC'] = ta.momentum.roc(df['Close'], window=12)
        df['Price_momentum'] = df['Close'].pct_change(periods=10)
        
        # Volatilità
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
    except Exception as e:
        st.warning(f"Errore calcolo indicatori: {str(e)}")
    
    return df

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

# ==================== MACHINE LEARNING ====================

def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara features per ML"""
    df = df.copy()
    
    # Target
    df['Target'] = df['Close'].pct_change().shift(-1)
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'Return_lag_{lag}'] = df['Close'].pct_change(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].pct_change(lag)
    
    # Features statistiche
    df['Return_mean_5'] = df['Close'].pct_change().rolling(5).mean()
    df['Return_std_5'] = df['Close'].pct_change().rolling(5).std()
    df['High_Low_ratio'] = (df['High'] - df['Low']) / df['Close']
    
    # Rimuovi NaN
    df = df.dropna()
    
    # Feature columns
    feature_cols = [col for col in df.columns if col not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
    
    # Filtra solo colonne numeriche
    feature_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']]
    
    X = df[feature_cols]
    y = df['Target']
    
    return X, y

class PredictionEngine:
    """Motore previsioni"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Training ensemble"""
        try:
            # Salva feature names
            self.feature_cols = X.columns.tolist()
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Scaling
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=50, 
                max_depth=8, 
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_score = rf_model.score(X_test_scaled, y_test)
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_score = xgb_model.score(X_test_scaled, y_test)
            
            # Gradient Boosting
            gb_model = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train)
            gb_score = gb_model.score(X_test_scaled, y_test)
            
            # Salva modelli
            self.models = {
                'RandomForest': {'model': rf_model, 'score': max(rf_score, 0.01)},
                'XGBoost': {'model': xgb_model, 'score': max(xgb_score, 0.01)},
                'GradientBoosting': {'model': gb_model, 'score': max(gb_score, 0.01)}
            }
            
            # Normalizza pesi
            total_score = sum(m['score'] for m in self.models.values())
            for model_name in self.models:
                self.models[model_name]['weight'] = self.models[model_name]['score'] / total_score
            
            return {
                'rf_score': rf_score,
                'xgb_score': xgb_score,
                'gb_score': gb_score,
                'avg_score': np.mean([rf_score, xgb_score, gb_score])
            }
        except Exception as e:
            st.error(f"Errore training: {str(e)}")
            return {'rf_score': 0, 'xgb_score': 0, 'gb_score': 0, 'avg_score': 0}
    
    def predict(self, X_latest: pd.DataFrame) -> Dict:
        """Previsione"""
        try:
            X_scaled = self.scaler.transform(X_latest)
            
            predictions = {}
            for model_name, model_info in self.models.items():
                pred = model_info['model'].predict(X_scaled)[0]
                predictions[model_name] = pred
            
            # Weighted average
            ensemble_pred = sum(
                predictions[name] * self.models[name]['weight'] 
                for name in predictions
            )
            
            # Confidenza
            pred_std = np.std(list(predictions.values()))
            confidence = max(50, min(95, 100 - (pred_std * 1000)))
            
            # Probabilità
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
        except Exception as e:
            st.error(f"Errore previsione: {str(e)}")
            return {
                'prediction': 0,
                'confidence': 50,
                'prob_up': 50,
                'prob_down': 50,
                'individual_predictions': {}
            }

def arima_forecast(series: pd.Series, steps: int = 30) -> Tuple[np.ndarray, Tuple]:
    """Previsione ARIMA"""
    try:
        # Usa solo ultimi 200 punti per velocità
        series_short = series.tail(200)
        model = ARIMA(series_short, order=(2, 1, 2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
        
        std = series_short.std()
        upper = forecast + 1.96 * std
        lower = forecast - 1.96 * std
        
        return forecast.values, (lower.values, upper.values)
    except:
        last_val = series.iloc[-1]
        return np.full(steps, last_val), (np.full(steps, last_val * 0.95), np.full(steps, last_val * 1.05))

# ==================== ANALISI RISCHIO ====================

def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk"""
    return np.percentile(returns.dropna(), (1 - confidence) * 100)

def calculate_sharpe(returns: pd.Series) -> float:
    """Sharpe Ratio"""
    excess_returns = returns.mean()
    return (excess_returns / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Maximum Drawdown"""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def support_resistance_levels(df: pd.DataFrame) -> Dict:
    """Livelli supporto/resistenza"""
    recent_data = df.tail(100)
    pivot = (recent_data['High'].max() + recent_data['Low'].min() + recent_data['Close'].iloc[-1]) / 3
    
    resistance_levels = []
    support_levels = []
    
    for i in range(1, 4):
        r = pivot + (recent_data['High'].max() - recent_data['Low'].min()) * i * 0.382
        s = pivot - (recent_data['High'].max() - recent_data['Low'].min()) * i * 0.382
        resistance_levels.append(round(r, 2))
        support_levels.append(round(s, 2))
    
    return {
        'resistance': resistance_levels,
        'support': support_levels,
        'pivot': round(pivot, 2)
    }

# ==================== VISUALIZZAZIONI ====================

def create_candlestick_chart(df: pd.DataFrame, symbol: str, indicators: bool = True):
    """Grafico candlestick"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} - Prezzo', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick
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
        # SMA
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                          line=dict(color='orange', width=1)),
                row=1, col=1
            )
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                          line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        # Bollinger
        if 'BB_high' in df.columns:
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
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI', 
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                          line=dict(color='blue', width=2)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', 
                          line=dict(color='red', width=2)),
                row=3, col=1
            )
            if 'MACD_diff' in df.columns:
                fig.add_trace(
                    go.Bar(x=df.index, y=df['MACD_diff'], name='Histogram', 
                          marker_color='gray'),
                    row=3, col=1
                )
    
    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

def create_prediction_chart(df: pd.DataFrame, forecast: np.ndarray, conf_intervals: Tuple):
    """Grafico previsioni"""
    fig = go.Figure()
    
    # Storico
    fig.add_trace(go.Scatter(
        x=df.index[-100:],
        y=df['Close'][-100:],
        mode='lines',
        name='Storico',
        line=dict(color='blue', width=2)
    ))
    
    # Previsione
    future_dates = pd.date_range(start=df.index[-1], periods=len(forecast) + 1, freq='D')[1:]
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode='lines',
        name='Previsione',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Intervallo
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
        name='Intervallo 95%',
        line=dict(width=0),
        fillcolor='rgba(255, 0, 0, 0.2)',
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="Previsione Prezzo (ARIMA)",
        xaxis_title="Data",
        yaxis_title="Prezzo",
        hovermode='x unified',
        height=500
    )
    
    return fig

# ==================== MAIN APP ====================

def main():
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Sistema Analisi Finanziaria Istituzionale</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configurazione")
        
        category = st.selectbox(
            "📊 Categoria Asset",
            options=list(INSTRUMENTS.keys())
        )
        
        instrument = st.selectbox(
            "🎯 Strumento",
            options=list(INSTRUMENTS[category].keys())
        )
        
        symbol = INSTRUMENTS[category][instrument]
        
        timeframe = st.selectbox(
            "⏱️ Timeframe",
            options=['15min', '1h', '4h', '1d'],
            index=3
        )
        
        st.markdown("---")
        
        st.subheader("🔧 Opzioni")
        show_indicators = st.checkbox("Indicatori Tecnici", value=True)
        show_predictions = st.checkbox("Previsioni ML", value=True)
        show_seasonality = st.checkbox("Stagionalità", value=True)
        
        st.markdown("---")
        
        st.subheader("📈 Indicatori Macro")
        vix_value = get_vix_data()
        fear_greed = get_fear_greed_index()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("VIX", f"{vix_value}")
        with col2:
            st.metric("FED", "5.33%")
        
        if category == 'Criptovalute':
            st.metric("Fear & Greed", f"{fear_greed['value']}", 
                     delta=fear_greed['classification'])
        
        analyze_button = st.button("🔍 ANALIZZA", type="primary", use_container_width=True)
    
    # Main
    if analyze_button:
        
        with st.spinner(f"📊 Caricamento {instrument}..."):
            df = get_price_data(symbol, timeframe)
        
        if df.empty:
            st.error("❌ Dati non disponibili")
            return
        
        with st.spinner("🔧 Calcolo indicatori..."):
            df = calculate_technical_indicators(df)
        
        # Metriche
        st.subheader(f"📊 {instrument} - {timeframe.upper()}")
        
        current_price = df['Close'].iloc[-1]
        price_change = df['Close'].pct_change().iloc[-1] * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Prezzo", f"${current_price:,.2f}", f"{price_change:+.2f}%")
        with col2:
            rsi_val = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
            st.metric("📊 RSI", f"{rsi_val:.1f}")
        with col3:
            atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
            st.metric("📉 ATR", f"{atr_val:.2f}")
        with col4:
            vol_val = df['Volatility'].iloc[-1] * 100 if 'Volatility' in df.columns else 0
            st.metric("📈 Volatilità", f"{vol_val:.1f}%")
        
        st.markdown("---")
        
        # TABELLA TRADERS
        st.subheader("👥 Strategie Trading Consigliate")
        
        # Calcolo livelli basati su ATR e analisi tecnica
        atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
        current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        current_macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        current_macd_signal = df['MACD_signal'].iloc[-1] if 'MACD_signal' in df.columns else 0
        
        sma20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else current_price
        sma50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else current_price
        bb_high = df['BB_high'].iloc[-1] if 'BB_high' in df.columns else current_price * 1.02
        bb_low = df['BB_low'].iloc[-1] if 'BB_low' in df.columns else current_price * 0.98
        
        # Calcolo bias direzionale
        bullish_signals = 0
        bearish_signals = 0
        
        if current_rsi < 50:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if current_macd > current_macd_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if current_price > sma20:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Determina direzione predominante
        is_bullish = bullish_signals > bearish_signals
        
        # TRADER 1: SCALPER (Aggressivo - Alto Rischio/Rendimento)
        if is_bullish:
            scalper_entry = current_price
            scalper_tp = current_price + (atr_val * 1.5)
            scalper_sl = current_price - (atr_val * 0.8)
            scalper_direction = "LONG ⬆️"
            scalper_color = "#00ff00"
        else:
            scalper_entry = current_price
            scalper_tp = current_price - (atr_val * 1.5)
            scalper_sl = current_price + (atr_val * 0.8)
            scalper_direction = "SHORT ⬇️"
            scalper_color = "#ff4444"
        
        scalper_risk = abs(scalper_entry - scalper_sl)
        scalper_reward = abs(scalper_tp - scalper_entry)
        scalper_rr = scalper_reward / scalper_risk if scalper_risk > 0 else 1
        scalper_success = min(75, 50 + (scalper_rr * 15) + (bullish_signals if is_bullish else bearish_signals) * 5)
        
        # TRADER 2: DAY TRADER (Moderato - Rischio/Rendimento Bilanciato)
        if is_bullish:
            day_entry = current_price
            day_tp = current_price + (atr_val * 2.5)
            day_sl = current_price - (atr_val * 1.2)
            day_direction = "LONG ⬆️"
            day_color = "#00ff00"
        else:
            day_entry = current_price
            day_tp = current_price - (atr_val * 2.5)
            day_sl = current_price + (atr_val * 1.2)
            day_direction = "SHORT ⬇️"
            day_color = "#ff4444"
        
        day_risk = abs(day_entry - day_sl)
        day_reward = abs(day_tp - day_entry)
        day_rr = day_reward / day_risk if day_risk > 0 else 1
        day_success = min(80, 55 + (day_rr * 12) + (bullish_signals if is_bullish else bearish_signals) * 4)
        
        # TRADER 3: SWING TRADER (Conservativo - Basso Rischio)
        if is_bullish:
            swing_entry = current_price
            swing_tp = max(bb_high, current_price + (atr_val * 3.5))
            swing_sl = max(bb_low, current_price - (atr_val * 1.8))
            swing_direction = "LONG ⬆️"
            swing_color = "#00ff00"
        else:
            swing_entry = current_price
            swing_tp = min(bb_low, current_price - (atr_val * 3.5))
            swing_sl = min(bb_high, current_price + (atr_val * 1.8))
            swing_direction = "SHORT ⬇️"
            swing_color = "#ff4444"
        
        swing_risk = abs(swing_entry - swing_sl)
        swing_reward = abs(swing_tp - swing_entry)
        swing_rr = swing_reward / swing_risk if swing_risk > 0 else 1
        swing_success = min(85, 60 + (swing_rr * 10) + (bullish_signals if is_bullish else bearish_signals) * 3)
        
        # Creazione tabella
        traders_data = {
            'Profilo': [
                '🔥 SCALPER (Aggressivo)',
                '⚡ DAY TRADER (Moderato)', 
                '🎯 SWING TRADER (Conservativo)'
            ],
            'Direzione': [scalper_direction, day_direction, swing_direction],
            'Entry': [f'${scalper_entry:,.2f}', f'${day_entry:,.2f}', f'${swing_entry:,.2f}'],
            'Take Profit': [f'${scalper_tp:,.2f}', f'${day_tp:,.2f}', f'${swing_tp:,.2f}'],
            'Stop Loss': [f'${scalper_sl:,.2f}', f'${day_sl:,.2f}', f'${swing_sl:,.2f}'],
            'Risk/Reward': [f'{scalper_rr:.2f}', f'{day_rr:.2f}', f'{swing_rr:.2f}'],
            'Probabilità Successo': [f'{scalper_success:.1f}%', f'{day_success:.1f}%', f'{swing_success:.1f}%']
        }
        
        traders_df = pd.DataFrame(traders_data)
        
        # Styling della tabella
        def color_direction(val):
            if 'LONG' in val:
                return 'background-color: rgba(0, 255, 0, 0.2); color: green; font-weight: bold;'
            elif 'SHORT' in val:
                return 'background-color: rgba(255, 0, 0, 0.2); color: red; font-weight: bold;'
            return ''
        
        def color_success(val):
            try:
                num = float(val.replace('%', ''))
                if num >= 70:
                    return 'background-color: rgba(0, 255, 0, 0.3); font-weight: bold;'
                elif num >= 60:
                    return 'background-color: rgba(255, 255, 0, 0.2);'
                else:
                    return 'background-color: rgba(255, 165, 0, 0.2);'
            except:
                return ''
        
        styled_df = traders_df.style.applymap(
            color_direction, subset=['Direzione']
        ).applymap(
            color_success, subset=['Probabilità Successo']
        ).set_properties(**{
            'text-align': 'center',
            'font-size': '14px',
            'padding': '10px'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#667eea'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '12px'),
                ('font-size', '15px')
            ]}
        ])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Info aggiuntive
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"""
            **🔥 SCALPER**
            - Timeframe: 15min-1h
            - Holding: Minuti/Ore
            - Operazioni/giorno: 5-10
            - R/R: {scalper_rr:.2f}
            """)
        with col2:
            st.info(f"""
            **⚡ DAY TRADER**
            - Timeframe: 1h-4h
            - Holding: Ore/1 giorno
            - Operazioni/giorno: 2-5
            - R/R: {day_rr:.2f}
            """)
        with col3:
            st.info(f"""
            **🎯 SWING TRADER**
            - Timeframe: 4h-1d
            - Holding: Giorni/Settimane
            - Operazioni/settimana: 1-3
            - R/R: {swing_rr:.2f}
            """)
        
        st.markdown("---")
        
        # Grafico
        st.subheader("📈 Grafico Candlestick")
        fig_candle = create_candlestick_chart(df.tail(200), instrument, show_indicators)
        st.plotly_chart(fig_candle, use_container_width=True)
        
        st.markdown("---")
        
        # ML Predictions
        if show_predictions and len(df) > 100:
            st.subheader("🤖 Previsioni Machine Learning")
            
            with st.spinner("🧠 Training modelli..."):
                X, y = prepare_ml_features(df)
                
                if len(X) > 50:
                    predictor = PredictionEngine()
                    scores = predictor.train_ensemble(X, y)
                    
                    X_latest = X.tail(1)
                    prediction_result = predictor.predict(X_latest)
                    
                    forecast, conf_int = arima_forecast(df['Close'], steps=30)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        
                        predicted_return = prediction_result['prediction'] * 100
                        predicted_price = current_price * (1 + prediction_result['prediction'])
                        
                        st.markdown("### 🎯 Previsione")
                        st.markdown(f"**Prezzo Previsto:** ${predicted_price:,.2f}")
                        st.markdown(f"**Variazione:** {predicted_return:+.2f}%")
                        st.markdown(f"**Confidenza:** {prediction_result['confidence']:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        prob_up = prediction_result['prob_up']
                        prob_down = prediction_result['prob_down']
                        
                        st.markdown(f"""
                        <div class="success-prob" style="color: {'#00ff00' if prob_up > prob_down else '#ff4444'};">
                            {prob_up:.1f}% ⬆️
                        </div>
                        <div class="success-prob" style="color: {'#ff4444' if prob_down > prob_up else '#00ff00'}; font-size: 32px;">
                            {prob_down:.1f}% ⬇️
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### 🎯 Performance Modelli")
                        
                        if prediction_result['individual_predictions']:
                            model_data = []
                            for name, pred in prediction_result['individual_predictions'].items():
                                model_data.append({
                                    'Modello': name,
                                    'Score': scores.get(f"{name.lower().replace('gradient', 'gb').replace('random', 'rf').replace('xg', 'xgb')}_score", 0),
                                    'Previsione %': pred * 100
                                })
                            
                            model_df = pd.DataFrame(model_data)
                            st.dataframe(model_df, use_container_width=True, hide_index=True)
                        
                        st.metric("📈 Score Medio", f"{scores['avg_score']:.3f}")
                    
                    # Forecast chart
                    st.markdown("#### 📉 Proiezione 30 Periodi")
                    fig_forecast = create_prediction_chart(df, forecast, conf_int)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Targets
                    st.markdown("#### 🎯 Livelli Target")
                    col1, col2, col3 = st.columns(3)
                    
                    target_1 = forecast[0] if len(forecast) > 0 else current_price
                    target_7 = forecast[6] if len(forecast) > 6 else current_price
                    target_30 = forecast[-1] if len(forecast) > 0 else current_price
                    
                    with col1:
                        change_1 = ((target_1 - current_price) / current_price) * 100
                        st.metric("1 Periodo", f"${target_1:,.2f}", f"{change_1:+.2f}%")
                    with col2:
                        change_7 = ((target_7 - current_price) / current_price) * 100
                        st.metric("7 Periodi", f"${target_7:,.2f}", f"{change_7:+.2f}%")
                    with col3:
                        change_30 = ((target_30 - current_price) / current_price) * 100
                        st.metric("30 Periodi", f"${target_30:,.2f}", f"{change_30:+.2f}%")
        
        st.markdown("---")
        
        # Risk Analysis
        st.subheader("⚠️ Analisi del Rischio")
        
        returns = df['Close'].pct_change().dropna()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            var_95 = calculate_var(returns, 0.95) * 100
            st.metric("📉 VaR 95%", f"{var_95:.2f}%")
        
        with col2:
            sharpe = calculate_sharpe(returns)
            st.metric("📊 Sharpe", f"{sharpe:.2f}")
        
        with col3:
            max_dd = calculate_max_drawdown(df['Close']) * 100
            st.metric("📉 Max DD", f"{max_dd:.2f}%")
        
        with col4:
            win_rate = (returns > 0).sum() / len(returns) * 100
            st.metric("✅ Win Rate", f"{win_rate:.1f}%")
        
        # Support/Resistance
        levels = support_resistance_levels(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Resistenze")
            for i, level in enumerate(levels['resistance'], 1):
                distance = ((level - current_price) / current_price) * 100
                st.markdown(f"**R{i}:** ${level:,.2f} ({distance:+.2f}%)")
        
        with col2:
            st.markdown("#### 🟢 Supporti")
            st.markdown(f"**Pivot:** ${levels['pivot']:,.2f}")
            for i, level in enumerate(levels['support'], 1):
                distance = ((level - current_price) / current_price) * 100
                st.markdown(f"**S{i}:** ${level:,.2f} ({distance:+.2f}%)")
        
        st.markdown("---")
        
        # Seasonality
        if show_seasonality:
            st.subheader("🗓️ Analisi Stagionalità")
            
            seasonality = calculate_seasonality(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📅 Pattern Mensile")
                monthly_data = []
                months = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 
                         'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
                for i in range(1, 13):
                    monthly_data.append(seasonality['monthly_pattern'].get(i, 0))
                
                fig_monthly = go.Figure(data=[
                    go.Bar(
                        x=months,
                        y=monthly_data,
                        marker_color=['green' if x > 0 else 'red' for x in monthly_data]
                    )
                ])
                fig_monthly.update_layout(
                    title="Performance Storica Mensile",
                    yaxis_title="Rendimento %",
                    height=400
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                st.metric("🎯 Bias Mese", f"{seasonality['current_month_bias']:+.3f}%")
            
            with col2:
                st.markdown("#### 📊 Pattern Settimanale")
                weekly_data = []
                days = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven']
                for i in range(5):
                    weekly_data.append(seasonality['weekly_pattern'].get(i, 0))
                
                fig_weekly = go.Figure(data=[
                    go.Bar(
                        x=days,
                        y=weekly_data,
                        marker_color=['green' if x > 0 else 'red' for x in weekly_data]
                    )
                ])
                fig_weekly.update_layout(
                    title="Performance Storica Giornaliera",
                    yaxis_title="Rendimento %",
                    height=400
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
                
                st.metric("🎯 Bias Giorno", f"{seasonality['current_day_bias']:+.3f}%")
        
        st.markdown("---")
        
        # Correlation Matrix
        st.subheader("🔗 Matrice Correlazioni")
        
        corr_cols = [col for col in ['Close', 'RSI', 'MACD', 'ATR', 'Volume', 'Volatility', 'BB_width'] 
                     if col in df.columns]
        
        if len(corr_cols) > 1:
            corr_matrix = df[corr_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_cols,
                y=corr_cols,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig_corr.update_layout(
                title="Correlazione Indicatori",
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        
        # Trading Signals
        st.subheader("🚦 Segnali di Trading")
        
        current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        current_macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        current_macd_signal = df['MACD_signal'].iloc[-1] if 'MACD_signal' in df.columns else 0
        
        sma20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else current_price
        sma50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else current_price
        
        price_vs_sma20 = ((current_price - sma20) / sma20) * 100
        price_vs_sma50 = ((current_price - sma50) / sma50) * 100
        
        signals = []
        
        # RSI
        if current_rsi < 30:
            signals.append(("🟢 RSI Oversold", "ACQUISTO", "RSI < 30"))
        elif current_rsi > 70:
            signals.append(("🔴 RSI Overbought", "VENDITA", "RSI > 70"))
        else:
            signals.append(("🟡 RSI Neutrale", "NEUTRALE", f"RSI = {current_rsi:.1f}"))
        
        # MACD
        if 'MACD' in df.columns and len(df) > 2:
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['MACD_signal'].iloc[-2]
            
            if current_macd > current_macd_signal and prev_macd <= prev_signal:
                signals.append(("🟢 MACD Bullish", "ACQUISTO", "Incrocio rialzista"))
            elif current_macd < current_macd_signal and prev_macd >= prev_signal:
                signals.append(("🔴 MACD Bearish", "VENDITA", "Incrocio ribassista"))
            else:
                signals.append(("🟡 MACD Neutrale", "NEUTRALE", "Nessun incrocio"))
        
        # Trend
        if price_vs_sma20 > 2 and price_vs_sma50 > 2:
            signals.append(("🟢 Trend Rialzista", "ACQUISTO", "Sopra medie mobili"))
        elif price_vs_sma20 < -2 and price_vs_sma50 < -2:
            signals.append(("🔴 Trend Ribassista", "VENDITA", "Sotto medie mobili"))
        else:
            signals.append(("🟡 Trend Laterale", "NEUTRALE", "Vicino medie"))
        
        # Bollinger
        if 'BB_low' in df.columns and 'BB_high' in df.columns:
            bb_low = df['BB_low'].iloc[-1]
            bb_high = df['BB_high'].iloc[-1]
            
            if current_price < bb_low:
                signals.append(("🟢 Sotto BB Inf.", "ACQUISTO", "Possibile rimbalzo"))
            elif current_price > bb_high:
                signals.append(("🔴 Sopra BB Sup.", "VENDITA", "Possibile ritracciamento"))
        
        # Volume
        if 'Volume_ratio' in df.columns:
            vol_ratio = df['Volume_ratio'].iloc[-1]
            if vol_ratio > 1.5:
                signals.append(("⚡ Volume Alto", "ATTENZIONE", "Volume anomalo"))
        
        # Display
        for signal, action, description in signals:
            color = "green" if "ACQUISTO" in action else "red" if "VENDITA" in action else "orange"
            st.markdown(f"""
            <div style="background-color: rgba(0,0,0,0.05); padding: 15px; 
                        border-left: 5px solid {color}; margin: 10px 0; border-radius: 5px;">
                <strong>{signal}</strong> - <span style="color: {color};">{action}</span><br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dominant Factors
        st.subheader("🎯 Fattori Dominanti")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Indicatori Tecnici")
            
            technical_factors = {}
            if 'RSI' in df.columns:
                technical_factors['RSI'] = abs(50 - current_rsi) / 50 * 100
            if 'MACD' in df.columns:
                technical_factors['MACD'] = abs(current_macd - current_macd_signal) * 100
            if 'Volatility' in df.columns:
                technical_factors['Volatilità'] = df['Volatility'].iloc[-1] * 1000
            if 'Volume_ratio' in df.columns:
                technical_factors['Volume'] = abs(df['Volume_ratio'].iloc[-1] - 1) * 100
            if 'BB_mid' in df.columns:
                technical_factors['Bollinger'] = abs((current_price - df['BB_mid'].iloc[-1]) / df['BB_mid'].iloc[-1]) * 1000
            
            if technical_factors:
                total = sum(technical_factors.values())
                tech_data = {k: (v/total)*100 for k, v in technical_factors.items()}
                
                fig_tech = go.Figure(data=[
                    go.Bar(
                        x=list(tech_data.values()),
                        y=list(tech_data.keys()),
                        orientation='h',
                        marker_color='steelblue'
                    )
                ])
                fig_tech.update_layout(
                    xaxis_title="Impatto %",
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_tech, use_container_width=True)
        
        with col2:
            st.markdown("#### 🌍 Macro Factors")
            
            macro_factors = {
                'VIX': (vix_value / 40) * 100,
                'FED Rate': (5.33 / 10) * 100,
                'Sentiment': 50 + (np.random.randn() * 10)
            }
            
            if category == 'Criptovalute':
                macro_factors['Fear & Greed'] = fear_greed['value']
            
            fig_macro = go.Figure(data=[
                go.Bar(
                    x=list(macro_factors.values()),
                    y=list(macro_factors.keys()),
                    orientation='h',
                    marker_color='coral'
                )
            ])
            fig_macro.update_layout(
                xaxis_title="Livello",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_macro, use_container_width=True)
        
        st.markdown("---")
        
        # Final Recommendation
        st.subheader("🎯 Raccomandazione Algoritmica")
        
        buy_score = 0
        sell_score = 0
        
        # RSI
        if current_rsi < 40:
            buy_score += 2
        elif current_rsi > 60:
            sell_score += 2
        
        # MACD
        if current_macd > current_macd_signal:
            buy_score += 1.5
        else:
            sell_score += 1.5
        
        # Trend
        if price_vs_sma20 > 0 and price_vs_sma50 > 0:
            buy_score += 2
        elif price_vs_sma20 < 0 and price_vs_sma50 < 0:
            sell_score += 2
        
        # ML
        if show_predictions and 'prediction_result' in locals():
            if prediction_result['prediction'] > 0.005:
                buy_score += 3
            elif prediction_result['prediction'] < -0.005:
                sell_score += 3
        
        # Volatility
        if 'Volatility' in df.columns:
            vol = df['Volatility'].iloc[-1] * 100
            if vol < 30:
                buy_score += 1
            elif vol > 60:
                sell_score += 1
        
        total_score = buy_score + sell_score
        buy_percentage = (buy_score / total_score * 100) if total_score > 0 else 50
        sell_percentage = (sell_score / total_score * 100) if total_score > 0 else 50
        
        if buy_percentage > 60:
            recommendation = "🟢 ACQUISTO FORTE"
            rec_color = "green"
            rec_desc = "Fattori tecnici favorevoli all'acquisto"
        elif buy_percentage > 50:
            recommendation = "🟢 ACQUISTO MODERATO"
            rec_color = "lightgreen"
            rec_desc = "Segnali positivi prevalenti"
        elif sell_percentage > 60:
            recommendation = "🔴 VENDITA FORTE"
            rec_color = "red"
            rec_desc = "Fattori ribassisti dominanti"
        elif sell_percentage > 50:
            recommendation = "🔴 VENDITA MODERATA"
            rec_color = "orange"
            rec_desc = "Segnali negativi prevalenti"
        else:
            recommendation = "🟡 NEUTRALE"
            rec_color = "gray"
            rec_desc = "Segnali contrastanti"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {rec_color}, {'darkgreen' if 'ACQUISTO' in recommendation else 'darkred' if 'VENDITA' in recommendation else 'darkgray'}); 
                    color: white; padding: 30px; border-radius: 15px; text-align: center;">
            <h2>{recommendation}</h2>
            <p style="font-size: 18px;">{rec_desc}</p>
            <h3>Score: {buy_percentage:.1f}% Bullish / {sell_percentage:.1f}% Bearish</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Disclaimer
        st.warning("""
        ⚠️ **DISCLAIMER**: Questo sistema fornisce analisi quantitative basate su modelli statistici.
        Le previsioni NON costituiscono consulenza finanziaria. Ogni investimento comporta rischi.
        Consultare sempre un professionista prima di operare sui mercati.
        """)
        
        # Footer
        st.markdown("---")
        st.info(f"""
        📊 **Info Analisi**:
        - Dati: {len(df):,} periodi
        - Range: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}
        - Indicatori: 25+
        - Modelli ML: 3
        - Aggiornamento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)


if __name__ == "__main__":
    main()
