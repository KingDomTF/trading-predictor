import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import io
import base64
import warnings
import math
import random

# Suppress warnings
warnings.filterwarnings("ignore")

# Configurazione pagina
st.set_page_config(
    page_title="Trading Success Predictor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stili CSS personalizzati
st.markdown("""
    <style>
        .main {
            background-color: #0b1220;
            color: #e5e7eb;
        }
        .stMetric {
            background-color: #111827;
            padding: 10px;
            border-radius: 10px;
        }
        .big-font {
            font-size: 28px !important;
            font-weight: bold;
            color: #e5e7eb;
        }
        .medium-font {
            font-size: 18px !important;
            color: #9ca3af;
        }
        .success-badge {
            background: linear-gradient(90deg, #22c55e, #16a34a);
            padding: 6px 12px;
            border-radius: 999px;
            color: white;
            font-weight: bold;
            font-size: 12px;
            display: inline-block;
        }
        .danger-badge {
            background: linear-gradient(90deg, #ef4444, #b91c1c);
            padding: 6px 12px;
            border-radius: 999px;
            color: white;
            font-weight: bold;
            font-size: 12px;
            display: inline-block;
        }
        .neutral-badge {
            background: linear-gradient(90deg, #6b7280, #4b5563);
            padding: 6px 12px;
            border-radius: 999px;
            color: white;
            font-weight: bold;
            font-size: 12px;
            display: inline-block;
        }
        .section-title {
            font-size: 22px !important;
            font-weight: bold;
            margin-top: 20px;
            color: #e5e7eb;
        }
        .section-subtitle {
            font-size: 16px !important;
            color: #9ca3af;
        }
        .sidebar .sidebar-content {
            background-color: #020617;
        }
        .risk-gauge {
            font-size: 40px;
            font-weight: bold;
        }
        .success-prob {
            font-size: 40px;
            font-weight: bold;
        }
        .profit-text {
            font-size: 26px;
            font-weight: bold;
        }
        .small-badge {
            background-color: #111827;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 11px;
            color: #9ca3af;
            display: inline-block;
        }
        .footer-text {
            font-size: 12px;
            color: #6b7280;
        }
    </style>
""", unsafe_allow_html=True)

# Funzione per il caching dei dati
@st.cache_data(ttl=3600)
def load_data(symbol, period="5y", interval="1d"):
    """
    Carica dati storici per il simbolo specificato da Yahoo Finance.
    """
    try:
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if data.empty:
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Errore nel download dei dati per {symbol}: {e}")
        return None

# Funzione per salvare e caricare il modello
def get_model_filename(symbol, risk_level, data_interval):
    return f"model_{symbol.replace('.', '_')}_{risk_level}_{data_interval}.pkl"

def save_model(model, symbol, risk_level, data_interval):
    filename = get_model_filename(symbol, risk_level, data_interval)
    joblib.dump(model, filename)

def load_model(symbol, risk_level, data_interval):
    filename = get_model_filename(symbol, risk_level, data_interval)
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

# Funzione per calcolare indicatori tecnici
def compute_indicators(df):
    """
    Calcolo indicatori tecnici principali.
    """
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=20).std() * np.sqrt(252)
    
    # RSI
    rsi_period = 14
    df["RSI"] = RSIIndicator(close=df["Close"], window=rsi_period).rsi()
    
    # MACD
    macd_indicator = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd_indicator.macd()
    df["MACD_Signal"] = macd_indicator.macd_signal()
    df["MACD_Diff"] = macd_indicator.macd_diff()
    
    # Bollinger Bands
    bb_indicator = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_High"] = bb_indicator.bollinger_hband()
    df["BB_Low"] = bb_indicator.bollinger_lband()
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["Close"]
    
    # EMAs
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["EMA_100"] = df["Close"].ewm(span=100).mean()
    
    # Supporti e resistenze semplificati (rolling max/min)
    df["Rolling_Max_20"] = df["High"].rolling(window=20).max()
    df["Rolling_Min_20"] = df["Low"].rolling(window=20).min()
    
    # Trend semplice
    df["Trend"] = df["EMA_20"] - df["EMA_50"]
    
    # Volume moving average
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Spike"] = df["Volume"] / df["Volume_MA_20"]
    
    # Lag features
    for lag in range(1, 6):
        df[f"Return_Lag_{lag}"] = df["Returns"].shift(lag)
        df[f"RSI_Lag_{lag}"] = df["RSI"].shift(lag)
    
    # Target: future return su 10 giorni
    horizon = 10
    df["Future_Return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    
    df.dropna(inplace=True)
    return df

# Funzione per creare dataset di training
def create_features_targets(df):
    feature_cols = [
        "Returns", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Diff",
        "BB_Width", "EMA_20", "EMA_50", "EMA_100", "Rolling_Max_20",
        "Rolling_Min_20", "Trend", "Volume_MA_20", "Volume_Spike"
    ]
    for lag in range(1, 6):
        feature_cols.append(f"Return_Lag_{lag}")
        feature_cols.append(f"RSI_Lag_{lag}")
    
    X = df[feature_cols].values
    y = df["Future_Return"].values
    return X, y, feature_cols

# Funzione per addestrare il modello
def train_model(X, y, risk_level):
    if risk_level == "Basso":
        n_estimators = 150
        max_depth = 6
        min_samples_leaf = 5
    elif risk_level == "Medio":
        n_estimators = 250
        max_depth = 8
        min_samples_leaf = 3
    else:  # Alto
        n_estimators = 350
        max_depth = 10
        min_samples_leaf = 2
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

# Funzione per valutare il modello
def evaluate_model(model, X, y):
    try:
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        return mae, r2, rmse
    except NotFittedError:
        return None, None, None

# Funzione per calcolare la probabilità di successo
def calculate_success_probability(predicted_return, volatility, risk_level):
    base_prob = 0.5
    if predicted_return > 0:
        base_prob += min(predicted_return * 2.5, 0.3)
    else:
        base_prob += max(predicted_return * 3.0, -0.25)
    
    if volatility is not None and not np.isnan(volatility):
        if volatility < 0.2:
            base_prob += 0.1
        elif volatility > 0.4:
            base_prob -= 0.1
    
    if risk_level == "Basso":
        base_prob -= 0.05
    elif risk_level == "Alto":
        base_prob += 0.05
    
    base_prob += random.uniform(-0.05, 0.05)
    
    success_prob = max(0.0, min(1.0, base_prob))
    return success_prob

# Funzione per generare analisi testuale
def generate_textual_analysis(success_prob, predicted_return, volatility, risk_level, price_target, symbol):
    if success_prob >= 0.75:
        sentiment_label = "Fortemente Positivo"
        badge_class = "success-badge"
    elif success_prob >= 0.6:
        sentiment_label = "Positivo"
        badge_class = "success-badge"
    elif success_prob >= 0.45:
        sentiment_label = "Neutro"
        badge_class = "neutral-badge"
    else:
        sentiment_label = "Negativo"
        badge_class = "danger-badge"
    
    if predicted_return >= 0.3:
        take_profit_suggestion = "Scenario di forte spinta rialzista: valuta un take profit tra +25% e +40% rispetto al prezzo attuale."
    elif predicted_return >= 0.15:
        take_profit_suggestion = "Scenario rialzista interessante: valuta un take profit tra +15% e +25%."
    elif predicted_return >= 0:
        take_profit_suggestion = "Scenario moderatamente rialzista: valuta un target prudente tra +8% e +15%."
    else:
        take_profit_suggestion = "Scenario ribassista o poco interessante: meglio mantenersi cauti o evitare nuovi ingressi."
    
    if volatility is not None and not np.isnan(volatility):
        if volatility < 0.18:
            volatility_desc = "bassa"
        elif volatility < 0.30:
            volatility_desc = "moderata"
        elif volatility < 0.45:
            volatility_desc = "elevata"
        else:
            volatility_desc = "molto elevata"
    else:
        volatility_desc = "non definita"
    
    if risk_level == "Basso":
        risk_profile_text = (
            "Con un profilo di rischio basso è fondamentale privilegiare operazioni ad alta probabilità "
            "di successo e con drawdown contenuto. È consigliabile usare stop loss stretti e posizioni ridotte."
        )
    elif risk_level == "Medio":
        risk_profile_text = (
            "Con un profilo di rischio medio puoi accettare una moderata volatilità in cambio di un rendimento potenzialmente maggiore, "
            "mantenendo comunque una buona disciplina nella gestione del rischio."
        )
    else:
        risk_profile_text = (
            "Con un profilo di rischio alto puoi puntare a operazioni più aggressive e speculative, "
            "ma è fondamentale monitorare attentamente la posizione ed essere pronto a uscire rapidamente se il setup si deteriora."
        )
    
    if price_target is not None:
        target_text = f"L'AI individua un possibile target di prezzo a **{price_target:.2f}** per {symbol}, in linea con lo scenario previsto."
    else:
        target_text = f"L'AI non individua un target preciso per {symbol}, ma suggerisce un monitoraggio attivo del trend e dei livelli chiave di supporto/resistenza."
    
    analysis_text = f"""
<div class="{badge_class}">Sentiment AI: {sentiment_label}</div>

**Interpretazione generale**

L'AI ha calcolato una probabilità di successo di circa **{success_prob*100:.1f}%** per questa operazione su **{symbol}**, 
con un rendimento atteso a 10 giorni di circa **{predicted_return*100:.1f}%**. La volatilità storica attuale è **{volatility_desc}**, 
il che influisce direttamente sul rapporto rischio/rendimento dell'operazione.

{target_text}

**Suggerimenti di gestione operativa**

- {take_profit_suggestion}
- Utilizza uno stop loss coerente con il tuo profilo di rischio e la volatilità attuale.
- Valuta eventuali ingressi graduali (scaling in) o uscite parziali (scaling out) nelle aree di supporto/resistenza chiave.
- Evita di sovraesporsi su un singolo titolo: mantieni sempre il portafoglio ben diversificato.

**Coerenza con il tuo profilo di rischio**

{risk_profile_text}

Ricorda: questa analisi non è un consiglio finanziario, ma uno strumento avanzato di supporto alle tue decisioni di trading.
"""
    
    return analysis_text

# Funzione per scaricare dati come CSV
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Scarica i dati in CSV</a>'
    return href

# Funzione per scaricare il modello
def get_model_download_link(model, filename="trading_model.pkl"):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="{filename}">📥 Scarica il modello AI</a>'
    return href

# Funzione per caricare modello da file
def load_uploaded_model(uploaded_file):
    try:
        model = joblib.load(uploaded_file)
        return model
    except Exception as e:
        st.error(f"Errore nel caricamento del modello: {e}")
        return None

# Funzione per recuperare prezzo live corrente
@st.cache_data(ttl=60)
def fetch_live_price(symbol):
    """
    Recupera il prezzo live corrente e il precedente close per il simbolo
    """
    try:
        ticker = yf.Ticker(symbol)
        live_data = ticker.history(period="1d", interval="1m")
        if not live_data.empty:
            live_price = live_data["Close"].iloc[-1]
        else:
            live_price = None
        
        prev_data = ticker.history(period="5d", interval="1d")
        if not prev_data.empty and len(prev_data) > 1:
            prev_close = prev_data["Close"].iloc[-2]
        else:
            prev_close = None
        
        return live_price, prev_close
    except Exception:
        return None, None

# SIDEBAR
with st.sidebar:
    st.markdown("<div class='big-font'>📈 Trading Success Predictor AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='medium-font'>Sviluppato per supportare decisioni di trading con modelli avanzati di Machine Learning.</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    symbol = st.text_input("Simbolo (Ticker)", value="AAPL", help="Inserisci il ticker (es. AAPL, TSLA, BTC-USD, EURUSD=X)")
    
    period = st.selectbox(
        "Periodo storico",
        options=["1y", "2y", "5y", "10y", "max"],
        index=2,
        help="Intervallo di dati storici da usare per il training del modello"
    )
    
    data_interval = st.selectbox(
        "Intervallo dati",
        options=["1d", "1h"],
        index=0,
        help="Risoluzione dei dati: daily o oraria"
    )
    
    risk_level = st.radio(
        "Profilo di rischio",
        options=["Basso", "Medio", "Alto"],
        index=1
    )
    
    refresh_data = st.checkbox("🔄 Aggiorna dati da Yahoo Finance", value=False)
    
    st.markdown("---")
    st.markdown("**Gestione modello AI**")
    
    model_action = st.radio(
        "Vuoi usare un modello esistente o crearne uno nuovo?",
        options=["Crea/Allena nuovo modello", "Carica modello esistente"],
        index=0
    )
    
    uploaded_model = None
    if model_action == "Carica modello esistente":
        uploaded_model_file = st.file_uploader("Carica modello .pkl", type="pkl")
        if uploaded_model_file is not None:
            uploaded_model = load_uploaded_model(uploaded_model_file)
    
    st.markdown("---")
    st.markdown("**Parametri operativi**")
    
    capital = st.number_input("Capitale da investire (€)", min_value=100.0, value=1000.0, step=100.0)
    max_risk_per_trade = st.slider("Rischio massimo per operazione (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    holding_period = st.slider("Orizzonte temporale (giorni)", min_value=5, max_value=60, value=10, step=5)
    
    st.markdown("---")
    st.markdown("**Azioni**")
    run_prediction = st.button("🚀 Avvia Analisi AI")
    
    st.markdown("---")
    st.markdown("<div class='footer-text'>⚠️ Disclaimer: questo strumento non costituisce consulenza finanziaria. Utilizzalo responsabilmente.</div>", unsafe_allow_html=True)

# HEADER
st.markdown("<div class='big-font'>AI Trading Success Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='medium-font'>Analisi avanzata di probabilità di successo, rischio e rendimento atteso per le tue operazioni di trading.</div>", unsafe_allow_html=True)
st.markdown("---")

# CARICAMENTO DATI
with st.spinner("📥 Caricamento dati storici..."):
    data = load_data(symbol, period=period, interval=data_interval)

if data is None or data.empty:
    st.error("Impossibile scaricare dati per questo simbolo. Verifica il ticker e riprova.")
    st.stop()

# MOSTRA INFO DI BASE
col_info1, col_info2, col_info3 = st.columns([2, 1, 1])

with col_info1:
    st.markdown("<div class='section-title'>📊 Panoramica del titolo</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-subtitle'>Simbolo analizzato: <b>{symbol}</b> | Periodo: <b>{period}</b> | Intervallo: <b>{data_interval}</b></div>", unsafe_allow_html=True)

with col_info2:
    st.metric("📈 Ultimo prezzo noto (Close)", f"{data['Close'].iloc[-1]:.2f}")

with col_info3:
    annual_volatility = data["Close"].pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252)
    st.metric("📉 Volatilità annualizzata (20d)", f"{annual_volatility*100:.2f}%")

st.markdown("---")

# GRAFICO PRINCIPALE
st.markdown("<div class='section-title'>📈 Grafico storico del prezzo</div>", unsafe_allow_html=True)

tab_price, tab_volume = st.tabs(["Prezzo & Medie", "Volume"])

with tab_price:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data["Date"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Prezzo"
    ))
    
    data["EMA_20"] = data["Close"].ewm(span=20).mean()
    data["EMA_50"] = data["Close"].ewm(span=50).mean()
    
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["EMA_20"],
        mode="lines",
        name="EMA 20",
        line=dict(width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["EMA_50"],
        mode="lines",
        name="EMA 50",
        line=dict(width=1.5)
    ))
    
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Prezzo",
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab_volume:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=data["Date"],
        y=data["Volume"],
        name="Volume"
    ))
    fig_vol.update_layout(
        xaxis_title="Data",
        yaxis_title="Volume",
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("---")

# PREZZO LIVE CORRENTE
live_price, prev_close = fetch_live_price(symbol)
col_live1, col_live2 = st.columns([1, 1])
with col_live1:
    if live_price is not None:
        delta_str = None
        if prev_close is not None and prev_close != 0:
            delta_pct = (live_price - prev_close) / prev_close * 100
            delta_str = f"{delta_pct:+.2f}%"
        display_price = f"{live_price:.4f}" if live_price < 10 else f"{live_price:.2f}"
        st.metric("💹 Prezzo live", display_price, delta_str)
    else:
        st.metric("💹 Prezzo live", "N/D")

with col_live2:
    st.caption(
        f"Aggiornato alle {datetime.datetime.now().strftime('%H:%M:%S')} "
        "(dati Yahoo Finance, possono essere ritardati)"
    )
    if st.button("🔁 Aggiorna prezzo live"):
        # Svuota la cache della funzione se è decorata con @st.cache_data
        try:
            fetch_live_price.clear()
        except AttributeError:
            pass
        # Ricarica l'app per ottenere un nuovo prezzo live
        st.rerun()

st.markdown("---")

# Inizializzazione modello
session_key = f"model_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Calcolo indicatori tecnici e preparazione dati..."):
        data_ind = compute_indicators(data)
        X, y, feature_cols = create_features_targets(data_ind)
        
        st.session_state[f"data_ind_{symbol}_{data_interval}"] = data_ind
        st.session_state[f"X_{symbol}_{data_interval}"] = X
        st.session_state[f"y_{symbol}_{data_interval}"] = y
        st.session_state[f"feature_cols_{symbol}_{data_interval}"] = feature_cols
        
        if uploaded_model is not None:
            st.session_state[session_key] = uploaded_model
        else:
            loaded_model = load_model(symbol, risk_level, data_interval)
            if loaded_model is not None:
                st.session_state[session_key] = loaded_model
            else:
                model = train_model(X, y, risk_level)
                st.session_state[session_key] = model
                save_model(model, symbol, risk_level, data_interval)
else:
    data_ind = st.session_state.get(f"data_ind_{symbol}_{data_interval}")
    X = st.session_state.get(f"X_{symbol}_{data_interval}")
    y = st.session_state.get(f"y_{symbol}_{data_interval}")
    feature_cols = st.session_state.get(f"feature_cols_{symbol}_{data_interval}")

model = st.session_state.get(session_key)

if model is None:
    st.error("Impossibile inizializzare il modello AI. Riprova.")
    st.stop()

if run_prediction:
    st.markdown("<div class='section-title'>🤖 Analisi AI dell'operazione</div>", unsafe_allow_html=True)
    
    with st.spinner("⚙️ L'AI sta analizzando il setup di mercato..."):
        latest_features = data_ind[feature_cols].iloc[-1].values.reshape(1, -1)
        predicted_return = model.predict(latest_features)[0]
        
        idx_for_vol = min(len(data_ind) - 1, holding_period * 2)
        holding_volatility = data_ind["Volatility"].iloc[-idx_for_vol]
        
        success_prob = calculate_success_probability(predicted_return, holding_volatility, risk_level)
        
        if live_price is None:
            current_price = data["Close"].iloc[-1]
        else:
            current_price = live_price
        
        suggested_price_target = current_price * (1 + predicted_return)
        
        try:
            mae, r2, rmse = evaluate_model(model, X, y)
        except Exception:
            mae, r2, rmse = None, None, None
    
    col_main1, col_main2 = st.columns([2, 1])
    
    with col_main1:
        st.markdown("<div class='section-subtitle'>Probabilità di successo & scenario atteso</div>", unsafe_allow_html=True)
        
        col_prob1, col_prob2, col_prob3 = st.columns(3)
        
        with col_prob1:
            st.markdown("**Probabilità di successo (AI)**")
            st.markdown(f"<div class='success-prob'>{success_prob*100:.1f}%</div>", unsafe_allow_html=True)
            if success_prob >= 0.75:
                st.markdown("<span class='small-badge'>Setup ad alta probabilità</span>", unsafe_allow_html=True)
            elif success_prob >= 0.6:
                st.markdown("<span class='small-badge'>Setup moderatamente favorevole</span>", unsafe_allow_html=True)
            elif success_prob >= 0.45:
                st.markdown("<span class='small-badge'>Setup neutro / incerto</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='small-badge'>Setup sfavorevole / rischioso</span>", unsafe_allow_html=True)
        
        with col_prob2:
            st.markdown("**Rendimento atteso (orizzonte 10 giorni)**")
            st.markdown(f"<div class='profit-text'>{predicted_return*100:.1f}%</div>", unsafe_allow_html=True)
            if predicted_return > 0:
                st.markdown("<span class='small-badge'>Potenziale scenario rialzista</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='small-badge'>Potenziale scenario ribassista</span>", unsafe_allow_html=True)
        
        with col_prob3:
            st.markdown("**Rischio & Volatilità**")
            if holding_volatility is not None and not np.isnan(holding_volatility):
                vol_perc = holding_volatility * 100
                st.markdown(f"<div class='risk-gauge'>{vol_perc:.1f}%</div>", unsafe_allow_html=True)
                if vol_perc < 18:
                    risk_label = "Basso"
                elif vol_perc < 30:
                    risk_label = "Medio"
                elif vol_perc < 45:
                    risk_label = "Elevato"
                else:
                    risk_label = "Molto elevato"
                st.markdown(f"<span class='small-badge'>Rischio attuale: {risk_label}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='risk-gauge'>N/D</div>", unsafe_allow_html=True)
                st.markdown("<span class='small-badge'>Volatilità non definita</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("<div class='section-subtitle'>Target di prezzo & gestione della posizione</div>", unsafe_allow_html=True)
        
        if suggested_price_target is not None and not np.isnan(suggested_price_target):
            st.markdown(f"- **Prezzo attuale stimato**: {current_price:.2f}")
            st.markdown(f"- **Target di prezzo AI (10 giorni)**: {suggested_price_target:.2f}")
            st.markdown(f"- **Variazione attesa**: {predicted_return*100:.1f}%")
            
            stop_loss_perc = max_risk_per_trade / 100
            if risk_level == "Basso":
                stop_loss_perc *= 0.6
            elif risk_level == "Alto":
                stop_loss_perc *= 1.3
            
            stop_loss_price = current_price * (1 - stop_loss_perc)
            st.markdown(f"- **Stop loss suggerito**: {stop_loss_price:.2f} ({-stop_loss_perc*100:.1f}%)")
            
            risk_per_share = current_price - stop_loss_price
            if risk_per_share > 0:
                max_loss_euro = capital * (max_risk_per_trade / 100)
                position_size = max_loss_euro / risk_per_share
                position_size_int = int(position_size)
                potential_profit = position_size_int * (suggested_price_target - current_price)
                potential_loss = position_size_int * (current_price - stop_loss_price)
                
                st.markdown(f"- **Dimensione posizione consigliata**: {position_size_int} unità")
                st.markdown(f"- **Perdita massima stimata**: {potential_loss:.2f} €")
                st.markdown(f"- **Profitto potenziale stimato**: {potential_profit:.2f} €")
            else:
                st.warning("Impossibile calcolare la dimensione della posizione (stop loss non valido).")
        else:
            st.warning("Impossibile calcolare un target di prezzo affidabile.")
        
        st.markdown("---")
        
        st.markdown("<div class='section-subtitle'>Report dettagliato dell'AI</div>", unsafe_allow_html=True)
        analysis_text = generate_textual_analysis(
            success_prob,
            predicted_return,
            holding_volatility,
            risk_level,
            suggested_price_target,
            symbol
        )
        st.markdown(analysis_text, unsafe_allow_html=True)
    
    with col_main2:
        st.markdown("<div class='section-subtitle'>📊 Statistiche modello AI</div>", unsafe_allow_html=True)
        
        if mae is not None:
            st.metric("MAE (errore medio assoluto)", f"{mae*100:.2f} %")
        else:
            st.metric("MAE (errore medio assoluto)", "N/D")
        
        if rmse is not None:
            st.metric("RMSE", f"{rmse*100:.2f} %")
        else:
            st.metric("RMSE", "N/D")
        
        if r2 is not None:
            st.metric("R² (spiegazione varianza)", f"{r2:.3f}")
        else:
            st.metric("R² (spiegazione varianza)", "N/D")
        
        st.markdown("---")
        
        st.markdown("**Importanza delle feature**")
        try:
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": importances
            }).sort_values("importance", ascending=False).head(10)
            
            fig_imp = px.bar(
                feature_importance_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 10 Feature più importanti",
            )
            fig_imp.update_layout(
                template="plotly_dark",
                height=400,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.info("Impossibile calcolare l'importanza delle feature per questo modello.")
        
        st.markdown("---")
        st.markdown("**Esporta dati e modello**")
        st.markdown(get_table_download_link(data_ind, filename=f"{symbol}_ai_dataset.csv"), unsafe_allow_html=True)
        st.markdown(get_model_download_link(model, filename=f"{symbol}_trading_model.pkl"), unsafe_allow_html=True)

else:
    st.info("Premi **🚀 Avvia Analisi AI** nella sidebar per eseguire la previsione sull'operazione.")
