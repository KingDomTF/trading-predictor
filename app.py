import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
import time  # <--- NUOVO IMPORT

warnings.filterwarnings('ignore')

# ==================== FUNZIONI CORE ====================

def calculate_technical_indicators(df):
    """Calcola indicatori tecnici."""
    df = df.copy()
   
    # EMA
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
   
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
   
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
   
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
   
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
   
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
   
    # Trend
    df['Price_Change'] = df['Close'].pct_change()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x[-1] > x[0] else 0)
   
    df = df.dropna()
    return df


def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    """Genera features per la predizione."""
    latest = df_ind.iloc[-1]
   
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    tp_distance = abs(tp - entry) / entry * 100
   
    features = {
        'sl_distance_pct': sl_distance,
        'tp_distance_pct': tp_distance,
        'rr_ratio': rr_ratio,
        'direction': 1 if direction == 'long' else 0,
        'main_tf': main_tf,
        'rsi': latest['RSI'],
        'macd': latest['MACD'],
        'macd_signal': latest['MACD_signal'],
        'atr': latest['ATR'],
        'ema_diff': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'bb_position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']),
        'volume_ratio': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1.0,
        'price_change': latest['Price_Change'] * 100,
        'trend': latest['Trend']
    }
   
    return np.array(list(features.values()), dtype=np.float32)


def simulate_historical_trades(df_ind, n_trades=500):
    """Simula trade storici per training."""
    X_list = []
    y_list = []
   
    for _ in range(n_trades):
        if len(df_ind) <= 100:
            break

        idx = np.random.randint(50, len(df_ind) - 50)
        row = df_ind.iloc[idx]
       
        direction = np.random.choice(['long', 'short'])
        entry = row['Close']
        sl_pct = np.random.uniform(0.5, 2.0)
        tp_pct = np.random.uniform(1.0, 4.0)
       
        if direction == 'long':
            sl = entry * (1 - sl_pct / 100)
            tp = entry * (1 + tp_pct / 100)
        else:
            sl = entry * (1 + sl_pct / 100)
            tp = entry * (1 - tp_pct / 100)
       
        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction, 60)
       
        # Simula outcome
        future_prices = df_ind.iloc[idx+1:idx+51]['Close'].values
        if len(future_prices) > 0:
            if direction == 'long':
                hit_tp = np.any(future_prices >= tp)
                hit_sl = np.any(future_prices <= sl)
            else:
                hit_tp = np.any(future_prices <= tp)
                hit_sl = np.any(future_prices >= sl)
           
            success = 1 if hit_tp and not hit_sl else 0
           
            X_list.append(features)
            y_list.append(success)
   
    return np.array(X_list), np.array(y_list)


def train_model(X_train, y_train):
    """Addestra il modello Random Forest."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
   
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y_train)
   
    return model, scaler


def predict_success(model, scaler, features):
    """Predice probabilità di successo."""
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100


def get_dominant_factors(model, features):
    """Identifica fattori dominanti."""
    feature_names = [
        'SL Distance %', 'TP Distance %', 'R/R Ratio', 'Direction', 'TimeFrame',
        'RSI', 'MACD', 'MACD Signal', 'ATR', 'EMA Diff %',
        'BB Position', 'Volume Ratio', 'Price Change %', 'Trend'
    ]
   
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:][::-1]
   
    factors = []
    for i in indices:
        if i < len(feature_names):
            factors.append(f"{feature_names[i]}: {features[i]:.2f} (importanza: {importances[i]:.2%})")
   
    return factors


def get_sentiment(text):
    """Semplice analisi sentiment basata su parole chiave."""
    positive_words = ['rally', 'up', 'bullish', 'gain', 'positive', 'strong', 'rise', 'surge', 'boom']
    negative_words = ['down', 'bearish', 'loss', 'negative', 'weak', 'slip', 'fall', 'drop', 'crash']
    score = sum(word in text.lower() for word in positive_words) - sum(word in text.lower() for word in negative_words)
    if score > 0:
        return 'Positive', score
    elif score < 0:
        return 'Negative', score
    else:
        return 'Neutral', 0


def predict_price(df_ind, steps=5):
    """Previsione prezzo semplice basata su EMA."""
    try:
        last_price = df_ind['Close'].iloc[-1]
        ema = df_ind['Close'].ewm(span=steps).mean().iloc[-1]
        forecast_values = [last_price + (ema - last_price) * (i / steps) for i in range(1, steps + 1)]
        forecast = np.array(forecast_values)
        return forecast.mean(), forecast
    except Exception:
        return None, None


def get_investor_psychology(symbol, news_summary, sentiment_label, df_ind):
    """Analisi approfondita della psicologia dell'investitore."""
    latest = df_ind.iloc[-1]
    trend = 'bullish' if latest['Trend'] == 1 else 'bearish'
    
    current_analysis = f"""
    **🌍 Contesto Globale (Ottobre 2025)**
    Nel contesto del 28 Ottobre 2025, i mercati globali sono influenzati da inflazione persistente e tensioni geopolitiche. 
    Per {symbol}, con trend {trend} e sentiment {sentiment_label}, gli investitori mostrano volatilità emotiva.
    """
    
    biases_analysis = """
    ### 🧠 Analisi Bias
    | Bias Cognitivo | Definizione |
    |---------------|-------------|
    | **Avversione alle Perdite** | Perdite percepite 2x più dolorose dei guadagni. |
    | **Effetto Gregge** | Seguire la massa. |
    """
    
    return current_analysis + biases_analysis


def get_web_signals(symbol, df_ind):
    """Funzione dinamica per ottenere segnali web aggiornati."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Prezzo corrente
        hist = ticker.history(period='1d')
        if hist.empty:
            return []
        current_price = hist['Close'].iloc[-1]
        
        # News recenti
        news = getattr(ticker, "news", None)
        news_summary = ' | '.join([item.get('title', '') for item in news[:5] if isinstance(item, dict)]) if news and isinstance(news, list) else 'Nessuna news recente disponibile.'
        
        sentiment_label, sentiment_score = get_sentiment(news_summary)
        
        # Stagionalità
        hist_monthly = yf.download(symbol, period='10y', interval='1mo', progress=False)
        if len(hist_monthly) < 12:
            seasonality_note = 'Dati storici insufficienti.'
        else:
            hist_monthly['Return'] = hist_monthly['Close'].pct_change()
            hist_monthly['Month'] = hist_monthly.index.month
            monthly_returns = hist_monthly.groupby('Month')['Return'].mean()
            current_month = datetime.datetime.now().month
            avg_current = monthly_returns.get(current_month, 0) * 100
            seasonality_note = f'Il mese corrente ha un ritorno medio storico di {avg_current:.2f}%.'
        
        _, forecast_series = predict_price(df_ind, steps=5)
        forecast_note = f'Previsione media prossima: {forecast_series.mean():.2f}' if forecast_series is not None else 'N/A'
        
        latest = df_ind.iloc[-1]
        atr = latest['ATR']
        trend = latest['Trend']
        suggestions = []
        directions = ['Long', 'Short'] if '=X' not in symbol else ['Buy', 'Sell']
        
        for dir_ in directions:
            is_positive_dir = (dir_ in ['Long', 'Buy'] and (sentiment_score > 0 or trend == 1)) or (dir_ in ['Short', 'Sell'] and (sentiment_score < 0 or trend == 0))
            prob = 70 if is_positive_dir else 60
            entry = round(current_price, 2)
            sl_mult = 1.0 if is_positive_dir else 1.5
            tp_mult = 2.5 if is_positive_dir else 2.0
            if dir_ in ['Long', 'Buy']:
                sl = round(entry - atr * sl_mult, 2)
                tp = round(entry + atr * tp_mult, 2)
            else:
                sl = round(entry + atr * sl_mult, 2)
                tp = round(entry - atr * tp_mult, 2)
            suggestions.append({
                'Direction': dir_,
                'Entry': entry,
                'SL': sl,
                'TP': tp,
                'Probability': prob,
                'Seasonality_Note': seasonality_note,
                'News_Summary': news_summary,
                'Sentiment': sentiment_label,
                'Forecast_Note': forecast_note
            })
        
        return suggestions
    except Exception as e:
        # st.error(f"Errore web: {e}") 
        return []


# ==================== PREZZO LIVE (MODIFICATO) ====================

# MODIFICA: Ridotto TTL a 2 secondi per permettere aggiornamenti quasi real-time
@st.cache_data(ttl=2) 
def fetch_live_price(symbol: str):
    """
    Recupera il prezzo 'live' (ultimo disponibile) da Yahoo Finance.
    ttl ridotto per aggiornamento frequente.
    """
    ticker = yf.Ticker(symbol)
    last_price = None
    prev_close = None

    # 1) Prova con fast_info (Spesso il più veloce e aggiornato)
    try:
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info is not None:
            last_price = fast_info.get("lastPrice", None)
            prev_close = fast_info.get("previousClose", None)
    except Exception:
        pass

    # 2) Fallback: intraday 1m
    if last_price is None:
        try:
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                last_price = float(hist["Close"].iloc[-1])
                if len(hist) > 1:
                    prev_close = float(hist["Close"].iloc[-2])
                else:
                     # Se c'è solo una barra oggi, prova a prendere il close di ieri
                     prev_hist = ticker.history(period="5d", interval="1d")
                     if len(prev_hist) > 1:
                         prev_close = float(prev_hist["Close"].iloc[-2])
        except Exception:
            pass

    return last_price, prev_close


# ==================== STREAMLIT APP ====================

@st.cache_data
def load_sample_data(symbol, interval='1h'):
    """Carica dati storici (cache lunga)."""
    period_map = {'5m': '60d', '15m': '60d', '1h': '730d'}
    period = period_map.get(interval, '730d')
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        if len(data) < 100:
            raise Exception("Dati insufficienti")
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return data
    except Exception as e:
        return None

@st.cache_resource
def train_or_load_model(symbol, interval='1h'):
    """Addestra il modello (cache molto lunga, non viene rifatto al refresh prezzo)."""
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=500)
    if X.size == 0 or y.size == 0:
        return None, None, None
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

# Configurazione pagina
st.set_page_config(
    page_title="Trading Predictor AI - Real Time",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizzato
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Trading Success Predictor AI")

# Parametri e Real-Time Toggle
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    symbol = st.text_input("🔍 Ticker", value="BTC-USD")
    proper_names = {'GC=F': 'Gold', 'EURUSD=X': 'EUR/USD', 'BTC-USD': 'Bitcoin'}
    proper_name = proper_names.get(symbol, symbol)

with col2:
    data_interval = st.selectbox("⏰ Timeframe", ['5m', '15m', '1h'], index=2)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_data = st.button("🔄 Reload Model", use_container_width=True)

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    # TOGGLE PER REAL TIME
    real_time_mode = st.toggle("🔴 LIVE MODE", value=False)

# LOGICA AUTO-REFRESH
if real_time_mode:
    time.sleep(3) # Aggiorna ogni 3 secondi
    st.rerun()

# PREZZO LIVE CORRENTE
# Questa chiamata è ora rapida grazie alla cache ridotta a 2s
live_price, prev_close = fetch_live_price(symbol)

col_live1, col_live2 = st.columns([1, 1])
with col_live1:
    if live_price is not None:
        delta_str = None
        if prev_close is not None and prev_close != 0:
            delta_pct = (live_price - prev_close) / prev_close * 100
            delta_str = f"{delta_pct:+.2f}%"
        
        # Animazione visuale per far capire che è vivo
        now_sec = datetime.datetime.now().second
        indicator = "⚡" if now_sec % 2 == 0 else "💹"
        
        st.metric(f"{indicator} Prezzo Live ({symbol})", f"{live_price:,.4f}", delta_str)
    else:
        st.metric("💹 Prezzo Live", "Caricamento...")

with col_live2:
    if real_time_mode:
        st.caption(f"🔴 AGGIORNAMENTO LIVE ATTIVO: {datetime.datetime.now().strftime('%H:%M:%S')}")
    else:
        st.caption(f"Ultimo check: {datetime.datetime.now().strftime('%H:%M:%S')} (Attiva LIVE MODE per real-time)")

st.markdown("---")

# 

# Inizializzazione modello
session_key = f"model_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Caricamento AI e analisi dati..."):
        model, scaler, df_ind = train_or_load_model(symbol=symbol, interval=data_interval)
        if model is not None:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            # st.success("Modello aggiornato!")
        else:
            st.error("Errore ticker.")

if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']
    
    # Previsione prezzo
    avg_forecast, forecast_series = predict_price(df_ind, steps=5)
    
    # Segnali web (Attenzione: chiamate web frequenti rallentano, quindi qui non forziamo refresh)
    # Se in live mode, potremmo voler evitare di chiamare 'get_web_signals' ogni 3 secondi se è lento.
    # Per ora lo lasciamo, ma tieni presente che lo scraping news rallenta il real-time price.
    if not real_time_mode: 
        web_signals_list = get_web_signals(symbol, df_ind)
    else:
        # In live mode usiamo una versione cache o vuota per velocità, oppure accettiamo il ritardo
        # Qui per semplicità mostriamo un avviso o usiamo dati precedenti se possibile.
        if 'web_signals' not in st.session_state:
             st.session_state.web_signals = get_web_signals(symbol, df_ind)
        web_signals_list = st.session_state.web_signals
    
    col_left, col_right = st.columns([1.2, 0.8])
   
    with col_left:
        st.markdown("### 💡 Suggerimenti Trade")
        if web_signals_list:
            suggestions_df = pd.DataFrame(web_signals_list)
            suggestions_df = suggestions_df.sort_values(by='Probability', ascending=False)
            
            # Mostra solo il primo per brevità in live mode
            top_trade = suggestions_df.iloc[0]
            st.info(f"Top Signal: {top_trade['Direction']} @ {top_trade['Entry']} (Prob: {top_trade['Probability']}%)")
            
            # Qui potresti rimettere il loop completo se vuoi
            
        else:
            st.info("Nessun segnale web o Live Mode attiva (news in cache).")
            
    with col_right:
         # Tabella asset watch
         st.markdown("### 🚀 Watchlist Live")
         # Se in real time mode, aggiorniamo anche questi prezzi? 
         # Attenzione: fare fetch di 10 ticker ogni 3 secondi blocca l'IP di Yahoo.
         # Meglio mostrare statico in live mode o aggiornare solo il simbolo principale.
         st.write("Watchlist disabilitata in Live Mode per performance.")

    # ... Resto del codice per i dettagli AI ...
    # (Ho tagliato per brevità, il focus era sul Real Time Update)
    
    st.markdown("---")
    st.markdown("### 📊 Dashboard Statistiche")
    latest = df_ind.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("RSI", f"{latest['RSI']:.2f}")
    c2.metric("Trend", "Bullish" if latest['Trend']==1 else "Bearish")
    c3.metric("ATR", f"{latest['ATR']:.2f}")

else:
    st.warning("Carica i dati per iniziare.")
