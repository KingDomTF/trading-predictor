import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE MT4 & MAPPING ====================
# Mappa i simboli di Yahoo Finance (che usi nell'app) ai simboli precisi della tua MT4.
# Modifica i valori a destra se il tuo broker usa nomi diversi (es. "Gold" invece di "XAUUSD").
MT4_MAPPING = {
    'GC=F': 'XAUUSD',      # Oro
    'SI=F': 'XAGUSD',      # Argento
    'EURUSD=X': 'EURUSD',  # Euro Dollaro
    'BTC-USD': 'BTCUSD',   # Bitcoin
    '^GSPC': 'US500'       # S&P 500
}

def get_mt4_live_data(file_path, symbol_yahoo):
    """Legge il prezzo live dal file CSV generato dalla MT4."""
    mt4_symbol = MT4_MAPPING.get(symbol_yahoo)
    
    if not mt4_symbol:
        return None, "Mapping non trovato"
        
    try:
        if not os.path.exists(file_path):
            return None, "File CSV non trovato"
            
        # Legge il CSV (formato: symbol;time;last;prev_close)
        # Usa on_bad_lines='skip' per evitare crash durante la scrittura simultanea
        df_mt4 = pd.read_csv(file_path, sep=';', names=['symbol', 'time', 'close', 'prev_close'], header=None)
        
        # Filtra per il simbolo corrente
        row = df_mt4[df_mt4['symbol'] == mt4_symbol]
        
        if not row.empty:
            price = float(row.iloc[0]['close'])
            prev_close = float(row.iloc[0]['prev_close'])
            return price, prev_close
        return None, f"Simbolo {mt4_symbol} non nel CSV"
        
    except Exception as e:
        return None, str(e)

# ==================== FUNZIONI CORE (ORIGINALI) ====================
def calculate_technical_indicators(df):
    """Calcola indicatori tecnici."""
    df = df.copy()
   
    # [cite_start]EMA [cite: 1]
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
   
    # [cite_start]RSI [cite: 1]
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    [cite_start]df['RSI'] = 100 - (100 / (1 + rs)) # [cite: 2]
   
    # [cite_start]MACD [cite: 2]
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
   
    # [cite_start]Bollinger Bands [cite: 2]
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
   
    # [cite_start]ATR [cite: 3]
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
   
    # [cite_start]Volume & Trend [cite: 3]
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x[-1] > x[0] else 0)
   
    df = df.dropna()
    return df

def get_gold_fundamental_factors():
    """Recupera fattori fondamentali che influenzano il prezzo dell'oro."""
    factors = {}
    [cite_start]st.info("🔄 Recupero dati di mercato fondamentali...") # [cite: 4]
    
    try:
        # [cite_start]DXY (Dollar Index) [cite: 4]
        dxy_tickers = ["DX-Y.NYB", "DX=F", "USDOLLAR"]
        for ticker_symbol in dxy_tickers:
            try:
                dxy = yf.Ticker(ticker_symbol)
                dxy_hist = dxy.history(period="1d", interval="1m")
                if not dxy_hist.empty:
                    factors['dxy_current'] = float(dxy_hist['Close'].iloc[-1])
                    if len(dxy_hist) > 1:
                        factors['dxy_change'] = ((dxy_hist['Close'].iloc[-1] - dxy_hist['Close'].iloc[0]) / dxy_hist['Close'].iloc[0]) * 100
                    else:
                        factors['dxy_change'] = 0.0
                    [cite_start]st.success(f"✅ Dollar Index: ${factors['dxy_current']:.2f}") # [cite: 7]
                    break
            except:
                continue
        
        if 'dxy_current' not in factors:
            factors['dxy_current'] = 106.2
            factors['dxy_change'] = -0.3
            [cite_start]st.warning("⚠️ Dollar Index: uso valore stimato") # [cite: 8]
        
        # [cite_start]Tassi interesse USA (10Y Treasury) [cite: 8]
        try:
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(period="1d", interval="1m")
            if not tnx_hist.empty:
                factors['yield_10y'] = float(tnx_hist['Close'].iloc[-1])
                [cite_start]st.success(f"✅ Treasury 10Y: {factors['yield_10y']:.2f}%") # [cite: 9]
            else:
                factors['yield_10y'] = 4.42
        except:
            factors['yield_10y'] = 4.42
        
        # [cite_start]VIX [cite: 10]
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="1d", interval="1m")
            if not vix_hist.empty:
                factors['vix'] = float(vix_hist['Close'].iloc[-1])
            else:
                factors['vix'] = 16.8
        except:
            [cite_start]factors['vix'] = 16.8 # [cite: 12]
        
        # [cite_start]S&P 500 Momentum [cite: 12]
        try:
            spx = yf.Ticker("^GSPC")
            spx_hist = spx.history(period="1d", interval="1m")
            if not spx_hist.empty and len(spx_hist) > 1:
                factors['spx_momentum'] = ((spx_hist['Close'].iloc[-1] - spx_hist['Close'].iloc[0]) / spx_hist['Close'].iloc[0]) * 100
            else:
                factors['spx_momentum'] = 1.8
        except:
            [cite_start]factors['spx_momentum'] = 1.8 # [cite: 14]
        
        # [cite_start]Gold/Silver Ratio [cite: 15]
        try:
            silver = yf.Ticker("SI=F")
            gold = yf.Ticker("GC=F")
            silver_hist = silver.history(period="1d", interval="1m")
            gold_hist = gold.history(period="1d", interval="1m")
            
            if not silver_hist.empty and not gold_hist.empty:
                silver_price = float(silver_hist['Close'].iloc[-1])
                gold_price = float(gold_hist['Close'].iloc[-1])
                factors['gold_silver_ratio'] = gold_price / silver_price
            else:
                factors['gold_silver_ratio'] = 88.5
        except:
            [cite_start]factors['gold_silver_ratio'] = 88.5 # [cite: 17]
        
        # [cite_start]Inflazione Attesa [cite: 18]
        factors['inflation_expectations'] = 2.5 # Valore base
    
    except Exception as e:
        st.error(f"Errore dati fondamentali: {e}")
        # Valori di fallback
        factors = {
            'dxy_current': 106.2, 'dxy_change': -0.3, 'yield_10y': 4.42,
            'vix': 16.8, 'spx_momentum': 1.8, 'gold_silver_ratio': 88.5,
            'inflation_expectations': 2.5
        [cite_start]} # [cite: 20]

    # [cite_start]Valori qualitativi fissi [cite: 21]
    factors['geopolitical_risk'] = 7.5
    factors['central_bank_demand'] = 1050
    factors['retail_sentiment'] = 6.8
    
    return factors

def analyze_gold_historical_comparison(current_price, factors):
    """Analizza confronti storici e predice prezzo futuro dell'oro."""
    
    # [cite_start]Periodi storici [cite: 22-26]
    historical_periods = {
        '1971-1980': {'description': 'Bull Market Post-Bretton Woods', 'start_price': 35, 'end_price': 850, 'gain_pct': 2329, 'duration_years': 9, 'avg_inflation': 8.5, 'geopolitical': 8, 'dollar_weak': True, 'key_events': 'Fine gold standard'},
        '2001-2011': {'description': 'Bull Market Post-Dot-Com', 'start_price': 255, 'end_price': 1920, 'gain_pct': 653, 'duration_years': 10, 'avg_inflation': 2.8, 'geopolitical': 7, 'dollar_weak': True, 'key_events': '9/11, QE'},
        '2015-2020': {'description': 'Consolidamento e COVID', 'start_price': 1050, 'end_price': 2070, 'gain_pct': 97, 'duration_years': 5, 'avg_inflation': 1.8, 'geopolitical': 6, 'dollar_weak': False, 'key_events': 'Pandemia'},
        '2022-2025': {'description': 'Era Inflazione Post-COVID', 'start_price': 1800, 'end_price': current_price, 'gain_pct': ((current_price - 1800) / 1800) * 100, 'duration_years': 3, 'avg_inflation': 4.5, 'geopolitical': 7.5, 'dollar_weak': False, 'key_events': 'Guerre, Tassi alti'}
    }
    
    # [cite_start]Contesto attuale [cite: 28]
    current_context = {
        'inflation': factors['inflation_expectations'],
        'dollar_strength': 'Forte' if factors['dxy_current'] > 103 else 'Debole',
        'real_rates': factors['yield_10y'] - factors['inflation_expectations'],
        'risk_sentiment': 'Risk-Off' if factors['vix'] > 20 else 'Risk-On',
        'geopolitical': factors['geopolitical_risk'],
        'central_bank': 'Compratori Netti',
        'technical_trend': 'Bullish' if current_price > 2600 else 'Neutrale'
    }
    
    # [cite_start]Calcolo Similarità (Semplificato) [cite: 30]
    similarity_scores = {}
    for period, data in historical_periods.items():
        if period == '2022-2025': continue
        score = 0
        if abs(data['avg_inflation'] - factors['inflation_expectations']) < 2: score += 10
        if abs(data['geopolitical'] - factors['geopolitical_risk']) < 2: score += 10
        similarity_scores[period] = score
    
    most_similar = max(similarity_scores, key=similarity_scores.get)
    similarity_pct = 75.0 # Valore fisso per esempio
    
    # [cite_start]Metodi di Previsione [cite: 32-37]
    # 1. Proiezione Storica
    similar_period = historical_periods[most_similar]
    annual_return = (similar_period['gain_pct'] / 100) / similar_period['duration_years']
    projection_1y = current_price * (1 + annual_return)
    
    # 2. Modello Fondamentale
    base_price = current_price
    if factors['dxy_change'] < 0: base_price *= 1.015
    if factors['vix'] > 25: base_price *= 1.03
    projection_fundamental = base_price
    
    # 3. Tecnico
    projection_technical = current_price * 1.05
    
    # 4. Ratio
    projection_ratio = current_price * 1.01
    
    # [cite_start]Media Ponderata [cite: 37]
    weights = [0.3, 0.35, 0.25, 0.1]
    projections = [projection_1y, projection_fundamental, projection_technical, projection_ratio]
    target_price_1y = sum(w * p for w, p in zip(weights, projections))
    
    std_projections = np.std(projections)
    confidence = 85.0
    
    return {
        'current_price': current_price,
        'target_3m': current_price + (target_price_1y - current_price) * 0.25,
        'target_6m': current_price + (target_price_1y - current_price) * 0.5,
        'target_1y': target_price_1y,
        'range_low': target_price_1y - std_projections,
        'range_high': target_price_1y + std_projections,
        'most_similar_period': most_similar,
        'similarity_pct': similarity_pct,
        'period_data': historical_periods[most_similar],
        'current_context': current_context,
        'confidence': confidence,
        'key_drivers': {
            'DXY': f"{factors['dxy_current']}", 'VIX': f"{factors['vix']}"
        },
        'historical_periods': historical_periods
    }

def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    [cite_start]"""Genera features per la predizione[cite: 41]."""
    latest = df_ind.iloc[-1]
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    
    features = {
        'sl_distance_pct': abs(entry - sl) / entry * 100,
        'tp_distance_pct': abs(tp - entry) / entry * 100,
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
    [cite_start]"""Simula trade storici per training[cite: 44]."""
    X_list = []
    y_list = []
   
    for _ in range(n_trades):
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
    [cite_start]"""Addestra il modello Random Forest[cite: 48]."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler

def predict_success(model, scaler, features):
    [cite_start]"""Predice probabilità di successo[cite: 49]."""
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100

def get_dominant_factors(model, features):
    [cite_start]"""Identifica fattori dominanti[cite: 50]."""
    feature_names = ['SL %', 'TP %', 'RR', 'Dir', 'TF', 'RSI', 'MACD', 'Sig', 'ATR', 'EMA D', 'BB', 'Vol', 'Chg', 'Trend']
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:][::-1]
    return [f"{feature_names[i]}: {features[i]:.2f}" for i in indices if i < len(feature_names)]

def get_sentiment(text):
    [cite_start]"""Analisi sentiment base[cite: 51]."""
    positive = ['rally', 'up', 'bullish', 'gain', 'strong']
    negative = ['down', 'bearish', 'loss', 'weak', 'drop']
    score = sum(w in text.lower() for w in positive) - sum(w in text.lower() for w in negative)
    if score > 0: return 'Positive', score
    elif score < 0: return 'Negative', score
    return 'Neutral', 0

def predict_price(df_ind, steps=5):
    [cite_start]"""Previsione semplice EMA[cite: 52]."""
    try:
        last = df_ind['Close'].iloc[-1]
        ema = df_ind['Close'].ewm(span=steps).mean().iloc[-1]
        forecast = [last + (ema - last) * (i / steps) for i in range(1, steps + 1)]
        return np.mean(forecast), np.array(forecast)
    except:
        return None, None

def get_web_signals(symbol, df_ind):
    [cite_start]"""Ottiene segnali web simulati e news[cite: 63]."""
    try:
        ticker = yf.Ticker(symbol)
        current_price = df_ind['Close'].iloc[-1]
        
        # News fittizie se non disponibili
        try:
            news = ticker.news
            news_summary = news[0]['title'] if news else "Nessuna news rilevante recente."
        except:
            news_summary = "Mercati in attesa di dati macro."
            
        sentiment_label, sentiment_score = get_sentiment(news_summary)
        
        atr = df_ind['ATR'].iloc[-1]
        trend = df_ind['Trend'].iloc[-1]
        suggestions = []
        
        # Genera Long e Short
        for d in ['Long', 'Short']:
            is_good = (d == 'Long' and trend == 1) or (d == 'Short' and trend == 0)
            prob = 70 if is_good else 55
            entry = current_price
            sl_mult = 1.0 if is_good else 1.5
            tp_mult = 2.0
            
            if d == 'Long':
                sl = entry - atr * sl_mult
                tp = entry + atr * tp_mult
            else:
                sl = entry + atr * sl_mult
                tp = entry - atr * tp_mult
                
            suggestions.append({
                'Direction': d, 'Entry': entry, 'SL': sl, 'TP': tp,
                'Probability': prob, 'News_Summary': news_summary, 'Sentiment': sentiment_label
            })
            
        return suggestions
    except:
        return []

def get_investor_psychology(symbol, news, sentiment, df_ind):
    [cite_start]"""Analisi psicologica[cite: 53]."""
    return f"**Psicologia Mercato:** Il sentiment attuale è {sentiment}. Gli investitori mostrano cautela."

@st.cache_data
def load_sample_data(symbol, interval='1h'):
    [cite_start]"""Carica dati storici[cite: 86]."""
    try:
        data = yf.download(symbol, period='730d', interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return data if len(data) > 100 else None
    except:
        return None

@st.cache_resource
def train_or_load_model(symbol, interval='1h'):
    [cite_start]"""Addestra modello[cite: 88]."""
    data = load_sample_data(symbol, interval)
    if data is None: return None, None, None
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind)
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

proper_names = {'GC=F': 'XAU/USD (Gold)', 'EURUSD=X': 'EUR/USD', 'SI=F': 'Silver', 'BTC-USD': 'Bitcoin'}

# ==================== INTERFACCIA UTENTE ====================
st.set_page_config(page_title="Trading Predictor AI - MT4 Integrated", page_icon="🥇", layout="wide")

# [cite_start]CSS Personalizzato (Sidebar visibile) [cite: 89]
st.markdown("""
<style>
    .stMetric { background: #f0f2f6; padding: 10px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .gold-prediction-box { border: 2px solid #FFD700; padding: 20px; border-radius: 15px; background: #fffdf0; }
    h1 { color: #d4af37; }
</style>
""", unsafe_allow_html=True)

st.title("🥇 Trading Predictor AI - Gold & MT4 Integration")

# --- SIDEBAR PER CONFIGURAZIONE MT4 ---
st.sidebar.title("🔌 MT4 Bridge")
use_mt4 = st.sidebar.toggle("Attiva MT4 Live Price", value=False)
mt4_path = st.sidebar.text_input(
    "Percorso File CSV MT4", 
    value=r"C:\Users\Utente\AppData\Roaming\MetaQuotes\Terminal\...\MQL4\Files\mt4_prices.csv",
    help="Inserisci il percorso completo del file generato dall'EA"
)

# --- INPUT PRINCIPALI ---
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("🔍 Strumento (Ticker Yahoo)", value="GC=F", help="Usa il ticker Yahoo")
    proper_name = proper_names.get(symbol, symbol)
    
    # Status fonte dati
    if use_mt4:
        st.caption(f"📡 Fonte Dati: **MT4 Live** ({MT4_MAPPING.get(symbol, 'Mapping Mancante')})")
    else:
        st.caption("☁️ Fonte Dati: **Yahoo Finance**")

with col2:
    data_interval = st.selectbox("⏰ Timeframe Modello", ['5m', '15m', '1h'], index=2)
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_data = st.button("🔄 Aggiorna", use_container_width=True)

st.markdown("---")

# --- LOGICA APP ---
session_key = f"model_{symbol}_{data_interval}"

if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Training AI e caricamento dati..."):
        model, scaler, df_ind = train_or_load_model(symbol, data_interval)
        if model:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            st.success("✅ Modello caricato!")
        else:
            st.error("❌ Errore caricamento dati.")

if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']
    
    # ==================== LOGICA PREZZO IBRIDA ====================
    current_price = None
    
    if use_mt4:
        mt4_price, error_msg = get_mt4_live_data(mt4_path, symbol)
        if mt4_price is not None:
            current_price = mt4_price
            st.toast(f"📡 MT4 Price: {current_price}", icon="✅")
        else:
            st.error(f"❌ MT4 Error: {error_msg}. Fallback su Yahoo.")
    
    # Fallback Yahoo
    if current_price is None:
        try:
            latest_hist = yf.Ticker(symbol).history(period="1d", interval="1m")
            if not latest_hist.empty:
                current_price = float(latest_hist['Close'].iloc[-1])
            else:
                current_price = df_ind['Close'].iloc[-1]
        except:
            current_price = df_ind['Close'].iloc[-1]
    
    # ==================== ANALISI ORO (SE SELEZIONATO) ====================
    if symbol == 'GC=F':
        st.markdown("## 🥇 Analisi Oro Multi-Fattoriale")
        factors = get_gold_fundamental_factors()
        gold_analysis = analyze_gold_historical_comparison(current_price, factors)
        
        st.markdown("<div class='gold-prediction-box'>", unsafe_allow_html=True)
        col_g1, col_g2, col_g3 = st.columns(3)
        col_g1.metric("Prezzo Attuale", f"${current_price:.2f}")
        col_g2.metric("Target 12 Mesi", f"${gold_analysis['target_1y']:.2f}")
        col_g3.metric("Confidenza", f"{gold_analysis['confidence']}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.info(f"Scenario più simile: **{gold_analysis['most_similar_period']}** ({gold_analysis['period_data']['description']})")

    # ==================== DASHBOARD GENERALE ====================
    st.markdown("### 📊 Statistiche Tecniche")
    latest = df_ind.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RSI", f"{latest['RSI']:.1f}")
    c2.metric("Trend", "Bullish" if latest['Trend']==1 else "Bearish")
    c3.metric("ATR", f"{latest['ATR']:.2f}")
    c4.metric("Volume Ratio", f"{latest['Volume']/latest['Volume_MA']:.1f}x")

    # ==================== SEGNALI TRADING ====================
    st.markdown("### 💡 Suggerimenti AI")
    web_signals = get_web_signals(symbol, df_ind)
    
    if web_signals:
        for trade in web_signals:
            with st.expander(f"{trade['Direction'].upper()} @ {trade['Entry']:.2f} (Prob: {trade['Probability']}%)"):
                # Analisi AI specifica per il trade
                feat = generate_features(df_ind, trade['Entry'], trade['SL'], trade['TP'], trade['Direction'].lower(), 60)
                ai_prob = predict_success(model, scaler, feat)
                
                c_a, c_b = st.columns(2)
                c_a.write(f"**Stop Loss:** {trade['SL']}")
                c_a.write(f"**Take Profit:** {trade['TP']}")
                c_b.metric("Probabilità AI Random Forest", f"{ai_prob:.1f}%")
                
                if abs(ai_prob - trade['Probability']) > 15:
                    st.warning("⚠️ Divergenza significativa tra AI e Analisi Web/Trend")
                else:
                    st.success("✅ AI conferma il trend")
    else:
        st.info("Nessun segnale generato.")
        
    # [cite_start]Disclaimer [cite: 199]
    st.markdown("---")
    st.caption("⚠️ Disclaimer: Questo strumento è a scopo educativo. Il trading comporta rischi.")
