import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
import time

warnings.filterwarnings('ignore')

# ==================== FUNZIONI CORE (AI & CALCOLI) ====================

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
    # Bollinger
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
    # Volume & Trend
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x[-1] > x[0] else 0)
    df = df.dropna()
    return df

def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    latest = df_ind.iloc[-1]
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    tp_distance = abs(tp - entry) / entry * 100
    features = {
        'sl_distance_pct': sl_distance, 'tp_distance_pct': tp_distance, 'rr_ratio': rr_ratio,
        'direction': 1 if direction == 'long' else 0, 'main_tf': main_tf, 'rsi': latest['RSI'],
        'macd': latest['MACD'], 'macd_signal': latest['MACD_signal'], 'atr': latest['ATR'],
        'ema_diff': (latest['EMA_20'] - latest['EMA_50']) / latest['Close'] * 100,
        'bb_position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']),
        'volume_ratio': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1.0,
        'price_change': latest['Price_Change'] * 100, 'trend': latest['Trend']
    }
    return np.array(list(features.values()), dtype=np.float32)

def simulate_historical_trades(df_ind, n_trades=500):
    X_list = []
    y_list = []
    for _ in range(n_trades):
        if len(df_ind) <= 100: break
        idx = np.random.randint(50, len(df_ind) - 50)
        row = df_ind.iloc[idx]
        direction = np.random.choice(['long', 'short'])
        entry = row['Close']
        sl_pct = np.random.uniform(0.5, 2.0)
        tp_pct = np.random.uniform(1.0, 4.0)
        if direction == 'long':
            sl = entry * (1 - sl_pct / 100); tp = entry * (1 + tp_pct / 100)
        else:
            sl = entry * (1 + sl_pct / 100); tp = entry * (1 - tp_pct / 100)
        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction, 60)
        future_prices = df_ind.iloc[idx+1:idx+51]['Close'].values
        if len(future_prices) > 0:
            if direction == 'long':
                hit_tp = np.any(future_prices >= tp); hit_sl = np.any(future_prices <= sl)
            else:
                hit_tp = np.any(future_prices <= tp); hit_sl = np.any(future_prices >= sl)
            success = 1 if hit_tp and not hit_sl else 0
            X_list.append(features); y_list.append(success)
    return np.array(X_list), np.array(y_list)

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y_train)
    return model, scaler

def predict_success(model, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    return model.predict_proba(features_scaled)[0][1] * 100

def get_dominant_factors(model, features):
    feature_names = ['SL %', 'TP %', 'R/R', 'Dir', 'TF', 'RSI', 'MACD', 'Sig', 'ATR', 'EMA D', 'BB Pos', 'Vol', 'P Chg', 'Trend']
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:][::-1]
    return [f"{feature_names[i]}: {features[i]:.2f} ({importances[i]:.0%})" for i in indices if i < len(feature_names)]

def get_sentiment(text):
    pos = ['rally', 'up', 'bullish', 'gain', 'strong', 'rise', 'boom']; neg = ['down', 'bearish', 'loss', 'weak', 'fall', 'drop', 'crash']
    score = sum(w in text.lower() for w in pos) - sum(w in text.lower() for w in neg)
    return ('Positive', score) if score > 0 else ('Negative', score) if score < 0 else ('Neutral', 0)

def predict_price(df_ind, steps=5):
    try:
        last = df_ind['Close'].iloc[-1]; ema = df_ind['Close'].ewm(span=steps).mean().iloc[-1]
        fc = [last + (ema - last) * (i / steps) for i in range(1, steps + 1)]
        return np.array(fc).mean(), np.array(fc)
    except: return None, None

def get_investor_psychology(symbol, news_summary, sentiment_label, df_ind):
    latest = df_ind.iloc[-1]
    trend = 'bullish' if latest['Trend'] == 1 else 'bearish'
    return f"""
    **🧠 Analisi Psicologica Real-Time (2025)**
    Prezzo: {latest['Close']:.2f} | Trend: {trend.upper()} | Sentiment: {sentiment_label}
    
    L'investitore medio sta sperimentando { 'FOMO e avidità' if trend == 'bullish' else 'Paura e incertezza' }. 
    Attenzione ai bias di conferma sulle news recenti: "{news_summary[:100]}...".
    """

# ==================== DATA FETCHING CON CACHE DIFFERENZIATA ====================

# 1. LIVE PRICE: Cache bassissima (2 secondi) per aggiornamento quasi istantaneo
@st.cache_data(ttl=2)
def fetch_live_price(symbol: str):
    ticker = yf.Ticker(symbol)
    last, prev = None, None
    try:
        fi = getattr(ticker, "fast_info", None)
        if fi: last = fi.get("lastPrice"); prev = fi.get("previousClose")
    except: pass
    if not last:
        try:
            h = ticker.history(period="1d", interval="1m")
            if not h.empty: last = h["Close"].iloc[-1]; prev = h["Close"].iloc[-2] if len(h)>1 else last
        except: pass
    return last, prev

# 2. WEB SIGNALS: Cache media (5 minuti). Non serve ricalcolare news/stagionalità ogni secondo.
@st.cache_data(ttl=300)
def get_web_signals_cached(symbol, last_price, last_atr, last_trend):
    """Calcola i segnali web ma viene cachata per evitare lentezza nel loop real-time."""
    try:
        ticker = yf.Ticker(symbol)
        news = getattr(ticker, "news", [])
        news_summary = ' | '.join([i.get('title','') for i in news[:3]]) if news else 'Nessuna news.'
        sent_lbl, sent_score = get_sentiment(news_summary)
        
        # Stagionalità
        hist_m = yf.download(symbol, period='5y', interval='1mo', progress=False)
        if len(hist_m) > 12:
            hist_m['Ret'] = hist_m['Close'].pct_change()
            avg = hist_m.groupby(hist_m.index.month)['Ret'].mean().get(datetime.datetime.now().month, 0) * 100
            season_note = f"Stagionalità mese corrente: {avg:+.2f}%"
        else: season_note = "Dati stagionali insufficienti."

        suggestions = []
        directions = ['Long', 'Short']
        for d in directions:
            is_pos = (d=='Long' and (sent_score>=0 or last_trend==1)) or (d=='Short' and (sent_score<0 or last_trend==0))
            prob = 70 if is_pos else 55
            sl_m = 1.0 if is_pos else 1.5; tp_m = 2.5 if is_pos else 2.0
            entry = last_price
            if d == 'Long': sl = entry - last_atr*sl_m; tp = entry + last_atr*tp_m
            else: sl = entry + last_atr*sl_m; tp = entry - last_atr*tp_m
            
            suggestions.append({
                'Direction': d, 'Entry': entry, 'SL': sl, 'TP': tp, 'Probability': prob,
                'Seasonality_Note': season_note, 'News_Summary': news_summary, 'Sentiment': sent_lbl
            })
        return suggestions
    except Exception as e: return []

# 3. WATCHLIST: Cache di 1 minuto. Evita di scaricare 10 ticker ogni 2 secondi.
@st.cache_data(ttl=60)
def get_watchlist_data(base_data):
    rows = []
    for item in base_data:
        p, prev = fetch_live_price(item["Ticker"]) # Usa la funzione base ma dentro una cached
        if p and prev:
            chg = (p - prev) / prev * 100
            rows.append({
                "Asset": item["Asset"], "Ticker": item["Ticker"], "Score": item["Score"],
                "Live Price": f"{p:.2f}", "Δ %": f"{chg:+.2f}%"
            })
    return pd.DataFrame(rows)

@st.cache_data
def load_sample_data(symbol, interval):
    p_map = {'5m':'60d', '15m':'60d', '1h':'730d'}
    try:
        d = yf.download(symbol, period=p_map.get(interval, '730d'), interval=interval, progress=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.droplevel(1)
        return d[['Open','High','Low','Close','Volume']] if len(d) > 100 else None
    except: return None

@st.cache_resource
def train_or_load_model(symbol, interval):
    data = load_sample_data(symbol, interval)
    if data is None: return None, None, None
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind)
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

# ==================== INTERFACCIA UTENTE ====================

st.set_page_config(page_title="Trading AI Real-Time", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stMetric { background: #f0f2f6; padding: 10px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .trade-card { background: white; padding: 15px; border-radius: 10px; border-left: 5px solid #667eea; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px; }
    .live-badge { background-color: #ff4b4b; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Trading Success Predictor AI (Live)")

# Inputs
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1: 
    symbol = st.text_input("Ticker", value="GC=F", help="Es: GC=F, BTC-USD, NVDA")
with c2: 
    tf = st.selectbox("Timeframe", ['5m', '15m', '1h'], index=2)
with c3: 
    if st.button("🔄 Reload All"): st.cache_data.clear()
with c4:
    # TOGGLE REAL TIME
    real_time = st.toggle("🔴 LIVE MODE", value=False)

# LOGICA LOOP REAL TIME
if real_time:
    time.sleep(2) # Attesa per non intasare la CPU/API
    st.rerun()

# 1. DISPLAY PREZZO LIVE (Update veloce)
lp, prev = fetch_live_price(symbol)
d_str = f"{(lp-prev)/prev*100:+.2f}%" if lp and prev else "0.00%"
now_str = datetime.datetime.now().strftime('%H:%M:%S')

col_h1, col_h2 = st.columns([1,3])
with col_h1:
    if real_time:
        st.markdown(f"### <span class='live-badge'>LIVE {now_str}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"### 🕒 Statico")
    st.metric("Prezzo Attuale", f"{lp:,.4f}" if lp else "Caricamento...", d_str)

# 2. CARICAMENTO MODELLO (Solo se cambia simbolo o refresh manuale)
k = f"model_{symbol}_{tf}"
if k not in st.session_state:
    with st.spinner("Addestramento AI..."):
        m, s, df = train_or_load_model(symbol, tf)
        if m: st.session_state[k] = {'m':m, 's':s, 'df':df}

if k in st.session_state:
    data = st.session_state[k]
    model, scaler, df_ind = data['m'], data['s'], data['df']
    
    # Aggiorniamo l'ultimo close del dataframe con il prezzo live per indicatori più freschi
    if lp:
        df_ind.iloc[-1, df_ind.columns.get_loc('Close')] = lp
        # Nota: ricalcolare tutti gli indicatori qui sarebbe pesante, usiamo approssimazione sull'ultima riga
    
    last_row = df_ind.iloc[-1]
    
    # Recupera SUGGERIMENTI WEB (Cachati per 5 min, quindi non rallentano il live)
    web_sigs = get_web_signals_cached(symbol, lp if lp else last_row['Close'], last_row['ATR'], last_row['Trend'])
    
    # LAYOUT PRINCIPALE
    left, right = st.columns([1.5, 1])
    
    with left:
        st.subheader("💡 Suggerimenti AI & Web")
        if web_sigs:
            for i, row in enumerate(web_sigs):
                # Card Grafica
                st.markdown(f"""
                <div class='trade-card'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span><strong>{row['Direction'].upper()}</strong> @ {row['Entry']:.2f}</span>
                        <span>Prob: <strong>{row['Probability']}%</strong></span>
                    </div>
                    <div style='font-size:0.9em; color:#555; margin-top:5px;'>
                        SL: {row['SL']:.2f} | TP: {row['TP']:.2f} | <em>{row['Sentiment']}</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Bottone Analisi (Funziona anche in live mode perché usa session state)
                if st.button(f"🔍 Analizza {row['Direction']} #{i}", key=f"btn_{i}"):
                    st.session_state.sel_trade = row
        
        # Dettaglio Analisi Trade Selezionato
        if 'sel_trade' in st.session_state:
            t = st.session_state.sel_trade
            st.markdown("---")
            st.markdown(f"### 🎯 Analisi Profonda: {t['Direction']}")
            
            # Calcolo predizione AI in tempo reale
            feats = generate_features(df_ind, t['Entry'], t['SL'], t['TP'], t['Direction'].lower(), 60)
            prob_ai = predict_success(model, scaler, feats)
            
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("🤖 Fiducia AI", f"{prob_ai:.1f}%", f"{prob_ai - t['Probability']:+.1f}% vs Web")
            c_b.metric("Risk/Reward", f"{(abs(t['TP']-t['Entry'])/abs(t['Entry']-t['SL'])):.2f}")
            c_c.info(t['Seasonality_Note'])
            
            st.caption(f"News: {t['News_Summary']}")
            
            dom_factors = get_dominant_factors(model, feats)
            st.write("**Fattori decisivi:** " + ", ".join(dom_factors[:3]))

    with right:
        st.subheader("🚀 Watchlist Live")
        # Lista asset da monitorare
        watchlist_items = [
            {"Asset": "Gold", "Ticker": "GC=F", "Score": "⭐5"},
            {"Asset": "Silver", "Ticker": "SI=F", "Score": "⭐4"},
            {"Asset": "Bitcoin", "Ticker": "BTC-USD", "Score": "⭐4"},
            {"Asset": "S&P 500", "Ticker": "^GSPC", "Score": "⭐4"},
            {"Asset": "Nvidia", "Ticker": "NVDA", "Score": "⭐5"},
            {"Asset": "Euro", "Ticker": "EURUSD=X", "Score": "⭐3"},
        ]
        
        # Recupera dati watchlist (Cachati per 60 sec)
        # In questo modo la pagina si ricarica ogni 2s, ma questa funzione
        # viene eseguita realmente solo una volta al minuto.
        wl_df = get_watchlist_data(watchlist_items)
        
        if not wl_df.empty:
            st.dataframe(wl_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Market Stats")
        m1, m2, m3 = st.columns(3)
        m1.metric("RSI (14)", f"{last_row['RSI']:.1f}")
        m2.metric("ATR", f"{last_row['ATR']:.2f}")
        m3.metric("Trend", "Bullish" if last_row['Trend']==1 else "Bearish")

else:
    st.info("Attendi il caricamento del modello...")
