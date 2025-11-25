import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import time
import warnings

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
   
    # Analisi generale attuale (2025)
    current_analysis = f"""
    **🌍 Contesto Globale (Ottobre 2025)**
   
    Nel contesto del 28 Ottobre 2025, i mercati globali sono influenzati da inflazione persistente (al 3.5% negli USA), tensioni geopolitiche e boom dell'IA. La psicologia degli investitori è segnata da un mix di ottimismo tecnologico e ansia macroeconomica. Per {symbol}, con trend {trend} e sentiment {sentiment_label}, gli investitori mostrano overreazioni emotive.
    """
   
    # Bias comportamentali
    biases_analysis = """
    ### 🧠 Analisi Approfondita dei Bias Comportamentali (2025)
   
    | Bias Cognitivo | Definizione | Esempio |
    |---------------|-------------|---------|
    | **Avversione alle Perdite** | Dolore perdita > Gioia guadagno. | Tenere asset in perdita. |
    | **FOMO** | Paura di perdere l'occasione. | Comprare sui massimi. |
    | **Recency Bias** | Peso eccessivo al breve termine. | Credere che il trend attuale sia infinito. |
    """
   
    # Analisi specifica per asset
    if symbol == 'GC=F':
        asset_specific = """
        ### 🥇 Focus su Oro (GC=F)
        Bene rifugio per eccellenza. Attenzione al **Safe-Haven Bias** durante le news geopolitiche.
        """
    elif symbol == 'BTC-USD':
        asset_specific = """
        ### ₿ Focus su Bitcoin (BTC-USD)
        Guidato dalla narrativa. Attenzione all'**Effetto Gregge** e alla volatilità estrema.
        """
    else:
        asset_specific = f"""
        ### 📈 Analisi Specifica per {symbol}
        La psicologia su questo asset seguirà pattern universali: paura nei ribassi, avidità nei rally.
        """
   
    return current_analysis + biases_analysis + asset_specific


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
        news_summary = ' | '.join([item.get('title', '') for item in news[:5] if isinstance(item, dict)]) if news and isinstance(news, list) else 'Nessuna news recente.'
       
        # Sentiment
        sentiment_label, sentiment_score = get_sentiment(news_summary)
       
        # Stagionalità
        hist_monthly = yf.download(symbol, period='10y', interval='1mo', progress=False)
        if len(hist_monthly) < 12:
            seasonality_note = 'Dati insufficienti.'
        else:
            hist_monthly['Return'] = hist_monthly['Close'].pct_change()
            hist_monthly['Month'] = hist_monthly.index.month
            monthly_returns = hist_monthly.groupby('Month')['Return'].mean()
            current_month = datetime.datetime.now().month
            avg_current = monthly_returns.get(current_month, 0) * 100
            seasonality_note = f'Ritorno storico mese corrente: {avg_current:.2f}%.'
       
        # Previsione
        _, forecast_series = predict_price(df_ind, steps=5)
        forecast_note = f'Media forecast (5 periodi): {forecast_series.mean():.2f}' if forecast_series is not None else 'N/A'
       
        # Generazione suggerimenti
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
        st.error(f"Errore web signals: {e}")
        return []


# ==================== PREZZO LIVE (INTERATTIVO) ====================

@st.cache_data(ttl=1)
def fetch_live_price(symbol: str):
    """
    Recupera il prezzo 'live' con cache minima (1 secondo)
    per permettere aggiornamenti frequenti.
    """
    ticker = yf.Ticker(symbol)
    last_price = None
    prev_close = None

    # 1) Fast Info
    try:
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info is not None:
            last_price = fast_info.get("lastPrice", None)
            prev_close = fast_info.get("previousClose", None)
    except Exception:
        pass

    # 2) Fallback Intraday
    if last_price is None:
        try:
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                last_price = float(hist["Close"].iloc[-1])
                if len(hist) > 1:
                    prev_close = float(hist["Close"].iloc[-2])
        except Exception:
            pass

    # 3) Fallback Daily
    if last_price is None:
        try:
            hist = ticker.history(period="2d", interval="1d")
            if not hist.empty:
                last_price = float(hist["Close"].iloc[-1])
                if len(hist) > 1:
                    prev_close = float(hist["Close"].iloc[-2])
        except Exception:
            pass

    return last_price, prev_close


@st.fragment
def show_interactive_price_panel(symbol_to_track):
    """
    Pannello isolato per il prezzo.
    Contiene:
    - Toggle per aggiornamento automatico
    - Pulsante per aggiornamento manuale
    """
    st.markdown("---")
    
    # --- CONTROLLI ---
    c_toggle, c_btn = st.columns([1, 3])
    
    with c_toggle:
        # Toggle: Se True, attiva il loop di aggiornamento
        auto_update = st.toggle("Live 🔴", value=False, help="Aggiorna automaticamente ogni 2 secondi")
    
    with c_btn:
        # Bottone: Se cliccato, ricarica il fragment (e quindi aggiorna il prezzo)
        if st.button("🔄 Aggiorna Prezzo", use_container_width=True):
            st.rerun()

    # --- DATI E VISUALIZZAZIONE ---
    live_price, prev_close = fetch_live_price(symbol_to_track)
    
    col_live1, col_live2 = st.columns([1, 1])
    with col_live1:
        if live_price is not None:
            delta_str = None
            if prev_close is not None and prev_close != 0:
                delta_pct = (live_price - prev_close) / prev_close * 100
                delta_str = f"{delta_pct:+.2f}%"
            display_price = f"{live_price:.4f}" if live_price < 10 else f"{live_price:.2f}"
            st.metric("💹 Prezzo di Mercato", display_price, delta_str)
        else:
            st.metric("💹 Prezzo di Mercato", "N/D")

    with col_live2:
        status = "🟢 Automatico (2s)" if auto_update else "⚪ Manuale"
        st.caption(
            f"Modalità: **{status}**\n\n"
            f"Ultimo check: {datetime.datetime.now().strftime('%H:%M:%S')}"
        )
    
    st.markdown("---")

    # --- LOGICA AUTO-UPDATE ---
    if auto_update:
        time.sleep(2)
        st.rerun()


# ==================== STREAMLIT APP ====================

@st.cache_data
def load_sample_data(symbol, interval='1h'):
    """Carica dati reali da yfinance."""
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
        st.error(f"Errore nel caricamento dati: {e}")
        return None


@st.cache_resource
def train_or_load_model(symbol, interval='1h'):
    """Addestra il modello."""
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=500)
    if X.size == 0 or y.size == 0:
        return None, None, None
    model, scaler = train_model(X, y)
    return model, scaler, df_ind


# Mappatura nomi propri
proper_names = {
    'GC=F': 'XAU/USD (Gold)',
    'EURUSD=X': 'EUR/USD',
    'SI=F': 'XAG/USD (Silver)',
    'BTC-USD': 'BTC/USD',
    '^GSPC': 'S&P 500',
}

# Configurazione pagina
st.set_page_config(
    page_title="Trading Predictor AI - Ultimate",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizzato
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 2rem; max-width: 1600px; }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
    }
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
    .stToggle { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Trading Success Predictor AI")
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;'>
    <p style='color: white; font-size: 1.1rem; margin: 0; text-align: center; font-weight: 500;'>
        🤖 Analisi predittiva avanzata • 📈 Prezzi Real-Time • 🧠 Psicologia
    </p>
</div>
""", unsafe_allow_html=True)

# Parametri Input
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("🔍 Ticker", value="GC=F", help="Es: GC=F, EURUSD=X, BTC-USD")
    proper_name = proper_names.get(symbol, symbol)
    st.markdown(f"**Asset:** `{proper_name}`")

with col2:
    data_interval = st.selectbox("⏰ Timeframe", ['5m', '15m', '1h'], index=2)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_data = st.button("🔄 Ricarica Modello AI", use_container_width=True)


# >>> QUI CHIAMIAMO IL PANNELLO PREZZO INTERATTIVO <<<
show_interactive_price_panel(symbol)


# Logica AI (Caricamento Modello)
session_key = f"model_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Training AI in corso..."):
        model, scaler, df_ind = train_or_load_model(symbol=symbol, interval=data_interval)
        if model is not None:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            st.success("✅ Modello pronto!")
        else:
            st.error("❌ Errore caricamento dati.")

# Display Dashboard AI
if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']
   
    # Previsione & Web Signals
    avg_forecast, forecast_series = predict_price(df_ind, steps=5)
    web_signals_list = get_web_signals(symbol, df_ind)
   
    col_left, col_right = st.columns([1.2, 0.8])
   
    # Colonna Sinistra: Suggerimenti Trade
    with col_left:
        st.markdown("### 💡 Suggerimenti Trade")
        if web_signals_list:
            suggestions_df = pd.DataFrame(web_signals_list).sort_values(by='Probability', ascending=False)
            
            for idx, row in suggestions_df.iterrows():
                sentiment_emoji = "🟢" if row['Sentiment'] == 'Positive' else "🔴" if row['Sentiment'] == 'Negative' else "🟡"
                
                c_trade, c_btn = st.columns([5, 1])
                with c_trade:
                    st.info(f"**{row['Direction'].upper()}** | Entry: {row['Entry']} | Prob: {row['Probability']}% {sentiment_emoji}")
                with c_btn:
                    if st.button("Analizza", key=f"an_{idx}"):
                        st.session_state.selected_trade = row
            
            # Espansione dettagli
            with st.expander("📊 Dettagli News & Sentiment"):
                st.write(suggestions_df.iloc[0]['News_Summary'])
                st.info(suggestions_df.iloc[0]['Seasonality_Note'])

        else:
            st.info("Nessun segnale web disponibile.")
   
    # Colonna Destra: Watchlist
    with col_right:
        st.markdown("### 🚀 Watchlist 2025")
        watchlist = ["GC=F", "SI=F", "BTC-USD", "NVDA", "^GSPC"]
        rows = []
        for tick in watchlist:
            p, prev = fetch_live_price(tick)
            if p:
                chg = ((p - prev)/prev)*100 if prev else 0
                rows.append({"Ticker": tick, "Price": f"{p:.2f}", "Change": f"{chg:+.2f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Analisi Dettagliata Trade Selezionato
    if 'selected_trade' in st.session_state:
        trade = st.session_state.selected_trade
        st.markdown("---")
        st.markdown(f"## 🎯 Analisi AI: {trade['Direction']} su {symbol}")
        
        # Rigeneriamo features al volo per il trade selezionato
        direction = 'long' if trade['Direction'].lower() in ['long', 'buy'] else 'short'
        features = generate_features(df_ind, trade['Entry'], trade['SL'], trade['TP'], direction, 60)
        prob_ai = predict_success(model, scaler, features)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilità AI", f"{prob_ai:.1f}%")
        c2.metric("Rischio/Rendimento", f"{(abs(trade['TP']-trade['Entry'])/abs(trade['Entry']-trade['SL'])):.2f}")
        c3.metric("Trend Attuale", "Bullish" if df_ind['Trend'].iloc[-1] == 1 else "Bearish")
        
        st.markdown("### 🧠 Psicologia")
        st.markdown(get_investor_psychology(symbol, trade['News_Summary'], trade['Sentiment'], df_ind))

else:
    st.warning("Carica i dati per vedere l'analisi.")
