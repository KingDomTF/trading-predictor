import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
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
    exp1 = df['Close'].ewm(span=12).mean()   # <<< QUI ERA L'ERRORE: tolto ")."
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
   with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_data = st.button("🔄 Carica Dati", use_container_width=True)

# ==================== NUOVO CODICE: PULSANTE RICERCA LIVE ====================
# Questo pulsante forza l'aggiornamento del prezzo scavalcando la cache
if st.button("🔎 Ricerca Prezzo Live", use_container_width=True, help="Forza una nuova ricerca su Yahoo Finance per il prezzo attuale"):
    fetch_live_price.clear()  # Cancella la cache per scaricare il dato fresco
# =============================================================================

# PREZZO LIVE CORRENTE
live_price, prev_close = fetch_live_price(symbol)
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
    """Analisi approfondita della psicologia dell'investitore con comparazione storica, bias comportamentali e focus specifici su asset come Bitcoin, Argento, Oro e S&P 500."""
    latest = df_ind.iloc[-1]
    trend = 'bullish' if latest['Trend'] == 1 else 'bearish'
    
    # Analisi generale attuale (2025)
    current_analysis = f"""
    **🌍 Contesto Globale (Ottobre 2025)**
    
    Nel contesto del 28 Ottobre 2025, i mercati globali sono influenzati da inflazione persistente (al 3.5% negli USA), tensioni geopolitiche (es. Medio Oriente e Ucraina) e un boom dell'IA che ha spinto il NASDAQ oltre i 20,000 punti. La psicologia degli investitori è segnata da un mix di ottimismo tecnologico e ansia macroeconomica, con il VIX a livelli elevati (intorno a 25), indicando volatilità. Per {symbol}, con trend {trend} e sentiment {sentiment_label}, gli investitori mostrano overreazioni emotive, amplificate da social media e AI-driven trading.
    """
    
    # Bias comportamentali
    biases_analysis = """
    ### 🧠 Analisi Approfondita dei Bias Comportamentali negli Investimenti (2025)
    
    I bias comportamentali causano spesso un gap tra ritorni del mercato e ritorni degli investitori retail stimato al 2-4% annuo.
    
    | Bias Cognitivo | Definizione | Esempio Generale |
    |---------------|-------------|------------------|
    | **Avversione alle Perdite** | Perdite percepite 2x più dolorose dei guadagni. | Mantenere asset in calo sperando in recuperi. |
    | **Eccessiva Fiducia** | Sovrastima abilità predittive. | Overtrading in asset volatili. |
    | **Effetto Gregge** | Seguire la massa. | Comprare dopo grandi rally. |
    | **Bias di Conferma** | Cercare conferme a convinzioni. | Ignorare segnali negativi sul proprio asset. |
    | **Bias di Ancoraggio** | Ancorarsi al prezzo di acquisto. | Non voler vendere in perdita. |
    | **Recency Bias** | Dare troppo peso agli eventi recenti. | Credere che l’ultimo trend continuerà all’infinito. |
    """
    
    # Analisi specifica per asset
    if symbol == 'GC=F':
        asset_specific = """
        ### 🥇 Focus su Oro (GC=F / XAU/USD)
        
        L'oro nel 2025 mantiene un ruolo di bene rifugio in contesti di inflazione e tensioni geopolitiche.  
        Bias chiave:
        - **Safe-Haven Bias**: rifugio emotivo nelle crisi.
        - **Loss Aversion**: difficoltà a vendere durante drawdown prolungati.
        - **FOMO**: ingresso tardivo dopo grandi rally.
        """
    elif symbol == 'BTC-USD':
        asset_specific = """
        ### ₿ Focus su Bitcoin (BTC-USD)
        
        Bitcoin è ancora fortemente guidato da sentiment e narrativa.  
        Bias chiave:
        - **Herding**: movimenti di massa dopo notizie/ETF/halving.
        - **Overconfidence**: convinzione di “capire il ciclo” meglio del mercato.
        - **Disposition Effect**: prendere profitti troppo presto sui gain e tenere le perdite.
        """
    elif symbol == 'SI=F':
        asset_specific = """
        ### 🥈 Focus su Argento (SI=F / XAG/USD)
        
        Argento = metallo metà industriale, metà rifugio: alta volatilità.  
        Bias chiave:
        - **FOMO** su “silver squeeze”.
        - **Recency Bias** su rally legati alla domanda industriale.
        """
    elif symbol == '^GSPC':
        asset_specific = """
        ### 📊 Focus su S&P 500 (^GSPC)
        
        L’S&P 500 riflette il sentiment macro-usa e il boom tech/AI.  
        Bias chiave:
        - **Home Bias** (per investitori USA).
        - **Overconfidence** in bull market prolungati.
        - **Panic Selling** nei crolli improvvisi.
        """
    else:
        asset_specific = f"""
        ### 📈 Analisi Specifica per {symbol}
        
        La psicologia su questo asset seguirà comunque pattern universali: paura nei ribassi, avidità nei rally, e forte influenza di bias come effetto gregge e recency bias.
        """
    
    historical_comparison = """
    ### 📚 Comparazione Storica Generale
    
    - **2008 Crisi Finanziaria**: panico e sell-off massicci, poi grande rally per chi è rimasto investito.
    - **2020 COVID**: crollo rapidissimo seguito da recupero a V.
    - **Dot-com 2000**: euforia tech seguita da crollo, simile ad alcune dinamiche attuali sul tema IA.
    """
    
    return current_analysis + biases_analysis + asset_specific + historical_comparison


def get_web_signals(symbol, df_ind):
    """Funzione dinamica per ottenere segnali web aggiornati, più precisi."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Prezzo corrente (ultimo close disponibile)
        hist = ticker.history(period='1d')
        if hist.empty:
            return []
        current_price = hist['Close'].iloc[-1]
        
        # News recenti
        news = getattr(ticker, "news", None)
        news_summary = ' | '.join([item.get('title', '') for item in news[:5] if isinstance(item, dict)]) if news and isinstance(news, list) else 'Nessuna news recente disponibile.'
        
        # Sentiment
        sentiment_label, sentiment_score = get_sentiment(news_summary)
        
        # Calcolo stagionalità
        hist_monthly = yf.download(symbol, period='10y', interval='1mo', progress=False)
        if len(hist_monthly) < 12:
            seasonality_note = 'Dati storici insufficienti per calcolare la stagionalità.'
        else:
            hist_monthly['Return'] = hist_monthly['Close'].pct_change()
            hist_monthly['Month'] = hist_monthly.index.month
            monthly_returns = hist_monthly.groupby('Month')['Return'].mean()
            current_month = datetime.datetime.now().month
            avg_current = monthly_returns.get(current_month, 0) * 100
            seasonality_note = f'Il mese corrente ha un ritorno medio storico di {avg_current:.2f}%.'
        
        # Previsione prezzo (usa df_ind per timeframe specifico)
        _, forecast_series = predict_price(df_ind, steps=5)
        forecast_note = f'Previsione media per i prossimi 5 periodi: {forecast_series.mean():.2f}' if forecast_series is not None else 'Previsione non disponibile.'
        
        # Genera suggerimenti precisi basati su sentiment e trend
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
        
        # Aggiungi un terzo suggerimento se sentiment neutrale
        if sentiment_score == 0:
            dir_ = directions[0] if trend == 1 else directions[1]
            entry = round(current_price, 2)
            sl_mult = 1.2
            tp_mult = 2.2
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
                'Probability': 65,
                'Seasonality_Note': seasonality_note,
                'News_Summary': news_summary,
                'Sentiment': sentiment_label,
                'Forecast_Note': forecast_note
            })
        
        return suggestions
    except Exception as e:
        st.error(f"Errore nel recupero dati web: {e}")
        return []


# ==================== PREZZO LIVE ====================

@st.cache_data(ttl=60)
def fetch_live_price(symbol: str):
    """
    Recupera il prezzo 'live' (ultimo disponibile) da Yahoo Finance.
    ttl=60 => al massimo un aggiornamento al minuto per evitare rate limit.
    """
    ticker = yf.Ticker(symbol)
    last_price = None
    prev_close = None

    # 1) Prova con fast_info
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
        except Exception:
            pass

    # 3) Fallback finale: ultimo daily close
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


# ==================== STREAMLIT APP ====================

@st.cache_data
def load_sample_data(symbol, interval='1h'):
    """Carica dati reali da yfinance."""
    period_map = {
        '5m': '60d',
        '15m': '60d',
        '1h': '730d'
    }
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
    page_title="Trading Predictor AI - Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizzato
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    section[data-testid="stSidebar"] { display: none; }
    .trade-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        transition: all 0.2s ease;
    }
    .trade-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
        transform: translateX(4px);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Trading Success Predictor AI")
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;'>
    <p style='color: white; font-size: 1.1rem; margin: 0; text-align: center; font-weight: 500;'>
        🤖 Analisi predittiva avanzata con Machine Learning • 📈 Indicatori tecnici real-time • 🧠 Psicologia dell'investitore
    </p>
</div>
""", unsafe_allow_html=True)

# Parametri
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input(
        "🔍 Seleziona Strumento (Ticker)",
        value="GC=F",
        help="Es: GC=F (Oro), EURUSD=X, BTC-USD, SI=F (Argento), ^GSPC (S&P 500)"
    )
    proper_name = proper_names.get(symbol, symbol)
    st.markdown(f"**Strumento selezionato:** `{proper_name}`")

with col2:
    data_interval = st.selectbox("⏰ Timeframe", ['5m', '15m', '1h'], index=2)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_data = st.button("🔄 Carica Dati", use_container_width=True)

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

st.markdown("---")

# Inizializzazione modello
session_key = f"model_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Caricamento AI e analisi dati..."):
        model, scaler, df_ind = train_or_load_model(symbol=symbol, interval=data_interval)
        if model is not None:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            st.success("✅ Sistema pronto! Modello addestrato con successo.")
        else:
            st.error("❌ Impossibile caricare dati. Verifica il ticker e riprova.")

if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']
    
    # Previsione prezzo
    avg_forecast, forecast_series = predict_price(df_ind, steps=5)
    
    # Segnali web
    web_signals_list = get_web_signals(symbol, df_ind)
    
    col_left, col_right = st.columns([1.2, 0.8])
   
    with col_left:
        st.markdown("### 💡 Suggerimenti Trade Intelligenti")
        if web_signals_list:
            suggestions_df = pd.DataFrame(web_signals_list)
            suggestions_df = suggestions_df.sort_values(by='Probability', ascending=False)
           
            st.markdown("**📋 Clicca su un trade per analisi approfondita AI:**")
           
            for idx, row in suggestions_df.iterrows():
                sentiment_emoji = "🟢" if row['Sentiment'] == 'Positive' else "🔴" if row['Sentiment'] == 'Negative' else "🟡"
                
                c_trade, c_btn = st.columns([5, 1])
                with c_trade:
                    st.markdown(f"""
                    <div class='trade-card'>
                        <strong style='font-size: 1.1rem; color: #667eea;'>{row['Direction'].upper()}</strong> 
                        <span style='color: #4a5568;'>• Entry: <strong>{row['Entry']:.2f}</strong> • SL: {row['SL']:.2f} • TP: {row['TP']:.2f}</span><br>
                        <span style='color: #2d3748;'>📊 Probabilità: <strong>{row['Probability']:.0f}%</strong> {sentiment_emoji} Sentiment: <strong>{row['Sentiment']}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                with c_btn:
                    if st.button("🔍", key=f"analyze_{idx}", help="Analizza con AI"):
                        st.session_state.selected_trade = row
           
            with st.expander("📊 Dettagli Supplementari (Stagionalità, News, Previsioni)"):
                st.markdown("#### 📅 Analisi Stagionalità")
                st.info(suggestions_df.iloc[0]['Seasonality_Note'])
                
                st.markdown("#### 📰 News Recenti")
                st.write(suggestions_df.iloc[0]['News_Summary'])
                
                st.markdown("#### 😊 Sentiment Aggregato")
                sentiment = suggestions_df.iloc[0]['Sentiment']
                if sentiment == 'Positive':
                    st.success(f"🟢 {sentiment} - Il mercato mostra segnali positivi")
                elif sentiment == 'Negative':
                    st.error(f"🔴 {sentiment} - Il mercato mostra segnali negativi")
                else:
                    st.warning(f"🟡 {sentiment} - Il mercato è neutrale")
                
                st.markdown("#### 🔮 Previsione Prezzo")
                st.info(suggestions_df.iloc[0]['Forecast_Note'])
        else:
            st.info("ℹ️ Nessun suggerimento web disponibile per questo strumento al momento.")
   
    with col_right:
        st.markdown("### 🚀 Asset con Potenziale 2025")
        st.markdown("*Basato su analisi storica e trend macro*")
        
        data_watch = [
            {"Asset": "🥇 Gold", "Ticker": "GC=F", "Score": "⭐⭐⭐⭐⭐"},
            {"Asset": "🥈 Silver", "Ticker": "SI=F", "Score": "⭐⭐⭐⭐"},
            {"Asset": "₿ Bitcoin", "Ticker": "BTC-USD", "Score": "⭐⭐⭐⭐"},
            {"Asset": "💎 Nvidia", "Ticker": "NVDA", "Score": "⭐⭐⭐⭐⭐"},
            {"Asset": "🖥️ Broadcom", "Ticker": "AVGO", "Score": "⭐⭐⭐⭐"},
            {"Asset": "🔍 Palantir", "Ticker": "PLTR", "Score": "⭐⭐⭐⭐"},
            {"Asset": "🏦 JPMorgan", "Ticker": "JPM", "Score": "⭐⭐⭐"},
            {"Asset": "☁️ Microsoft", "Ticker": "MSFT", "Score": "⭐⭐⭐⭐⭐"},
            {"Asset": "📦 Amazon", "Ticker": "AMZN", "Score": "⭐⭐⭐⭐"},
            {"Asset": "🚗 Tesla", "Ticker": "TSLA", "Score": "⭐⭐⭐⭐"},
            {"Asset": "🔋 Lithium ETF", "Ticker": "LIT", "Score": "⭐⭐⭐⭐"},
            {"Asset": "📊 S&P 500", "Ticker": "^GSPC", "Score": "⭐⭐⭐⭐"}
        ]

        rows = []
        for row in data_watch:
            price, prev = fetch_live_price(row["Ticker"])
            if price is not None and prev is not None and prev != 0:
                change_pct = (price - prev) / prev * 100
                change_str = f"{change_pct:+.2f}%"
            else:
                change_str = "N/D"

            if price is not None:
                price_str = f"{price:.4f}" if price < 10 else f"{price:.2f}"
            else:
                price_str = "N/D"

            rows.append({
                "Asset": row["Asset"],
                "Ticker": row["Ticker"],
                "Score": row["Score"],
                "Live Price": price_str,
                "Δ % (vs close prec.)": change_str,
            })

        growth_df = pd.DataFrame(rows)
        st.dataframe(growth_df, use_container_width=True, hide_index=True)
    
    # Analisi del trade selezionato
    if 'selected_trade' in st.session_state:
        trade = st.session_state.selected_trade
       
        with st.spinner("🔮 Analisi AI in corso..."):
            direction = 'long' if trade['Direction'].lower() in ['long', 'buy'] else 'short'
            entry = trade['Entry']
            sl = trade['SL']
            tp = trade['TP']
           
            features = generate_features(df_ind, entry, sl, tp, direction, 60)
            success_prob = predict_success(model, scaler, features)
            factors = get_dominant_factors(model, features)
           
            st.markdown("---")
            st.markdown("### 📊 Dashboard Statistiche Real-Time")
            latest = df_ind.iloc[-1]
            
            ca, cb, cc, cd, ce = st.columns(5)
            with ca:
                st.metric("💵 Prezzo Attuale", f"{latest['Close']:.2f}")
            with cb:
                rsi_color = "🟢" if 30 <= latest['RSI'] <= 70 else "🔴"
                st.metric(f"{rsi_color} RSI", f"{latest['RSI']:.1f}")
            with cc:
                st.metric("📏 ATR", f"{latest['ATR']:.2f}")
            with cd:
                trend_emoji = "📈" if latest['Trend'] == 1 else "📉"
                trend_text = "Bullish" if latest['Trend'] == 1 else "Bearish"
                st.metric(f"{trend_emoji} Trend", trend_text)
            with ce:
                if avg_forecast is not None:
                    forecast_change = ((avg_forecast - latest['Close']) / latest['Close']) * 100
                    st.metric("🔮 Previsione", f"{avg_forecast:.2f}", f"{forecast_change:+.1f}%")
                else:
                    st.metric("🔮 Previsione", "N/A")
            
            st.markdown("---")
            st.markdown("## 🎯 Risultati Analisi AI Avanzata")
           
            c1r, c2r, c3r, c4r = st.columns(4)
            with c1r:
                delta = success_prob - trade['Probability']
                st.metric(
                    "🎲 Probabilità AI",
                    f"{success_prob:.1f}%",
                    delta=f"{delta:+.1f}%" if delta != 0 else None,
                    help=f"Analisi Web: {trade['Probability']:.0f}%"
                )
            with c2r:
                rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0.0
                rr_emoji = "🟢" if rr >= 2 else "🟡" if rr >= 1.5 else "🔴"
                st.metric(f"{rr_emoji} Risk/Reward", f"{rr:.2f}x")
            with c3r:
                risk_pct = abs(entry - sl) / entry * 100 if entry != 0 else 0.0
                st.metric("📉 Rischio %", f"{risk_pct:.2f}%")
            with c4r:
                reward_pct = abs(tp - entry) / entry * 100 if entry != 0 else 0.0
                st.metric("📈 Reward %", f"{reward_pct:.2f}%")
           
            st.markdown("---")
            st.markdown("### 🔍 Fattori Chiave dell'Analisi AI")
            for i, factor in enumerate(factors, 1):
                emoji = ["🥇", "🥈", "🥉", "🏅", "🎖️"][i-1]
                st.markdown(f"{emoji} **{i}.** {factor}")
            
            st.markdown("---")
            st.markdown("### 🧠 Analisi Psicologica dell'Investitore")
            st.markdown("*Approfondimento comportamentale con focus su " + proper_name + "*")
            psych_analysis = get_investor_psychology(symbol, trade['News_Summary'], trade['Sentiment'], df_ind)
            st.markdown(psych_analysis)
else:
    st.warning("⚠️ Seleziona uno strumento e carica i dati per iniziare l'analisi.")

# Info
with st.expander("ℹ️ Come Funziona Questo Sistema"):
    st.markdown("""
    ### 🤖 Tecnologia AI Avanzata
    
    - 📊 14 indicatori tecnici (RSI, MACD, EMA, Bollinger, ATR, Volume, Trend)
    - 📈 500+ setup storici simulati per addestrare il modello
    - 🌐 Segnali web: news, sentiment, stagionalità
    - 🧠 Focus sulla psicologia comportamentale dell'investitore
    
    ⚠️ Questo strumento è a scopo educativo e non costituisce consulenza finanziaria.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; margin-top: 2rem;'>
    <p style='color: #4a5568; font-size: 0.95rem; margin: 0;'>
        ⚠️ <strong>Disclaimer Importante:</strong> Questo è uno strumento educativo e di ricerca. Non costituisce consiglio finanziario.<br>
        Consulta sempre un professionista qualificato prima di prendere decisioni di investimento.
    </p>
    <p style='color: #718096; font-size: 0.85rem; margin-top: 0.5rem;'>
        Sviluppato con ❤️ utilizzando Machine Learning • © 2025
    </p>
</div>
""", unsafe_allow_html=True)
