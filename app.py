import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
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

# ==================== STREAMLIT APP ====================
@st.cache_data
def load_sample_data(symbol='GC=F', interval='1h'):
    """Carica dati reali da yfinance con timeframe specificato."""
    period_map = {
        '5m': '60d',
        '15m': '60d',
        '1h': '730d'
    }
    period = period_map.get(interval, '730d')
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        
        # Fix per multi-index colonne in versioni recenti di yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)  # Rimuove livello 'Ticker'
        
        if len(data) < 100:
            raise Exception("Dati insufficienti")
        
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return data
    except Exception as e:
        st.error(f"Errore nel caricamento dati: {e}")
        # Dati simulati come fallback
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
        base_price = 2000
        trend = np.cumsum(np.random.randn(1000) * 0.3)
        noise = np.random.randn(1000) * 2
      
        close_prices = base_price + trend + noise
      
        data = pd.DataFrame({
            'Open': close_prices + np.random.randn(1000) * 0.5,
            'High': close_prices + np.abs(np.random.randn(1000) * 1.5),
            'Low': close_prices - np.abs(np.random.randn(1000) * 1.5),
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        return data

@st.cache_resource
def train_or_load_model(symbol='GC=F', interval='1h'):
    """Addestra il modello con dati al timeframe specificato."""
    data = load_sample_data(symbol, interval)
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=500)
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

# Configurazione pagina
st.set_page_config(
    page_title="Trading Predictor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📈 Trading Success Predictor AI")
st.markdown("**Analisi predittiva per operazioni XAU/USD con Machine Learning**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Parametri Trade")
  
    direction = st.selectbox("📊 Direzione", ["long", "short"], index=0)
  
    col1, col2 = st.columns(2)
    with col1:
        entry = st.number_input("💰 Entry", value=2000.0, step=0.5, format="%.2f")
    with col2:
        main_tf = st.selectbox("⏰ TF Analisi", [5, 15, 60], index=1,
                               format_func=lambda x: f"{x}m" if x < 60 else f"{x//60}H")
  
    sl = st.number_input("🛑 Stop Loss", value=1980.0, step=0.5, format="%.2f")
    tp = st.number_input("🎯 Take Profit", value=2050.0, step=0.5, format="%.2f")
  
    st.markdown("---")
  
    st.header("📊 Timeframe Dati")
    data_interval = st.selectbox("⏰ Timeframe Dati", ['5m', '15m', '1h'], index=2)
  
    # Validazione input
    if direction == 'long':
        if sl >= entry:
            st.error("❌ SL deve essere < Entry")
        if tp <= entry:
            st.error("❌ TP deve essere > Entry")
    else:
        if sl <= entry:
            st.error("❌ SL deve essere > Entry")
        if tp >= entry:
            st.error("❌ TP deve essere < Entry")
  
    refresh_data = st.button("🔄 Aggiorna Dati", use_container_width=True)

# Inizializzazione
session_key = f"model_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Caricamento AI e dati..."):
        model, scaler, df_ind = train_or_load_model(symbol='GC=F', interval=data_interval)
        st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
        st.success("✅ Sistema pronto!")

state = st.session_state[session_key]
model = state['model']
scaler = state['scaler']
df_ind = state['df_ind']

# Statistiche correnti
col1, col2, col3, col4 = st.columns(4)
latest = df_ind.iloc[-1]
with col1:
    st.metric("Prezzo Corrente", f"${latest['Close']:.2f}")
with col2:
    st.metric("RSI", f"{latest['RSI']:.1f}")
with col3:
    st.metric("ATR", f"{latest['ATR']:.2f}")
with col4:
    trend_emoji = "📈" if latest['Trend'] == 1 else "📉"
    st.metric("Trend", trend_emoji)

# Predizione
st.markdown("---")
predict_btn = st.button("🚀 ANALIZZA TRADE", type="primary", use_container_width=True)
if predict_btn:
    # Validazione
    valid = True
    if direction == 'long' and (sl >= entry or tp <= entry):
        st.error("❌ Verifica i livelli: per LONG, SL < Entry < TP")
        valid = False
    elif direction == 'short' and (sl <= entry or tp >= entry):
        st.error("❌ Verifica i livelli: per SHORT, TP < Entry < SL")
        valid = False
  
    if valid:
        with st.spinner("🔮 Analisi in corso..."):
            features = generate_features(df_ind, entry, sl, tp, direction, main_tf)
            success_prob = predict_success(model, scaler, features)
            factors = get_dominant_factors(model, features)
          
            # Risultato principale
            st.markdown("## 🎯 Risultato Analisi")
          
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎲 Probabilità Successo", f"{success_prob:.1f}%",
                         delta=f"{success_prob-50:.1f}%" if success_prob > 50 else None)
            with col2:
                rr = abs(tp - entry) / abs(entry - sl)
                st.metric("⚖️ Risk/Reward", f"{rr:.2f}x")
            with col3:
                risk_pct = abs(entry - sl) / entry * 100
                st.metric("📉 Risk %", f"{risk_pct:.2f}%")
            with col4:
                reward_pct = abs(tp - entry) / entry * 100
                st.metric("📈 Reward %", f"{reward_pct:.2f}%")
          
            # Interpretazione
            st.markdown("### 💡 Valutazione")
            if success_prob >= 65:
                st.success(f"✅ **SETUP FAVOREVOLE** ({success_prob:.1f}% probabilità)")
                st.write("Il modello indica condizioni favorevoli per questo trade.")
            elif success_prob >= 50:
                st.warning(f"⚠️ **SETUP NEUTRALE** ({success_prob:.1f}% probabilità)")
                st.write("Probabilità equilibrata. Valuta attentamente altri fattori.")
            else:
                st.error(f"❌ **SETUP SFAVOREVOLE** ({success_prob:.1f}% probabilità)")
                st.write("Il modello suggerisce di evitare questo trade.")
          
            # Fattori dominanti
            st.markdown("### 📊 Fattori Chiave dell'Analisi")
            for i, factor in enumerate(factors, 1):
                st.write(f"**{i}.** {factor}")
          
            # Grafici
            col1, col2 = st.columns(2)
          
            with col1:
                st.markdown("### 📈 Prezzo e Medie Mobili")
                chart_data = df_ind.tail(100)[['Close', 'EMA_20', 'EMA_50']].copy()
                chart_data.columns = ['Prezzo', 'EMA 20', 'EMA 50']
                st.line_chart(chart_data, height=300)
          
            with col2:
                st.markdown("### 📉 RSI (14)")
                rsi_data = df_ind.tail(100)[['RSI']].copy()
                st.line_chart(rsi_data, height=300)

# Info
with st.expander("ℹ️ Come funziona"):
    st.markdown("""
    **Questo strumento usa Machine Learning (Random Forest) per:**
    - Analizzare indicatori tecnici (RSI, MACD, EMA, Bollinger Bands, ATR)
    - Valutare setup di trading basati su dati storici
    - Stimare probabilità di successo del trade
  
    **Indicatori analizzati:**
    - 📊 RSI: forza relativa del trend
    - 📈 MACD: momentum e direzione
    - 🎯 EMA: medie mobili esponenziali
    - 📉 Bollinger Bands: volatilità
    - ⚡ ATR: volatilità media
    - 📊 Volume: forza del movimento
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    ⚠️ <strong>Disclaimer:</strong> Questo strumento è per scopi educativi e di analisi.<br>
    Non costituisce consiglio finanziario. Fai sempre le tue ricerche prima di operare.
</div>
""", unsafe_allow_html=True)pagine webmi dici, in maniera elementare, cosa fa questa App??Ho pensato per 8sQuesta app è uno strumento semplice per chi fa trading su oro (XAU/USD). Ecco cosa fa, spiegato facile:

Carica dati: Prende prezzi storici dell'oro da internet (o usa dati finti se non riesce) e calcola indicatori tecnici come RSI, MACD, medie mobili, bande di Bollinger e altri, per capire come si muove il mercato.
Inserisci il tuo trade: Nella barra laterale, metti i dettagli del tuo trade: direzione (long o short), prezzo di entry, stop loss e take profit, e il timeframe (come 5m, 15m o 1h).
Analizza con AI: Usa un modello di machine learning (Random Forest) addestrato su trade simulati per prevedere la probabilità che il tuo trade vada bene (raggiunga il take profit senza toccare lo stop loss).
Mostra risultati: Ti dice la probabilità di successo (es. 60%), il rischio/ricompensa, e fattori chiave che influenzano l'analisi. Mostra anche grafici del prezzo, medie mobili e RSI per vedere il contesto.
Avvertenze: È solo per studio, non dà consigli finanziari veri. Ti avvisa di fare ricerche tue.

In pratica, aiuta a valutare se un'idea di trade è buona o no, ma non è magica – è basata su dati passati!2,8sModifica il codice sopra in maniera che: ricerca dal web solo dati reali. mi suggerisce i trade con probabilita' di successo alta, quando inserisco nella barra laterale lo strumento su cui voglio tradare. Non solo oro.Ho pensato per 27spythonimport streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
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

def suggest_trades(model, scaler, df_ind, main_tf, num_suggestions=5, prob_threshold=65.0):
    """Suggerisce trade con alta probabilità di successo."""
    latest = df_ind.iloc[-1]
    entry = latest['Close']
    atr = latest['ATR']
    
    suggestions = []
    for _ in range(50):  # Genera molti per filtrare
        direction = np.random.choice(['long', 'short'])
        sl_pct = np.random.uniform(0.5, 2.0)
        tp_pct = np.random.uniform(1.0, 4.0)
        
        if direction == 'long':
            sl = entry - (atr * sl_pct / 100 * 10)  # Scala con ATR
            tp = entry + (atr * tp_pct / 100 * 10)
        else:
            sl = entry + (atr * sl_pct / 100 * 10)
            tp = entry - (atr * tp_pct / 100 * 10)
        
        features = generate_features(df_ind, entry, sl, tp, direction, main_tf)
        success_prob = predict_success(model, scaler, features)
        
        if success_prob >= prob_threshold:
            suggestions.append({
                'Direction': direction,
                'Entry': entry,
                'SL': sl,
                'TP': tp,
                'Probability': success_prob
            })
        
        if len(suggestions) >= num_suggestions:
            break
    
    return pd.DataFrame(suggestions)

# ==================== STREAMLIT APP ====================
@st.cache_data
def load_sample_data(symbol, interval='1h'):
    """Carica dati reali da yfinance con timeframe specificato."""
    period_map = {
        '5m': '60d',
        '15m': '60d',
        '1h': '730d'
    }
    period = period_map.get(interval, '730d')
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        
        # Fix per multi-index colonne in versioni recenti di yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)  # Rimuove livello 'Ticker'
        
        if len(data) < 100:
            raise Exception("Dati insufficienti")
        
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return data
    except Exception as e:
        st.error(f"Errore nel caricamento dati reali: {e}")
        return None  # No fallback to simulated data

@st.cache_resource
def train_or_load_model(symbol, interval='1h'):
    """Addestra il modello con dati al timeframe specificato."""
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=500)
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

# Configurazione pagina
st.set_page_config(
    page_title="Trading Predictor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📈 Trading Success Predictor AI")
st.markdown("**Analisi predittiva per operazioni su vari strumenti con Machine Learning**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Parametri Trade")
    
    symbol = st.text_input("📈 Strumento (Ticker)", value="GC=F")
  
    direction = st.selectbox("📊 Direzione", ["long", "short"], index=0)
  
    col1, col2 = st.columns(2)
    with col1:
        entry = st.number_input("💰 Entry", value=2000.0, step=0.5, format="%.2f")
    with col2:
        main_tf = st.selectbox("⏰ TF Analisi", [5, 15, 60], index=1,
                               format_func=lambda x: f"{x}m" if x < 60 else f"{x//60}H")
  
    sl = st.number_input("🛑 Stop Loss", value=1980.0, step=0.5, format="%.2f")
    tp = st.number_input("🎯 Take Profit", value=2050.0, step=0.5, format="%.2f")
  
    st.markdown("---")
  
    st.header("📊 Timeframe Dati")
    data_interval = st.selectbox("⏰ Timeframe Dati", ['5m', '15m', '1h'], index=2)
  
    # Validazione input
    if direction == 'long':
        if sl >= entry:
            st.error("❌ SL deve essere < Entry")
        if tp <= entry:
            st.error("❌ TP deve essere > Entry")
    else:
        if sl <= entry:
            st.error("❌ SL deve essere > Entry")
        if tp >= entry:
            st.error("❌ TP deve essere < Entry")
  
    refresh_data = st.button("🔄 Aggiorna Dati", use_container_width=True)

# Inizializzazione
session_key = f"model_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Caricamento AI e dati..."):
        model, scaler, df_ind = train_or_load_model(symbol=symbol, interval=data_interval)
        if model is not None:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            st.success("✅ Sistema pronto!")
        else:
            st.error("Impossibile caricare dati reali. Prova un altro ticker o timeframe.")

if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']

    # Statistiche correnti
    col1, col2, col3, col4 = st.columns(4)
    latest = df_ind.iloc[-1]
    with col1:
        st.metric("Prezzo Corrente", f"${latest['Close']:.2f}")
    with col2:
        st.metric("RSI", f"{latest['RSI']:.1f}")
    with col3:
        st.metric("ATR", f"{latest['ATR']:.2f}")
    with col4:
        trend_emoji = "📈" if latest['Trend'] == 1 else "📉"
        st.metric("Trend", trend_emoji)

    # Suggerimenti trade
    suggest_btn = st.button("💡 Suggerisci Trade", use_container_width=True)
    if suggest_btn:
        with st.spinner("🔍 Generazione suggerimenti..."):
            suggestions_df = suggest_trades(model, scaler, df_ind, main_tf)
            if not suggestions_df.empty:
                st.markdown("### 📈 Suggerimenti Trade con Alta Probabilità")
                st.dataframe(suggestions_df.style.format({
                    'Entry': '{:.2f}',
                    'SL': '{:.2f}',
                    'TP': '{:.2f}',
                    'Probability': '{:.1f}%'
                }))
            else:
                st.warning("Nessun suggerimento con probabilità alta trovato. Prova a modificare i parametri.")

    # Predizione manuale
    st.markdown("---")
    predict_btn = st.button("🚀 ANALIZZA TRADE", type="primary", use_container_width=True)
    if predict_btn:
        # Validazione
        valid = True
        if direction == 'long' and (sl >= entry or tp <= entry):
            st.error("❌ Verifica i livelli: per LONG, SL < Entry < TP")
            valid = False
        elif direction == 'short' and (sl <= entry or tp >= entry):
            st.error("❌ Verifica i livelli: per SHORT, TP < Entry < SL")
            valid = False
      
        if valid:
            with st.spinner("🔮 Analisi in corso..."):
                features = generate_features(df_ind, entry, sl, tp, direction, main_tf)
                success_prob = predict_success(model, scaler, features)
                factors = get_dominant_factors(model, features)
              
                # Risultato principale
                st.markdown("## 🎯 Risultato Analisi")
              
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎲 Probabilità Successo", f"{success_prob:.1f}%",
                             delta=f"{success_prob-50:.1f}%" if success_prob > 50 else None)
                with col2:
                    rr = abs(tp - entry) / abs(entry - sl)
                    st.metric("⚖️ Risk/Reward", f"{rr:.2f}x")
                with col3:
                    risk_pct = abs(entry - sl) / entry * 100
                    st.metric("📉 Risk %", f"{risk_pct:.2f}%")
                with col4:
                    reward_pct = abs(tp - entry) / entry * 100
                    st.metric("📈 Reward %", f"{reward_pct:.2f}%")
              
                # Interpretazione
                st.markdown("### 💡 Valutazione")
                if success_prob >= 65:
                    st.success(f"✅ **SETUP FAVOREVOLE** ({success_prob:.1f}% probabilità)")
                    st.write("Il modello indica condizioni favorevoli per questo trade.")
                elif success_prob >= 50:
                    st.warning(f"⚠️ **SETUP NEUTRALE** ({success_prob:.1f}% probabilità)")
                    st.write("Probabilità equilibrata. Valuta attentamente altri fattori.")
                else:
                    st.error(f"❌ **SETUP SFAVOREVOLE** ({success_prob:.1f}% probabilità)")
                    st.write("Il modello suggerisce di evitare questo trade.")
              
                # Fattori dominanti
                st.markdown("### 📊 Fattori Chiave dell'Analisi")
                for i, factor in enumerate(factors, 1):
                    st.write(f"**{i}.** {factor}")
              
                # Grafici
                col1, col2 = st.columns(2)
              
                with col1:
                    st.markdown("### 📈 Prezzo e Medie Mobili")
                    chart_data = df_ind.tail(100)[['Close', 'EMA_20', 'EMA_50']].copy()
                    chart_data.columns = ['Prezzo', 'EMA 20', 'EMA 50']
                    st.line_chart(chart_data, height=300)
              
                with col2:
                    st.markdown("### 📉 RSI (14)")
                    rsi_data = df_ind.tail(100)[['RSI']].copy()
                    st.line_chart(rsi_data, height=300)
else:
    st.warning("Carica i dati per lo strumento selezionato.")

# Info
with st.expander("ℹ️ Come funziona"):
    st.markdown("""
    **Questo strumento usa Machine Learning (Random Forest) per:**
    - Analizzare indicatori tecnici (RSI, MACD, EMA, Bollinger Bands, ATR)
    - Valutare setup di trading basati su dati storici reali
    - Stimare probabilità di successo del trade
    - Suggerire trade con alta probabilità
  
    **Indicatori analizzati:**
    - 📊 RSI: forza relativa del trend
    - 📈 MACD: momentum e direzione
    - 🎯 EMA: medie mobili esponenziali
    - 📉 Bollinger Bands: volatilità
    - ⚡ ATR: volatilità media
    - 📊 Volume: forza del movimento
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    ⚠️ <strong>Disclaimer:</strong> Questo strumento è per scopi educativi e di analisi.<br>
    Non costituisce consiglio finanziario. Fai sempre le tue ricerche prima di operare.
</div>
""", unsafe_allow_html=True)
