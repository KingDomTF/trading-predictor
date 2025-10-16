import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== DATI SCT DA RICERCA WEB ====================
# Pattern stagionali medi mensili (approssimati da chart storici; + positivo, - negativo)
SEASONAL_PATTERNS = {
    'GC=F': {1: 1.5, 2: 0.5, 3: 0.5, 4: -0.5, 5: -0.5, 6: 0.0, 7: 1.5, 8: 1.5, 9: 1.0, 10: -0.5, 11: 0.0, 12: 0.5},  # Oro
    'EURUSD=X': {1: -0.8, 2: 0.2, 3: -0.3, 4: -0.5, 5: -0.4, 6: 0.5, 7: 0.6, 8: 0.4, 9: -0.2, 10: 0.1, 11: 0.2, 12: 1.2},  # EUR/USD
    'CL=F': {1: 1.0, 2: -1.5, 3: 2.0, 4: 1.5, 5: 0.5, 6: 2.5, 7: 5.0, 8: 3.0, 9: -0.5, 10: 0.0, 11: -2.0, 12: -3.0},  # Petrolio
    # Aggiungi altri asset se necessario, basati su ricerche simili
}

# Correlazioni principali (es. oro vs USD: -0.7; valori da dati storici)
CORRELATIONS = {
    'GC=F': {'^TNX': -0.4, 'DX-Y.NYB': -0.7},  # Oro vs Tassi USA, USD Index
    'EURUSD=X': {'DX-Y.NYB': -0.9, '^GSPC': 0.5},  # EUR/USD vs USD, S&P500
    'CL=F': {'DX-Y.NYB': -0.3, '^GSPC': 0.6},  # Petrolio vs USD, S&P500
}

def get_seasonal_bias(symbol, current_month):
    """Calcola bias stagionale giornaliero (up/down/neutrale)."""
    if symbol in SEASONAL_PATTERNS:
        monthly_avg = SEASONAL_PATTERNS[symbol].get(current_month, 0)
        if monthly_avg > 0.5:
            return 1  # Uptrend
        elif monthly_avg < -0.5:
            return -1  # Downtrend
        else:
            return 0  # Neutrale
    return 0

def get_correlation_factor(symbol, df_ind, correlated_symbols):
    """Calcola fattore correlazione basato su asset correlati."""
    factor = 0.0
    for corr_sym, corr_coef in correlated_symbols.items():
        try:
            corr_data = yf.download(corr_sym, period='1mo', progress=False)['Close']
            if not corr_data.empty:
                corr_change = corr_data.pct_change().mean() * 100
                factor += corr_coef * corr_change  # Aggiusta bias
        except:
            pass
    return factor

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

def generate_features(df_ind, entry, sl, tp, direction, main_tf, symbol):
    """Genera features per la predizione, inclusi SCT."""
    latest = df_ind.iloc[-1]
  
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 1.0
    sl_distance = abs(entry - sl) / entry * 100
    tp_distance = abs(tp - entry) / entry * 100
  
    current_month = datetime.now().month
    seasonal_bias = get_seasonal_bias(symbol, current_month)
    
    corr_symbols = CORRELATIONS.get(symbol, {})
    corr_factor = get_correlation_factor(symbol, df_ind, corr_symbols)
  
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
        'trend': latest['Trend'],
        'seasonal_bias': seasonal_bias,  # Nuovo: Bias SCT
        'corr_factor': corr_factor  # Nuovo: Fattore correlazione
    }
  
    return np.array(list(features.values()), dtype=np.float32)

def simulate_historical_trades(df_ind, symbol, n_trades=500):
    """Simula trade storici per training, inclusi SCT."""
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
      
        # Simula mese casuale per historical
        sim_month = np.random.randint(1, 13)
        features = generate_features(df_ind.iloc[:idx+1], entry, sl, tp, direction, 60, symbol)
        features[-2] = get_seasonal_bias(symbol, sim_month)  # Adatta seasonal
      
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
        'BB Position', 'Volume Ratio', 'Price Change %', 'Trend',
        'Seasonal Bias', 'Corr Factor'  # Nuovi
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
def load_sample_data(symbol='GC=F', period='1y'):
    """Carica dati reali da yfinance."""
    try:
        data = yf.download(symbol, period=period, progress=False)
        if len(data) < 100:
            raise Exception("Dati insufficienti")
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return data
    except:
        # Dati simulati fallback
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
def train_or_load_model(symbol):
    """Addestra il modello per l'asset."""
    data = load_sample_data(symbol)
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, symbol, n_trades=500)
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

# Configurazione pagina
st.set_page_config(
    page_title="SCT Trading Predictor AI",
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
st.title("📈 SCT Trading Success Predictor AI")
st.markdown("**Analisi predittiva con Machine Learning e Seasonal Correlation Trend (SCT) per vari asset**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Parametri Trade")
  
    symbol = st.selectbox("📊 Asset", ["GC=F (Oro)", "EURUSD=X (EUR/USD)", "CL=F (Petrolio)"], index=0)
    symbol = symbol.split()[0]  # Estrai ticker
  
    direction = st.selectbox("📊 Direzione", ["long", "short"], index=0)
  
    col1, col2 = st.columns(2)
    with col1:
        entry = st.number_input("💰 Entry", value=2000.0, step=0.5, format="%.2f")
    with col2:
        main_tf = st.selectbox("⏰ TF", [15, 60, 240, 1440], index=1,
                               format_func=lambda x: f"{x}m" if x < 60 else (f"{x//60}H" if x < 1440 else "D1"))
  
    sl = st.number_input("🛑 Stop Loss", value=1980.0, step=0.5, format="%.2f")
    tp = st.number_input("🎯 Take Profit", value=2050.0, step=0.5, format="%.2f")
  
    st.markdown("---")
  
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
if 'model' not in st.session_state or refresh_data or 'symbol' not in st.session_state or st.session_state.symbol != symbol:
    with st.spinner("🧠 Caricamento AI e dati..."):
        model, scaler, df_ind = train_or_load_model(symbol)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.df_ind = df_ind
        st.session_state.symbol = symbol
        st.success("✅ Sistema pronto!")

model = st.session_state.model
scaler = st.session_state.scaler
df_ind = st.session_state.df_ind
symbol = st.session_state.symbol

# Statistiche correnti
col1, col2, col3, col4, col5 = st.columns(5)
latest = df_ind.iloc[-1]
with col1:
    st.metric("Prezzo Corrente", f"{latest['Close']:.2f}")
with col2:
    st.metric("RSI", f"{latest['RSI']:.1f}")
with col3:
    st.metric("ATR", f"{latest['ATR']:.2f}")
with col4:
    trend_emoji = "📈" if latest['Trend'] == 1 else "📉"
    st.metric("Trend", trend_emoji)
with col5:
    current_month = datetime.now().month
    s_bias = get_seasonal_bias(symbol, current_month)
    s_emoji = "📈" if s_bias > 0 else ("📉" if s_bias < 0 else "➡️")
    st.metric("SCT Bias", s_emoji)

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
            features = generate_features(df_ind, entry, sl, tp, direction, main_tf, symbol)
            success_prob = predict_success(model, scaler, features)
            factors = get_dominant_factors(model, features)
          
            # Risultato principale
            st.markdown("## 🎯 Risultato Analisi SCT")
          
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
                st.write("Il modello SCT indica condizioni favorevoli per questo trade.")
            elif success_prob >= 50:
                st.warning(f"⚠️ **SETUP NEUTRALE** ({success_prob:.1f}% probabilità)")
                st.write("Probabilità equilibrata. Valuta attentamente altri fattori.")
            else:
                st.error(f"❌ **SETUP SFAVOREVOLE** ({success_prob:.1f}% probabilità)")
                st.write("Il modello suggerisce di evitare questo trade.")
          
            # Fattori dominanti
            st.markdown("### 📊 Fattori Chiave dell'Analisi (inclusi SCT)")
            for i, factor in enumerate(factors, 1):
                st.write(f"**{i}.** {factor}")
          
            # Tendenza Giornaliera SCT
            st.markdown("### 📅 Tendenza Giornaliera SCT")
            bias_desc = "Rialzista" if features[-2] > 0 else ("Ribassista" if features[-2] < 0 else "Neutrale")
            st.write(f"Basato su stagionalità storica: {bias_desc}. Correlazione fattore: {features[-1]:.2f}")
          
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
    **Questo strumento usa Machine Learning (Random Forest) integrato con SCT per:**
    - Analizzare indicatori tecnici (RSI, MACD, EMA, Bollinger Bands, ATR)
    - Incorporare bias stagionale e correlazioni da dati storici web
    - Valutare setup di trading su vari asset e timeframe
    - Stimare probabilità di successo e tendenza giornaliera
  
    **Elementi SCT aggiunti:**
    - 📅 Stagionalità: Pattern mensili da analisi storiche
    - 🔗 Correlazioni: Relazioni con asset come USD, azioni
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    ⚠️ <strong>Disclaimer:</strong> Questo strumento è per scopi educativi e di analisi.<br>
    Non costituisce consiglio finanziario. Fai sempre le tue ricerche prima di operare.
</div>
""", unsafe_allow_html=True)
