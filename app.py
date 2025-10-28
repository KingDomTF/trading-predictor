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

# Suggerimenti basati su ricerca web
web_signals = {
    'GC=F': [
        {'Direction': 'Long', 'Entry': 4315, 'SL': 4300, 'TP': 4350, 'Probability': 65, 'Seasonality_Note': 'Ottobre bilanciato, forte finestra intorno al 11/14. Aprile quieter.', 'News_Summary': 'Gold a $4218, rally verso $4440, volatility alta.'},
        {'Direction': 'Short', 'Entry': 4350, 'SL': 4380, 'TP': 4316, 'Probability': 70, 'Seasonality_Note': 'Ottobre bilanciato, forte finestra intorno al 11/14. Aprile quieter.', 'News_Summary': 'Gold a $4218, rally verso $4440, volatility alta.'},
        {'Direction': 'Long', 'Entry': 4320, 'SL': 4273, 'TP': 4400, 'Probability': 60, 'Seasonality_Note': 'Ottobre bilanciato, forte finestra intorno al 11/14. Aprile quieter.', 'News_Summary': 'Gold a $4218, rally verso $4440, volatility alta.'},
    ],
    'EURUSD=X': [
        {'Direction': 'Sell', 'Entry': 1.17, 'SL': 1.18, 'TP': 1.15, 'Probability': 65, 'Seasonality_Note': 'Ottobre modestly bullish, average +0.45% da 1971.', 'News_Summary': 'EUR/USD a 1.16995, bullish verso 1.18-1.20 in Q4.'},
        {'Direction': 'Buy', 'Entry': 1.16, 'SL': 1.15, 'TP': 1.18, 'Probability': 60, 'Seasonality_Note': 'Ottobre modestly bullish, average +0.45% da 1971.', 'News_Summary': 'EUR/USD a 1.16995, bullish verso 1.18-1.20 in Q4.'},
    ],
    'SI=F': [
        {'Direction': 'Buy', 'Entry': 50.49, 'SL': 48.0, 'TP': 53.0, 'Probability': 65, 'Seasonality_Note': 'Forte da June a Sept, peaks in March/Sept, lows in June.', 'News_Summary': 'Silver a $52.84, up 76% in 2025, rally.'},
        {'Direction': 'Sell', 'Entry': 52.653, 'SL': 53.765, 'TP': 52.25, 'Probability': 60, 'Seasonality_Note': 'Forte da June a Sept, peaks in March/Sept, lows in June.', 'News_Summary': 'Silver a $52.84, up 76% in 2025, rally.'},
    ],
    'BTC-USD': [
        {'Direction': 'Sell', 'Entry': 110888.48, 'SL': 113154.63, 'TP': 105705.99, 'Probability': 65, 'Seasonality_Note': 'Ottobre favors long bias, forte intorno 11/16.', 'News_Summary': 'Bitcoin down to $105,732, slip below 200-day SMA, oversold vs gold.'},
        {'Direction': 'Buy', 'Entry': 112500, 'SL': 110000, 'TP': 115000, 'Probability': 60, 'Seasonality_Note': 'Ottobre favors long bias, forte intorno 11/16.', 'News_Summary': 'Bitcoin down to $105,732, slip below 200-day SMA, oversold vs gold.'},
    ],
}

def suggest_trades(model, scaler, df_ind, main_tf, num_suggestions=5, prob_threshold=65.0):
    """Suggerisce trade con alta probabilità."""
    latest = df_ind.iloc[-1]
    entry = latest['Close']
    atr = latest['ATR']
    
    suggestions = []
    for _ in range(50):
        direction = np.random.choice(['long', 'short'])
        sl_pct = np.random.uniform(0.5, 2.0)
        tp_pct = np.random.uniform(1.0, 4.0)
        
        if direction == 'long':
            sl = entry - (atr * sl_pct / 100 * 10)
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
    model, scaler = train_model(X, y)
    return model, scaler, df_ind

# Configurazione pagina
st.set_page_config(
    page_title="Trading Predictor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizzato
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📈 Trading Success Predictor AI")
st.markdown("**Analisi predittiva per operazioni su vari strumenti con Machine Learning**")

# Parametri semplificati
col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.text_input("📈 Seleziona Strumento (Ticker)", value="GC=F", help="Es: GC=F (Oro), EURUSD=X, BTC-USD, SI=F (Argento)")
with col2:
    data_interval = st.selectbox("⏰ Timeframe Dati", ['5m', '15m', '1h'], index=2)

main_tf = 60  # Fisso per analisi

refresh_data = st.button("🔄 Carica/Aggiorna Dati", use_container_width=True)

st.markdown("---")

# Inizializzazione modello
session_key = f"model_{symbol}_{data_interval}"
if session_key not in st.session_state or refresh_data:
    with st.spinner("🧠 Caricamento AI e dati..."):
        model, scaler, df_ind = train_or_load_model(symbol=symbol, interval=data_interval)
        if model is not None:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            st.success("✅ Sistema pronto!")
        else:
            st.error("Impossibile caricare dati. Prova un altro ticker.")

if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']

    # Layout: Statistiche correnti
    st.markdown("### 📊 Statistiche Correnti")
    latest = df_ind.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prezzo", f"${latest['Close']:.2f}")
    with col2:
        st.metric("RSI", f"{latest['RSI']:.1f}")
    with col3:
        st.metric("ATR", f"{latest['ATR']:.2f}")
    with col4:
        trend_emoji = "📈" if latest['Trend'] == 1 else "📉"
        st.metric("Trend", trend_emoji)

    st.markdown("---")

    # Layout a due colonne: Suggerimenti Web + Grafici
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 💡 Suggerimenti Trade (Web)")
        if symbol in web_signals:
            suggestions_df = pd.DataFrame(web_signals[symbol])
            suggestions_df = suggestions_df.sort_values(by='Probability', ascending=False)
            st.dataframe(suggestions_df[['Direction', 'Entry', 'SL', 'TP', 'Probability']].style.format({
                'Entry': '{:.2f}',
                'SL': '{:.2f}',
                'TP': '{:.2f}',
                'Probability': '{:.0f}%'
            }), use_container_width=True)
            
            with st.expander("📅 Stagionalità & 📰 News"):
                st.write("**Stagionalità:**", suggestions_df.iloc[0]['Seasonality_Note'])
                st.write("**News:**", suggestions_df.iloc[0]['News_Summary'])
        else:
            st.info("Nessun suggerimento web per questo strumento.")
    
    with col_right:
        st.markdown("### 📈 Grafici Tecnici")
        chart_data = df_ind.tail(100)[['Close', 'EMA_20', 'EMA_50']].copy()
        chart_data.columns = ['Prezzo', 'EMA 20', 'EMA 50']
        st.line_chart(chart_data, height=250)
        
        rsi_data = df_ind.tail(100)[['RSI']].copy()
        st.line_chart(rsi_data, height=200)

    # Predizione manuale
    if predict_btn:
        valid = True
        if direction == 'long' and (sl >= entry or tp <= entry):
            st.error("❌ Verifica i livelli per LONG")
            valid = False
        elif direction == 'short' and (sl <= entry or tp >= entry):
            st.error("❌ Verifica i livelli per SHORT")
            valid = False
        
        if valid:
            with st.spinner("🔮 Analisi in corso..."):
                features = generate_features(df_ind, entry, sl, tp, direction, main_tf)
                success_prob = predict_success(model, scaler, features)
                factors = get_dominant_factors(model, features)
                
                st.markdown("---")
                st.markdown("## 🎯 Risultato Analisi")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎲 Probabilità", f"{success_prob:.1f}%")
                with col2:
                    rr = abs(tp - entry) / abs(entry - sl)
                    st.metric("⚖️ R/R", f"{rr:.2f}x")
                with col3:
                    risk_pct = abs(entry - sl) / entry * 100
                    st.metric("📉 Risk", f"{risk_pct:.2f}%")
                with col4:
                    reward_pct = abs(tp - entry) / entry * 100
                    st.metric("📈 Reward", f"{reward_pct:.2f}%")
                
                # Interpretazione
                if success_prob >= 65:
                    st.success(f"✅ **SETUP FAVOREVOLE** ({success_prob:.1f}%)")
                elif success_prob >= 50:
                    st.warning(f"⚠️ **SETUP NEUTRALE** ({success_prob:.1f}%)")
                else:
                    st.error(f"❌ **SETUP SFAVOREVOLE** ({success_prob:.1f}%)")
                
                # Fattori
                st.markdown("### 📊 Fattori Chiave")
                for i, factor in enumerate(factors, 1):
                    st.write(f"**{i}.** {factor}")

else:
    st.warning("Carica i dati per lo strumento selezionato.")

# Info
with st.expander("ℹ️ Come funziona"):
    st.markdown("""
    **Machine Learning (Random Forest) analizza:**
    - 📊 Indicatori tecnici (RSI, MACD, EMA, Bollinger, ATR)
    - 📈 Setup storici e probabilità di successo
    - 💡 Suggerimenti web con stagionalità e news
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    ⚠️ <strong>Disclaimer:</strong> Strumento educativo. Non è consiglio finanziario.
</div>
""", unsafe_allow_html=True)
