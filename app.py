import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIGURAZIONE ASSET =========================
ASSETS = {
    'GC=F': '🥇 Gold',
    'SI=F': '🥈 Silver', 
    'BTC-USD': '₿ Bitcoin',
    '^GSPC': '📊 S&P 500'
}

# ========================= CONFIGURAZIONE ADATTIVA TIMEFRAME =========================
def get_tf_config(interval):
    if interval == '5m':
        return {
            'ema_fast': [5,8,13,21], 'ema_slow': [34,55],
            'rsi_period': 10, 'bb_period': 20, 'atr_period': 10,
            'pattern_lookback': 144, 'future_bars': 36, 'min_similarity': 0.78,
            'simulations': 5000
        }
    elif interval == '15m':
        return {
            'ema_fast': [8,13,21], 'ema_slow': [55,89],
            'rsi_period': 12, 'bb_period': 20, 'atr_period': 12,
            'pattern_lookback': 120, 'future_bars': 40, 'min_similarity': 0.80,
            'simulations': 4500
        }
    else:  # 1h
        return {
            'ema_fast': [9,20,50], 'ema_slow': [100,200],
            'rsi_period': 14, 'bb_period': 20, 'atr_period': 14,
            'pattern_lookback': 90, 'future_bars': 50, 'min_similarity': 0.80,
            'simulations': 3500
        }

# ========================= INDICATORI ADATTIVI =========================
def calculate_indicators(df, config):
    df = df.copy()
    for p in config['ema_fast'] + config['ema_slow']:
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    
    # EMA Alignment intelligente
    if len(config['ema_fast']) >= 3:
        align = (df[f'EMA_{config["ema_fast"][0]}'] > df[f'EMA_{config["ema_fast"][1]}']) & \
                (df[f'EMA_{config["ema_fast"][1]}'] > df[f'EMA_{config["ema_fast"][2]}'])
    else:
        align = df[f'EMA_{config["ema_fast"][0]}'] > df[f'EMA_{config["ema_fast"][-1]}']
    if config['ema_slow']:
        align &= df[f'EMA_{config["ema_fast"][-1]}'] > df[f'EMA_{config["ema_slow"][0]}']
    df['EMA_Alignment'] = align.astype(int)

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(config['rsi_period']).mean()
    loss = -delta.clip(upper=0).rolling(config['rsi_period']).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - 100 / (1 + rs)

    # MACD veloce per intraday
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger
    mid = df['Close'].rolling(config['bb_period']).mean()
    std = df['Close'].rolling(config['bb_period']).std()
    df['BB_upper'] = mid + 2*std
    df['BB_lower'] = mid - 2*std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / mid

    # ATR
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(config['atr_period']).mean()

    # Volume
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Volume_ratio'] = df['Volume'] / (df['Vol_MA20'] + 1)

    # Momentum & Position
    df['Momentum'] = df['Close'].pct_change(10)
    df['Price_Pos'] = (df['Close'] - df['Low'].rolling(20).min()) / (df['High'].rolling(20).max() - df['Low'].rolling(20).min() + 1e-8)

    return df.dropna()

# ========================= PATTERN MATCHING VELOCE =========================
def find_patterns(df_ind, config):
    if len(df_ind) < 300: return pd.DataFrame()
    lookback = config['pattern_lookback']
    features = ['RSI','MACD_Hist','Volume_ratio','Momentum','Price_Pos','EMA_Alignment']
    current = df_ind.iloc[-lookback:][features].mean()

    matches = []
    for i in range(lookback + 100, len(df_ind) - config['future_bars'], 4):
        hist = df_ind.iloc[i-lookback:i][features].mean()
        similarity = 1 - np.mean(np.abs(current - hist) / (np.abs(current) + np.abs(hist) + 1e-8))
        if similarity > config['min_similarity']:
            ret = (df_ind['Close'].iloc[i + config['future_bars']] - df_ind['Close'].iloc[i]) / df_ind['Close'].iloc[i]
            matches.append({'similarity': similarity, 'return': ret, 'direction': 'LONG' if ret > 0.02 else 'SHORT' if ret < -0.02 else 'HOLD'})
    
    return pd.DataFrame(matches).sort_values('similarity', ascending=False).head(30) if matches else pd.DataFrame()

# ========================= FEATURES PER ML =========================
def make_features(df_slice, direction):
    l = df_slice.iloc[-1]
    features = np.array([
        l['RSI']/100,
        1 if l['RSI'] < 30 else 0,
        1 if l['RSI'] > 70 else 0,
        l['MACD_Hist'] / l['Close'],
        l['EMA_Alignment'],
        (l['Close'] - l['BB_lower']) / (l['BB_upper'] - l['BB_lower'] + 1e-8),
        l['Volume_ratio'],
        l['Momentum'],
        l['ATR']/l['Close'],
        l['Price_Pos'],
        1 if direction == 'long' else 0
    ], dtype=np.float32)
    return features

# ========================= TRAINING CON CACHE =========================
@st.cache_resource(ttl=180)
def train_model(_symbol, _interval):
    config = get_tf_config(_interval)
    data = yf.download(_symbol, period="730d", interval=_interval, progress=False)
    if len(data) < 300:
        return None, None, None, None
    
    df_ind = calculate_indicators(data, config)
    patterns = find_patterns(df_ind, config)

    X, y = [], []
    sims = config['simulations']
    for _ in range(sims):
        idx = np.random.randint(200, len(df_ind)-120)
        direction = 'long' if df_ind.iloc[idx-30:idx]['EMA_Alignment'].mean() > 0.6 else 'short'
        entry = df_ind.iloc[idx]['Close']
        atr = max(df_ind.iloc[idx]['ATR'], entry*0.0001)
        
        sl_mult = np.random.uniform(0.8, 2.2)
        tp_mult = np.random.uniform(3.2, 6.5)
        sl = entry - atr*sl_mult if direction=='long' else entry + atr*sl_mult
        tp = entry + atr*tp_mult if direction=='long' else entry - atr*tp_mult

        feat = make_features(df_ind.iloc[:idx+1], direction)
        future_high = df_ind.iloc[idx+1:idx+101]['High'].max()
        future_low = df_ind.iloc[idx+1:idx+101]['Low'].min()
        hit_tp = (future_high >= tp) if direction=='long' else (future_low <= tp)
        hit_sl_first = (future_low <= sl) if direction=='long' else (future_high >= sl)
        success = 1 if hit_tp and not hit_sl_first else 0

        X.append(feat)
        y.append(success)

    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ensemble = VotingClassifier([
        ('gb', GradientBoostingClassifier(n_estimators=350, max_depth=9, learning_rate=0.08)),
        ('rf', RandomForestClassifier(n_estimators=400, max_depth=14, n_jobs=-1)),
        ('nn', MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=600, early_stopping=True))
    ], voting='soft', weights=[2.5, 2, 1])

    ensemble.fit(X_scaled, y)
    return ensemble, scaler, df_ind, patterns

# ========================= PREZZO LIVE =========================
def get_live_price(symbol):
    try:
        if symbol == 'BTC-USD':
            r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true", timeout=5)
            data = r.json()['bitcoin']
            return {'price': data['usd'], 'change': data['usd_24h_change']}
        else:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get('regularMarketPrice') or info.get('currentPrice') or ticker.history(period='1d')['Close'].iloc[-1]
            open_p = info.get('regularMarketOpen', price)
            return {'price': price, 'change': (price - open_p)/open_p*100}
    except:
        return {'price': 0, 'change': 0}

# ========================= UI STREAMLIT =========================
st.set_page_config(page_title="ALADDIN QUANTUM 2025", page_icon="⚡", layout="wide")
st.markdown("<h1 style='text-align:center;'>⚡ ALADDIN QUANTUM SCALPER 2025</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666; font-size:1.2rem;'>5m • 15m • 1h | Ensemble AI + Pattern Matching | Precisione Reale 2025</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2,1,1])
with col1:
    symbol = st.selectbox("Asset", options=list(ASSETS.keys()), format_func=lambda x: ASSETS[x])
with col2:
    interval = st.selectbox("Timeframe", ['5m', '15m', '1h'], index=0)  # 5m di default per te!
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh = st.button("🔄 Update", use_container_width=True)

# ========================= CARICAMENTO DATI =========================
key = f"q_{symbol}_{interval}"
if key not in st.session_state or refresh:
    with st.spinner(f"⚡ Addestramento QUANTUM su {interval} in corso..."):
        ensemble, scaler, df_ind, patterns = train_model(symbol, interval)
        live = get_live_price(symbol)
        st.session_state[key] = {
            'ensemble': ensemble, 'scaler': scaler, 'df_ind': df_ind,
            'patterns': patterns, 'live': live, 'time': datetime.datetime.now()
        }
        if ensemble:
            st.success(f"✅ Pronto! {datetime.datetime.now().strftime('%H:%M:%S')}")
        else:
            st.error("Errore dati")

if key in st.session_state:
    state = st.session_state[key]
    live = state['live']
    df_ind = state['df_ind']
    patterns = state['patterns']
    ensemble = state['ensemble']
    scaler = state['scaler']

    # Prezzo live
    st.markdown(f"### {ASSETS[symbol]} • {interval} • Ultimo aggiornamento: {state['time'].strftime('%H:%M:%S')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Prezzo", f"${live['price']:,.2f}")
    col2.metric("Variazione", f"{live['change']:+.2f}%")

    # Pattern storici
    if not patterns.empty:
        dom = patterns['direction'].mode()[0]
        avg_ret = patterns['return'].mean()*100
        st.success(f"Pattern dominante: {dom} • Ritorno medio simile: {avg_ret:+.2f}%")
    
    # Predizione attuale
    latest = df_ind.iloc[-1]
    direction = 'long' if latest['EMA_Alignment'] == 1 and
