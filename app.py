import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
import json
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== MT4 BRIDGE SYSTEM ====================
class MT4Bridge:
    """Gestisce comunicazione bidirezionale con MT4"""
    
    def __init__(self, bridge_folder= r"C:\Users\dcbat\AppData\Roaming\MetaQuotes\Terminal\B8925BF731C22E88F33C7A8D7CD3190E\MQL4\Files"):
        self.bridge_folder = Path(bridge_folder)
        self.signals_file = self.bridge_folder / "signals.json"
        self.status_file = self.bridge_folder / "status.json"
        self.trades_file = self.bridge_folder / "trades.json"
        self.heartbeat_file = self.bridge_folder / "heartbeat.json"
        
        # Crea cartella se non esiste
        try:
            self.bridge_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            st.error(f"❌ Errore creazione cartella bridge: {e}")
    
    def send_signal(self, signal_data):
        """Invia segnale a MT4"""
        try:
            signal = {
                "timestamp": datetime.datetime.now().isoformat(),
                "symbol": signal_data.get("symbol", "XAUUSD"),
                "direction": signal_data.get("direction", "BUY"),
                "entry": float(signal_data.get("entry", 0)),
                "stop_loss": float(signal_data.get("sl", 0)),
                "take_profit": float(signal_data.get("tp", 0)),
                "lot_size": float(signal_data.get("lot_size", 0.01)),
                "probability": float(signal_data.get("probability", 0)),
                "ai_confidence": float(signal_data.get("ai_confidence", 0)),
                "risk_reward": float(signal_data.get("rr_ratio", 0)),
                "comment": signal_data.get("comment", "AI_Signal"),
                "magic_number": int(signal_data.get("magic", 12345)),
                "status": "PENDING"
            }
            
            with open(self.signals_file, 'w') as f:
                json.dump(signal, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Errore invio segnale: {e}")
            return False
    
    def get_status(self):
        """Legge status MT4"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def get_open_trades(self):
        """Recupera trade aperti"""
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def is_mt4_connected(self):
        """Verifica connessione MT4"""
        try:
            if self.heartbeat_file.exists():
                with open(self.heartbeat_file, 'r') as f:
                    data = json.load(f)
                    last_beat = datetime.datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                    age = (datetime.datetime.now() - last_beat).total_seconds()
                    return age < 10  # Connesso se heartbeat < 10 secondi
        except:
            pass
        return False
    
    def get_live_price(self):
        """Recupera prezzo live da MT4 con gestione errori di lettura"""
        json_path = self.bridge_folder / "live_price.json"
        try:
            if json_path.exists():
                # Tenta di leggere fino a 3 volte se il file è occupato
                for _ in range(3):
                    try:
                        with open(json_path, 'r') as f:
                            return json.load(f)
                    except json.JSONDecodeError:
                        # Il file è vuoto o corrotto mentre MT4 scriveva
                        time.sleep(0.01)
                        continue
                    except Exception:
                        time.sleep(0.01)
                        continue
        except:
            pass
        return None
    
    def clear_signal(self):
        """Pulisce file segnale"""
        try:
            if self.signals_file.exists():
                self.signals_file.unlink()
            return True
        except:
            return False

# ==================== HELPER FUNCTIONS ====================
def convert_yfinance_to_mt4_symbol(yf_symbol):
    """Converte ticker yfinance in simbolo MT4"""
    mapping = {
        'GC=F': 'XAUUSD',
        'SI=F': 'XAGUSD',
        'EURUSD=X': 'EURUSD',
        'GBPUSD=X': 'GBPUSD',
        'USDJPY=X': 'USDJPY',
        'BTC-USD': 'BTCUSD',
        '^GSPC': 'US500',
        '^DJI': 'US30',
    }
    return mapping.get(yf_symbol, yf_symbol.replace('=F', '').replace('=X', ''))

def calculate_lot_size(balance, risk_pct, entry, sl, pip_value=10, contract_size=100):
    """Calcola lot size ottimale"""
    risk_amount = balance * (risk_pct / 100)
    sl_distance = abs(entry - sl)
    
    # Per oro: 1 lotto = 100 oz, 1 pip = 0.01, valore pip = $10
    sl_pips = sl_distance / 0.01
    
    if sl_pips <= 0:
        return 0.01
    
    lot_size = risk_amount / (sl_pips * pip_value)
    lot_size = max(0.01, min(round(lot_size, 2), 10.0))
    
    return lot_size

# ==================== TECHNICAL INDICATORS ====================
def calculate_technical_indicators(df):
    """Calcola indicatori tecnici"""
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

# ==================== GOLD ANALYSIS ====================
def get_gold_fundamental_factors():
    """Recupera fattori fondamentali oro"""
    factors = {}
    
    try:
        # Dollar Index
        for ticker_symbol in ["DX-Y.NYB", "DX=F"]:
            try:
                dxy = yf.Ticker(ticker_symbol)
                dxy_hist = dxy.history(period="5d", interval="1d")
                if not dxy_hist.empty:
                    factors['dxy_current'] = float(dxy_hist['Close'].iloc[-1])
                    factors['dxy_change'] = ((dxy_hist['Close'].iloc[-1] - dxy_hist['Close'].iloc[0]) / dxy_hist['Close'].iloc[0]) * 100
                    break
            except:
                continue
        
        if 'dxy_current' not in factors:
            factors['dxy_current'] = 106.2
            factors['dxy_change'] = -0.3
        
        # Treasury 10Y
        try:
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(period="5d")
            if not tnx_hist.empty:
                factors['yield_10y'] = float(tnx_hist['Close'].iloc[-1])
            else:
                factors['yield_10y'] = 4.42
        except:
            factors['yield_10y'] = 4.42
        
        # VIX
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="5d")
            if not vix_hist.empty:
                factors['vix'] = float(vix_hist['Close'].iloc[-1])
            else:
                factors['vix'] = 16.8
        except:
            factors['vix'] = 16.8
        
        # S&P 500 Momentum
        try:
            spx = yf.Ticker("^GSPC")
            spx_hist = spx.history(period="5d")
            if not spx_hist.empty and len(spx_hist) > 1:
                factors['spx_momentum'] = ((spx_hist['Close'].iloc[-1] - spx_hist['Close'].iloc[0]) / spx_hist['Close'].iloc[0]) * 100
            else:
                factors['spx_momentum'] = 1.8
        except:
            factors['spx_momentum'] = 1.8
        
        # Gold/Silver Ratio
        factors['gold_silver_ratio'] = 88.5
        factors['inflation_expectations'] = 2.5
        factors['geopolitical_risk'] = 7.5
        factors['central_bank_demand'] = 1050
        
    except Exception as e:
        st.warning(f"⚠️ Alcuni dati fondamentali non disponibili")
        factors = {
            'dxy_current': 106.2, 'dxy_change': -0.3, 'yield_10y': 4.42,
            'vix': 16.8, 'spx_momentum': 1.8, 'gold_silver_ratio': 88.5,
            'inflation_expectations': 2.5, 'geopolitical_risk': 7.5,
            'central_bank_demand': 1050
        }
    
    return factors

def analyze_gold_historical_comparison(current_price, factors):
    """Analisi oro con confronto storico"""
    
    historical_periods = {
        '1971-1980': {
            'description': 'Bull Market Post-Bretton Woods',
            'start_price': 35, 'end_price': 850, 'gain_pct': 2329,
            'duration_years': 9, 'avg_inflation': 8.5, 'geopolitical': 8,
            'dollar_weak': True, 'key_events': 'Fine gold standard, inflazione alta'
        },
        '2001-2011': {
            'description': 'Bull Market Post-Dot-Com',
            'start_price': 255, 'end_price': 1920, 'gain_pct': 653,
            'duration_years': 10, 'avg_inflation': 2.8, 'geopolitical': 7,
            'dollar_weak': True, 'key_events': '9/11, QE, crisi 2008'
        },
        '2022-2025': {
            'description': 'Era Inflazione Post-COVID',
            'start_price': 1800, 'end_price': current_price,
            'gain_pct': ((current_price - 1800) / 1800) * 100,
            'duration_years': 3, 'avg_inflation': 4.5, 'geopolitical': 7.5,
            'dollar_weak': False, 'key_events': 'Inflazione persistente, tensioni'
        }
    }
    
    # Trova periodo simile
    similarity_scores = {}
    for period, data in historical_periods.items():
        if period == '2022-2025':
            continue
        score = 0
        score += max(0, 10 - abs(data['avg_inflation'] - factors['inflation_expectations']) * 2)
        score += max(0, 10 - abs(data['geopolitical'] - factors['geopolitical_risk']) * 2)
        if data['dollar_weak'] == (factors['dxy_current'] < 100):
            score += 15
        similarity_scores[period] = score
    
    most_similar = max(similarity_scores, key=similarity_scores.get)
    similar_period = historical_periods[most_similar]
    
    # Calcola target
    annual_return = (similar_period['gain_pct'] / 100) / similar_period['duration_years']
    target_1y = current_price * (1 + annual_return)
    target_6m = current_price + (target_1y - current_price) * 0.5
    target_3m = current_price + (target_1y - current_price) * 0.25
    
    confidence = min(100, similarity_scores[most_similar] * 2)
    
    return {
        'current_price': current_price,
        'target_3m': target_3m,
        'target_6m': target_6m,
        'target_1y': target_1y,
        'most_similar_period': most_similar,
        'similarity_pct': (similarity_scores[most_similar] / 45) * 100,
        'confidence': confidence,
        'key_drivers': {
            'Dollar Index': f"${factors['dxy_current']:.2f}",
            'VIX': f"{factors['vix']:.1f}",
            'Inflazione': f"{factors['inflation_expectations']:.2f}%",
            'Rischio Geo': f"{factors['geopolitical_risk']}/10"
        }
    }

# ==================== ML MODEL ====================
def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    """Genera features per ML"""
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
    """Simula trade storici"""
    X_list, y_list = [], []
    
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
    """Addestra Random Forest"""
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
    """Predice probabilità successo"""
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100

# ==================== WEB SIGNALS ====================
def get_web_signals(symbol, df_ind):
    """Genera segnali intelligenti"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", interval="1h")
        
        if hist.empty:
            return []
        
        current_price = float(hist['Close'].iloc[-1])
        latest = df_ind.iloc[-1]
        atr = latest['ATR']
        trend = latest['Trend']
        rsi = latest['RSI']
        
        suggestions = []
        
        # Long signal
        if rsi < 40 or trend == 1:
            suggestions.append({
                'Direction': 'Long',
                'Entry': round(current_price, 2),
                'SL': round(current_price - atr * 1.5, 2),
                'TP': round(current_price + atr * 3.0, 2),
                'Probability': 72 if rsi < 40 else 65
            })
        
        # Short signal
        if rsi > 60 or trend == 0:
            suggestions.append({
                'Direction': 'Short',
                'Entry': round(current_price, 2),
                'SL': round(current_price + atr * 1.5, 2),
                'TP': round(current_price - atr * 3.0, 2),
                'Probability': 72 if rsi > 60 else 65
            })
        
        return suggestions
        
    except Exception as e:
        st.error(f"Errore generazione segnali: {e}")
        return []

# ==================== DATA LOADING ====================
@st.cache_data(ttl=300)
def load_sample_data(symbol, interval='1h'):
    """Carica dati da yfinance"""
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
        st.error(f"Errore caricamento dati: {e}")
        return None

@st.cache_resource
def train_or_load_model(symbol, interval='1h'):
    """Addestra modello ML"""
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None
    
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=500)
    model, scaler = train_model(X, y)
    
    return model, scaler, df_ind

# ==================== STREAMLIT UI ====================
st.set_page_config(
    page_title="AI Trading Predictor + MT4",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; max-width: 1600px;}
    h1 {background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 700; font-size: 2.5rem !important;}
    .stMetric {background: linear-gradient(135deg, #FFF8DC 0%, #FFE4B5 100%);
        padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(255,215,0,0.2);}
    .stButton > button {background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: white; border: none; border-radius: 8px; padding: 0.6rem 1.5rem;
        font-weight: 600; box-shadow: 0 4px 6px rgba(255,165,0,0.3);}
    .trade-card {background: white; border-radius: 10px; padding: 1rem;
        margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(255,215,0,0.15);
        border-left: 4px solid #FFD700;}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎯 AI Trading Predictor + MT4 Integration")
st.markdown("""
<div style='background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 1rem; 
     border-radius: 10px; margin-bottom: 1rem;'>
    <p style='color: white; font-size: 1.1rem; margin: 0; text-align: center; font-weight: 500;'>
        🤖 Machine Learning • 📊 Real-Time Analysis • 🔗 MT4 Auto-Trading • ⚡ Live Prices
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize
if 'mt4_bridge' not in st.session_state:
    st.session_state.mt4_bridge = MT4Bridge()
    st.session_state.live_price_active = False
    st.session_state.last_price_update = time.time()

bridge = st.session_state.mt4_bridge

# Configurazione
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    symbol = st.text_input("🎯 Ticker Symbol", value="GC=F", 
                           help="GC=F (Gold), SI=F (Silver), EURUSD=X, BTC-USD")
    
with col2:
    interval = st.selectbox("⏱️ Timeframe", ['5m', '15m', '1h'], index=2)

# Load Model logic happens later, but we need df_ind for suggestions
# So we run train_or_load_model early to get df_ind
model_placeholder, scaler_placeholder, df_ind_placeholder = train_or_load_model(symbol, interval)
df_ind = df_ind_placeholder

with col3:
    st.markdown("---")
    
    # Trade Suggestions
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 💡 AI Trade Suggestions")
        
        if df_ind is not None:
            web_signals = get_web_signals(symbol, df_ind)
            
            if web_signals:
                for idx, signal in enumerate(web_signals):
                    col_trade, col_analyze = st.columns([4, 1])
                    
                    with col_trade:
                        direction_emoji = "🟢" if signal['Direction'] == 'Long' else "🔴"
                        st.markdown(f"""
                        <div class='trade-card'>
                        <strong style='font-size: 1.1rem;'>{direction_emoji} {signal['Direction'].upper()}</strong><br>
                            Entry: <strong>${signal['Entry']:.2f}</strong> • 
                            SL: ${signal['SL']:.2f} • 
                            TP: ${signal['TP']:.2f}<br>
                            📊 Probability: <strong>{signal['Probability']:.0f}%</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_analyze:
                        if st.button("🔍 Analyze", key=f"btn_{idx}"):
                            st.session_state.selected_signal = signal
            else:
                st.info("No signals available at the moment")
    
    with col_right:
        st.markdown("### 📊 MT4 Status")
        
        status = bridge.get_status()
        if status:
            st.success("🟢 Connected")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Balance", f"${status.get('balance', 0):.2f}")
            with col_b:
                st.metric("Equity", f"${status.get('equity', 0):.2f}")
            st.metric("Open Trades", status.get('open_trades', 0))
        else:
            st.warning("🟡 Waiting for MT4...")
        
        # Open Trades
        st.markdown("#### 📋 Open Trades")
        trades = bridge.get_open_trades()
        if trades:
            for trade in trades[:3]:
                profit = trade.get('profit', 0)
                emoji = "🟢" if profit >= 0 else "🔴"
                st.markdown(f"{emoji} **{trade.get('symbol')}** P/L: ${profit:.2f}")
        else:
            st.info("No open trades")
    
    # AI Analysis
    if 'selected_signal' in st.session_state and df_ind is not None:
        signal = st.session_state.selected_signal
        
        st.markdown("---")
        st.markdown("## 🤖 AI Deep Analysis")
        
        direction = 'long' if signal['Direction'].lower() == 'long' else 'short'
        entry = signal['Entry']
        sl = signal['SL']
        tp = signal['TP']
        
        # Generate features and predict
        features = generate_features(df_ind, entry, sl, tp, direction, 60)
        ai_prob = predict_success(model_placeholder, scaler_placeholder, features)
        
        col1_a, col2_a, col3_a, col4_a = st.columns(4)
        with col1_a:
            delta = ai_prob - signal['Probability']
            st.metric("🤖 AI Confidence", f"{ai_prob:.1f}%", f"{delta:+.1f}%")
        with col2_a:
            rr = abs(tp - entry) / abs(entry - sl)
            st.metric("⚖️ Risk/Reward", f"{rr:.2f}x")
        with col3_a:
            risk = abs(entry - sl) / entry * 100
            st.metric("📉 Risk %", f"{risk:.2f}%")
        with col4_a:
            reward = abs(tp - entry) / entry * 100
            st.metric("📈 Reward %", f"{reward:.2f}%")
        
        st.markdown("---")
        
        # MT4 Integration
        st.markdown("### 🔗 Send to MT4")
        
        col_config, col_send = st.columns([2, 1])
        
        with col_config:
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                account_balance = st.number_input(
                    "💰 Account Balance ($)",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=10000.0,
                    step=100.0
                )
            
            with col_b:
                risk_pct = st.number_input(
                    "📊 Risk per Trade (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.1
                )
            
            with col_c:
                lot_size = calculate_lot_size(account_balance, risk_pct, entry, sl)
                st.metric("📦 Lot Size", f"{lot_size:.2f}")
            
            # Trade Info
            mt4_symbol = convert_yfinance_to_mt4_symbol(symbol)
            direction_mt4 = "BUY" if direction == 'long' else "SELL"
            
            st.info(f"""
            **📋 Trade Details:**
            - Symbol: `{mt4_symbol}`
            - Direction: `{direction_mt4}`
            - Entry: `{entry:.2f}`
            - Stop Loss: `{sl:.2f}`
            - Take Profit: `{tp:.2f}`
            - Lot Size: `{lot_size:.2f}`
            - AI Confidence: `{ai_prob:.1f}%`
            - Risk Amount: `${account_balance * risk_pct / 100:.2f}`
            """)
        
        with col_send:
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            mt4_connected = bridge.is_mt4_connected()
            if st.button("🚀 SEND TO MT4", type="primary", use_container_width=True):
                if not mt4_connected:
                    st.error("❌ MT4 not connected! Start EA first.")
                else:
                    signal_data = {
                        "symbol": mt4_symbol,
                        "direction": direction_mt4,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "lot_size": lot_size,
                        "probability": signal['Probability'],
                        "ai_confidence": ai_prob,
                        "rr_ratio": rr,
                        "comment": f"AI_{mt4_symbol}_{datetime.datetime.now().strftime('%H%M')}",
                        "magic": 12345
                    }
                    
                    if bridge.send_signal(signal_data):
                        st.success("✅ Signal sent to MT4!")
                        st.balloons()
                        # Log
                        st.info(f"📝 Signal saved to: {bridge.signals_file}")
                    else:
                        st.error("❌ Failed to send signal")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🗑️ Clear Signal", use_container_width=True):
                if bridge.clear_signal():
                    st.success("✅ Signal cleared")

# Disclaimer
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #FFF8DC 0%, #FFE4B5 100%);
padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
    <p style='color: #8B4513; font-size: 0.9rem; margin: 0;
text-align: center;'>
        ⚠️ <strong>Risk Warning:</strong> Trading involves substantial risk. 
        This is educational software.
        Always test in demo first. Past performance 
        does not guarantee future results.
        © 2025 AI Trading System
    </p>
</div>
""", unsafe_allow_html=True)

# CORREZIONE APPLICATA QUI SOTTO
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh = st.button("🔄 Load Data", use_container_width=True)

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    live_toggle = st.checkbox("⚡ Live Price", value=st.session_state.live_price_active)
    st.session_state.live_price_active = live_toggle

# MT4 Connection Status
col_status1, col_status2 = st.columns([3, 1])

with col_status1:
    mt4_connected = bridge.is_mt4_connected()
    if mt4_connected:
        st.success("🟢 MT4 Connected")
    else:
        st.warning("🟡 MT4 Disconnected - Start EA in MT4")

with col_status2:
    status = bridge.get_status()
    if status:
        st.metric("💰 Balance", f"${status.get('balance', 0):.2f}")

st.markdown("---")

# Load Model
session_key = f"model_{symbol}_{interval}"
if session_key not in st.session_state or refresh:
    with st.spinner("🧠 Training AI Model..."):
        model, scaler, df_ind = train_or_load_model(symbol=symbol, interval=interval)
        if model is not None:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            st.success("✅ AI Model Ready!")
        else:
            st.error("❌ Failed to load data")

# Main Content
if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']
    
    # Get current price
    ticker = yf.Ticker(symbol)
    
    # Live price from MT4 if available
    mt4_price = bridge.get_live_price()
    
    if mt4_price and st.session_state.live_price_active:
        # Use MT4 live price
        current_price = float(mt4_price.get('bid', df_ind['Close'].iloc[-1]))
        mt4_spread = mt4_price.get('spread', 0)
        
        # Auto-refresh every second
        current_time = time.time()
        if current_time - st.session_state.last_price_update >= 1:
            st.session_state.last_price_update = current_time
            time.sleep(1)
            st.rerun()
    elif st.session_state.live_price_active:
        # Fallback to yfinance if MT4 not available
        current_time = time.time()
        if current_time - st.session_state.last_price_update >= 1:
            try:
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    st.session_state.last_price_update = current_time
                else:
                    current_price = df_ind['Close'].iloc[-1]
            except:
                current_price = df_ind['Close'].iloc[-1]
            
            time.sleep(1)
            st.rerun()
        else:
            try:
                hist = ticker.history(period="1d", interval="1m")
                current_price = float(hist['Close'].iloc[-1]) if not hist.empty else df_ind['Close'].iloc[-1]
            except:
                current_price = df_ind['Close'].iloc[-1]
        mt4_spread = 0
    else:
        # Static price
        try:
            hist = ticker.history(period="1d", interval="1h")
            current_price = float(hist['Close'].iloc[-1]) if not hist.empty else df_ind['Close'].iloc[-1]
        except:
            current_price = df_ind['Close'].iloc[-1]
        mt4_spread = 0
    
    # Gold Analysis
    if symbol == 'GC=F':
        st.markdown("## 🥇 GOLD MARKET ANALYSIS")
        
        with st.spinner("📊 Analyzing fundamentals..."):
            factors = get_gold_fundamental_factors()
            gold_analysis = analyze_gold_historical_comparison(current_price, factors)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💵 Current", f"${gold_analysis['current_price']:.2f}")
        with col2:
            change_3m = ((gold_analysis['target_3m'] - current_price) / current_price) * 100
            st.metric("📅 Target 3M", f"${gold_analysis['target_3m']:.2f}", f"{change_3m:+.1f}%")
        with col3:
            change_6m = ((gold_analysis['target_6m'] - current_price) / current_price) * 100
            st.metric("📅 Target 6M", f"${gold_analysis['target_6m']:.2f}", f"{change_6m:+.1f}%")
        with col4:
            st.metric("🎯 Confidence", f"{gold_analysis['confidence']:.0f}%")
        
        st.markdown("---")
    
    # Market Stats
    st.markdown("### 📊 Market Statistics")
    latest = df_ind.iloc[-1]
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if len(df_ind) > 1:
            price_change = ((current_price - df_ind['Close'].iloc[-2]) / df_ind['Close'].iloc[-2]) * 100
        else:
            price_change = 0
        st.metric("💵 Live Price", f"${current_price:.2f}", f"{price_change:+.2f}%")
    with col2:
        if mt4_price and st.session_state.live_price_active:
            st.metric("📊 Spread", f"{mt4_spread:.1f}")
        else:
            st.metric("📊 Spread", "N/A")
    with col3:
        rsi_color = "🟢" if 30 <= latest['RSI'] <= 70 else "🔴"
        st.metric(f"{rsi_color} RSI", f"{latest['RSI']:.1f}")
    with col4:
        st.metric("📏 ATR", f"{latest['ATR']:.2f}")
    with col5:
        trend_text = "📈 Bull" if latest['Trend'] == 1 else "📉 Bear"
        st.metric("Trend", trend_text)
    with col6:
        if mt4_price and st.session_state.live_price_active:
            st.metric("⚡ Source", "MT4")
        elif st.session_state.live_price_active:
            st.metric("⚡ Source", "LIVE")
        else:
            st.metric("⚡ Source", "STATIC")
