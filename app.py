"""
Trading Predictor AI - MT4 Integration
Versione Finale: Supporto 1 Minuto + Prezzo Live Reale da MT4
"""

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
    """Sistema di comunicazione con MetaTrader 4"""
    
    # ⚠️ IMPORTANTE: Assicurati che questo percorso corrisponda alla cartella MQL4/Files del tuo MT4
    def __init__(self, bridge_folder= r"C:\Users\dcbat\AppData\Roaming\MetaQuotes\Terminal\B8925BF731C22E88F33C7A8D7CD3190E\MQL4\Files"):
        self.bridge_folder = Path(bridge_folder)
        self.signals_file = self.bridge_folder / "signals.json"
        self.status_file = self.bridge_folder / "status.json"
        self.trades_file = self.bridge_folder / "trades.json"
        
        try:
            self.bridge_folder.mkdir(parents=True, exist_ok=True)
        except:
            pass
    
    def send_signal(self, signal_data):
        """Invia segnale a MT4"""
        try:
            signal = {
                "timestamp": datetime.datetime.now().isoformat(),
                "symbol": str(signal_data.get("symbol", "XAUUSD")),
                "direction": str(signal_data.get("direction", "BUY")),
                "entry": float(signal_data.get("entry", 0)),
                "stop_loss": float(signal_data.get("sl", 0)),
                "take_profit": float(signal_data.get("tp", 0)),
                "lot_size": float(signal_data.get("lot_size", 0.01)),
                "probability": float(signal_data.get("probability", 0)),
                "ai_confidence": float(signal_data.get("ai_confidence", 0)),
                "risk_reward": float(signal_data.get("rr_ratio", 0)),
                "comment": str(signal_data.get("comment", "AI_Signal")),
                "magic_number": int(signal_data.get("magic", 12345)),
                "status": "PENDING"
            }
            
            with open(self.signals_file, 'w', encoding='utf-8') as f:
                json.dump(signal, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            st.error(f"❌ Errore invio: {str(e)}")
            return False
    
    def get_status(self):
        """Leggi status MT4"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def get_open_trades(self):
        """Leggi trade aperti"""
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def clear_signal(self):
        """Pulisci segnale"""
        try:
            if self.signals_file.exists():
                self.signals_file.unlink()
            return True
        except:
            return False

# ==================== HELPER FUNCTIONS ====================

def convert_to_mt4_symbol(yf_symbol):
    """Converte ticker yfinance in simbolo MT4"""
    mapping = {
        'GC=F': 'XAUUSD',
        'SI=F': 'XAGUSD',
        'EURUSD=X': 'EURUSD',
        'GBPUSD=X': 'GBPUSD',
        'USDJPY=X': 'USDJPY',
        'BTC-USD': 'BTCUSD',
        '^GSPC': 'US500',
    }
    return mapping.get(yf_symbol, yf_symbol.replace('=F', '').replace('=X', ''))

def calculate_lot_size(balance, risk_pct, sl_distance_pips, pip_value=10):
    """Calcola lot size con risk management"""
    risk_amount = balance * (risk_pct / 100)
    lot_size = risk_amount / (sl_distance_pips * pip_value)
    lot_size = round(lot_size, 2)
    return max(0.01, min(lot_size, 10.0))

# ==================== TECHNICAL INDICATORS ====================

def calculate_technical_indicators(df):
    """Calcola indicatori tecnici"""
    df = df.copy()
    
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Trend'] = df['Close'].rolling(window=20).apply(lambda x: 1 if x[-1] > x[0] else 0)
    
    df = df.dropna()
    return df

# ==================== AI MODEL ====================

def generate_features(df_ind, entry, sl, tp, direction, main_tf):
    """Genera features per predizione"""
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
    """Simula trade storici per training"""
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
    """Addestra modello"""
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
    """Predice probabilità"""
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    return prob * 100

# ==================== DATA LOADING ====================

@st.cache_data
def load_sample_data(symbol, interval='1h'):
    """Carica dati da yfinance"""
    period_map = {
        '1m': '7d',   # Timeframe 1 minuto supportato
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
        st.error(f"Errore caricamento dati: {e}")
        return None

@st.cache_resource
def train_or_load_model(symbol, interval='1h'):
    """Addestra modello"""
    data = load_sample_data(symbol, interval)
    if data is None:
        return None, None, None
    
    df_ind = calculate_technical_indicators(data)
    X, y = simulate_historical_trades(df_ind, n_trades=500)
    model, scaler = train_model(X, y)
    
    return model, scaler, df_ind

# ==================== WEB SIGNALS ====================

def get_web_signals(symbol, df_ind, current_price):
    """Genera suggerimenti trade"""
    try:
        latest = df_ind.iloc[-1]
        atr = latest['ATR']
        trend = latest['Trend']
        rsi = latest['RSI']
        
        suggestions = []
        
        # Segnale LONG
        if rsi < 70 and trend == 1:
            entry = round(current_price, 2)
            sl = round(entry - atr * 1.5, 2)
            tp = round(entry + atr * 3.0, 2)
            prob = 72 if rsi < 50 else 68
            
            suggestions.append({
                'Direction': 'LONG',
                'Entry': entry,
                'SL': sl,
                'TP': tp,
                'Probability': prob,
                'Signal': 'Trend Bullish + RSI OK'
            })
        
        # Segnale SHORT
        if rsi > 30 and trend == 0:
            entry = round(current_price, 2)
            sl = round(entry + atr * 1.5, 2)
            tp = round(entry - atr * 3.0, 2)
            prob = 72 if rsi > 50 else 68
            
            suggestions.append({
                'Direction': 'SHORT',
                'Entry': entry,
                'SL': sl,
                'TP': tp,
                'Probability': prob,
                'Signal': 'Trend Bearish + RSI OK'
            })
        
        # Segnale NEUTRAL
        if not suggestions:
            entry = round(current_price, 2)
            direction = 'LONG' if trend == 1 else 'SHORT'
            
            if direction == 'LONG':
                sl = round(entry - atr * 1.2, 2)
                tp = round(entry + atr * 2.5, 2)
            else:
                sl = round(entry + atr * 1.2, 2)
                tp = round(entry - atr * 2.5, 2)
            
            suggestions.append({
                'Direction': direction,
                'Entry': entry,
                'SL': sl,
                'TP': tp,
                'Probability': 65,
                'Signal': 'Neutral Market'
            })
        
        return suggestions
        
    except Exception as e:
        st.error(f"Errore generazione segnali: {e}")
        return []

# ==================== LIVE PRICE FETCHER (MT4 PRIORITY) ====================

def get_live_price(symbol):
    """
    Recupera prezzo live PRIMA da MT4 (preciso), 
    poi da YFinance (fallback ritardato)
    """
    # 1. Prova a leggere dal Bridge MT4 (Prezzo Reale del Broker)
    if 'mt4_bridge' in st.session_state:
        status = st.session_state.mt4_bridge.get_status()
        if status and 'live_price' in status:
            return float(status['live_price'])

    # 2. Fallback su Yahoo Finance (Se MT4 è chiuso)
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
        
    return None

# ==================== STREAMLIT UI ====================

proper_names = {
    'GC=F': 'XAU/USD (Gold)',
    'EURUSD=X': 'EUR/USD',
    'SI=F': 'XAG/USD (Silver)',
    'BTC-USD': 'BTC/USD',
    '^GSPC': 'S&P 500',
}

st.set_page_config(
    page_title="Trading AI - MT4 Bridge",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; max-width: 1600px;}
    h1 {color: #2E86AB; font-weight: 700; font-size: 2.5rem !important;}
    h2 {color: #A23B72; font-weight: 600; font-size: 1.8rem !important;}
    h3 {color: #F18F01; font-weight: 600; font-size: 1.4rem !important;}
    .stMetric {background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
               padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .stButton>button {background: linear-gradient(135deg, #2E86AB 0%, #1e5f8a 100%);
                      color: white; border: none; border-radius: 8px; padding: 0.6rem 1.5rem;
                      font-weight: 600; transition: all 0.3s;}
    .stButton>button:hover {transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Trading AI - MetaTrader 4 Bridge")
st.markdown("**Sistema di Trading Automatizzato con Intelligenza Artificiale**")

if 'live_price_active' not in st.session_state:
    st.session_state.live_price_active = False

if 'mt4_bridge' not in st.session_state:
    st.session_state.mt4_bridge = MT4Bridge()

bridge = st.session_state.mt4_bridge

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    symbol = st.selectbox("📊 Strumento", list(proper_names.keys()), format_func=lambda x: proper_names[x])

with col2:
    interval = st.selectbox("⏱️ Timeframe", ['1m', '5m', '15m', '1h'], index=3)

with col3:
    load_btn = st.button("🔄 Carica Dati", use_container_width=True)

with col4:
    if st.button("📡 " + ("STOP Live" if st.session_state.live_price_active else "START Live"), 
                 use_container_width=True,
                 type="primary" if not st.session_state.live_price_active else "secondary"):
        st.session_state.live_price_active = not st.session_state.live_price_active
        st.rerun()

st.markdown("---")

session_key = f"model_{symbol}_{interval}"

if session_key not in st.session_state or load_btn:
    with st.spinner("🧠 Caricamento AI..."):
        model, scaler, df_ind = train_or_load_model(symbol=symbol, interval=interval)
        if model is not None:
            st.session_state[session_key] = {'model': model, 'scaler': scaler, 'df_ind': df_ind}
            st.success("✅ Sistema pronto!")
        else:
            st.error("❌ Errore caricamento dati")

if session_key in st.session_state:
    state = st.session_state[session_key]
    model = state['model']
    scaler = state['scaler']
    df_ind = state['df_ind']
    
    # LIVE PRICE DISPLAY LOOP
    if st.session_state.live_price_active:
        placeholder_price = st.empty()
        
        while st.session_state.live_price_active:
            live_price = get_live_price(symbol)
            
            if live_price:
                current_price = live_price
                
                # Check source
                status = bridge.get_status()
                if status and 'live_price' in status:
                     source_text = "MT4 REAL-TIME" 
                else: 
                     source_text = "YAHOO (DELAYED)"
                
                placeholder_price.success(f"📡 **LIVE ({source_text}):** ${current_price:.2f}")
            else:
                current_price = df_ind['Close'].iloc[-1]
                placeholder_price.warning(f"⚠️ Prezzo da cache: ${current_price:.2f}")
            
            time.sleep(1)
    else:
        current_price = df_ind['Close'].iloc[-1]
        st.info(f"💵 **Prezzo Attuale:** ${current_price:.2f} (Click 'START Live' per aggiornamento real-time)")
    
    st.markdown("### 📊 Dashboard Indicatori")
    latest = df_ind.iloc[-1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("💵 Prezzo", f"${current_price:.2f}")
    with col2: st.metric(f"{'🟢' if 30 <= latest['RSI'] <= 70 else '🔴'} RSI", f"{latest['RSI']:.1f}")
    with col3: st.metric("📏 ATR", f"{latest['ATR']:.2f}")
    with col4: st.metric(f"{'📈' if latest['Trend'] == 1 else '📉'} Trend", "Bullish" if latest['Trend'] == 1 else "Bearish")
    with col5: st.metric("📊 MACD", "🟢 BUY" if latest['MACD'] > latest['MACD_signal'] else "🔴 SELL")
    
    st.markdown("---")
    
    web_signals = get_web_signals(symbol, df_ind, current_price)
    
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown("### 💡 Suggerimenti AI")
        if web_signals:
            for idx, signal in enumerate(web_signals):
                with st.container():
                    col_sig, col_btn = st.columns([5, 1])
                    with col_sig:
                        st.markdown(f"**{'🟢' if signal['Direction'] == 'LONG' else '🔴'} {signal['Direction']}** - Entry: `${signal['Entry']:.2f}` | SL: `${signal['SL']:.2f}` | TP: `${signal['TP']:.2f}` | Prob: **{signal['Probability']}%** | {signal['Signal']}")
                    with col_btn:
                        if st.button("🎯", key=f"sel_{idx}"):
                            st.session_state.selected_trade = signal
        else:
            st.info("Nessun segnale disponibile")
    
    with col_right:
        st.markdown("### 🔗 MT4 Connection")
        status = bridge.get_status()
        if status:
            st.success("🟢 **CONNESSO**")
            c1, c2 = st.columns(2)
            c1.metric("💰 Balance", f"${status.get('balance', 0):,.2f}")
            c2.metric("📈 Equity", f"${status.get('equity', 0):,.2f}")
            st.metric("📊 Margin", f"{status.get('margin_level', 0):.1f}%")
            st.caption(f"🕐 {status.get('timestamp', 'N/A')}")
        else:
            st.warning("🟡 **DISCONNESSO**")
            st.caption("Avvia EA in MT4")
        
        st.markdown("#### 📋 Trade Aperti")
        trades = bridge.get_open_trades()
        if trades:
            for trade in trades[:5]:
                profit = trade.get('profit', 0)
                st.markdown(f"{'🟢' if profit >= 0 else '🔴'} **{trade.get('symbol')}** ${profit:.2f}")
        else:
            st.info("Nessun trade")
    
    if 'selected_trade' in st.session_state:
        trade = st.session_state.selected_trade
        st.markdown("---")
        st.markdown("## 🎯 Analisi Trade Selezionato")
        
        direction = 'long' if trade['Direction'].lower() == 'long' else 'short'
        entry = trade['Entry']
        sl = trade['SL']
        tp = trade['TP']
        
        features = generate_features(df_ind, entry, sl, tp, direction, 60)
        ai_prob = predict_success(model, scaler, features)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎲 AI Confidence", f"{ai_prob:.1f}%")
        c2.metric("⚖️ Risk/Reward", f"{abs(tp - entry) / abs(entry - sl):.2f}x")
        c3.metric("📉 Risk", f"{abs(entry - sl) / entry * 100:.2f}%")
        c4.metric("📈 Reward", f"{abs(tp - entry) / entry * 100:.2f}%")
        
        st.markdown("---")
        st.markdown("### 🚀 Invia a MetaTrader 4")
        
        cc1, cc2, cc3 = st.columns(3)
        acc_bal = cc1.number_input("💰 Balance Account ($)", min_value=100.0, value=10000.0, step=100.0)
        r_pct = cc2.number_input("📊 Rischio (%)", min_value=0.5, value=2.0, step=0.5)
        
        sl_pips = abs(entry - sl) / 0.01
        lot_size = calculate_lot_size(acc_bal, r_pct, sl_pips)
        cc3.metric("📦 Lot Size", f"{lot_size:.2f}")
        
        mt4_symbol = convert_to_mt4_symbol(symbol)
        dir_mt4 = "BUY" if trade['Direction'].lower() == 'long' else "SELL"
        
        st.info(f"**Ordine:** `{mt4_symbol}` | `{dir_mt4}` | Lotti: `{lot_size}` | SL: `{sl}` | TP: `{tp}`")
        
        cs, cc = st.columns(2)
        if cs.button("🚀 INVIA ORDINE A MT4", type="primary", use_container_width=True):
            sig_data = {
                "symbol": mt4_symbol, "direction": dir_mt4, "entry": entry,
                "sl": sl, "tp": tp, "lot_size": lot_size,
                "probability": trade['Probability'], "ai_confidence": ai_prob,
                "rr_ratio": abs(tp - entry) / abs(entry - sl),
                "comment": f"AI_{mt4_symbol}", "magic": 12345
            }
            if bridge.send_signal(sig_data):
                st.success("✅ Inviato!"); st.balloons()
            else: st.error("❌ Errore")
        
        if cc.button("🗑️ Pulisci", use_container_width=True):
            bridge.clear_signal()
            st.success("✅ Cancellato")
            del st.session_state.selected_trade
            st.rerun()
else:
    st.warning("⚠️ Carica i dati per iniziare l'analisi")

st.markdown("---")
st.markdown("<center><p style='color: #64748b;'>⚠️ Disclaimer: Sistema educativo.</p></center>", unsafe_allow_html=True)
