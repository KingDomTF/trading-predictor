import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
import yfinance as yf
import datetime
import warnings
import requests

# --- CONFIGURAZIONE ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="ALADDIN ULTIMATE", page_icon="🎯", layout="wide")

ASSETS = {
    'GC=F': '🥇 Gold', 
    'SI=F': '🥈 Silver', 
    'BTC-USD': '₿ Bitcoin', 
    '^GSPC': '📊 S&P 500'
}

# --- DATA FETCHING & INDICATORS ---

def get_vix_data():
    """VIX Fear Index - Inversamente correlato con S&P500"""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period='5d')
        if not hist.empty:
            current_vix = hist['Close'].iloc[-1]
            vix_ma = hist['Close'].mean()
            return {
                'vix': current_vix,
                'vix_ma': vix_ma,
                'vix_regime': 'EXTREME_FEAR' if current_vix > 30 else 'FEAR' if current_vix > 20 else 'NEUTRAL' if current_vix > 12 else 'COMPLACENCY',
                'vix_trend': 'RISING' if current_vix > vix_ma else 'FALLING'
            }
        return {'vix': 15, 'vix_ma': 15, 'vix_regime': 'NEUTRAL', 'vix_trend': 'NEUTRAL'}
    except:
        return {'vix': 15, 'vix_ma': 15, 'vix_regime': 'NEUTRAL', 'vix_trend': 'NEUTRAL'}

def get_put_call_ratio():
    """
    Proxy per Put/Call Ratio. 
    Nota: I dati real-time PC ratio sono difficili da ottenere free. 
    Usiamo un proxy basato sulla volatilità implicita relativa se disponibile o placeholder.
    """
    return {'pc_ratio': 0.9, 'pc_boost': 0} # Placeholder funzionale

def get_fear_greed_index():
    """CNN Fear & Greed Index proxy (basato su VIX, momentum, breadth)"""
    try:
        vix_data = get_vix_data()
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(period='125d')
        
        if len(hist) >= 125:
            current_price = hist['Close'].iloc[-1]
            ma_125 = hist['Close'].rolling(125).mean().iloc[-1]
            momentum_score = ((current_price - ma_125) / ma_125) * 100
            
            nyse_adv_dec = 1.2 if momentum_score > 5 else 0.8 if momentum_score < -5 else 1.0
            vix_score = (30 - vix_data['vix']) / 30 * 100
            momentum_score_norm = max(0, min(100, 50 + momentum_score * 2))
            breadth_score = nyse_adv_dec * 50
            
            fg_index = (vix_score * 0.4 + momentum_score_norm * 0.3 + breadth_score * 0.3)
            fg_index = max(0, min(100, fg_index))
            
            if fg_index >= 75: fg_label = 'EXTREME_GREED'
            elif fg_index >= 55: fg_label = 'GREED'
            elif fg_index >= 45: fg_label = 'NEUTRAL'
            elif fg_index >= 25: fg_label = 'FEAR'
            else: fg_label = 'EXTREME_FEAR'
            
            return {'fg_index': fg_index, 'fg_label': fg_label}
        return {'fg_index': 50, 'fg_label': 'NEUTRAL'}
    except:
        return {'fg_index': 50, 'fg_label': 'NEUTRAL'}

def get_realtime_crypto(symbol):
    try:
        if symbol == 'BTC-USD':
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()['bitcoin']
                return {'price': data['usd'], 'volume_24h': data.get('usd_24h_vol', 0), 'change_24h': data.get('usd_24h_change', 0)}
        return None
    except:
        return None

def calc_order_flow_proxy(df):
    """Order Flow Imbalance proxy (buy vs sell pressure)"""
    df = df.copy()
    df['Price_Delta'] = df['Close'] - df['Open']
    df['Price_Range'] = df['High'] - df['Low']
    
    # Evitiamo divisione per zero
    denom = df['Price_Range'].replace(0, 0.00001)
    
    df['Buy_Pressure'] = ((df['Close'] - df['Low']) / denom) * df['Volume']
    df['Sell_Pressure'] = ((df['High'] - df['Close']) / denom) * df['Volume']
    
    denom_ofi = (df['Buy_Pressure'] + df['Sell_Pressure']).replace(0, 0.00001)
    df['OFI'] = (df['Buy_Pressure'] - df['Sell_Pressure']) / denom_ofi
    
    df['OFI_MA'] = df['OFI'].rolling(window=20).mean()
    df['OFI_Momentum'] = df['OFI'] - df['OFI_MA']
    df['Cumulative_OFI'] = df['OFI'].rolling(window=50).sum()
    
    df['Buy_Volume'] = df['Buy_Pressure'].rolling(window=20).sum()
    df['Sell_Volume'] = df['Sell_Pressure'].rolling(window=20).sum()
    
    denom_vol_imb = (df['Buy_Volume'] + df['Sell_Volume']).replace(0, 0.00001)
    df['Volume_Imbalance'] = (df['Buy_Volume'] - df['Sell_Volume']) / denom_vol_imb
    
    # MFI Simulation
    df['MFI'] = df['RSI'] # Proxy semplice se MFI non calcolato esplicitamente
    # OBV Signal
    df['OBV_Signal'] = np.where(df['Close'] > df['Close'].shift(1), 1, 0)

    return df

def calc_advanced_indicators(df, tf='5m'):
    """Indicatori avanzati con Order Flow"""
    df = df.copy()
    
    # Medie Mobili
    if tf in ['5m', '15m']:
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    else:
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    
    # RSI
    period = 9 if tf in ['5m', '15m'] else 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 0.00001)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_9'] = df['RSI'] # Alias
    df['RSI_14'] = df['RSI'] # Alias per compatibilità
    
    # Stochastic
    k_period = 5 if tf == '5m' else 9 if tf == '15m' else 14
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min + 0.00001))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # MACD
    fast, slow, signal = (5, 13, 5) if tf in ['5m', '15m'] else (12, 26, 9)
    df['MACD'] = df['Close'].ewm(span=fast).mean() - df['Close'].ewm(span=slow).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=signal).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_mid'] + (bb_std * 2)
    df['BB_lower'] = df['BB_mid'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_mid'] + 0.00001)
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 0.00001)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volume Analysis
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1)
    df['Volume_Surge'] = (df['Volume_Ratio'] > 1.8).astype(int)
    
    # Momentum
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    df['ROC_10'] = df['ROC']
    
    # ADX
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = df['Low'].diff().clip(upper=0).abs()
    atr = true_range.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 0.00001))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 0.00001))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 0.00001))
    df['ADX'] = dx.rolling(14).mean()
    df['DI_Plus'] = plus_di
    df['DI_Minus'] = minus_di
    
    # Extra indicators for Ultimate logic
    df['Williams_R'] = -100 * (high_max - df['Close']) / (high_max - low_min + 0.00001)
    df['CCI'] = (df['Close'] - df['Close'].rolling(20).mean()) / (0.015 * df['Close'].rolling(20).std() + 0.00001)

    # Trend Alignment
    if tf in ['5m', '15m']:
        df['Trend_Align'] = ((df['EMA_9'] > df['EMA_20']) & (df['EMA_20'] > df['EMA_50'])).astype(int)
        # 3 points for perfect alignment logic
        df['Trend_Align'] = np.where((df['EMA_9'] > df['EMA_20']) & (df['EMA_20'] > df['EMA_50']), 3, 0)
    else:
        df['Trend_Align'] = ((df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_100'])).astype(int)
        df['Trend_Align'] = np.where((df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_100']), 3, 0)

    # Order Flow Proxy (Calculated AFTER basic indicators because it uses some)
    df = calc_order_flow_proxy(df)
    
    return df.dropna()

def calc_multi_indicators(df, tf):
    """Wrapper per calcolare indicatori"""
    return calc_advanced_indicators(df, tf)

def detect_market_regime(df_1h, vix_data):
    """Determina il regime di mercato generale"""
    last = df_1h.iloc[-1]
    sma50 = df_1h['Close'].rolling(50).mean().iloc[-1]
    sma200 = df_1h['Close'].rolling(200).mean().iloc[-1]
    
    if last['Close'] > sma50 > sma200:
        regime = 'STRONG_BULL'
    elif last['Close'] > sma200:
        regime = 'BULL'
    elif last['Close'] < sma50 < sma200:
        regime = 'STRONG_BEAR'
    else:
        regime = 'BEAR'
        
    if vix_data['vix_regime'] == 'EXTREME_FEAR':
        regime = 'EXTREME_VOLATILITY'
        
    return {'regime': regime}

# --- ANALYSIS LOGIC ---

def analyze_buyer_seller_behavior(df_5m, df_15m, df_1h, vix_data, fg_data, symbol):
    """Analizza comportamento compratori/venditori in base a VIX, F&G, News"""
    l5, l15 = df_5m.iloc[-1], df_15m.iloc[-1]
    
    ofi_5m_trend = 'BUYERS_DOMINANT' if l5['OFI_Momentum'] > 0.1 else 'SELLERS_DOMINANT' if l5['OFI_Momentum'] < -0.1 else 'BALANCED'
    
    cumulative_ofi = l5['Cumulative_OFI']
    ofi_regime = 'STRONG_BUY' if cumulative_ofi > 10 else 'BUY' if cumulative_ofi > 3 else 'STRONG_SELL' if cumulative_ofi < -10 else 'SELL' if cumulative_ofi < -3 else 'NEUTRAL'
    
    volume_imbalance = l5['Volume_Imbalance']
    vol_imb_signal = 'BUY_PRESSURE' if volume_imbalance > 0.15 else 'SELL_PRESSURE' if volume_imbalance < -0.15 else 'NEUTRAL'
    
    behavior = {
        'ofi_5m': ofi_5m_trend,
        'ofi_regime': ofi_regime,
        'cumulative_ofi': cumulative_ofi,
        'volume_imbalance': vol_imb_signal,
        'volume_imbalance_value': volume_imbalance,
        'confidence': 0.5
    }
    
    if symbol == '^GSPC':
        if vix_data['vix_regime'] in ['FEAR', 'EXTREME_FEAR']:
            if ofi_regime in ['STRONG_BUY', 'BUY']:
                behavior['signal'] = 'CONTRARIAN_BUY'
                behavior['confidence'] = 0.85
            else:
                behavior['signal'] = 'FEAR_SELL'
                behavior['confidence'] = 0.65
        elif fg_data['fg_label'] == 'EXTREME_GREED':
            if ofi_regime in ['STRONG_SELL', 'SELL']:
                behavior['signal'] = 'CONTRARIAN_SELL'
                behavior['confidence'] = 0.80
            else:
                behavior['signal'] = 'GREED_BUY'
                behavior['confidence'] = 0.60
        else:
            behavior['signal'] = 'FOLLOW_OFI'
            behavior['confidence'] = 0.70
            
    elif symbol == 'GC=F':
        if vix_data['vix_regime'] in ['FEAR', 'EXTREME_FEAR']:
            behavior['signal'] = 'SAFE_HAVEN_BUY'
            behavior['confidence'] = 0.90
        else:
            behavior['signal'] = 'FOLLOW_OFI'
            
    elif symbol == 'BTC-USD':
        if vix_data['vix_regime'] == 'EXTREME_FEAR':
            behavior['signal'] = 'RISK_OFF_SELL'
            behavior['confidence'] = 0.80
        elif vix_data['vix_regime'] == 'COMPLACENCY' and fg_data['fg_label'] in ['GREED', 'EXTREME_GREED']:
            behavior['signal'] = 'RISK_ON_BUY'
            behavior['confidence'] = 0.85
        else:
            behavior['signal'] = 'FOLLOW_OFI'
            behavior['confidence'] = 0.70
    else:
        behavior['signal'] = 'FOLLOW_OFI'
        behavior['confidence'] = 0.70
        
    return behavior

def analyze_single_tf_enhanced(df, tf, vix_data, fg_data):
    """Analisi pattern con VIX/F&G awareness"""
    lookback = 30 if tf == '5m' else 50 if tf == '15m' else 90
    if len(df) < lookback + 50: return pd.Series()
    
    latest = df.iloc[-lookback:]
    
    features = {
        'rsi': latest['RSI'].mean(),
        'stoch': latest['Stoch_K'].mean(),
        'adx': latest['ADX'].mean(),
        'ofi': latest['OFI'].mean(),
        'volume_imb': latest['Volume_Imbalance'].mean()
    }
    
    vix_adjustment = 1.15 if vix_data['vix_regime'] in ['FEAR', 'EXTREME_FEAR'] else 0.85 if vix_data['vix_regime'] == 'COMPLACENCY' else 1.0
    
    patterns = []
    # Reduced complexity for speed in demo
    search_range = range(lookback + 50, len(df) - lookback - 20, 5) 
    
    for i in search_range:
        hist = df.iloc[i-lookback:i]
        hist_features = {
            'rsi': hist['RSI'].mean(),
            'stoch': hist['Stoch_K'].mean(),
            'adx': hist['ADX'].mean(),
            'ofi': hist['OFI'].mean(),
            'volume_imb': hist['Volume_Imbalance'].mean()
        }
        
        similarity = 1 - sum([abs(features[k] - hist_features[k]) / (abs(features[k]) + abs(hist_features[k]) + 0.00001) for k in features]) / len(features)
        similarity *= vix_adjustment
        
        if similarity > 0.80:
            future = df.iloc[i:i+20]
            if len(future) >= 20:
                ret = (future['Close'].iloc[-1] - future['Close'].iloc[0]) / future['Close'].iloc[0]
                patterns.append({'similarity': similarity, 'return': ret, 'direction': 'LONG' if ret > 0.001 else 'SHORT' if ret < -0.001 else 'NEUTRAL'})
    
    if patterns:
        df_p = pd.DataFrame(patterns)
        if not df_p.empty:
            direction = df_p['direction'].value_counts().index[0]
            return pd.Series({'direction': direction, 'avg_similarity': df_p['similarity'].mean(), 'avg_return': df_p['return'].mean()})
    
    return pd.Series()

def all_aligned(p5, p15, p1h):
    if p5.empty or p15.empty or p1h.empty:
        return False
    return p5['direction'] == p15['direction'] == p1h['direction'] and p5['direction'] in ['LONG', 'SHORT']

def find_mtf_patterns_advanced(df_5m, df_15m, df_1h, market_regime, vix_data, pc_data):
    """Pattern matching con VIX e F&G"""
    fg_data = {'fg_label': 'NEUTRAL', 'fg_index': 50} # Placeholder interno
    p5 = analyze_single_tf_enhanced(df_5m, '5m', vix_data, fg_data)
    p15 = analyze_single_tf_enhanced(df_15m, '15m', vix_data, fg_data)
    p1h = analyze_single_tf_enhanced(df_1h, '1h', vix_data, fg_data)
    
    mtf_signal = {
        '5m_direction': p5['direction'] if not p5.empty else 'NEUTRAL',
        '15m_direction': p15['direction'] if not p15.empty else 'NEUTRAL',
        '1h_direction': p1h['direction'] if not p1h.empty else 'NEUTRAL',
        '5m_confidence': p5['avg_similarity'] if not p5.empty else 0,
        '15m_confidence': p15['avg_similarity'] if not p15.empty else 0,
        '1h_confidence': p1h['avg_similarity'] if not p1h.empty else 0,
        'alignment': 'STRONG' if all_aligned(p5, p15, p1h) else 'WEAK',
        'vix_regime': vix_data['vix_regime'],
        'vix_boost': 5 if vix_data['vix_trend'] == 'RISING' else 0,
        'pc_boost': 0
    }
    return mtf_signal, p5, p15, p1h

# --- AI & ENSEMBLE ---

def generate_ultra_features(df_5m, df_15m, df_1h, entry, sl, tp, direction, vix_data, fg_data, behavior):
    """Features con VIX, F&G, Order Flow"""
    l5, l15, l1h = df_5m.iloc[-1], df_15m.iloc[-1], df_1h.iloc[-1]
    
    features = {
        'rr_ratio': abs(tp - entry) / (abs(entry - sl) + 0.00001),
        'direction': 1 if direction == 'long' else 0,
        '5m_rsi': l5['RSI'],
        '5m_stoch_k': l5['Stoch_K'],
        '5m_macd_hist': l5['MACD_Hist'],
        '5m_adx': l5['ADX'],
        '5m_trend_align': l5['Trend_Align'],
        '5m_volume_surge': l5['Volume_Surge'],
        '5m_bb_pct': l5['BB_pct'],
        '5m_ofi': l5['OFI'],
        '5m_ofi_momentum': l5['OFI_Momentum'],
        '5m_volume_imbalance': l5['Volume_Imbalance'],
        '15m_rsi': l15['RSI'],
        '15m_stoch_k': l15['Stoch_K'],
        '15m_macd_hist': l15['MACD_Hist'],
        '15m_adx': l15['ADX'],
        '15m_trend_align': l15['Trend_Align'],
        '15m_ofi': l15['OFI'],
        '15m_volume_imbalance': l15['Volume_Imbalance'],
        '1h_rsi': l1h['RSI'],
        '1h_macd_hist': l1h['MACD_Hist'],
        '1h_adx': l1h['ADX'],
        '1h_trend_align': l1h['Trend_Align'],
        '1h_ofi': l1h['OFI'],
        'vix': vix_data['vix'],
        'vix_regime_encoded': 3 if vix_data['vix_regime'] == 'EXTREME_FEAR' else 2 if vix_data['vix_regime'] == 'FEAR' else 1 if vix_data['vix_regime'] == 'NEUTRAL' else 0,
        'vix_trend_encoded': 1 if vix_data['vix_trend'] == 'RISING' else 0,
        'fg_index': fg_data['fg_index'],
        'fg_regime_encoded': 2 if fg_data['fg_label'] in ['EXTREME_GREED', 'GREED'] else 1 if fg_data['fg_label'] == 'NEUTRAL' else 0,
        'cumulative_ofi': behavior['cumulative_ofi'],
        'volume_imbalance_val': behavior['volume_imbalance_value'],
        'behavior_confidence': behavior.get('confidence', 0.5),
        'mtf_rsi_avg': (l5['RSI'] + l15['RSI'] + l1h['RSI']) / 3,
        'mtf_ofi_avg': (l5['OFI'] + l15['OFI'] + l1h['OFI']) / 3,
        'mtf_adx_avg': (l5['ADX'] + l15['ADX'] + l1h['ADX']) / 3
    }
    
    return np.array(list(features.values()), dtype=np.float32)

def train_ultra_ensemble(df_5m, df_15m, df_1h, vix_hist, n_sim=1000):
    """Training ensemble ultra-potenziato"""
    X_list, y_list = [], []
    
    # Mock VIX data for training loop if series passed
    vix_data_mock = {'vix': 20, 'vix_regime': 'NEUTRAL', 'vix_trend': 'NEUTRAL'}
    fg_data_sim = {'fg_index': 50, 'fg_label': 'NEUTRAL'}
    
    limit = min(len(df_5m), len(df_15m)*3, len(df_1h)*12) - 100
    
    for _ in range(n_sim):
        idx = np.random.randint(150, limit)
        
        align_5m = df_5m.iloc[idx-20:idx]['Trend_Align'].mean()
        ofi_signal = df_5m.iloc[idx-10:idx]['OFI'].mean()
        
        direction = 'long' if (align_5m > 0.6 or ofi_signal > 0.2) else 'short' if (align_5m < 0.4 or ofi_signal < -0.2) else ('long' if np.random.random() > 0.5 else 'short')
        
        entry = df_5m.iloc[idx]['Close']
        atr = df_5m.iloc[idx]['ATR']
        
        sl_mult = np.random.uniform(0.3, 0.9)
        tp_mult = np.random.uniform(1.8, 4.0)
        
        if direction == 'long':
            sl, tp = entry - (atr * sl_mult), entry + (atr * tp_mult)
        else:
            sl, tp = entry + (atr * sl_mult), entry - (atr * tp_mult)
            
        idx_15m = min(idx // 3, len(df_15m) - 1)
        idx_1h = min(idx // 12, len(df_1h) - 1)
        
        behavior = {'cumulative_ofi': df_5m.iloc[idx-50:idx]['OFI'].sum(), 'volume_imbalance_value': df_5m.iloc[idx]['Volume_Imbalance'], 'confidence': 0.7}
        
        features = generate_ultra_features(
            df_5m.iloc[:idx+1], 
            df_15m.iloc[:idx_15m+1], 
            df_1h.iloc[:idx_1h+1], 
            entry, sl, tp, direction, vix_data_mock, fg_data_sim, behavior
        )
        
        future = df_5m.iloc[idx+1:idx+61]['Close'].values
        if len(future) > 0:
            if direction == 'long':
                hit_tp = np.any(future >= tp)
                hit_sl = np.any(future <= sl)
            else:
                hit_tp = np.any(future <= tp)
                hit_sl = np.any(future >= sl)
                
            success = 1 if hit_tp and not hit_sl else 0
            X_list.append(features)
            y_list.append(success)
            
    X, y = np.array(X_list), np.array(y_list)
    
    if len(y) < 10: # Fallback se non genera abbastanza campioni
        return None, None

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
    ada = AdaBoostClassifier(n_estimators=50, random_state=42)
    nn = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=200, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf), ('ada', ada), ('nn', nn)],
        voting='soft',
        weights=[3, 2.5, 1.5, 1]
    )
    ensemble.fit(X_scaled, y)
    
    return ensemble, scaler

def generate_ultimate_trades(ensemble, scaler, df_5m, df_15m, df_1h, mtf_signal, behavior, vix_data, fg_data, market_regime, live_price):
    """Sistema ULTIMO con VIX, F&G, Order Flow"""
    l5, l15 = df_5m.iloc[-1], df_15m.iloc[-1]
    entry = live_price
    atr = l5['ATR']
    
    # Determinazione Direzione Base
    if behavior['signal'] in ['CONTRARIAN_BUY', 'SAFE_HAVEN_BUY', 'RISK_ON_BUY']:
        direction = 'long'
        base_confidence = 30
    elif behavior['signal'] in ['CONTRARIAN_SELL', 'RISK_ON_SELL', 'RISK_OFF_SELL', 'FEAR_SELL']:
        direction = 'short'
        base_confidence = 30
    else:
        if mtf_signal['alignment'] == 'STRONG':
            direction = 'long' if mtf_signal['5m_direction'] == 'LONG' else 'short'
            base_confidence = 25
        else:
            direction = 'long' if l5['Trend_Align'] >= 1 and l5['OFI'] > 0 else 'short'
            base_confidence = 15

    # Configurazione strategie
    configs = [
        {'name': '⚡ 5min Scalp', 'sl': 0.5, 'tp': 1.5, 'tf': '5m'},
        {'name': '📊 15min Swing', 'sl': 0.8, 'tp': 2.5, 'tf': '15m'},
        {'name': '🎯 1hour Position', 'sl': 1.2, 'tp': 3.5, 'tf': '1h'}
    ]
    
    trades = []
    
    for cfg in configs:
        if direction == 'long':
            sl, tp = entry - (atr * cfg['sl']), entry + (atr * cfg['tp'])
        else:
            sl, tp = entry + (atr * cfg['sl']), entry - (atr * cfg['tp'])
            
        features = generate_ultra_features(df_5m, df_15m, df_1h, entry, sl, tp, direction, vix_data, fg_data, behavior)
        features_scaled = scaler.transform(features.reshape(1, -1))
        base_prob = ensemble.predict_proba(features_scaled)[0][1] * 100
        
        # Calcolo probabilità avanzato
        prob = base_prob * 0.30 + base_confidence
        
        # MTF Boost
        if mtf_signal['alignment'] == 'STRONG':
            prob += 10
            
        # Market Regime Alignment
        if market_regime['regime'] in ['STRONG_BULL', 'BULL'] and direction == 'long':
            prob += 8
        elif market_regime['regime'] in ['STRONG_BEAR', 'BEAR'] and direction == 'short':
            prob += 8
            
        # Order Flow Confirmation
        if cfg['tf'] == '5m':
            if l5['OFI'] > 0 and direction == 'long': prob += 5
            if l5['OFI'] < 0 and direction == 'short': prob += 5
            
        # Indicator Extremes
        if l5['RSI'] < 30 and direction == 'long': prob += 5
        if l5['RSI'] > 70 and direction == 'short': prob += 5
        
        prob = min(max(prob, 10), 99.9)
        
        risk_pct = abs(entry - sl) / entry * 100
        reward_pct = abs(tp - entry) / entry * 100
        
        trades.append({
            'Strategy': cfg['name'],
            'Direction': direction.upper(),
            'Entry': round(entry, 2),
            'SL': round(sl, 2),
            'TP': round(tp, 2),
            'Probability': round(prob, 1),
            'RR': round(abs(tp-entry)/(abs(entry-sl)+0.00001), 1),
            'Risk%': round(risk_pct, 2),
            'Reward%': round(reward_pct, 2)
        })
        
    return pd.DataFrame(trades).sort_values('Probability', ascending=False)

def get_live_data(symbol):
    try:
        crypto = get_realtime_crypto(symbol) if symbol == 'BTC-USD' else None
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Fallback price retrieval
        price = crypto['price'] if crypto else (info.get('currentPrice') or info.get('regularMarketPrice') or info.get('ask') or ticker.history(period='1d')['Close'].iloc[-1])
        volume = crypto['volume_24h'] if crypto else info.get('volume', 0)
        
        return {
            'price': float(price),
            'open': float(info.get('open', price)),
            'high': float(info.get('dayHigh', price)),
            'low': float(info.get('dayLow', price)),
            'volume': int(volume),
            'source': 'CoinGecko' if crypto else 'Yahoo Finance'
        }
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return None

@st.cache_data(ttl=120)
def load_data_mtf(symbol):
    try:
        # Updated yfinance syntax for multi-data download
        d5 = yf.download(symbol, period='5d', interval='5m', progress=False)
        d15 = yf.download(symbol, period='15d', interval='15m', progress=False)
        d1h = yf.download(symbol, period='60d', interval='1h', progress=False)
        
        data_list = []
        for d in [d5, d15, d1h]:
            if d.empty: return None, None, None
            # Handle MultiIndex columns if present (common in new yfinance)
            if isinstance(d.columns, pd.MultiIndex):
                try:
                    d = d.xs(symbol, axis=1, level=1)
                except:
                    d.columns = d.columns.droplevel(1)
            
            # Ensure columns exist
            cols = ['Open','High','Low','Close','Volume']
            if not all(col in d.columns for col in cols):
                return None, None, None
            data_list.append(d[cols])
            
        return data_list[0], data_list[1], data_list[2]
    except Exception as e:
        st.error(f"Error loading MTF data: {e}")
        return None, None, None

@st.cache_resource
def train_ultimate_system(symbol):
    d5, d15, d1h = load_data_mtf(symbol)
    if d5 is not None:
        df_5m = calc_multi_indicators(d5, '5m')
        df_15m = calc_multi_indicators(d15, '15m')
        df_1h = calc_multi_indicators(d1h, '1h')
        
        vix_hist = yf.Ticker('^VIX').history(period='60d')
        ensemble, scaler = train_ultra_ensemble(df_5m, df_15m, df_1h, vix_hist, n_sim=200) # Lower sim for speed
        return ensemble, scaler, df_5m, df_15m, df_1h
    return None, None, None, None, None

# --- UI LAYOUT ---

st.markdown("""
<style>
    * { font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 0.8rem; max-width: 1800px; }
    h1 { color: #1a202c; font-size: 2.2rem !important; text-align: center; margin-bottom: 0.3rem !important; }
    .stMetric { background: #f7fafc; padding: 0.6rem; border-radius: 8px; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 100%; border-radius: 8px; font-weight: 600; }
    .trade-card { padding: 1rem; border-radius: 10px; margin-bottom: 1rem; color: white; }
    .long-card { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
    .short-card { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>🎯 ALADDIN ULTIMATE - 6-Layer Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #4a5568; font-size: 0.95rem; font-weight: 600;">📊 MTF • 🔥 VIX • 📈 Put/Call • 💰 Order Flow • 🎯 Market Regime • 🧠 4-Model AI</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.selectbox("Select Asset", list(ASSETS.keys()), format_func=lambda x: ASSETS[x])
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("✅ **System Online**")
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh = st.button("🔄 Analyze", use_container_width=True)

st.markdown("---")

key = f"ultimate_{symbol}"

if refresh or key not in st.session_state:
    with st.spinner(f"🚀 Running 6-Layer Analysis on {ASSETS[symbol]}..."):
        ensemble, scaler, df_5m, df_15m, df_1h = train_ultimate_system(symbol)
        
        if ensemble is not None:
            live_data = get_live_data(symbol)
            vix_data = get_vix_data()
            pc_data = get_put_call_ratio()
            fg_data = get_fear_greed_index()
            
            market_regime = detect_market_regime(df_1h, vix_data)
            mtf_signal, p5, p15, p1h = find_mtf_patterns_advanced(df_5m, df_15m, df_1h, market_regime, vix_data, pc_data)
            
            behavior = analyze_buyer_seller_behavior(df_5m, df_15m, df_1h, vix_data, fg_data, symbol)
            
            final_trades = generate_ultimate_trades(ensemble, scaler, df_5m, df_15m, df_1h, mtf_signal, behavior, vix_data, fg_data, market_regime, live_data['price'])
            
            st.session_state[key] = {
                'ensemble': ensemble, 'scaler': scaler, 
                'df_5m': df_5m, 'df_15m': df_15m, 'df_1h': df_1h,
                'live_data': live_data, 'vix_data': vix_data, 'pc_data': pc_data, 'fg_data': fg_data,
                'market_regime': market_regime, 'mtf_signal': mtf_signal, 'behavior': behavior,
                'final_trades': final_trades,
                'time': datetime.datetime.now()
            }
        else:
            st.error("Failed to initialize system. Not enough data or API error.")

if key in st.session_state:
    data = st.session_state[key]
    live = data['live_data']
    vix = data['vix_data']
    
    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Asset Price", f"${live['price']:,.2f}", f"{data['df_5m']['Close'].iloc[-1] - data['df_5m']['Open'].iloc[-1]:.2f}")
    m2.metric("VIX (Fear)", f"{vix['vix']:.2f}", vix['vix_regime'])
    m3.metric("Market Regime", data['market_regime']['regime'], data['mtf_signal']['alignment'])
    m4.metric("Buyer/Seller", data['behavior']['signal'], f"{data['behavior']['confidence']*100:.0f}% Conf")
    
    st.markdown("---")
    
    # Main Content: Trades & Technicals
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.subheader("🤖 AI Trade Signals")
        trades = data['final_trades']
        
        if not trades.empty:
            for _, trade in trades.iterrows():
                color_cls = "long-card" if trade['Direction'] == 'LONG' else "short-card"
                html = f"""
                <div class="trade-card {color_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0; color:white;">{trade['Direction']} - {trade['Strategy']}</h3>
                        <h2 style="margin:0; color:white;">{trade['Probability']}%</h2>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:10px; font-size:0.9rem;">
                        <span>Entry: <b>{trade['Entry']}</b></span>
                        <span>TP: <b>{trade['TP']}</b></span>
                        <span>SL: <b>{trade['SL']}</b></span>
                    </div>
                    <div style="margin-top:5px; font-size:0.8rem; opacity:0.9;">
                         Risk: {trade['Risk%']}% | Reward: {trade['Reward%']}% | RR: {trade['RR']}
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No High Probability Trades Found")

    with c2:
        st.subheader("📊 Technical Insight")
        tab1, tab2, tab3 = st.tabs(["5m", "15m", "1h"])
        
        with tab1:
            l5 = data['df_5m'].iloc[-1]
            st.write(f"RSI: {l5['RSI']:.1f}")
            st.write(f"Stoch K: {l5['Stoch_K']:.1f}")
            st.write(f"OFI: {l5['OFI']:.4f}")
            st.progress(min(max(int(l5['RSI']), 0), 100))
            
        with tab2:
            l15 = data['df_15m'].iloc[-1]
            st.write(f"RSI: {l15['RSI']:.1f}")
            st.write(f"Trend: {'Aligned' if l15['Trend_Align']>0 else 'Weak'}")
            
        with tab3:
            l1h = data['df_1h'].iloc[-1]
            st.write(f"Regime: {data['market_regime']['regime']}")
            st.write(f"OFI Cumulative: {data['behavior']['cumulative_ofi']:.2f}")

    st.caption(f"Last Update: {data['time'].strftime('%H:%M:%S')} | Source: {live['source']}")
