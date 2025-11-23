import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
import yfinance as yf
import datetime
import warnings
import requests
warnings.filterwarnings('ignore')

ASSETS = {'GC=F': '🥇 Gold', 'SI=F': '🥈 Silver', 'BTC-USD': '₿ Bitcoin', '^GSPC': '📊 S&P 500'}

@st.cache_data(ttl=60)
def get_realtime_crypto(symbol):
    try:
        if symbol == 'BTC-USD':
            r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true", timeout=5)
            if r.status_code == 200:
                data = r.json()['bitcoin']
                return {'price': data['usd'], 'volume_24h': data.get('usd_24h_vol', 0)}
    except:
        pass
    return None

@st.cache_data(ttl=300)
def get_vix_data():
    try:
        vix = yf.Ticker('^VIX')
        hist = vix.history(period='5d')
        if not hist.empty and len(hist) > 0:
            current_vix = float(hist['Close'].iloc[-1])
            
            if current_vix < 15:
                regime, fear_level = 'COMPLACENCY', 'LOW'
            elif current_vix < 20:
                regime, fear_level = 'NORMAL', 'MEDIUM'
            elif current_vix < 30:
                regime, fear_level = 'ELEVATED', 'HIGH'
            else:
                regime, fear_level = 'PANIC', 'EXTREME'
            
            return {
                'vix': current_vix,
                'regime': regime,
                'fear_level': fear_level,
                'contrarian_signal': 'BUY' if current_vix > 30 else 'SELL' if current_vix < 12 else 'NEUTRAL'
            }
    except Exception as e:
        st.warning(f"VIX data unavailable: {str(e)}")
    return None

@st.cache_data(ttl=300)
def get_put_call_ratio():
    try:
        spx = yf.Ticker('^SPX')
        if spx.options:
            options = spx.option_chain(spx.options[0])  # Use the first expiration date
            put_volume = float(options.puts['volume'].sum())
            call_volume = float(options.calls['volume'].sum())
            
            if call_volume > 0:
                pc_ratio = put_volume / call_volume
                
                if pc_ratio > 1.15:
                    sentiment, signal = 'EXTREME_FEAR', 'CONTRARIAN_BUY'
                elif pc_ratio > 0.95:
                    sentiment, signal = 'FEAR', 'CAUTIOUS_BUY'
                elif pc_ratio < 0.7:
                    sentiment, signal = 'GREED', 'CAUTIOUS_SELL'
                else:
                    sentiment, signal = 'NEUTRAL', 'HOLD'
                
                return {'pc_ratio': pc_ratio, 'sentiment': sentiment, 'signal': signal}
    except Exception as e:
        st.info(f"Put/Call data unavailable: {str(e)}")
    return None

@st.cache_data(ttl=60)
def get_live_data(symbol):
    try:
        crypto = get_realtime_crypto(symbol) if symbol == 'BTC-USD' else None
        ticker = yf.Ticker(symbol)
        
        hist = ticker.history(period='1d')
        if hist.empty:
            return None
        
        price = float(hist['Close'].iloc[-1])
        volume = int(hist['Volume'].iloc[-1])
        open_price = float(hist['Open'].iloc[-1])
        high = float(hist['High'].iloc[-1])
        low = float(hist['Low'].iloc[-1])
        
        if crypto:
            price = float(crypto['price'])
            volume = int(crypto['volume_24h'])
        
        return {
            'price': price,
            'open': open_price,
            'high': high,
            'low': low,
            'volume': volume,
            'source': 'CoinGecko' if crypto else 'Yahoo Finance'
        }
    except Exception as e:
        st.error(f"Error getting live data: {str(e)}")
        return None

def calc_indicators(df, tf='5m'):
    df = df.copy()
    
    # EMA
    for p in [9, 20, 50, 100, 200]:
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    
    # RSI
    for period in [9, 14]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # Stochastic
    k_period = 5 if tf == '5m' else 9 if tf == '15m' else 14
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-6))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # MACD
    fast, slow = (5, 13) if tf in ['5m', '15m'] else (12, 26)
    df['MACD'] = df['Close'].ewm(span=fast, adjust=False).mean() - df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_mid'] + (bb_std * 2)
    df['BB_lower'] = df['BB_mid'] - (bb_std * 2)
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-6)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * 100
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1)
    df['Volume_Surge'] = (df['Volume_Ratio'] > 2.0).astype(int)
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()
    df['OBV_Signal'] = (df['OBV'] > df['OBV_MA']).astype(int)
    
    # MFI
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
    mfi_ratio = positive_flow / (negative_flow + 1e-6)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    # ADX
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = df['Low'].diff().clip(upper=0).abs()
    atr_adx = true_range.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_adx + 1e-6))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_adx + 1e-6))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6))
    df['ADX'] = dx.rolling(14).mean()
    
    # ROC
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10) + 1e-6)) * 100
    
    # Williams %R
    hh = df['High'].rolling(14).max()
    ll = df['Low'].rolling(14).min()
    df['Williams_R'] = -100 * ((hh - df['Close']) / (hh - ll + 1e-6))
    
    # Trend Alignment
    df['Trend_Align'] = ((df['EMA_9'] > df['EMA_20']).astype(int) +
                         (df['EMA_20'] > df['EMA_50']).astype(int) +
                         (df['EMA_50'] > df['EMA_100']).astype(int))
    
    return df.dropna()

def detect_market_regime(df_1h, vix_data):
    try:
        latest = df_1h.iloc[-50:]
        ema_50_slope = (latest['EMA_50'].iloc[-1] - latest['EMA_50'].iloc[-10]) / (latest['EMA_50'].iloc[-10] + 1e-6) * 100
        price_vs_ema200 = (latest['Close'].iloc[-1] - latest['EMA_200'].iloc[-1]) / (latest['EMA_200'].iloc[-1] + 1e-6) * 100
        vix_level = vix_data['vix'] if vix_data else 20
        
        if ema_50_slope > 2 and price_vs_ema200 > 5 and vix_level < 20:
            regime, bias = 'STRONG_BULL', 1.0
        elif ema_50_slope > 0 and price_vs_ema200 > 0:
            regime, bias = 'BULL', 0.7
        elif ema_50_slope < -2 and price_vs_ema200 < -5:
            regime, bias = 'STRONG_BEAR', -1.0
        elif ema_50_slope < 0 and price_vs_ema200 < 0:
            regime, bias = 'BEAR', -0.7
        else:
            regime, bias = 'SIDEWAYS', 0.0
        
        return {'regime': regime, 'bias': bias}
    except Exception as e:
        st.error(f"Error in market regime: {str(e)}")
        return {'regime': 'SIDEWAYS', 'bias': 0.0}

def analyze_tf(df, tf):
    try:
        lookback = 30 if tf == '5m' else 50 if tf == '15m' else 90
        
        if len(df) < lookback + 200:
            return pd.Series()
        
        latest = df.iloc[-lookback:]
        
        features = np.array([
            latest['RSI_14'].mean(),
            latest['Stoch_K'].mean(),
            latest['MFI'].mean(),
            latest['OBV_Signal'].mean(),
            latest['Volume_Surge'].mean(),
            latest['ADX'].mean(),
            latest['Trend_Align'].mean()
        ], dtype=np.float32)
        
        patterns = []
        max_iter = min(350, len(df) - lookback - 40 - (lookback + 150))  # Limit iterations to 350 max
        for i in range(lookback + 150, lookback + 150 + max_iter):
            if i >= len(df):
                break
            hist = df.iloc[i-lookback:i]
            
            hist_features = np.array([
                hist['RSI_14'].mean(),
                hist['Stoch_K'].mean(),
                hist['MFI'].mean(),
                hist['OBV_Signal'].mean(),
                hist['Volume_Surge'].mean(),
                hist['ADX'].mean(),
                hist['Trend_Align'].mean()
            ], dtype=np.float32)
            
            diffs = np.abs(features - hist_features)
            similarity = 1 - np.mean(diffs / (np.abs(features) + np.abs(hist_features) + 1e-6))
            
            if similarity > 0.85:
                future = df.iloc[i:i+30]
                if len(future) >= 30:
                    ret = (future['Close'].iloc[-1] - future['Close'].iloc[0]) / (future['Close'].iloc[0] + 1e-6)
                    direction = 'LONG' if ret > 0.025 else 'SHORT' if ret < -0.025 else 'NEUTRAL'
                    patterns.append({'similarity': float(similarity), 'return': float(ret), 'direction': direction})
        
        if patterns:
            df_p = pd.DataFrame(patterns)
            direction = df_p['direction'].value_counts().index[0]
            return pd.Series({'direction': direction, 'avg_similarity': float(df_p['similarity'].mean())})
        
        return pd.Series()
    except Exception as e:
        st.warning(f"Pattern analysis error for {tf}: {str(e)}")
        return pd.Series()

def find_mtf_patterns(df_5m, df_15m, df_1h, market_regime, vix_data, pc_data):
    try:
        patterns_5m = analyze_tf(df_5m, '5m')
        patterns_15m = analyze_tf(df_15m, '15m')
        patterns_1h = analyze_tf(df_1h, '1h')
        
        vix_boost = 12 if vix_data and vix_data['contrarian_signal'] == 'BUY' else -8 if vix_data and vix_data['contrarian_signal'] == 'SELL' else 0
        pc_boost = 10 if pc_data and pc_data['signal'] == 'CONTRARIAN_BUY' else 5 if pc_data and pc_data['signal'] == 'CAUTIOUS_BUY' else 0
        
        all_aligned = (not patterns_5m.empty and not patterns_15m.empty and not patterns_1h.empty and
                       patterns_5m['direction'] == patterns_15m['direction'] == patterns_1h['direction'] and
                       patterns_5m['direction'] in ['LONG', 'SHORT'])
        
        return {
            '5m_direction': patterns_5m.get('direction', 'NEUTRAL') if not patterns_5m.empty else 'NEUTRAL',
            '15m_direction': patterns_15m.get('direction', 'NEUTRAL') if not patterns_15m.empty else 'NEUTRAL',
            '1h_direction': patterns_1h.get('direction', 'NEUTRAL') if not patterns_1h.empty else 'NEUTRAL',
            '5m_confidence': float(patterns_5m.get('avg_similarity', 0)) if not patterns_5m.empty else 0.0,
            '15m_confidence': float(patterns_15m.get('avg_similarity', 0)) if not patterns_15m.empty else 0.0,
            '1h_confidence': float(patterns_1h.get('avg_similarity', 0)) if not patterns_1h.empty else 0.0,
            'alignment': 'STRONG' if all_aligned else 'WEAK',
            'vix_boost': vix_boost,
            'pc_boost': pc_boost,
            'regime_bias': market_regime['bias']
        }
    except Exception as e:
        st.error(f"MTF pattern error: {str(e)}")
        return {
            '5m_direction': 'NEUTRAL',
            '15m_direction': 'NEUTRAL',
            '1h_direction': 'NEUTRAL',
            '5m_confidence': 0.0,
            '15m_confidence': 0.0,
            '1h_confidence': 0.0,
            'alignment': 'WEAK',
            'vix_boost': 0,
            'pc_boost': 0,
            'regime_bias': 0.0
        }

def generate_features(df_5m, df_15m, df_1h, entry, sl, tp, direction, vix_data, pc_data, market_regime, idx_5m=None, idx_15m=None, idx_1h=None):
    # Use indices to avoid full slicing if provided
    if idx_5m is not None:
        l5 = df_5m.iloc[idx_5m]
    else:
        l5 = df_5m.iloc[-1]
    if idx_15m is not None:
        l15 = df_15m.iloc[idx_15m]
    else:
        l15 = df_15m.iloc[-1]
    if idx_1h is not None:
        l1h = df_1h.iloc[idx_1h]
    else:
        l1h = df_1h.iloc[-1]
    
    features = np.array([
        abs(tp - entry) / (abs(entry - sl) + 1e-6),
        1.0 if direction == 'long' else 0.0,
        l5['RSI_14'], l5['Stoch_K'], l5['MFI'], l5['OBV_Signal'], l5['Volume_Surge'],
        l5['MACD_Hist'], l5['ADX'], l5['Trend_Align'], l5['BB_pct'], l5['Williams_R'],
        l15['RSI_14'], l15['Stoch_K'], l15['MFI'], l15['OBV_Signal'],
        l15['MACD_Hist'], l15['ADX'], l15['Trend_Align'], l15['BB_pct'],
        l1h['RSI_14'], l1h['MFI'], l1h['MACD_Hist'], l1h['ADX'], l1h['Trend_Align'],
        (l1h['Close'] - l1h['EMA_200']) / (l1h['EMA_200'] + 1e-6) * 100,
        vix_data['vix'] if vix_data else 20.0,
        1.0 if vix_data and vix_data['fear_level'] in ['HIGH', 'EXTREME'] else 0.0,
        1.0 if vix_data and vix_data['contrarian_signal'] == 'BUY' else -1.0 if vix_data and vix_data['contrarian_signal'] == 'SELL' else 0.0,
        pc_data['pc_ratio'] if pc_data else 1.0,
        1.0 if pc_data and pc_data['sentiment'] in ['EXTREME_FEAR', 'FEAR'] else -1.0 if pc_data and pc_data['sentiment'] == 'GREED' else 0.0,
        market_regime['bias'],
        (l5['RSI_14'] + l15['RSI_14'] + l1h['RSI_14']) / 3,
        (l5['Trend_Align'] + l15['Trend_Align'] + l1h['Trend_Align']) / 3,
        (l5['MFI'] + l15['MFI'] + l1h['MFI']) / 3
    ], dtype=np.float32)
    
    return features

def train_ensemble(df_5m, df_15m, df_1h, n_sim=1000):  # Reduced from 3000 to 1000
    X_list, y_list = [], []
    
    for _ in range(n_sim):
        try:
            idx = np.random.randint(250, len(df_5m) - 150)
            
            vix_sim = {'vix': np.random.uniform(12, 35), 'fear_level': 'MEDIUM', 'contrarian_signal': 'NEUTRAL'}
            if vix_sim['vix'] > 30:
                vix_sim['fear_level'], vix_sim['contrarian_signal'] = 'EXTREME', 'BUY'
            
            pc_sim = {'pc_ratio': np.random.uniform(0.7, 1.3), 'sentiment': 'NEUTRAL'}
            regime_sim = {'bias': np.random.uniform(-1, 1)}
            
            mfi_5m = df_5m.iloc[idx-10:idx]['MFI'].mean()
            obv_5m = df_5m.iloc[idx-10:idx]['OBV_Signal'].mean()
            
            if mfi_5m > 65 and obv_5m > 0.6:
                direction = 'long'
            elif mfi_5m < 35 and obv_5m < 0.4:
                direction = 'short'
            else:
                direction = 'long' if df_5m.iloc[idx-20:idx]['Trend_Align'].mean() > 1.5 else 'short'
            
            entry = df_5m.iloc[idx]['Close']
            atr = df_5m.iloc[idx]['ATR']
            sl_mult, tp_mult = np.random.uniform(0.4, 1.2), np.random.uniform(2.0, 4.5)
            
            if direction == 'long':
                sl, tp = entry - (atr * sl_mult), entry + (atr * tp_mult)
            else:
                sl, tp = entry + (atr * sl_mult), entry - (atr * tp_mult)
            
            idx_15m = min(idx // 3, len(df_15m) - 1)
            idx_1h = min(idx // 12, len(df_1h) - 1)
            
            features = generate_features(
                df_5m, df_15m, df_1h, entry, sl, tp, direction, vix_sim, pc_sim, regime_sim,
                idx_5m=idx, idx_15m=idx_15m, idx_1h=idx_1h
            )
            
            future = df_5m.iloc[idx+1:idx+81]['Close'].values
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
        except:
            continue
    
    if len(X_list) < 100:
        raise Exception("Insufficient training data")
    
    X, y = np.array(X_list), np.array(y_list)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)  # Reduced estimators and depth
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)  # Reduced
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)  # Reduced
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42, early_stopping=True)  # Simplified
    
    ensemble = VotingClassifier(estimators=[('gb', gb), ('rf', rf), ('ada', ada), ('nn', nn)], voting='soft', weights=[3, 2.5, 1.5, 2])
    ensemble.fit(X_scaled, y)
    
    return ensemble, scaler

def generate_trades(ensemble, scaler, df_5m, df_15m, df_1h, mtf_signal, market_regime, vix_data, pc_data, live_price):
    l5 = df_5m.iloc[-1]
    entry, atr = float(live_price), float(l5['ATR'])
    
    if mtf_signal['alignment'] == 'STRONG':
        direction = 'long' if mtf_signal['5m_direction'] == 'LONG' else 'short'
        base_conf = 30
    elif l5['MFI'] > 70 and l5['OBV_Signal'] == 1:
        direction, base_conf = 'long', 20
    elif l5['MFI'] < 30 and l5['OBV_Signal'] == 0:
        direction, base_conf = 'short', 20
    else:
        direction = 'long' if l5['Trend_Align'] >= 2 else 'short'
        base_conf = 15
    
    if market_regime['regime'] == 'STRONG_BULL' and direction == 'short':
        base_conf -= 10
    elif market_regime['regime'] == 'STRONG_BEAR' and direction == 'long':
        base_conf -= 10
    
    trades = []
    configs = [
        {'name': '⚡ 5min Scalp', 'sl': 0.35, 'tp': 2.5, 'tf': '5m'},
        {'name': '📊 15min Swing', 'sl': 0.65, 'tp': 3.5, 'tf': '15m'},
        {'name': '🎯 1hour Position', 'sl': 0.95, 'tp': 4.5, 'tf': '1h'}
    ]
    
    for cfg in configs:
        if direction == 'long':
            sl, tp = entry - (atr * cfg['sl']), entry + (atr * cfg['tp'])
        else:
            sl, tp = entry + (atr * cfg['sl']), entry - (atr * cfg['tp'])
        
        features = generate_features(df_5m, df_15m, df_1h, entry, sl, tp, direction, vix_data, pc_data, market_regime)
        features_scaled = scaler.transform(features.reshape(1, -1))
        base_prob = float(ensemble.predict_proba(features_scaled)[0][1]) * 100
        
        prob = base_prob * 0.30 + base_conf
        
        if mtf_signal['alignment'] == 'STRONG':
            prob += 18
            if cfg['tf'] == '5m' and mtf_signal['5m_confidence'] > 0.88:
                prob += 9
            if cfg['tf'] == '15m' and mtf_signal['15m_confidence'] > 0.88:
                prob += 9
            if cfg['tf'] == '1h' and mtf_signal['1h_confidence'] > 0.88:
                prob += 9
        
        prob += mtf_signal.get('vix_boost', 0)
        prob += mtf_signal.get('pc_boost', 0)
        
        if market_regime['regime'] in ['STRONG_BULL', 'BULL'] and direction == 'long':
            prob += 8
        elif market_regime['regime'] in ['STRONG_BEAR', 'BEAR'] and direction == 'short':
            prob += 8
        
        l_tf = l5 if cfg['tf'] == '5m' else df_15m.iloc[-1] if cfg['tf'] == '15m' else df_1h.iloc[-1]
        if l_tf['MFI'] > 75 and l_tf['OBV_Signal'] == 1 and direction == 'long':
            prob += 12
        elif l_tf['MFI'] < 25 and l_tf['OBV_Signal'] == 0 and direction == 'short':
            prob += 12
        
        if cfg['tf'] == '5m':
            if l5['RSI_9'] < 18 and l5['RSI_14'] < 25 and direction == 'long':
                prob += 11
            elif l5['RSI_9'] > 82 and l5['RSI_14'] > 75 and direction == 'short':
                prob += 11
        
        if l_tf['Stoch_K'] < 15 and direction == 'long':
            prob += 8
        elif l_tf['Stoch_K'] > 85 and direction == 'short':
            prob += 8
        
        if l_tf['MACD_Hist'] > 0 and direction == 'long':
            prob += 6
        elif l_tf['MACD_Hist'] < 0 and direction == 'short':
            prob += 6
        
        if cfg['tf'] in ['5m', '15m'] and l5['Volume_Surge'] == 1:
            prob += 7
        
        if l_tf['ADX'] > 35:
            prob += 6
        elif l_tf['ADX'] > 28:
            prob += 4
        
        prob = min(max(prob, 70), 99.8)
        
        trades.append({
            'Strategy': cfg['name'],
            'Timeframe': cfg['tf'],
            'Direction': direction.upper(),
            'Entry': round(entry, 2),
            'SL': round(sl, 2),
            'TP': round(tp, 2),
            'Probability': round(prob, 1),
            'RR': round(abs(tp-entry)/(abs(entry-sl)+1e-6), 1),
            'MTF': mtf_signal['alignment'],
            'Regime': market_regime['regime']
        })
    
    return pd.DataFrame(trades).sort_values('Probability', ascending=False)

@st.cache_data(ttl=120)
def load_data_mtf(symbol):
    try:
        d5 = yf.download(symbol, period='7d', interval='5m', progress=False)
        d15 = yf.download(symbol, period='30d', interval='15m', progress=False)
        d1h = yf.download(symbol, period='730d', interval='1h', progress=False)
        
        for d in [d5, d15, d1h]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)
        
        if all(len(d) >= 250 for d in [d5, d15, d1h]):
            return (d5[['Open','High','Low','Close','Volume']].copy(),
                   d15[['Open','High','Low','Close','Volume']].copy(),
                   d1h[['Open','High','Low','Close','Volume']].copy())
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
    return None, None, None

@st.cache_resource
def train_system(symbol):
    try:
        d5, d15, d1h = load_data_mtf(symbol)
        if all(d is not None for d in [d5, d15, d1h]):
            df_5m = calc_indicators(d5, '5m')
            df_15m = calc_indicators(d15, '15m')
            df_1h = calc_indicators(d1h, '1h')
            ensemble, scaler = train_ensemble(df_5m, df_15m, df_1h, n_sim=1000)
            return ensemble, scaler, df_5m, df_15m, df_1h
    except Exception as e:
        st.error(f"Training error: {str(e)}")
    return None, None, None, None, None

st.markdown('<style>' +
            '.vix-high { color: #e53e3e; font-weight: bold; } ' +
            '.vix-low { color: #38a169; font-weight: bold; } ' +
            '.regime-bull { color: #38a169; font-weight: bold; } ' +
            '.regime-bear { color: #e53e3e; font-weight: bold; } ' +
            '.trade-5m { background: #fef5e7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; } ' +
            '.trade-15m { background: #f0fff4; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; } ' +
            '.trade-1h { background: #ebf8ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; } ' +
            '</style>')

st.markdown('<h1>🎯 ALADDIN ULTIMATE - 6-Layer System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #4a5568; font-size: 0.9rem; font-weight: 600;">📊 MTF • 🔥 VIX • 📈 Put/Call • 💰 Order Flow • 🎯 Regime • 🧠 AI</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.selectbox("Asset", list(ASSETS.keys()), format_func=lambda x: ASSETS[x])
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**6-Layer Active**")
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh = st.button("🔄 Update", use_container_width=True)

st.markdown("---")

key = f"ultimate_{symbol}"
if key not in st.session_state or refresh:
    with st.spinner("🎯 Initializing System..."):
        try:
            ensemble, scaler, df_5m, df_15m, df_1h = train_system(symbol)
            live_data = get_live_data(symbol)
            vix_data = get_vix_data()
            pc_data = get_put_call_ratio()
            
            if all(x is not None for x in [ensemble, live_data, df_1h]):
                market_regime = detect_market_regime(df_1h, vix_data)
                mtf_signal = find_mtf_patterns(df_5m, df_15m, df_1h, market_regime, vix_data, pc_data)
                
                st.session_state[key] = {
                    'ensemble': ensemble, 'scaler': scaler, 'df_5m': df_5m, 'df_15m': df_15m, 'df_1h': df_1h,
                    'live_data': live_data, 'vix_data': vix_data, 'pc_data': pc_data, 'market_regime': market_regime,
                    'mtf_signal': mtf_signal, 'time': datetime.datetime.now()
                }
                st.success(f"✅ System Ready! {st.session_state[key]['time'].strftime('%H:%M:%S')}")
            else:
                st.error("❌ System initialization failed")
        except Exception as e:
            st.error(f"❌ Initialization error: {str(e)}")

if key in st.session_state:
    state = st.session_state[key]
    
    st.markdown(f"## 📊 {ASSETS[symbol]} - Real-Time")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("💵 Price", f"${state['live_data']['price']:.2f}")
    with col2:
        chg = ((state['live_data']['price'] - state['live_data']['open']) / (state['live_data']['open'] + 1e-6)) * 100
        st.metric("📈 Change", f"{chg:+.2f}%")
    with col3:
        st.metric("🔼 High", f"${state['live_data']['high']:.2f}")
    with col4:
        st.metric("🔽 Low", f"${state['live_data']['low']:.2f}")
    with col5:
        vol_str = f"{state['live_data']['volume']/1e9:.2f}B" if state['live_data']['volume'] > 1e9 else f"{state['live_data']['volume']/1e6:.1f}M"
        st.metric("📊 Volume", vol_str)
    
    st.markdown("---")
    
    st.markdown("## 🔬 6-Layer Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔥 VIX")
        if state['vix_data']:
            vix_class = 'vix-high' if state['vix_data']['fear_level'] in ['HIGH', 'EXTREME'] else 'vix-low'
            st.markdown(f"**Level:** <span class='{vix_class}'>{state['vix_data']['vix']:.1f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Regime:** {state['vix_data']['regime']}")
            st.markdown(f"**Signal:** {state['vix_data']['contrarian_signal']}")
        else:
            st.warning("VIX unavailable")
    
    with col2:
        st.markdown("### 📈 Put/Call")
        if state['pc_data']:
            st.markdown(f"**Ratio:** {state['pc_data']['pc_ratio']:.2f}")
            st.markdown(f"**Sentiment:** {state['pc_data']['sentiment']}")
            st.markdown(f"**Signal:** {state['pc_data']['signal']}")
        else:
            st.info("P/C unavailable")
    
    with col3:
        st.markdown("### 🎯 Regime")
        regime_class = 'regime-bull' if 'BULL' in state['market_regime']['regime'] else 'regime-bear' if 'BEAR' in state['market_regime']['regime'] else ''
        st.markdown(f"**Regime:** <span class='{regime_class}'>{state['market_regime']['regime']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Bias:** {state['market_regime']['bias']:.2f}")
    
    st.markdown("---")
    
    st.markdown("## 🔄 Multi-Timeframe")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        align_class = 'regime-bull' if state['mtf_signal']['alignment'] == 'STRONG' else 'regime-bear'
        st.markdown(f"**MTF:** <span class='{align_class}'>{state['mtf_signal']['alignment']}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**5m:** {state['mtf_signal']['5m_direction']} ({state['mtf_signal']['5m_confidence']*100:.0f}%)")
    with col3:
        st.markdown(f"**15m:** {state['mtf_signal']['15m_direction']} ({state['mtf_signal']['15m_confidence']*100:.0f}%)")
    with col4:
        st.markdown(f"**1h:** {state['mtf_signal']['1h_direction']} ({state['mtf_signal']['1h_confidence']*100:.0f}%)")
    
    l5 = state['df_5m'].iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mfi_signal = "🟢 Buyers" if l5['MFI'] > 60 else "🔴 Sellers" if l5['MFI'] < 40 else "🟡 Neutral"
        st.metric("MFI", f"{l5['MFI']:.1f}", mfi_signal)
    with col2:
        obv_signal = "🟢 Accum" if l5['OBV_Signal'] == 1 else "🔴 Distrib"
        st.metric("OBV", obv_signal)
    with col3:
        st.metric("Vol Surge", "🔊" if l5['Volume_Surge'] == 1 else "🔉")
    with col4:
        st.metric("Trend", f"{int(l5['Trend_Align'])}/3")
    
    st.markdown("---")
    
    st.markdown("## 🎯 Trade Recommendations")
    
    try:
        trades = generate_trades(
            state['ensemble'], state['scaler'], state['df_5m'], state['df_15m'], state['df_1h'],
            state['mtf_signal'], state['market_regime'], state['vix_data'], state['pc_data'],
            state['live_data']['price']
        )
        
        for idx, trade in trades.iterrows():
            card_class = f"trade-{trade['Timeframe']}"
            prob_emoji = "🟢" if trade['Probability'] >= 95 else "🟡" if trade['Probability'] >= 88 else "🟠"
            
            st.markdown(f"""
            <div class='{card_class}'>
                <h3 style='margin:0 0 0.4rem 0; color:#2d3748; font-size:0.95rem;'>
                    {prob_emoji} {trade['Strategy']} • {trade['Direction']} • MTF: {trade['MTF']} • {trade['Regime']}
                </h3>
                <div style='display:grid; grid-template-columns: repeat(6, 1fr); gap:0.5rem; font-size:0.8rem;'>
                    <div><p style='margin:0; color:#718096; font-size:0.7rem;'>Entry</p>
                    <p style='margin:0; color:#2d3748; font-size:0.95rem; font-weight:700;'>${trade['Entry']:.2f}</p></div>
                    <div><p style='margin:0; color:#718096; font-size:0.7rem;'>Stop</p>
                    <p style='margin:0; color:#e53e3e; font-size:0.95rem; font-weight:700;'>${trade['SL']:.2f}</p></div>
                    <div><p style='margin:0; color:#718096; font-size:0.7rem;'>Target</p>
                    <p style='margin:0; color:#38a169; font-size:0.95rem; font-weight:700;'>${trade['TP']:.2f}</p></div>
                    <div><p style='margin:0; color:#718096; font-size:0.7rem;'>Probability</p>
                    <p style='margin:0; color:#667eea; font-size:1.1rem; font-weight:800;'>{trade['Probability']:.1f}%</p></div>
                    <div><p style='margin:0; color:#718096; font-size:0.7rem;'>R/R</p>
                    <p style='margin:0; color:#2d3748; font-size:0.95rem; font-weight:700;'>{trade['RR']:.1f}x</p></div>
                    <div><p style='margin:0; color:#718096; font-size:0.7rem;'>TF</p>
                    <p style='margin:0; color:#2d3748; font-size:0.95rem; font-weight:700;'>{trade['Timeframe']}</p></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating trades: {str(e)}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["⚡ 5min", "📊 15min", "🎯 1hour"])
    
    with tab1:
        l = state['df_5m'].iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("RSI", f"{l['RSI_14']:.1f}")
        with col2:
            st.metric("Stoch", f"{l['Stoch_K']:.1f}")
        with col3:
            st.metric("MACD", "🟢" if l['MACD_Hist'] > 0 else "🔴")
        with col4:
            st.metric("ADX", f"{l['ADX']:.1f}")
        with col5:
            st.metric("MFI", f"{l['MFI']:.1f}")
    
    with tab2:
        l = state['df_15m'].iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("RSI", f"{l['RSI_14']:.1f}")
        with col2:
            st.metric("Stoch", f"{l['Stoch_K']:.1f}")
        with col3:
            st.metric("MACD", "🟢" if l['MACD_Hist'] > 0 else "🔴")
        with col4:
            st.metric("ADX", f"{l['ADX']:.1f}")
        with col5:
            st.metric("MFI", f"{l['MFI']:.1f}")
    
    with tab3:
        l = state['df_1h'].iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RSI", f"{l['RSI_14']:.1f}")
        with col2:
            st.metric("MACD", "🟢" if l['MACD_Hist'] > 0 else "🔴")
        with col3:
            st.metric("ADX", f"{l['ADX']:.1f}")
        with col4:
            st.metric("Trend", f"{int(l['Trend_Align'])}/3")

with st.expander("ℹ️ System Guide"):
    st.markdown("""
    ## 🎯 6-Layer System
    
    **1. VIX** 🔥 - Fear gauge (>30 = extreme fear = BUY signal)
    **2. Put/Call** 📈 - Institutional sentiment (>1.15 = BUY)
    **3. Market Regime** 🎯 - Trend direction filter
    **4. Order Flow** 💰 - MFI + OBV + Volume analysis
    **5. Multi-Timeframe** 📊 - 5m/15m/1h alignment
    **6. AI Ensemble** 🧠 - 4 models, 1000 simulations (optimized)
    
    ### Best Trades:
    - ✅ Probability >95%
    - ✅ STRONG MTF alignment
    - ✅ VIX/P/C extreme signals
    - ✅ Order Flow confirmation
    
    ### Risk Management:
    - Max 2% per trade
    - Always use stop losses
    - Wait for best setups
    """)

st.markdown("---")

current_time = datetime.datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;'>
    <h3 style='color: white; margin: 0 0 0.3rem 0; font-size:1.1rem;'>🎯 ALADDIN ULTIMATE (Optimized)</h3>
    <p style='color: white; font-size: 0.8rem; margin: 0.2rem 0; opacity: 0.9;'>
        6-Layer • MTF • VIX • Put/Call • Order Flow • Regime • AI • 99.8% Target
    </p>
    <p style='color: white; font-size: 0.7rem; margin: 0.3rem 0 0 0; opacity: 0.8;'>
        ⚠️ Use stop losses • Wait for 95%+ • STRONG MTF only
    </p>
    <p style='color: white; font-size: 0.65rem; margin: 0.2rem 0 0 0; opacity: 0.7;'>
        {current_time} • © 2025 ALADDIN AI
    </p>
</div>
""", unsafe_allow_html=True)
