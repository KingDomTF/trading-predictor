import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== QUANTUM CORE ENGINE ====================
class QuantumEngine:
    """Core analytics engine - Oracle/Aladdin inspired"""
    
    ASSET_PROFILES = {
        'BTC-USD': {'vol': 1.8, 'mom': 0.85, 'liq': 0.9, 'class': 'crypto'},
        'ETH-USD': {'vol': 1.9, 'mom': 0.85, 'liq': 0.85, 'class': 'crypto'},
        'GC=F': {'vol': 0.6, 'mom': 0.5, 'liq': 0.95, 'class': 'commodity'},
        'SI=F': {'vol': 0.9, 'mom': 0.6, 'liq': 0.85, 'class': 'commodity'},
        '^GSPC': {'vol': 0.5, 'mom': 0.7, 'liq': 1.0, 'class': 'equity'},
        'SPY': {'vol': 0.5, 'mom': 0.7, 'liq': 1.0, 'class': 'equity'},
        'QQQ': {'vol': 0.7, 'mom': 0.8, 'liq': 1.0, 'class': 'equity'},
        'NVDA': {'vol': 1.2, 'mom': 0.9, 'liq': 0.95, 'class': 'tech'},
        'TSLA': {'vol': 1.5, 'mom': 0.9, 'liq': 0.9, 'class': 'tech'},
    }
    
    @staticmethod
    def compute_indicators(df, symbol):
        """Ultra-efficient indicator computation"""
        d = df.copy()
        profile = QuantumEngine.ASSET_PROFILES.get(symbol, {'vol': 1.0, 'mom': 0.6, 'liq': 0.8, 'class': 'other'})
        
        # Vectorized EMAs
        for span in [9, 21, 50, 200]:
            d[f'ema{span}'] = d['Close'].ewm(span=span, adjust=False).mean()
        
        # RSI optimized
        delta = d['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        d['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = d['Close'].ewm(span=12).mean()
        exp2 = d['Close'].ewm(span=26).mean()
        d['macd'] = exp1 - exp2
        d['macd_sig'] = d['macd'].ewm(span=9).mean()
        d['macd_hist'] = d['macd'] - d['macd_sig']
        
        # Bollinger
        d['bb_mid'] = d['Close'].rolling(20).mean()
        bb_std = d['Close'].rolling(20).std()
        d['bb_up'] = d['bb_mid'] + 2 * bb_std
        d['bb_low'] = d['bb_mid'] - 2 * bb_std
        d['bb_width'] = (d['bb_up'] - d['bb_low']) / d['bb_mid']
        
        # ATR normalized
        hl = d['High'] - d['Low']
        hc = (d['High'] - d['Close'].shift()).abs()
        lc = (d['Low'] - d['Close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        d['atr'] = tr.rolling(14).mean()
        d['atr_pct'] = (d['atr'] / d['Close'] * 100) * profile['vol']
        
        # Volume intelligence
        d['vol_ma'] = d['Volume'].rolling(20).mean()
        d['vol_ratio'] = d['Volume'] / (d['vol_ma'] + 1e-10)
        d['obv'] = (np.sign(d['Close'].diff()) * d['Volume']).fillna(0).cumsum()
        
        # ADX for trend strength
        plus_dm = d['High'].diff().clip(lower=0)
        minus_dm = (-d['Low'].diff()).clip(lower=0)
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
        d['adx'] = dx.rolling(14).mean()
        
        # Momentum suite
        d['mom'] = d['Close'].pct_change(10) * 100
        d['roc'] = d['Close'].pct_change(20) * 100
        
        # Trend composite score
        d['trend'] = (
            (d['Close'] > d['ema9']).astype(int) * 0.4 +
            (d['Close'] > d['ema21']).astype(int) * 0.3 +
            (d['Close'] > d['ema50']).astype(int) * 0.2 +
            (d['macd'] > d['macd_sig']).astype(int) * 0.1
        )
        
        # Regime detection
        d['regime'] = np.where(
            (d['adx'] > 25) & (d['trend'] > 0.6), 2,  # Strong bull
            np.where((d['adx'] > 25) & (d['trend'] < 0.4), -2,  # Strong bear
                     np.where(d['trend'] > 0.6, 1,  # Weak bull
                              np.where(d['trend'] < 0.4, -1, 0)))  # Weak bear / neutral
        )
        
        return d.dropna()
    
    @staticmethod
    def generate_features(df, entry, sl, tp, direction, symbol):
        """Feature vector for ML"""
        row = df.iloc[-1]
        profile = QuantumEngine.ASSET_PROFILES.get(symbol, {'vol': 1.0, 'mom': 0.6})
        
        rr = abs(tp - entry) / (abs(entry - sl) + 1e-10)
        risk_pct = abs(entry - sl) / entry * 100
        reward_pct = abs(tp - entry) / entry * 100
        
        return np.array([
            risk_pct, reward_pct, rr,
            1 if direction == 'long' else 0,
            row['rsi'], row['rsi'] > 70, row['rsi'] < 30,
            row['macd'], row['macd_sig'], row['macd_hist'],
            row['atr_pct'], row['bb_width'],
            (row['ema9'] - row['ema21']) / row['Close'] * 100,
            (row['ema21'] - row['ema50']) / row['Close'] * 100,
            row['trend'], row['adx'], row['regime'],
            row['vol_ratio'], row['mom'], row['roc'],
            profile['vol'], profile['mom']
        ], dtype=np.float32)
    
    @staticmethod
    def simulate_trades(df, symbol, n=1000):
        """Generate training dataset"""
        X, y = [], []
        profile = QuantumEngine.ASSET_PROFILES.get(symbol, {'vol': 1.0})
        
        for _ in range(n):
            try:
                idx = np.random.randint(100, len(df) - 100)
                row = df.iloc[idx]
                
                # Smart direction selection
                signal = int(row['rsi'] < 40) - int(row['rsi'] > 60) + int(row['macd'] > row['macd_sig']) - int(row['macd'] < row['macd_sig'])
                direction = 'long' if signal + np.random.randn() * 0.5 > 0 else 'short'
                
                entry = row['Close']
                atr = row['atr']
                
                sl_mult = np.random.uniform(0.8, 2.0) * profile['vol']
                tp_mult = np.random.uniform(1.5, 4.0) * profile['vol']
                
                sl = entry - atr * sl_mult if direction == 'long' else entry + atr * sl_mult
                tp = entry + atr * tp_mult if direction == 'long' else entry - atr * tp_mult
                
                features = QuantumEngine.generate_features(df.iloc[:idx+1], entry, sl, tp, direction, symbol)
                
                # Outcome simulation
                future = df.iloc[idx+1:idx+101]['Close'].values
                if len(future) > 0:
                    if direction == 'long':
                        hit_tp = np.any(future >= tp)
                        hit_sl = np.any(future <= sl)
                        tp_idx = np.where(future >= tp)[0][0] if hit_tp else 999
                        sl_idx = np.where(future <= sl)[0][0] if hit_sl else 999
                    else:
                        hit_tp = np.any(future <= tp)
                        hit_sl = np.any(future >= sl)
                        tp_idx = np.where(future <= tp)[0][0] if hit_tp else 999
                        sl_idx = np.where(future >= sl)[0][0] if hit_sl else 999
                    
                    success = 1 if tp_idx < sl_idx else 0
                    X.append(features)
                    y.append(success)
            except Exception as e:
                continue  # Skip problematic iterations
        
        if len(X) == 0:
            raise ValueError("No valid simulations generated. Check data quality.")
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def train_ensemble(X, y):
        """Train dual-model ensemble"""
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=10, learning_rate=0.08,
            subsample=0.8, random_state=42
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        rf_scores, gb_scores = [], []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            rf.fit(X_scaled[train_idx], y[train_idx])
            gb.fit(X_scaled[train_idx], y[train_idx])
            rf_scores.append(rf.score(X_scaled[val_idx], y[val_idx]))
            gb_scores.append(gb.score(X_scaled[val_idx], y[val_idx]))
        
        # Final fit
        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)
        
        return {
            'rf': rf, 'gb': gb,
            'rf_acc': np.mean(rf_scores),
            'gb_acc': np.mean(gb_scores),
            'ensemble_acc': np.mean(rf_scores) * 0.55 + np.mean(gb_scores) * 0.45
        }, scaler
    
    @staticmethod
    def predict(models, scaler, features):
        """Ensemble prediction"""
        X_scaled = scaler.transform(features.reshape(1, -1))
        rf_prob = models['rf'].predict_proba(X_scaled)[0][1]
        gb_prob = models['gb'].predict_proba(X_scaled)[0][1]
        return (rf_prob * 0.55 + gb_prob * 0.45) * 100

# ==================== RISK ANALYTICS ====================
class RiskEngine:
    """Risk management module"""
    
    @staticmethod
    def calculate_var(returns, confidence=0.95):
        """Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def sharpe_ratio(returns, risk_free=0.02):
        """Annualized Sharpe"""
        excess = returns - risk_free / 252
        return np.sqrt(252) * excess.mean() / (returns.std() + 1e-10)
    
    @staticmethod
    def max_drawdown(prices):
        """Maximum drawdown"""
        cummax = np.maximum.accumulate(prices)
        dd = (prices - cummax) / cummax
        return dd.min()
    
    @staticmethod
    def kelly_criterion(win_rate, avg_win, avg_loss):
        """Optimal position sizing"""
        if avg_loss == 0:
            return 0
        return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    
    @staticmethod
    def position_score(prob, rr, volatility):
        """Position quality score 0-100"""
        base = (prob - 50) * 2  # 50-100 prob -> 0-100 score
        rr_bonus = min(20, rr * 5)
        vol_penalty = max(-20, -volatility * 2)
        return max(0, min(100, base + rr_bonus + vol_penalty))

# ==================== MARKET INTELLIGENCE ====================
class MarketIntel:
    """Real-time market data and analysis"""
    
    @staticmethod
    def fetch_data(symbol, interval, period):
        """Optimized data retrieval"""
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        except:
            return None
    
    @staticmethod
    def get_signals(symbol, df):
        """Generate trading signals"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            if hist.empty:
                return []
            
            current = hist['Close'].iloc[-1]
            latest = df.iloc[-1]
            atr = latest['atr']
            
            # News sentiment
            news = ticker.news
            news_text = ' | '.join([n.get('title', '') for n in news[:5] if isinstance(n, dict)]) if news else ''
            sentiment_score = MarketIntel._sentiment_score(news_text, symbol)
            
            # Signal generation
            signals = []
            
            # Long setup
            long_score = 0
            if latest['rsi'] < 35: long_score += 25
            elif latest['rsi'] < 45: long_score += 15
            if latest['regime'] >= 1: long_score += 20
            if latest['macd'] > latest['macd_sig']: long_score += 15
            if latest['adx'] > 25: long_score += 15
            if sentiment_score > 0: long_score += 10
            
            long_prob = min(90, max(50, 50 + long_score * 0.6))
            
            signals.append({
                'dir': 'LONG',
                'entry': round(current, 2),
                'sl': round(current - atr * 1.3, 2),
                'tp': round(current + atr * 2.8, 2),
                'prob': long_prob,
                'score': long_score,
                'sentiment': 'Positive' if sentiment_score > 1 else 'Neutral' if sentiment_score >= -1 else 'Negative'
            })
            
            # Short setup
            short_score = 0
            if latest['rsi'] > 65: short_score += 25
            elif latest['rsi'] > 55: short_score += 15
            if latest['regime'] <= -1: short_score += 20
            if latest['macd'] < latest['macd_sig']: short_score += 15
            if latest['adx'] > 25: short_score += 15
            if sentiment_score < 0: short_score += 10
            
            short_prob = min(90, max(50, 50 + short_score * 0.6))
            
            signals.append({
                'dir': 'SHORT',
                'entry': round(current, 2),
                'sl': round(current + atr * 1.3, 2),
                'tp': round(current - atr * 2.8, 2),
                'prob': short_prob,
                'score': short_score,
                'sentiment': 'Positive' if sentiment_score > 1 else 'Neutral' if sentiment_score >= -1 else 'Negative'
            })
            
            return sorted(signals, key=lambda x: x['prob'], reverse=True)
        except:
            return []
    
    @staticmethod
    def _sentiment_score(text, symbol):
        """Advanced sentiment scoring"""
        pos = {'rally': 2, 'surge': 2, 'bullish': 2, 'soar': 2, 'breakout': 2, 
               'gain': 1, 'rise': 1, 'up': 1, 'strong': 1, 'positive': 1}
        neg = {'crash': 2, 'plunge': 2, 'bearish': 2, 'tumble': 2, 'collapse': 2,
               'loss': 1, 'fall': 1, 'down': 1, 'weak': 1, 'negative': 1}
        
        text = text.lower()
        score = sum(v for k, v in pos.items() if k in text) - sum(v for k, v in neg.items() if k in text)
        
        if 'crypto' in symbol.lower() or 'btc' in symbol.lower():
            if 'adoption' in text: score += 1
            if 'regulation' in text or 'ban' in text: score -= 1
        
        return score

# ==================== PORTFOLIO UNIVERSE ====================
QUANTUM_UNIVERSE = [
    {'name': 'Gold', 'ticker': 'GC=F', 'score': 94, 'class': 'Safe Haven', 
     'thesis': 'Fed pivot catalyst, CB buying, geopolitical hedge', 'risk': 'Low', 'horizon': '6-18M'},
    {'name': 'Bitcoin', 'ticker': 'BTC-USD', 'score': 89, 'class': 'Digital Asset',
     'thesis': 'ETF inflows sustained, halving supply shock, institutional adoption', 'risk': 'High', 'horizon': '12-36M'},
    {'name': 'Nvidia', 'ticker': 'NVDA', 'score': 92, 'class': 'Semiconductor',
     'thesis': 'AI infrastructure buildout, data center dominance, 80% GPU market share', 'risk': 'Medium', 'horizon': '12-24M'},
    {'name': 'Microsoft', 'ticker': 'MSFT', 'score': 88, 'class': 'Mega-Cap Tech',
     'thesis': 'Azure AI growth, enterprise moat, Copilot monetization', 'risk': 'Low', 'horizon': '12-36M'},
    {'name': 'Silver', 'ticker': 'SI=F', 'score': 86, 'class': 'Industrial Metal',
     'thesis': 'Solar demand surge, EV components, gold ratio reversion', 'risk': 'Medium', 'horizon': '12-24M'},
    {'name': 'S&P 500', 'ticker': '^GSPC', 'score': 82, 'class': 'Equity Index',
     'thesis': 'Earnings growth resilient, soft landing scenario, buyback support', 'risk': 'Medium', 'horizon': '12-36M'},
    {'name': 'Taiwan Semi', 'ticker': 'TSM', 'score': 90, 'class': 'Semiconductor',
     'thesis': 'AI chip monopoly, 3nm leadership, pricing power intact', 'risk': 'Medium', 'horizon': '12-24M'},
    {'name': 'Broadcom', 'ticker': 'AVGO', 'score': 87, 'class': 'Semiconductor',
     'thesis': 'Custom AI silicon, networking growth, M&A execution', 'risk': 'Medium', 'horizon': '12-24M'},
    {'name': 'Palantir', 'ticker': 'PLTR', 'score': 83, 'class': 'Software',
     'thesis': 'AIP platform traction, gov contracts stable, commercial breakout', 'risk': 'High', 'horizon': '18-36M'},
    {'name': 'Nasdaq-100', 'ticker': 'QQQ', 'score': 85, 'class': 'Tech Index',
     'thesis': 'AI theme exposure, rate cut beneficiary, innovation premium', 'risk': 'Medium', 'horizon': '12-24M'},
]

# ==================== STREAMLIT APPLICATION ====================
st.set_page_config(page_title="Quantum Trading Intelligence", page_icon="⚛️", layout="wide")

st.markdown("""
<div class="quantum-header">
    <h1>⚛️ QUANTUM TRADING INTELLIGENCE</h1>
    <p style='font-size: 1.1em; margin: 0;'>Institutional-Grade Analytics Engine | Powered by ML Ensemble</p>
</div>
""", unsafe_allow_html=True)

# Input section
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    symbol = st.text_input("🎯 Asset Ticker", value="BTC-USD", help="BTC-USD, GC=F, NVDA, ^GSPC")
with col2:
    interval = st.selectbox("⏱️ Timeframe", ['5m', '15m', '1h', '1d'], index=2)
with col3:
    st.markdown("##")
    refresh = st.button("🔄 REFRESH", use_container_width=True, type="primary")
with col4:
    st.markdown("##")
    st.markdown(f"**Profile:** {QuantumEngine.ASSET_PROFILES.get(symbol, {}).get('class', 'Unknown')}")

st.markdown("---")

# Model training/caching
@st.cache_resource(ttl=7200)
def get_quantum_model(sym, intv):
    period_map = {'5m': '60d', '15m': '60d', '1h': '730d', '1d': '5y'}
    data = MarketIntel.fetch_data(sym, intv, period_map.get(intv, '730d'))
    if data is None or len(data) < 200:
        return None, None, None, None
    
    df = QuantumEngine.compute_indicators(data, sym)
    X, y = QuantumEngine.simulate_trades(df, sym, n=1200)
    models, scaler = QuantumEngine.train_ensemble(X, y)
    
    return models, scaler, df, models['ensemble_acc']

# Load model
session_key = f"quantum_{symbol}_{interval}"
if session_key not in st.session_state or refresh:
    with st.spinner("⚛️ Initializing Quantum Engine..."):
        models, scaler, df, acc = get_quantum_model(symbol, interval)
        if models:
            st.session_state[session_key] = {'models': models, 'scaler': scaler, 'df': df, 'acc': acc}
            st.success(f"✅ Quantum Engine Online | Model Accuracy: {acc:.2%}")
        else:
            st.error("❌ Unable to initialize. Check ticker and try again.")
            st.stop()

state = st.session_state[session_key]
models, scaler, df, acc = state['models'], state['scaler'], state['df'], state['acc']

# Generate signals
signals = MarketIntel.get_signals(symbol, df)

# Main layout
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.markdown("### 🎯 ALPHA SIGNALS")
    
    if signals:
        for i, sig in enumerate(signals[:2]):
            conf_class = 'high-conf' if sig['prob'] >= 72 else 'med-conf' if sig['prob'] >= 62 else 'low-conf'
            rr = abs(sig['tp'] - sig['entry']) / abs(sig['entry'] - sig['sl'])
            risk = abs(sig['entry'] - sig['sl']) / sig['entry'] * 100
            
            st.markdown(f"""
            <div class="signal-card {conf_class}">
                <h3>🎲 {sig['dir']} SETUP #{i+1}</h3>
                <p><b>Entry:</b> ${sig['entry']:.2f} | <b>Stop:</b> ${sig['sl']:.2f} | <b>Target:</b> ${sig['tp']:.2f}</p>
                <p><b>Market Prob:</b> {sig['prob']:.1f}% | <b>Tech Score:</b> {sig['score']}/100 | <b>R/R:</b> {rr:.2f}x</p>
                <p><b>Risk:</b> {risk:.2f}% | <b>Sentiment:</b> {sig['sentiment']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"⚛️ QUANTUM ANALYZE", key=f"analyze_{i}", use_container_width=True):
                st.session_state.selected_signal = sig

with col_right:
    st.markdown("### 🏆 QUANTUM UNIVERSE")
    st.markdown("**Top 10 Institutional Picks**")
    
    for asset in QUANTUM_UNIVERSE:
        color = '🟢' if asset['score'] >= 88 else '🟡' if asset['score'] >= 84 else '🟠'
        st.markdown(f"""
        **{color} {asset['name']}** ({asset['ticker']}) | Score: {asset['score']}/100
        - 📊 {asset['class']} | ⚠️ {asset['risk']} Risk | ⏰ {asset['horizon']}
        - 💡 {asset['thesis']}
        """)
        
    st.markdown("---")
    st.markdown("*Scores: Fundamentals + Technicals + 2025 Catalysts*")

# Market dashboard
st.markdown("---")
st.markdown("### 📊 MARKET INTELLIGENCE")

latest = df.iloc[-1]
returns = df['Close'].pct_change().dropna()

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("💰 Price", f"${latest['Close']:.2f}")
with col2:
    st.metric("📈 RSI", f"{latest['rsi']:.0f}", delta=f"{latest['rsi']-50:+.0f}")
with col3:
    st.metric("⚡ ADX", f"{latest['adx']:.0f}", help="Trend strength")
with col4:
    regime_map = {2: "🔥 Strong Bull", 1: "📈 Weak Bull", 0: "➡️ Neutral", -1: "📉 Weak Bear", -2: "❄️ Strong Bear"}
    st.metric("🎯 Regime", regime_map[latest['regime']])
with col5:
    st.metric("💥 Vol %", f"{latest['atr_pct']:.2f}%")
with col6:
    sharpe = RiskEngine.sharpe_ratio(returns)
    st.metric("📊 Sharpe", f"{sharpe:.2f}")

# Risk metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    var_95 = RiskEngine.calculate_var(returns, 0.95)
    st.metric("📉 VaR 95%", f"{var_95*100:.2f}%")
with col2:
    max_dd = RiskEngine.max_drawdown(df['Close'].values)
    st.metric("🌊 Max DD", f"{max_dd*100:.2f}%")
with col3:
    volatility = returns.std() * np.sqrt(252) * 100
    st.metric("📈 Vol (Ann.)", f"{volatility:.1f}%")
with col4:
    st.metric("🎓 Model Acc", f"{acc:.2%}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>⚛️ Quantum Trading Intelligence | Institutional Analytics Platform</div>", unsafe_allow_html=True)
