# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime as dt

# Custom CSS for modern look
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 5px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def fetch_live_prices(symbols):
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            prices[symbol] = ticker.history(period='1d')['Close'].iloc[-1]
        except Exception as e:
            st.error(f"Error fetching live price for {symbol}: {e}")
            prices[symbol] = np.nan
    return prices

@st.cache_data
def fetch_historical_data(symbols, period_years=5):
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365 * period_years)
    try:
        data = yf.download(symbols, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def calculate_metrics(hist_data, current_period_days=30):
    if hist_data.empty:
        return None, None, None
    
    closes = hist_data['Close']
    returns = closes.pct_change().dropna()
    
    # Historical metrics
    avg_returns = returns.mean() * 252  # Annualized
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    # Correlations
    correlations = closes.corr()
    
    # Current period
    current_data = closes.iloc[-current_period_days:]
    current_returns = current_data.pct_change().dropna()
    current_avg_returns = current_returns.mean() * 252
    current_volatility = current_returns.std() * np.sqrt(252)
    
    comparisons = pd.DataFrame({
        'Historical Avg Return': avg_returns,
        'Current Avg Return': current_avg_returns,
        'Historical Volatility': volatility,
        'Current Volatility': current_volatility
    })
    
    return comparisons, correlations, closes

def backtest_strategy(data, strategy_func):
    signals = strategy_func(data)
    positions = signals.shift(1)  # Lag signal
    returns = data['Close'].pct_change()
    strategy_returns = positions * returns
    trades = positions.diff().abs() > 0
    profitable = strategy_returns[trades.shift(-1).fillna(False)] > 0
    win_rate = profitable.sum() / profitable.count() * 100 if profitable.count() > 0 else 0
    return win_rate

def define_strategies():
    def rsi_strategy(data):
        rsi = RSIIndicator(data['Close'], window=14).rsi()
        signal = pd.Series(0, index=data.index)
        signal[rsi < 30] = 1  # Buy
        signal[rsi > 70] = -1  # Sell
        return signal
    
    def macd_strategy(data):
        macd = MACD(data['Close'])
        signal = pd.Series(0, index=data.index)
        signal[(macd.macd_diff() > 0) & (macd.macd_diff().shift(1) < 0)] = 1  # Crossover buy
        signal[(macd.macd_diff() < 0) & (macd.macd_diff().shift(1) > 0)] = -1  # Crossover sell
        return signal
    
    def bb_strategy(data):
        bb = BollingerBands(data['Close'])
        signal = pd.Series(0, index=data.index)
        signal[data['Close'] < bb.bollinger_lband()] = 1
        signal[data['Close'] > bb.bollinger_hband()] = -1
        return signal
    
    def ma_crossover(data):
        ma_short = data['Close'].rolling(50).mean()
        ma_long = data['Close'].rolling(200).mean()
        signal = pd.Series(0, index=data.index)
        signal[(ma_short > ma_long) & (ma_short.shift(1) < ma_long.shift(1))] = 1
        signal[(ma_short < ma_long) & (ma_short.shift(1) > ma_long.shift(1))] = -1
        return signal
    
    def volume_strategy(data):
        vol_ma = data['Volume'].rolling(20).mean()
        signal = pd.Series(0, index=data.index)
        signal[(data['Volume'] > vol_ma * 1.5) & (data['Close'] > data['Open'])] = 1
        signal[(data['Volume'] > vol_ma * 1.5) & (data['Close'] < data['Open'])] = -1
        return signal
    
    def ml_strategy(data):
        df = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        features = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], axis=1, errors='ignore')
        target = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        train_features, test_features, train_target, test_target = train_test_split(
            features[:-1], target[:-1], test_size=0.2, shuffle=False
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(train_features, train_target)
        
        predictions = model.predict(features)
        signal = pd.Series(predictions * 2 - 1, index=data.index)  # 1 to 1, 0 to -1
        return signal
    
    def combined_strategy(data):
        rsi_sig = rsi_strategy(data)
        macd_sig = macd_strategy(data)
        signal = pd.Series(0, index=data.index)
        signal[(rsi_sig == 1) & (macd_sig == 1)] = 1
        signal[(rsi_sig == -1) & (macd_sig == -1)] = -1
        return signal
    
    return [
        ('RSI', rsi_strategy),
        ('MACD', macd_strategy),
        ('Bollinger Bands', bb_strategy),
        ('MA Crossover', ma_crossover),
        ('Volume Breakout', volume_strategy),
        ('ML RFC', ml_strategy),
        ('Combined RSI+MACD', combined_strategy)
    ]

def generate_proposals(symbols, live_prices, hist_data, backtest_period_years=3):
    proposals = []
    strategies = define_strategies()
    
    for symbol in symbols:
        asset_data = hist_data.loc[:, (slice(None), symbol)].droplevel(1, axis=1)
        backtest_start = dt.date.today() - dt.timedelta(days=365 * backtest_period_years)
        backtest_data = asset_data[asset_data.index >= backtest_start]
        
        if backtest_data.empty:
            continue
        
        win_rates = {}
        for name, func in strategies:
            win_rates[name] = backtest_strategy(backtest_data, func)
        
        top_strategies = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for strat_name, win_rate in top_strategies:
            if win_rate > 0:
                entry = live_prices.get(symbol, np.nan)
                if np.isnan(entry):
                    continue
                vol = backtest_data['Close'].pct_change().std() * np.sqrt(252)
                tp = entry * (1 + 0.05 * vol)  # +5% adjusted by vol
                sl = entry * (1 - 0.03 * vol)  # -3% adjusted
                proposals.append({
                    'Asset': symbol,
                    'Strategy': strat_name,
                    'Entry': entry,
                    'TP': tp,
                    'SL': sl,
                    'Win %': win_rate
                })
    
    # Select top 3 overall
    proposals = sorted(proposals, key=lambda x: x['Win %'], reverse=True)[:3]
    return pd.DataFrame(proposals)

# Main app
st.title("Financial Analysis Dashboard")

default_symbols = ['GC=F', 'SI=F', '^GSPC', 'BTC-USD']
asset_names = {'GC=F': 'Gold', 'SI=F': 'Silver', '^GSPC': 'US500', 'BTC-USD': 'Bitcoin'}

with st.sidebar:
    st.header("Settings")
    hist_years = st.slider("Historical Years", 1, 10, 5)
    current_days = st.slider("Current Period Days", 10, 90, 30)
    backtest_years = st.slider("Backtest Years", 1, 5, 3)
    additional_symbols = st.text_input("Additional Symbols (comma-separated, e.g., 'AAPL,GOOGL')")
    
    if additional_symbols:
        extra_symbols = [sym.strip() for sym in additional_symbols.split(',') if sym.strip()]
        symbols = default_symbols + extra_symbols
        for sym in extra_symbols:
            asset_names[sym] = sym  # Use symbol as name for additional
    else:
        symbols = default_symbols

live_prices = fetch_live_prices(symbols)
hist_data = fetch_historical_data(symbols, hist_years)

if not hist_data.empty:
    comparisons, correlations, closes = calculate_metrics(hist_data, current_days)
    
    st.header("Live Prices")
    num_cols = 4
    for i in range(0, len(symbols), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j < len(symbols):
                symbol = symbols[i + j]
                with cols[j]:
                    st.metric(asset_names.get(symbol, symbol), f"${live_prices[symbol]:.2f}" if not np.isnan(live_prices[symbol]) else "N/A")
    
    st.header("Price Charts")
    st.line_chart(closes)
    
    st.header("Metrics Comparison")
    st.dataframe(comparisons.style.format("{:.2%}"))
    
    st.header("Correlations")
    st.dataframe(correlations)
    
    st.header("Trading Proposals")
    proposals_df = generate_proposals(symbols, live_prices, hist_data, backtest_years)
    if not proposals_df.empty:
        st.dataframe(proposals_df.style.format({
            'Entry': '${:.2f}',
            'TP': '${:.2f}',
            'SL': '${:.2f}',
            'Win %': '{:.1f}%'
        }))
    else:
        st.info("No valid proposals generated.")
else:
    st.error("No historical data available.")
