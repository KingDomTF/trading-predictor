# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

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
def fetch_live_prices(symbols, interval):
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d', interval=interval)
            prices[symbol] = hist['Close'].iloc[-1]
        except Exception as e:
            st.error(f"Error fetching live price for {symbol}: {e}")
            prices[symbol] = np.nan
    return prices

@st.cache_data
def fetch_historical_data(symbols, period, interval):
    end_date = dt.date.today()
    try:
        data = yf.download(symbols, period=period, interval=interval)
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

def calculate_risk(returns_series):
    if len(returns_series.dropna()) < 2:
        return np.nan, np.nan, np.nan
    returns = returns_series.dropna()
    var = np.percentile(returns, 5)
    mean_ret = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = mean_ret / vol if vol != 0 else np.nan
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()
    return var, sharpe, max_dd

def backtest_strategy(data, strategy_func):
    signals = strategy_func(data)
    positions = signals.shift(1)
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
        signal[rsi < 30] = 1
        signal[rsi > 70] = -1
        return signal
    
    def macd_strategy(data):
        macd = MACD(data['Close'])
        signal = pd.Series(0, index=data.index)
        signal[(macd.macd_diff() > 0) & (macd.macd_diff().shift(1) < 0)] = 1
        signal[(macd.macd_diff() < 0) & (macd.macd_diff().shift(1) > 0)] = -1
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
    
    def combined_strategy(data):
        rsi_sig = rsi_strategy(data)
        macd_sig = macd_strategy(data)
        signal = pd.Series(0, index=data.index)
        signal[(rsi_sig == 1) & (macd_sig == 1)] = 1
        signal[(rsi_sig == -1) & (macd_sig == -1)] = -1
        return signal
    
    def ml_ensemble_strategy(data):
        signal = pd.Series(0, index=data.index)
        try:
            df = data.copy()
            df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
            lags = [1,2,3,5,10]
            for l in lags:
                df[f'ret_lag_{l}'] = df['Close'].pct_change(l)
            df = df.dropna()
            if len(df) < 50:
                return signal
            features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            X = df[features].iloc[:-1]
            y = df['Close'].iloc[1:]
            current = df['Close'].iloc[:-1]
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X, y)
            xgb = XGBRegressor(n_estimators=100, random_state=42)
            xgb.fit(X, y)
            preds = (rf.predict(X) + gb.predict(X) + xgb.predict(X)) / 3
            signal.iloc[:-1] = np.where(preds > current, 1, -1)
            # For next
            X_last = df[features].iloc[-1:]
            pred_next = (rf.predict(X_last) + gb.predict(X_last) + xgb.predict(X_last)) / 3
            signal.iloc[-1] = 1 if pred_next[0] > df['Close'].iloc[-1] else -1
        except Exception as e:
            st.error(f"ML Ensemble error: {e}")
        return signal
    
    def arima_strategy(data):
        signal = pd.Series(0, index=data.index)
        try:
            model = ARIMA(data['Close'], order=(5,1,0)).fit()
            fitted = model.fittedvalues
            signal = np.where(fitted > data['Close'].shift(1), 1, -1)
            signal = pd.Series(signal, index=data.index).fillna(0)
            # For next
            pred_next = model.forecast(1)[0]
            signal.iloc[-1] = 1 if pred_next > data['Close'].iloc[-1] else -1
        except Exception as e:
            st.error(f"ARIMA error: {e}")
        return signal
    
    return [
        ('RSI', rsi_strategy),
        ('MACD', macd_strategy),
        ('Bollinger Bands', bb_strategy),
        ('MA Crossover', ma_crossover),
        ('Volume Breakout', volume_strategy),
        ('Combined RSI+MACD', combined_strategy),
        ('Ensemble ML', ml_ensemble_strategy),
        ('ARIMA', arima_strategy)
    ]

def generate_proposals(main_symbols, live_prices, hist_data, backtest_period_years=3):
    proposals = []
    strategies = define_strategies()
    
    for symbol in main_symbols:
        asset_data = hist_data.loc[:, (slice(None), symbol)].droplevel(1, axis=1)
        backtest_start_date = dt.date.today() - dt.timedelta(days=365 * backtest_period_years)
        backtest_start = pd.to_datetime(backtest_start_date)
        backtest_data = asset_data[asset_data.index >= backtest_start]
        
        if backtest_data.empty or len(backtest_data) < 20:
            continue
        
        win_rates = {}
        signals = {}
        for name, func in strategies:
            sig = func(backtest_data)
            signals[name] = sig.iloc[-1]
            win_rates[name] = backtest_strategy(backtest_data, func)
        
        top_strategies = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for strat_name, win_rate in top_strategies:
            sig = signals[strat_name]
            if win_rate > 0 and sig != 0:
                entry = live_prices.get(symbol, np.nan)
                if np.isnan(entry):
                    continue
                vol = backtest_data['Close'].pct_change().std() * np.sqrt(252)
                if sig > 0:  # Buy
                    tp = entry * (1 + 0.05 * vol)
                    sl = entry * (1 - 0.03 * vol)
                    direction = "Buy"
                else:  # Sell
                    tp = entry * (1 - 0.05 * vol)
                    sl = entry * (1 + 0.03 * vol)
                    direction = "Sell"
                proposals.append({
                    'Asset': symbol,
                    'Strategy': strat_name,
                    'Direction': direction,
                    'Entry': entry,
                    'TP': tp,
                    'SL': sl,
                    'Win %': win_rate
                })
    
    proposals = sorted(proposals, key=lambda x: x['Win %'], reverse=True)[:3]
    return pd.DataFrame(proposals)

# Main app
st.title("Sistema Analisi Finanziaria Istituzionale")

categories = {
    'Metalli Preziosi': ['GC=F', 'SI=F', 'PL=F', 'PA=F'],
    'Criptovalute': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD'],
    'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X'],
    'Commodities': ['CL=F', 'BZ=F', 'NG=F', 'HG=F']
}

asset_names = {
    'GC=F': 'Oro', 'SI=F': 'Argento', 'PL=F': 'Platino', 'PA=F': 'Palladio',
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'BNB-USD': 'BNB', 'ADA-USD': 'Cardano',
    'EURUSD=X': 'EUR/USD', 'GBPUSD=X': 'GBP/USD', 'USDJPY=X': 'USD/JPY', 'USDCHF=X': 'USD/CHF', 'AUDUSD=X': 'AUD/USD',
    'CL=F': 'Petrolio WTI', 'BZ=F': 'Petrolio Brent', 'NG=F': 'Gas Naturale', 'HG=F': 'Rame',
    '^VIX': 'VIX', '^TNX': '10Y Yield'
}

with st.sidebar:
    st.header("Impostazioni")
    category = st.selectbox("Categoria", list(categories.keys()))
    timeframe = st.selectbox("Timeframe", ['15min', '1h', '4h', '1d'])
    hist_years = st.slider("Anni Storici", 1, 10, 5)
    current_days = st.slider("Giorni Periodo Corrente", 10, 90, 30)
    backtest_years = st.slider("Anni Backtest", 1, 5, 3)

timeframe_dict = {
    '15min': {'period': '60d', 'interval': '15m'},
    '1h': {'period': '730d', 'interval': '1h'},
    '4h': {'period': '1825d', 'interval': '4h'},
    '1d': {'period': f"{365 * hist_years}d", 'interval': '1d'}
}

main_symbols = categories[category]
macro_symbols = ['^VIX', '^TNX']
symbols = main_symbols + macro_symbols

tf_params = timeframe_dict[timeframe]
live_prices = fetch_live_prices(symbols, tf_params['interval'])
hist_data = fetch_historical_data(symbols, tf_params['period'], tf_params['interval'])

if not hist_data.empty:
    comparisons, correlations, closes = calculate_metrics(hist_data, current_days)
    
    st.header("Prezzi Live")
    num_cols = 4
    for i in range(0, len(main_symbols), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j < len(main_symbols):
                symbol = main_symbols[i + j]
                with cols[j]:
                    st.metric(asset_names.get(symbol, symbol), f"${live_prices[symbol]:.2f}" if not np.isnan(live_prices[symbol]) else "N/A")
    
    st.header("Indicatori Macroeconomici")
    cols = st.columns(2)
    with cols[0]:
        st.metric("VIX (Indice della Paura)", f"{live_prices['^VIX']:.2f}" if not np.isnan(live_prices['^VIX']) else "N/A")
    with cols[1]:
        st.metric("Rendimento Treasury 10Y", f"{live_prices['^TNX']:.2f}%" if not np.isnan(live_prices['^TNX']) else "N/A")
    
    if category == 'Criptovalute':
        try:
            fng_response = requests.get('https://api.alternative.me/fng/?limit=1').json()
            fng = fng_response['data'][0]['value']
            st.metric("Crypto Fear & Greed Index", fng)
        except:
            pass
    
    primary_asset = st.selectbox("Asset Principale per Grafico", main_symbols)
    st.header(f"Grafico Interattivo per {asset_names.get(primary_asset, primary_asset)}")
    asset_data = hist_data.loc[:, (slice(None), primary_asset)].droplevel(1, axis=1)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.6, 0.2, 0.2], subplot_titles=('Candlestick', 'MACD', 'RSI'))
    fig.add_trace(go.Candlestick(x=asset_data.index,
                                 open=asset_data['Open'],
                                 high=asset_data['High'],
                                 low=asset_data['Low'],
                                 close=asset_data['Close'],
                                 name='Candlestick'), row=1, col=1)
    ma50 = asset_data['Close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=asset_data.index, y=ma50, name='MA50', line=dict(color='orange')), row=1, col=1)
    bb = BollingerBands(asset_data['Close'])
    fig.add_trace(go.Scatter(x=asset_data.index, y=bb.bollinger_hband(), name='Upper BB', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=asset_data.index, y=bb.bollinger_lband(), name='Lower BB', line=dict(color='gray')), row=1, col=1)
    support = asset_data['Low'].rolling(20).min().iloc[-1]
    resistance = asset_data['High'].rolling(20).max().iloc[-1]
    fig.add_hline(y=support, line_dash="dash", annotation_text="Support", row=1, col=1)
    fig.add_hline(y=resistance, line_dash="dash", annotation_text="Resistance", row=1, col=1)
    macd = MACD(asset_data['Close'])
    fig.add_trace(go.Scatter(x=asset_data.index, y=macd.macd(), name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=asset_data.index, y=macd.macd_signal(), name='Signal', line=dict(color='red')), row=2, col=1)
    rsi = RSIIndicator(asset_data['Close'], window=14).rsi()
    fig.add_trace(go.Scatter(x=asset_data.index, y=rsi, name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", row=3, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.header("Confronto Metriche")
    st.dataframe(comparisons.style.format("{:.2%}"))
    
    st.header("Correlazioni")
    st.dataframe(correlations)
    
    st.header("Analisi del Rischio")
    returns = closes.pct_change()
    risk_df = pd.DataFrame(index=main_symbols, columns=['VaR 95%', 'Sharpe Ratio', 'Max Drawdown'])
    for symbol in main_symbols:
        r = returns[symbol]
        var, sharpe, max_dd = calculate_risk(r)
        risk_df.loc[symbol] = [var, sharpe, max_dd]
    st.dataframe(risk_df.style.format("{:.2%}"))
    
    if timeframe == '1d':
        st.header("Analisi Stagionalità (Rendimenti Medi Mensili)")
        returns = returns.copy()
        returns['month'] = returns.index.month
        monthly = returns.groupby('month').mean()[main_symbols]
        st.bar_chart(monthly)
    
    st.header("Proposte di Trading")
    proposals_df = generate_proposals(main_symbols, live_prices, hist_data, backtest_years)
    if not proposals_df.empty:
        st.dataframe(proposals_df.style.format({
            'Entry': '${:.2f}',
            'TP': '${:.2f}',
            'SL': '${:.2f}',
            'Win %': '{:.1f}%'
        }))
    else:
        st.info("Nessuna proposta valida generata.")
else:
    st.error("Nessun dato storico disponibile.")
