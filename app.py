import time
import json
import os
import glob
import pandas as pd
import numpy as np
import yfinance as yf
from supabase import create_client
from datetime import datetime, timedelta

# ================= CONFIGURAZIONE TITAN V6.2 (SYNTAX FIXED) =================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

# PERCORSO MT4 (Già inserito corretto)
MT4_PATH_INPUT = r"C:\Users\dcbat\AppData\Roaming\MetaQuotes\Terminal\B8925BF731C22E88F33C7A8D7CD3190E\MQL4\Files\mt4_live_data.json"

SYMBOL_MAP = {
    "XAUUSD": "GC=F", "GOLD": "GC=F", "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
    "US500": "ES=F", "SPX": "^GSPC", "XAGUSD": "SI=F", "EURUSD": "EURUSD=X"
}

MACRO_TICKERS = { "DXY": "DX-Y.NYB", "US10Y": "^TNX", "VIX": "^VIX" }

# ================= CLASSE DI CACHING (Performance Engine) =================
class MarketDataCache:
    def __init__(self, update_interval_minutes=30):
        self.cache = {}
        self.macro_cache = None
        self.last_macro_update = datetime.min
        self.update_interval = timedelta(minutes=update_interval_minutes)

    def _clean_df(self, df):
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df

    def get_macro_context(self):
        now = datetime.now()
        if self.macro_cache is None or (now - self.last_macro_update) > self.update_interval:
            try:
                tickers = list(MACRO_TICKERS.values())
                df = yf.download(tickers, period="5d", interval="1d", progress=False, auto_adjust=False)['Close']
                if df is not None and not df.empty:
                    self.macro_cache = self._process_macro(df)
                    self.last_macro_update = now
            except: pass
        return self.macro_cache

    def _process_macro(self, df):
        ctx = {}
        try:
            dxy_curr = df[MACRO_TICKERS['DXY']].iloc[-1]
            dxy_prev = df[MACRO_TICKERS['DXY']].iloc[-2]
            vix_curr = df[MACRO_TICKERS['VIX']].iloc[-1]
            
            ctx['USD_TREND'] = "BULL" if dxy_curr > dxy_prev else "BEAR"
            ctx['RISK_ON'] = True if vix_curr < 20 else False
            
            if MACRO_TICKERS['US10Y'] in df.columns:
                tnx_curr = df[MACRO_TICKERS['US10Y']].iloc[-1]
                tnx_prev = df[MACRO_TICKERS['US10Y']].iloc[-2]
                ctx['RATES_TREND'] = "BULL" if tnx_curr > tnx_prev else "BEAR"
        except: return None
        return ctx

    def get_asset_history(self, yf_symbol):
        now = datetime.now()
        if yf_symbol not in self.cache or (now - self.cache[yf_symbol]['timestamp']) > self.update_interval:
            try:
                daily = self._clean_df(yf.download(yf_symbol, period="1y", interval="1d", progress=False, auto_adjust=False))
                hourly = self._clean_df(yf.download(yf_symbol, period="60d", interval="60m", progress=False, auto_adjust=False))
                minute = self._clean_df(yf.download(yf_symbol, period="5d", interval="15m", progress=False, auto_adjust=False))
                
                if daily is not None and hourly is not None and minute is not None:
                    self.cache[yf_symbol] = {
                        'daily': daily, 'hourly': hourly, 'minute': minute, 'timestamp': now
                    }
            except: pass
        return self.cache.get(yf_symbol)

# ================= TITAN BRAIN (Logic Engine) =================
class TitanEngine:
    def __init__(self):
        self.data_manager = MarketDataCache()

    def calculate_indicators(self, df, price):
        last_idx = df.index[-1] + pd.Timedelta(minutes=15)
        # Fix per dataframe con colonne extra
        row_data = [price] * len(df.columns)
        new_row = pd.DataFrame([row_data], columns=df.columns, index=[last_idx])
        df = pd.concat([df, new_row])
        
        ema_21 = df['Close'].ewm(span=21).mean().iloc[-1]
        ema_50 = df['Close'].ewm(span=50).mean().iloc[-1]
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        high_low = df['High'] - df['Low']
        tr = np.maximum(high_low, np.abs(df['High'] - df['Close'].shift()))
        atr = tr.rolling(14).mean().iloc[-1]
        
        return ema_21, ema_50, rsi, atr

    def calculate_vwap(self, df):
        try:
            v = df['Volume'].values
            p = (df['High'] + df['Low'] + df['Close']) / 3
            return ((p * v).cumsum() / v.cumsum()).iloc[-1]
        except: return df['Close'].iloc[-1]

    def analyze(self, symbol, live_price):
        yf_sym = SYMBOL_MAP.get(symbol, symbol)
        
        history = self.data_manager.get_asset_history(yf_sym)
        macro = self.data_manager.get_macro_context()
        
        if not history: return None

        daily = history['daily']
        hourly = history['hourly']
        minute = history['minute'].copy()

        # Strategia
        sma_200 = daily['Close'].rolling(200).mean().iloc[-1]
        trend_long = "BULL" if live_price > sma_200 else "BEAR"
        
        vwap = self.calculate_vwap(hourly)
        ema_21, ema_50, rsi, atr = self.calculate_indicators(minute, live_price)
        
        # Scoring
        score_bull = 0
        score_bear = 0
        
        if trend_long == "BULL": score_bull += 20
        else: score_bear += 20
        
        if live_price > vwap: score_bull += 25
        else: score_bear += 25
        
        if ema_21 > ema_50: score_bull += 25
        else: score_bear += 25
        
        if macro:
            if symbol in ["XAUUSD", "GOLD"]:
                if macro.get('USD_TREND') == "BEAR": score_bull += 30
                else: score_bear += 30
            elif symbol in ["BTCUSD", "US500"]:
                if macro.get('RISK_ON'): score_bull += 30
                else: score_bear += 30
        
        signal = "WAIT"
        final_score = 50
        detail = f"Trend: {trend_long} | RSI: {rsi:.1f}"
        
        if score_bull >= 75 and rsi < 75:
            signal = "BUY"
            final_score = score_bull
            detail = "Strong Bullish Confluence"
        elif score_bear >= 75 and rsi > 25:
            signal = "SELL"
            final_score = score_bear
            detail = "Strong Bearish Confluence"
        
        # Regime & Risk
        mean_h = hourly['Close'].rolling(50).mean().iloc[-1]
        std_h = hourly['Close'].rolling(50).std().iloc[-1]
        z = (live_price - mean_h) / (std_h + 1e-9)
        regime = "VOLATILE" if abs(z) > 2.5 else ("TRENDING" if abs(ema_21-ema_50) > live_price*0.001 else "RANGING")

        if np.isnan(atr) or atr == 0: atr = live_price * 0.005
        sl = live_price - (atr * 2.0) if signal == "BUY" else live_price + (atr * 2.0)
        tp = live_price + (atr * 3.0) if signal == "BUY" else live_price - (atr * 3.0)
        rr = round(abs(tp - live_price) / abs(live_price - sl), 2) if sl != 0 else 0

        return {
            "symbol": symbol, "current_price": round(live_price, 2), "recommendation": signal, "details": detail,
            "entry_price": round(live_price, 2), "stop_loss": round(sl, 2), "take_profit": round(tp, 2),
            "risk_reward": rr, "prob_buy": final_score if signal == "BUY" else (100-final_score if signal=="SELL" else 50),
            "prob_sell": final_score if signal == "SELL" else (100-final_score if signal=="BUY" else 50),
            "confidence_score": final_score, "market_regime": regime
        }

# ================= MAIN LOOP =================
def main():
    print("---------------------------------------")
    print("🚀 TITAN V6.2 OPTIMUM: READY")
    
    # Risoluzione percorso cartella
    if MT4_PATH_INPUT.endswith(".json") or "." in os.path.basename(MT4_PATH_INPUT):
        mt4_folder = os.path.dirname(MT4_PATH_INPUT)
    else:
        mt4_folder = MT4_PATH_INPUT
        
    print(f"📂 Watch Folder: {mt4_folder}")
    print("---------------------------------------")
    
    try: supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except: 
        print("Error connecting to database")
        return

    engine = TitanEngine()
    print("🧠 Inizializzazione Cache Dati (Attendi)...")
    engine.data_manager.get_macro_context() # Preload

    last_files_mod = {}
    spinner = ["|", "/", "-", "\\"]
    idx = 0
    
    while True:
        try:
            if not os.path.exists(mt4_folder):
                time.sleep(2)
                continue

            files = glob.glob(os.path.join(mt4_folder, "*_data.json"))
            activity = False
            
            for file_path in files:
                try:
                    current_mod = os.path.getmtime(file_path)
                    if current_mod != last_files_mod.get(file_path, 0):
                        last_files_mod[file_path] = current_mod
                        activity = True
                        
                        with open(file_path, "r", encoding='utf-8-sig') as f: content = f.read().strip()
                        if not content: continue
                        data = json.loads(content)
                        
                        sym = data.get("symbol").replace(".m", "").replace(".pro", "")
                        price = float(data.get("price"))
                        
                        # Update DB
                        supabase.table("mt4_feed").insert({"symbol": sym, "price": price, "equity": data.get("equity")}).execute()
                        print(f"⚡ {sym}: ${price}", end="")
                        
                        # Analisi
                        payload = engine.analyze(sym, price)
                        
                        if payload:
                            supabase.table("ai_oracle").insert(payload).execute()
                            if payload['recommendation'] != "WAIT":
                                print(f" -> 💎 {payload['recommendation']} ({payload['confidence_score']}%)")
                            else:
                                print(f" -> 💤 WAIT")
                        else:
                            print(" -> ⏳ Loading History...")

                except Exception: continue

            if not activity:
                idx = (idx + 1) % 4
                print(f"{spinner[idx]} Titan Engine Running...", end="\r")
                
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
