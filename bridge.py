import os, sys, time, json, logging, numpy as np
from collections import deque
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    MT4_PATH = os.getenv("MT4_PATH", "").rstrip(os.sep)
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    MIN_TICKS = 50

class TradingEngine:
    def __init__(self):
        self.price_buffer = {s: deque(maxlen=100) for s in Config.ASSETS}
        self.last_signal_time = {s: 0 for s in Config.ASSETS}

    def analyze(self, symbol, price):
        self.price_buffer[symbol].append(price)
        history = list(self.price_buffer[symbol])
        if len(history) < Config.MIN_TICKS: return None

        prices = np.array(history)
        deltas = np.abs(np.diff(prices))
        avg_vol = np.mean(deltas[-30:]) # Volatilità media
        curr_vol = deltas[-1] if len(deltas) > 0 else 0

        # LOGICA ISTITUZIONALE: Squeeze & Impulse
        is_squeeze = curr_vol < (avg_vol * 0.5) # Accumulo
        is_whale_entry = curr_vol > (avg_vol * 2.5) # Esplosione

        ma_fast, ma_slow = np.mean(prices[-10:]), np.mean(prices[-50:])
        signal = None
        
        if is_whale_entry:
            if price > ma_slow and price > ma_fast: signal = "BUY"
            elif price < ma_slow and price < ma_fast: signal = "SELL"

        if signal and (time.time() - self.last_signal_time[symbol] > 300):
            self.last_signal_time[symbol] = time.time()
            return {'signal': signal, 'conf': 95 if is_squeeze else 85, 'reason': 'Institutional Breakout'}
        return None

def main():
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    engine = TradingEngine()
    print("🚀 TITAN WHALE RADAR ONLINE...")
    
    while True:
        for symbol in Config.ASSETS:
            path = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            if not os.path.exists(path): continue
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                price = (float(data['bid']) + float(data['ask'])) / 2
                
                # Invia Prezzo a price_history ogni 5s
                supabase.table("price_history").insert({"symbol": symbol, "price": price}).execute()
                
                res = engine.analyze(symbol, price)
                if res:
                    sl = price * 0.995 if res['signal'] == "BUY" else price * 1.005
                    tp = price * 1.015 if res['signal'] == "BUY" else price * 0.985
                    supabase.table("trading_signals").insert({
                        "symbol": symbol, "recommendation": res['signal'], "current_price": price,
                        "entry_price": price, "stop_loss": round(sl, 2), "take_profit": round(tp, 2),
                        "confidence_score": res['conf'], "details": res['reason']
                    }).execute()
                    print(f"🎯 {symbol} {res['signal']} Sent!")
            except: continue
        time.sleep(1)

if __name__ == "__main__": main()
