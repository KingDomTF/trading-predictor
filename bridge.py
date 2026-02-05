import os, sys, time, json, numpy as np
from collections import deque
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    MT4_PATH = os.getenv("MT4_PATH", "").rstrip(os.sep)
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    # Maggiore storia per stabilità
    MIN_TICKS = 100 
    SIGNAL_COOLDOWN = 600 # 10 minuti di pausa tra segnali

class TradingEngine:
    def __init__(self):
        self.price_buffer = {s: deque(maxlen=200) for s in Config.ASSETS}
        self.last_signal_time = {s: 0 for s in Config.ASSETS}
        self.trend_confirmation = {s: {"dir": None, "count": 0} for s in Config.ASSETS}

    def analyze(self, symbol, price):
        self.price_buffer[symbol].append(price)
        history = list(self.price_buffer[symbol])
        if len(history) < Config.MIN_TICKS: return None

        prices = np.array(history)
        ma_fast, ma_slow = np.mean(prices[-20:]), np.mean(prices[-100:])
        
        # Calcolo Volatilità (ATR Semplificato)
        vol = np.mean(np.abs(np.diff(prices[-20:])))
        
        # Identifica Direzione Potenziale
        current_dir = "BUY" if price > ma_slow and price > ma_fast else "SELL" if price < ma_slow and price < ma_fast else None
        
        # Filtro di Conferma: La direzione deve persistere per 5 tick
        if current_dir == self.trend_confirmation[symbol]["dir"]:
            self.trend_confirmation[symbol]["count"] += 1
        else:
            self.trend_confirmation[symbol] = {"dir": current_dir, "count": 1}

        # Segnale valido solo dopo 5 conferme e se fuori dal cooldown
        if self.trend_confirmation[symbol]["count"] >= 5 and current_dir:
            if (time.time() - self.last_signal_time[symbol] > Config.SIGNAL_COOLDOWN):
                self.last_signal_time[symbol] = time.time()
                # SL/TP basati sulla volatilità reale (ATR * 3) per dare "respiro" al trade
                sl_dist = vol * 3
                sl = price - sl_dist if current_dir == "BUY" else price + sl_dist
                tp = price + (sl_dist * 2) if current_dir == "BUY" else price - (sl_dist * 2)
                
                return {
                    'signal': current_dir, 
                    'conf': 88, 
                    'sl': round(sl, 2), 
                    'tp': round(tp, 2)
                }
        return None

def main():
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    engine = TradingEngine()
    print("🛡️ TITAN GUARDIAN ONLINE - Modalità Stabilità Attiva")
    
    while True:
        for symbol in Config.ASSETS:
            path = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            if not os.path.exists(path): continue
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                price = (float(data['bid']) + float(data['ask'])) / 2
                
                # Prezzo Live
                supabase.table("price_history").insert({"symbol": symbol, "price": price}).execute()
                
                res = engine.analyze(symbol, price)
                if res:
                    supabase.table("trading_signals").insert({
                        "symbol": symbol, "recommendation": res['signal'], "current_price": price,
                        "entry_price": price, "stop_loss": res['sl'], "take_profit": res['tp'],
                        "confidence_score": res['conf'], "details": "Trend Confirmed"
                    }).execute()
                    print(f"🎯 {symbol} {res['signal']} Sent | SL: {res['sl']}")
            except: continue
        time.sleep(1)

if __name__ == "__main__": main()
