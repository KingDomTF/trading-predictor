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
    MIN_DATA_POINTS = 60 # Almeno 1 ora di dati per essere accurati
    COOLDOWN = 900 # 15 minuti di pausa dopo un segnale

class WhaleEngine:
    def __init__(self):
        self.prices = {s: deque(maxlen=300) for s in Config.ASSETS}
        self.last_signal = {s: 0 for s in Config.ASSETS}

    def analyze(self, symbol, current_p):
        self.prices[symbol].append(current_p)
        hist = list(self.prices[symbol])
        if len(hist) < Config.MIN_DATA_POINTS: return None

        # 1. ANALISI VOLATILITÀ (Per capire se le Whale si muovono)
        arr = np.array(hist)
        vols = np.abs(np.diff(arr[-30:]))
        avg_vol = np.mean(vols)
        current_vol = vols[-1] if len(vols) > 0 else 0

        # 2. LOGICA DI ESPANSIONE (Il momento dell'entrata)
        # Cerchiamo un movimento che sia 3 volte la media -> Entrata Istituzionale
        is_explosion = current_vol > (avg_vol * 3.0)
        
        # 3. FILTRO TREND (Conferma istituzionale)
        ma_fast = np.mean(arr[-10:])
        ma_slow = np.mean(arr[-100:])
        
        signal = None
        if is_explosion:
            if current_p > ma_slow and ma_fast > ma_slow: signal = "BUY"
            elif current_p < ma_slow and ma_fast < ma_slow: signal = "SELL"

        # 4. CALCOLO SL ACCURATO (Dietro la zona di stasi)
        if signal and (time.time() - self.last_signal[symbol] > Config.COOLDOWN):
            self.last_signal[symbol] = time.time()
            
            # SL protetto: 2.5 volte la volatilità media (per non essere preso dal "rumore")
            dist = avg_vol * 2.5
            sl = current_p - dist if signal == "BUY" else current_p + dist
            tp = current_p + (dist * 3) if signal == "BUY" else current_p - (dist * 3)
            
            return {'sig': signal, 'sl': round(sl, 2), 'tp': round(tp, 2), 'conf': 92}
        return None

def main():
    db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    engine = WhaleEngine()
    print("💎 TITAN PRECISION ENGINE ONLINE")

    while True:
        for symbol in Config.ASSETS:
            path = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            if not os.path.exists(path): continue
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                price = (float(data['bid']) + float(data['ask'])) / 2
                
                # Invia prezzo live per allineamento istantaneo
                db.table("price_history").insert({"symbol": symbol, "price": price}).execute()
                
                res = engine.analyze(symbol, price)
                if res:
                    db.table("trading_signals").insert({
                        "symbol": symbol, "recommendation": res['sig'], "current_price": price,
                        "entry_price": price, "stop_loss": res['sl'], "take_profit": res['tp'],
                        "confidence_score": res['conf'], "details": "Institutional Expansion detected"
                    }).execute()
                    print(f"✅ {symbol} {res['sig']} - SL PROTETTO: {res['sl']}")
            except: continue
        time.sleep(0.5) # Massima velocità di calcolo

if __name__ == "__main__": main()
