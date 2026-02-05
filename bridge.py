"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 BRIDGE - ULTRA FAST REAL-TIME EDITION
═══════════════════════════════════════════════════════════════════════════════
Ottimizzato per latenza minima (1s) e rilevamento istituzionale "Whale Radar".
"""

import os
import sys
import time
import json
import logging
import codecs
import numpy as np
from collections import deque
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

# Fix encoding per Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

load_dotenv()

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    MT4_PATH = os.getenv("MT4_PATH", "").rstrip(os.sep)
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    
    # PARAMETRI DI VELOCITÀ - Ridotti per aggiornamento immediato
    PRICE_HISTORY_INTERVAL = 1 # Aggiorna il prezzo ogni secondo
    MIN_TICKS_WARMUP = 30
    SIGNAL_COOLDOWN = 60

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TITAN")

class TradingEngine:
    def __init__(self):
        self.price_buffer = {symbol: deque(maxlen=200) for symbol in Config.ASSETS}
        self.last_signal = {symbol: None for symbol in Config.ASSETS}
        self.last_signal_time = {symbol: 0 for symbol in Config.ASSETS}

    def analyze(self, symbol, current_price):
        self.price_buffer[symbol].append(current_price)
        history = list(self.price_buffer[symbol])
        
        if len(history) < Config.MIN_TICKS_WARMUP:
            return {'status': 'WARMUP', 'progress': f"{len(history)}/{Config.MIN_TICKS_WARMUP}"}
        
        prices = np.array(history)
        
        # LOGICA WHALE RADAR (Istituzionali)
        # Calcolo Volatilità Relativa
        deltas = np.abs(np.diff(prices))
        avg_vol = np.mean(deltas[-20:])
        curr_vol = deltas[-1] if len(deltas) > 0 else 0
        
        # Squeeze (Accumulo) e Impulse (Esplosione delle Whale)
        is_squeeze = curr_vol < (avg_vol * 0.5)
        is_impulse = curr_vol > (avg_vol * 2.0)
        
        ma_fast = np.mean(prices[-10:])
        ma_slow = np.mean(prices[-50:])
        
        signal = None
        if is_impulse:
            if current_price > ma_slow and ma_fast > ma_slow:
                signal = "BUY"
            elif current_price < ma_slow and ma_fast < ma_slow:
                signal = "SELL"
        
        if signal and (signal != self.last_signal[symbol] or time.time() - self.last_signal_time[symbol] > Config.SIGNAL_COOLDOWN):
            self.last_signal[symbol] = signal
            self.last_signal_time[symbol] = time.time()
            return {
                'status': 'SIGNAL',
                'signal': signal,
                'entry': current_price,
                'confidence': 95 if is_squeeze else 85,
                'reason': 'Institutional Impulse'
            }
            
        return {'status': 'SCANNING', 'message': f'Vol: {curr_vol:.2f}'}

def main():
    if not Config.SUPABASE_URL: return
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    engine = TradingEngine()
    last_update = {s: 0 for s in Config.ASSETS}

    print("🚀 TITAN BRIDGE FAST START...")

    while True:
        cycle_start = time.time()
        for symbol in Config.ASSETS:
            file_path = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            if not os.path.exists(file_path): continue
            
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                # Calcolo prezzo medio istantaneo
                price = (float(data.get('bid', 0)) + float(data.get('ask', 0))) / 2.0
                if price <= 0: continue

                # INVIO PREZZO LIVE (Ogni secondo)
                if time.time() - last_update[symbol] >= Config.PRICE_HISTORY_INTERVAL:
                    supabase.table("price_history").insert({
                        "symbol": symbol, 
                        "price": price, 
                        "volume": int(data.get('volume', 1000))
                    }).execute()
                    last_update[symbol] = time.time()

                # ANALISI SEGNALE
                result = engine.analyze(symbol, price)
                if result['status'] == 'SIGNAL':
                    # Calcolo SL/TP dinamico
                    sl = price * 0.995 if result['signal'] == "BUY" else price * 1.005
                    tp = price * 1.015 if result['signal'] == "BUY" else price * 0.985
                    
                    supabase.table("trading_signals").insert({
                        "symbol": symbol,
                        "recommendation": result['signal'],
                        "current_price": price,
                        "entry_price": price,
                        "stop_loss": round(sl, 2),
                        "take_profit": round(tp, 2),
                        "confidence_score": result['confidence'],
                        "details": f"Whale Radar: {result['reason']}"
                    }).execute()
                    logger.info(f"🎯 {symbol} {result['signal']} SENT @ {price}")
                    
            except Exception: continue
        
        # Pausa minima per non bloccare la CPU ma garantire reattività
        time.sleep(0.1)

if __name__ == "__main__":
    main()
