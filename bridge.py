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

load_dotenv()

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    MT4_PATH = os.getenv("MT4_PATH", "").rstrip(os.sep)
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    MIN_TICKS_WARMUP = 30
    PRICE_HISTORY_INTERVAL = 5

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
        rsi = self.calculate_rsi(prices)
        ma_fast = np.mean(prices[-10:])
        ma_slow = np.mean(prices[-50:])

        signal = "BUY" if rsi < 30 else "SELL" if rsi > 70 else None
        if signal and (signal != self.last_signal[symbol] or time.time() - self.last_signal_time[symbol] > 60):
            self.last_signal[symbol] = signal
            self.last_signal_time[symbol] = time.time()
            return {'status': 'SIGNAL', 'signal': signal, 'entry': current_price, 'confidence': 85, 'reason': 'RSI Extreme'}
        return {'status': 'SCANNING', 'message': f'RSI: {rsi:.1f}'}

    def calculate_rsi(self, prices, period=14):
        deltas = np.diff(prices)
        gains = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0.0001
        losses = np.mean(-deltas[deltas < 0]) if any(deltas < 0) else 0.0001
        return 100 - (100 / (1 + (gains / losses)))

def main():
    if not Config.SUPABASE_URL: return
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    engine = TradingEngine()
    last_history_update = {s: 0 for s in Config.ASSETS}

    while True:
        for symbol in Config.ASSETS:
            file_path = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            if not os.path.exists(file_path): continue
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                price = (float(data['bid']) + float(data['ask'])) / 2
                
                # Invia Prezzo ogni 5 secondi
                if time.time() - last_history_update[symbol] > Config.PRICE_HISTORY_INTERVAL:
                    supabase.table("price_history").insert({"symbol": symbol, "price": price}).execute()
                    last_history_update[symbol] = time.time()

                # Analisi Segnale
                res = engine.analyze(symbol, price)
                if res['status'] == 'SIGNAL':
                    payload = {
                        "symbol": symbol, "recommendation": res['signal'], "current_price": price,
                        "entry_price": price, "stop_loss": price * 0.99, "take_profit": price * 1.02,
                        "confidence_score": res['confidence'], "details": res['reason']
                    }
                    supabase.table("trading_signals").insert(payload).execute()
                    logger.info(f"🎯 Segnale {res['signal']} inviato per {symbol}")
            except: continue
        time.sleep(1)

if __name__ == "__main__": main()
