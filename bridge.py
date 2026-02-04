"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 'ORACLE PRIME' - LOGIC FIX (BUY vs SELL MATH)
═══════════════════════════════════════════════════════════════════════════════
Backend engine.
CORREZIONE CRITICA:
- Ora calcola SL e TP in modo diverso per BUY e SELL.
- Risolve il bug dove i SELL mostravano numeri da BUY.
═══════════════════════════════════════════════════════════════════════════════
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
        
        # Se non abbiamo abbastanza dati, aspettiamo
        if len(history) < Config.MIN_TICKS_WARMUP:
            return {'status': 'WARMUP', 'progress': f"{len(history)}/{Config.MIN_TICKS_WARMUP}"}
        
        prices = np.array(history)
        rsi = self.calculate_rsi(prices)
        
        # LOGICA DI TRADING SEMPLIFICATA E ROBUSTA
        signal = None
        reason = ""
        
        # RSI Strategy
        if rsi < 30:
            signal = "BUY"
            reason = "Oversold Bounce"
        elif rsi > 70:
            signal = "SELL"
            reason = "Overbought Reversal"
            
        # Filtro Anti-Spam (Evita di mandare lo stesso segnale ogni secondo)
        if signal and (signal != self.last_signal[symbol] or time.time() - self.last_signal_time[symbol] > 120):
            self.last_signal[symbol] = signal
            self.last_signal_time[symbol] = time.time()
            return {
                'status': 'SIGNAL', 
                'signal': signal, 
                'entry': current_price, 
                'confidence': 85, 
                'reason': reason
            }
            
        return {'status': 'SCANNING', 'message': f'RSI: {rsi:.1f}'}

    def calculate_rsi(self, prices, period=14):
        deltas = np.diff(prices)
        if len(deltas) == 0: return 50
        
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0001
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

def main():
    if not Config.SUPABASE_URL: 
        logger.error("❌ SUPABASE_URL mancante nel file .env")
        return
        
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    engine = TradingEngine()
    last_history_update = {s: 0 for s in Config.ASSETS}

    print("🚀 TITAN BRIDGE AVVIATO - In attesa di dati MT4...")

    while True:
        for symbol in Config.ASSETS:
            file_path = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            if not os.path.exists(file_path): continue
            
            try:
                # Legge il file JSON generato da MT4
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read().strip()
                    if not content: continue
                    data = json.loads(content)
                
                bid = float(data.get('bid', 0))
                ask = float(data.get('ask', 0))
                price = (bid + ask) / 2
                
                # 1. Invia Prezzo per il Grafico (ogni 5 secondi)
                if time.time() - last_history_update[symbol] > Config.PRICE_HISTORY_INTERVAL:
                    supabase.table("price_history").insert({"symbol": symbol, "price": price}).execute()
                    last_history_update[symbol] = time.time()

                # 2. Analizza per Segnali
                res = engine.analyze(symbol, price)
                
               if res['status'] == 'SIGNAL':
                    # === CALCOLO MATEMATICO CORRETTO (BUY vs SELL) ===
                    if res['signal'] == 'BUY':
                        # BUY: SL sotto (-0.5%), TP sopra (+1.0%)
                        sl = price * 0.995  
                        tp = price * 1.010  
                    else: 
                        # SELL: SL sopra (+0.5%), TP sotto (-1.0%)
                        sl = price * 1.005  
                        tp = price * 0.990  

                    payload = {
                        "symbol": symbol, 
                        "recommendation": res['signal'], 
                        "current_price": price,
                        "entry_price": price, 
                        "stop_loss": round(sl, 2), 
                        "take_profit": round(tp, 2),
                        "confidence_score": res['confidence'], 
                        "details": res['reason']
                    }
                    
                    supabase.table("trading_signals").insert(payload).execute()
                    logger.info(f"🎯 {symbol} {res['signal']} inviato! Entry: {price} SL: {round(sl,2)} TP: {round(tp,2)}") Entry: {price} SL: {round(sl,2)} TP: {round(tp,2)}")
                    
            except Exception as e:
                # logger.error(f"Errore loop: {e}")
                continue
                
        time.sleep(0.5) # Pausa breve per non sovraccaricare la CPU

if __name__ == "__main__": 
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bridge fermato dall'utente.")
