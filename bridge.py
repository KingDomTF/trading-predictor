import MetaTrader4 as mt4
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from supabase import create_client
import os, time, json
from dotenv import load_dotenv

load_dotenv()

class TitanInstitutionalEngine:
    def __init__(self):
        self.supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        if not mt4.initialize():
            print("❌ Errore critico: MT5 non inizializzato.")
            exit()
            
    def get_kalman_signal(self, symbol):
        # Recupera 200 tick in tempo reale dalla RAM (non da file!)
        ticks = mt4.copy_ticks_from(symbol, time.time(), 200, mt4.COPY_TICKS_ALL)
        if ticks is None: return 0, 0
        
        df = pd.DataFrame(ticks)
        prices = df['last'].values
        
        # Filtro di Kalman per rimozione rumore microstrutturale
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], 
                          initial_state_mean=prices[0], observation_covariance=1, 
                          transition_covariance=0.01)
        
        state_means, _ = kf.filter(prices)
        estimate = state_means.flatten()[-1]
        
        # Calcolo Z-Score sul residuo
        resid = prices[-1] - estimate
        z_score = resid / np.std(prices - state_means.flatten())
        
        return z_score, prices[-1]

    def execute_institutional_order(self, symbol, action, price):
        """Esecuzione con Fill Policy IOC (Immediate or Cancel)"""
        lot = 1.0 # Qui andrebbe la logica di Risk Management 1%
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt4.ORDER_TYPE_BUY if action == "BUY" else mt4.ORDER_TYPE_SELL,
            "price": price,
            "magic": 2026001,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC, # Cruciale per istituzionali
        }
        
        start = time.perf_counter()
        result = mt5.order_send(request)
        latency = int((time.perf_counter() - start) * 1000)
        
        # Log su Supabase per analisi slippage
        self.supabase.table("execution_logs").insert({
            "symbol": symbol,
            "latency_ms": latency,
            "requested_price": price,
            "fill_price": result.price if result else 0
        }).execute()

    def run(self):
        print("🏛️ TITAN V2: Esecuzione Istituzionale Avviata...")
        while True:
            for symbol in ["XAUUSD", "BTCUSD"]:
                z, current_price = self.get_kalman_signal(symbol)
                
                # Soglia istituzionale: 2.5 Sigma
                if z > 2.5: self.execute_institutional_order(symbol, "SELL", current_price)
                if z < -2.5: self.execute_institutional_order(symbol, "BUY", current_price)
            
            time.sleep(0.1) # Loop a 10Hz

if __name__ == "__main__":
    engine = TitanInstitutionalEngine()
    engine.run()
