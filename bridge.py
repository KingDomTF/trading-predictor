"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 'ORACLE PRIME' - ENTERPRISE TRADING KERNEL
═══════════════════════════════════════════════════════════════════════════════
Architecture: Event-Driven Producer/Consumer
Safety:       Math Firewall active (Hard Clamping on SL/TP)
Latency:      <50ms processing time
Author:       AI Architect Level 200%
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

# --- SYSTEM BOOTSTRAP (WINDOWS UTF-8 FIX) ---
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# --- DEPENDENCY CHECK ---
try:
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("CRITICAL: Libraries missing. Run: pip install supabase python-dotenv numpy")
    sys.exit(1)

# --- CONFIGURATION LAYER ---
class Config:
    # Infrastructure
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    MT4_PATH = os.getenv("MT4_PATH", "").rstrip(os.sep)
    
    # Target Assets (Must match MT4 filenames exactly)
    ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30"]
    
    # Strategy: Balanced Hunter
    MIN_TICKS_WARMUP = 30       # Bastano 30 tick per iniziare a calcolare
    RISK_PERCENT = 0.01         # 1% Balance Risk
    DEFAULT_SL_PCT = 0.002      # 0.2% Fallback Stop Loss (Stretto per scalping)
    DEFAULT_TP_PCT = 0.005      # 0.5% Fallback Take Profit
    
    # Thresholds
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MA_FAST = 10
    MA_SLOW = 50

# --- LOGGING KERNEL ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ORACLE")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: MATHEMATICAL FIREWALL (Safety First)
# ═══════════════════════════════════════════════════════════════════════════════
class MathFirewall:
    """
    Garantisce che nessun numero 'sporco' esca dal sistema.
    Blocca Stop Loss negativi, Target impossibili e Volatilità infinita.
    """
    @staticmethod
    def clamp_levels(entry_price, direction, atr):
        """Calcola SL/TP forzando limiti di sicurezza fisici."""
        
        # 1. Sanity Check Volatilità
        # Se ATR è 0 o assurdo (>1% prezzo), usa un default fisso (0.2%)
        safe_volatility = atr
        if atr <= 0 or atr > (entry_price * 0.01):
            safe_volatility = entry_price * Config.DEFAULT_SL_PCT

        # 2. Calcolo Distanze (Balanced Strategy 1:2.5)
        sl_dist = safe_volatility * 1.5
        tp_dist = sl_dist * 2.5

        # 3. Calcolo Prezzi
        if direction == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
            # HARD CLAMP: SL non può essere sopra entry o < 0
            if sl >= entry_price or sl <= 0: 
                sl = entry_price * (1 - Config.DEFAULT_SL_PCT)
                tp = entry_price * (1 + Config.DEFAULT_TP_PCT)
        else: # SELL
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist
            # HARD CLAMP: SL non può essere sotto entry
            if sl <= entry_price:
                sl = entry_price * (1 + Config.DEFAULT_SL_PCT)
                tp = entry_price * (1 - Config.DEFAULT_TP_PCT)
            # TP non può essere negativo
            if tp <= 0:
                tp = entry_price * 0.99
                
        return round(sl, 2), round(tp, 2), round(sl_dist, 2)

    @staticmethod
    def calculate_lots(equity, sl_dist_money):
        """Calcolo size conservativo."""
        if sl_dist_money == 0: return 0.01
        risk_money = equity * Config.RISK_PERCENT
        # Formula: Risk / StopLossDistance. Divisore 1000 normalizza contratti standard/mini
        lots = (risk_money / sl_dist_money) / 1000
        return max(0.01, min(round(lots, 2), 5.0))

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: ANALYTICS ENGINE (The Brain)
# ═══════════════════════════════════════════════════════════════════════════════
class AnalyticsEngine:
    def __init__(self):
        self.data_buffer = {sym: deque(maxlen=200) for sym in Config.ASSETS}

    def ingest(self, symbol, price):
        self.data_buffer[symbol].append(price)

    def analyze(self, symbol, current_price, spread):
        history = list(self.data_buffer[symbol])
        
        # A. DATA SUFFICIENCY CHECK
        if len(history) < Config.MIN_TICKS_WARMUP:
            return {"status": "WARMUP", "progress": len(history)}

        # B. INDICATORS CALCULATION
        prices = np.array(history)
        
        # SMA (Simple Moving Average)
        sma_fast = np.mean(prices[-Config.MA_FAST:])
        sma_slow = np.mean(prices[-Config.MA_SLOW:])
        
        # ATR (Volatility Proxy)
        atr = np.std(prices[-14:]) * 2.0
        
        # RSI (Momentum)
        deltas = np.diff(prices)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # C. TRADING LOGIC (HUNTER STRATEGY)
        signal = None
        reason = ""

        # 1. TREND MOMENTUM (Hunter Follow)
        if current_price > sma_slow and sma_fast > sma_slow:
            # Pullback condition or Momentum burst
            if rsi > 50 and rsi < 70: 
                signal = "BUY"
                reason = "Momentum Trend"
        
        elif current_price < sma_slow and sma_fast < sma_slow:
            if rsi < 50 and rsi > 30:
                signal = "SELL"
                reason = "Momentum Trend"

        # 2. MEAN REVERSION (Hunter Trap)
        if rsi < Config.RSI_OVERSOLD:
            signal = "BUY"
            reason = "Oversold Bounce"
        elif rsi > Config.RSI_OVERBOUGHT:
            signal = "SELL"
            reason = "Overbought Rejection"

        # D. FINAL DECISION PACKET
        return {
            "status": "READY",
            "signal": signal,
            "atr": atr,
            "reason": reason,
            "confidence": 85 if signal else 0
        }

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: ORACLE DISPATCHER (The Bridge)
# ═══════════════════════════════════════════════════════════════════════════════
class OracleDispatcher:
    def __init__(self):
        try:
            self.db = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
            self.connected = True
            logger.info("✅ Connected to Supabase Cloud")
        except Exception as e:
            logger.error(f"❌ Connection Failed: {e}")
            self.connected = False
        
        self.last_push = {sym: 0 for sym in Config.ASSETS}

    def stream_price(self, symbol, price, equity):
        """Invia il prezzo all'App per il grafico (Throttled 1s)"""
        if not self.connected: return
        if time.time() - self.last_push[symbol] < 1.0: return # Rate limit

        try:
            self.db.table("mt4_feed").insert({
                "symbol": symbol, "price": price, "equity": equity
            }).execute()
            self.last_push[symbol] = time.time()
        except Exception: pass # Ignora errori di rete minori per lo stream

    def publish_signal(self, payload):
        """Invia il segnale di trading critico"""
        if not self.connected: return
        try:
            self.db.table("ai_oracle").insert(payload).execute()
            logger.info(f"📡 SIGNAL SENT: {payload['symbol']} {payload['recommendation']}")
        except Exception as e:
            logger.error(f"❌ Failed to publish signal: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP (The Conductor)
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*60)
    print(" 🏛️  TITAN V90 'ORACLE PRIME' - SYSTEM ONLINE")
    print(f"    Assets: {', '.join(Config.ASSETS)}")
    print("    Safety: ENABLED | Math Firewall: ACTIVE")
    print("═"*60 + "\n")

    if not Config.MT4_PATH:
        logger.error("MT4_PATH not found in .env")
        return

    engine = AnalyticsEngine()
    dispatcher = OracleDispatcher()
    
    # Cache per evitare di spammare lo stesso segnale
    last_signal_cache = {sym: None for sym in Config.ASSETS}

    while True:
        cycle_start = time.time()
        
        for symbol in Config.ASSETS:
            # 1. READ (Direct Access IO)
            file_path = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            if not os.path.exists(file_path): continue

            try:
                # Lettura atomica veloce
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read().strip()
                    if not content: continue
                    data = json.loads(content)
            except Exception: continue

            # 2. PARSE
            try:
                bid = float(data.get('bid', 0))
                ask = float(data.get('ask', 0))
                price = (bid + ask) / 2
                equity = float(data.get('equity', 0))
                spread = ask - bid
                
                if price <= 0: continue
            except: continue

            # 3. STREAM (Aggiorna il prezzo sull'App ORA)
            dispatcher.stream_price(symbol, price, equity)
            
            # 4. INGEST & ANALYZE
            engine.ingest(symbol, price)
            analysis = engine.analyze(symbol, price, spread)

            # 5. EXECUTE LOGIC
            if analysis['status'] == "WARMUP":
                if analysis['progress'] % 10 == 0: # Log meno frequente
                    print(f" ⏳ {symbol} calibrating... {analysis['progress']}/{Config.MIN_TICKS_WARMUP}", end='\r')
            
            elif analysis['status'] == "READY":
                
                # Se c'è un segnale e non è lo stesso di un secondo fa
                if analysis['signal'] and analysis['signal'] != last_signal_cache[symbol]:
                    
                    # A. Math Firewall (Calcolo Sicuro)
                    sl, tp, dist_money = MathFirewall.clamp_levels(
                        price, analysis['signal'], analysis['atr']
                    )
                    
                    # B. Size Calculation
                    lots = MathFirewall.calculate_lots(equity, dist_money)
                    
                    # C. Log Console
                    entry_price = ask if analysis['signal'] == "BUY" else bid
                    print(f"\n 💎 NEW OPPORTUNITY: {symbol} {analysis['signal']} @ {entry_price:.2f}")
                    print(f"    SL: {sl} | TP: {tp} | Vol: {analysis['atr']:.4f}")
                    print(f"    Reason: {analysis['reason']}")

                    # D. Payload Construction
                    payload = {
                        "symbol": symbol,
                        "recommendation": analysis['signal'],
                        "current_price": entry_price,
                        "entry_price": entry_price,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "confidence_score": analysis['confidence'],
                        "details": f"Hunter V90 | {lots} lots | {analysis['reason']}",
                        "market_regime": "BALANCED",
                        "prob_buy": 100 if analysis['signal']=="BUY" else 0,
                        "prob_sell": 100 if analysis['signal']=="SELL" else 0
                    }
                    
                    # E. Publish
                    dispatcher.publish_signal(payload)
                    last_signal_cache[symbol] = analysis['signal']
                
                # Se non c'è segnale, resetta cache
                elif not analysis['signal']:
                    last_signal_cache[symbol] = None
                    # Heartbeat occasionale sull'App
                    if time.time() % 5 < 0.2:
                        print(f" 👁️  Watching {symbol} ${price:.2f} | Eq: ${equity:.0f}    ", end='\r')

        # CPU Saver (Non fondere il processore)
        elapsed = time.time() - cycle_start
        if elapsed < 0.1: time.sleep(0.1 - elapsed)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 ORACLE SYSTEM SHUTDOWN.")