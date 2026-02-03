"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 'ORACLE PRIME' - SIMPLIFIED FOR APP.PY STRUCTURE
═══════════════════════════════════════════════════════════════════════════════
Backend engine optimized for your Streamlit dashboard structure
Generates clean signals compatible with your app.py visualization
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

# Windows UTF-8 fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Dependencies check
try:
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("❌ Missing libraries. Run: pip install supabase python-dotenv numpy")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """System configuration from .env file"""
    
    # Supabase credentials
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # MT4 data path
    MT4_PATH = os.getenv("MT4_PATH", "").rstrip(os.sep)
    
    # --- LISTA ASSET AGGIORNATA ---
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    
    # Strategy parameters
    MIN_TICKS_WARMUP = 30          # Minimum data points before trading
    RISK_PERCENT = 0.01            # 1% risk per trade
    
    # Technical indicators
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MA_FAST = 10
    MA_SLOW = 50
    
    # Safety limits
    MAX_SL_PERCENT = 0.03          # Max 3% stop loss
    MIN_RR_RATIO = 2.0             # Minimum 2:1 risk/reward
    
    @classmethod
    def validate(cls):
        """Check if configuration is valid"""
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            print("❌ Missing SUPABASE_URL or SUPABASE_KEY in .env")
            return False
        if not cls.MT4_PATH:
            print("❌ Missing MT4_PATH in .env")
            return False
        if not os.path.exists(cls.MT4_PATH):
            print(f"❌ MT4_PATH does not exist: {cls.MT4_PATH}")
            return False
        return True

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TITAN")

# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL SAFETY LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class SafeMath:
    """Ensures all calculations produce valid, safe values"""
    
    @staticmethod
    def calculate_levels(entry_price, direction, volatility):
        """
        Calculate Stop Loss and Take Profit with ROBUST safety checks
        Returns: (stop_loss, take_profit)
        """
        
        # === SAFETY CHECK 1: Validate volatility ===
        min_volatility = entry_price * 0.001  # Min 0.1%
        max_volatility = entry_price * 0.05   # Max 5%
        
        if volatility <= 0 or volatility < min_volatility or volatility > max_volatility:
            volatility = entry_price * 0.003  # Fallback 0.3%
        
        # === CALCULATE DISTANCES ===
        sl_distance = volatility * 1.5
        tp_distance = sl_distance * Config.MIN_RR_RATIO
        
        # === SAFETY CHECK 2: Minimum distances ===
        min_sl_distance = entry_price * 0.002  # Min 0.2%
        if sl_distance < min_sl_distance:
            sl_distance = min_sl_distance
            tp_distance = sl_distance * Config.MIN_RR_RATIO
        
        # === CALCULATE LEVELS ===
        if direction == "BUY":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SELL
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
            
        return round(stop_loss, 5), round(take_profit, 5)
    
    @staticmethod
    def validate_price(price):
        """Check if price is valid"""
        return price > 0 and price < 1000000  # Reasonable range

# ═══════════════════════════════════════════════════════════════════════════════
# TRADING BRAIN
# ═══════════════════════════════════════════════════════════════════════════════

class TradingEngine:
    """Analyzes market data and generates trading signals"""
    
    def __init__(self):
        # Price history buffer for each symbol
        self.price_buffer = {symbol: deque(maxlen=200) for symbol in Config.ASSETS}
        
        # Last signal sent (avoid spam)
        self.last_signal = {symbol: None for symbol in Config.ASSETS}
        self.last_signal_time = {symbol: 0 for symbol in Config.ASSETS}
    
    def add_tick(self, symbol, price):
        """Add new price to history"""
        if symbol in self.price_buffer:
            self.price_buffer[symbol].append(price)
    
    def analyze(self, symbol, current_price):
        """
        Analyze market and return trading signal
        """
        # Get price history
        history = list(self.price_buffer[symbol])
        
        # Need minimum data
        if len(history) < Config.MIN_TICKS_WARMUP:
            return {
                'status': 'WARMUP',
                'progress': f"{len(history)}/{Config.MIN_TICKS_WARMUP}",
                'message': 'Collecting data...'
            }
        
        # Convert to numpy for calculations
        prices = np.array(history)
        
        # === TECHNICAL INDICATORS ===
        ma_fast = np.mean(prices[-Config.MA_FAST:])
        ma_slow = np.mean(prices[-Config.MA_SLOW:])
        
        # ATR (Volatility)
        price_changes = np.abs(np.diff(prices))
        atr = np.mean(price_changes[-14:]) * 2.0 if len(price_changes) >= 14 else np.std(prices) * 0.5
        
        # RSI
        deltas = np.diff(prices)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0001
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # === TRADING LOGIC ===
        signal = None
        reason = ""
        confidence = 0
        
        # Trend Following
        if current_price > ma_slow and ma_fast > ma_slow:
            if rsi > 50 and rsi < 70:
                signal = "BUY"
                reason = "Momentum Trend Up"
                confidence = 85
        elif current_price < ma_slow and ma_fast < ma_slow:
            if rsi < 50 and rsi > 30:
                signal = "SELL"
                reason = "Momentum Trend Down"
                confidence = 85
        
        # Mean Reversion
        if rsi < Config.RSI_OVERSOLD:
            signal = "BUY"
            reason = "Oversold Reversal"
            confidence = 75
        elif rsi > Config.RSI_OVERBOUGHT:
            signal = "SELL"
            reason = "Overbought Reversal"
            confidence = 75
        
        # === SIGNAL FILTERING ===
        if signal:
            # Check for duplicates or timing
            if signal == self.last_signal[symbol]:
                if time.time() - self.last_signal_time[symbol] < 30:
                    signal = None  # Skip duplicate
        
        # === RESULT ===
        if signal:
            sl, tp = SafeMath.calculate_levels(current_price, signal, atr)
            self.last_signal[symbol] = signal
            self.last_signal_time[symbol] = time.time()
            
            return {
                'status': 'SIGNAL',
                'signal': signal,
                'entry': current_price,
                'stop_loss': sl,
                'take_profit': tp,
                'confidence': confidence,
                'reason': reason
            }
        else:
            return {
                'status': 'SCANNING',
                'message': f'RSI: {rsi:.1f} | Trend: {"UP" if ma_fast > ma_slow else "DOWN"}'
            }

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseConnector:
    """Handles all Supabase communication"""
    
    def __init__(self):
        try:
            self.client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
            self.connected = True
            logger.info("✅ Connected to Supabase")
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            self.connected = False
        
        self.last_feed_update = {symbol: 0 for symbol in Config.ASSETS}
    
    def send_price_feed(self, symbol, price, equity):
        if not self.connected: return
        if time.time() - self.last_feed_update[symbol] < 1.0: return
        
        try:
            self.client.table("mt4_feed").insert({
                "symbol": symbol, "price": round(price, 5), "equity": round(equity, 2)
            }).execute()
            self.last_feed_update[symbol] = time.time()
        except: pass
    
    def send_trading_signal(self, symbol, signal_data):
        if not self.connected: return
        
        try:
            payload = {
                "symbol": symbol,
                "recommendation": signal_data.get('signal', 'WAIT'),
                "current_price": round(signal_data.get('entry', 0), 2),
                "entry_price": round(signal_data.get('entry', 0), 2),
                "stop_loss": round(signal_data.get('stop_loss', 0), 2),
                "take_profit": round(signal_data.get('take_profit', 0), 2),
                "confidence_score": signal_data.get('confidence', 0),
                "details": f"TITAN V90 | {signal_data.get('reason', 'Scanning')}",
                "market_regime": "BALANCED_HUNTER",
                "prob_buy": 100 if signal_data.get('signal') == 'BUY' else 0,
                "prob_sell": 100 if signal_data.get('signal') == 'SELL' else 0
            }
            self.client.table("ai_oracle").insert(payload).execute()
            logger.info(f"📡 Signal sent: {symbol} {signal_data.get('signal', 'WAIT')}")
        except Exception as e:
            logger.error(f"❌ Failed to send signal: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n🏛️  TITAN V90 ORACLE PRIME\n{'═'*30}\nAssets: {', '.join(Config.ASSETS)}\nMT4 Path: {Config.MT4_PATH}\n")
    
    if not Config.validate(): return
    
    engine = TradingEngine()
    db = DatabaseConnector()
    
    logger.info("🚀 System online - Processing market data...")
    iteration = 0
    
    while True:
        iteration += 1
        cycle_start = time.time()
        
        for symbol in Config.ASSETS:
            file_path = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            if not os.path.exists(file_path): continue
            
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read().strip()
                if not content: continue
                
                data = json.loads(content)
                bid = float(data.get('bid', 0))
                ask = float(data.get('ask', 0))
                equity = float(data.get('equity', 10000))
                price = (bid + ask) / 2.0 if bid > 0 else float(data.get('price', 0))
                
                if not SafeMath.validate_price(price): continue
                
                db.send_price_feed(symbol, price, equity)
                engine.add_tick(symbol, price)
                result = engine.analyze(symbol, price)
                
                if result['status'] == 'SIGNAL':
                    print(f"\n🎯 {symbol} {result['signal']} @ ${price:.2f} | Conf: {result['confidence']}%")
                    db.send_trading_signal(symbol, result)
                elif result['status'] == 'SCANNING':
                    if iteration % 100 == 0:
                        print(f"👁️  {symbol}: ${price:.2f} | {result['message']}", end='\r')
            
            except Exception: continue
        
        if iteration % 1000 == 0: logger.info(f"💓 System healthy | Iteration {iteration}")
        
        elapsed = time.time() - cycle_start
        if elapsed < 0.1: time.sleep(0.1 - elapsed)

if __name__ == "__main__":
    main()
