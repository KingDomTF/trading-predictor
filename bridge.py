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
    
    # Trading assets (adjust to your needs)
    ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "US30", "ETHUSD"]
    
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
        Calculate Stop Loss and Take Profit with safety checks
        Returns: (stop_loss, take_profit)
        """
        
        # Ensure volatility is reasonable
        if volatility <= 0 or volatility > entry_price * 0.05:
            # Fallback to 0.3% if volatility is invalid
            volatility = entry_price * 0.003
        
        # Calculate distances
        sl_distance = volatility * 1.5  # 1.5x ATR for stop
        tp_distance = sl_distance * Config.MIN_RR_RATIO  # 2:1 minimum
        
        # Calculate levels based on direction
        if direction == "BUY":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
            
            # Safety check: SL must be below entry
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.997  # 0.3% below
                take_profit = entry_price * 1.006  # 0.6% above
                
        else:  # SELL
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
            
            # Safety check: SL must be above entry
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.003  # 0.3% above
                take_profit = entry_price * 0.994  # 0.6% below
        
        # Final validation
        if stop_loss <= 0 or take_profit <= 0:
            # Emergency fallback
            if direction == "BUY":
                stop_loss = entry_price * 0.997
                take_profit = entry_price * 1.006
            else:
                stop_loss = entry_price * 1.003
                take_profit = entry_price * 0.994
        
        # Round to 2 decimals for clean display
        return round(stop_loss, 2), round(take_profit, 2)
    
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
        Returns: dict with signal details or None
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
        
        # Moving Averages
        ma_fast = np.mean(prices[-Config.MA_FAST:])
        ma_slow = np.mean(prices[-Config.MA_SLOW:])
        
        # ATR (Average True Range) - volatility measure
        price_changes = np.abs(np.diff(prices))
        atr = np.mean(price_changes[-14:]) * 2.0 if len(price_changes) >= 14 else np.std(prices) * 0.5
        
        # RSI (Relative Strength Index)
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
        
        # Strategy 1: Trend Following (Momentum)
        if current_price > ma_slow and ma_fast > ma_slow:
            # Uptrend confirmed
            if rsi > 50 and rsi < 70:
                signal = "BUY"
                reason = "Momentum Trend Up"
                confidence = 85
        
        elif current_price < ma_slow and ma_fast < ma_slow:
            # Downtrend confirmed
            if rsi < 50 and rsi > 30:
                signal = "SELL"
                reason = "Momentum Trend Down"
                confidence = 85
        
        # Strategy 2: Mean Reversion (Oversold/Overbought)
        if rsi < Config.RSI_OVERSOLD:
            signal = "BUY"
            reason = "Oversold Reversal"
            confidence = 75
        
        elif rsi > Config.RSI_OVERBOUGHT:
            signal = "SELL"
            reason = "Overbought Reversal"
            confidence = 75
        
        # === SIGNAL FILTERING ===
        
        # Avoid sending same signal repeatedly
        if signal:
            # Check if signal changed or enough time passed (30 seconds)
            if signal == self.last_signal[symbol]:
                if time.time() - self.last_signal_time[symbol] < 30:
                    signal = None  # Skip duplicate
        
        # === PREPARE RESULT ===
        
        if signal:
            # Calculate Stop Loss and Take Profit
            sl, tp = SafeMath.calculate_levels(current_price, signal, atr)
            
            # Update cache
            self.last_signal[symbol] = signal
            self.last_signal_time[symbol] = time.time()
            
            return {
                'status': 'SIGNAL',
                'signal': signal,
                'entry': current_price,
                'stop_loss': sl,
                'take_profit': tp,
                'confidence': confidence,
                'reason': reason,
                'atr': round(atr, 5),
                'rsi': round(rsi, 1)
            }
        
        else:
            # No signal, market scanning
            return {
                'status': 'SCANNING',
                'message': f'RSI: {rsi:.1f} | Trend: {"UP" if ma_fast > ma_slow else "DOWN"}',
                'rsi': round(rsi, 1)
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
        
        # Rate limiting
        self.last_feed_update = {symbol: 0 for symbol in Config.ASSETS}
        self.last_oracle_update = {symbol: 0 for symbol in Config.ASSETS}
    
    def send_price_feed(self, symbol, price, equity):
        """
        Send price to mt4_feed table (for charts)
        Rate limited to 1 update per second per symbol
        """
        if not self.connected:
            return
        
        # Rate limit check
        if time.time() - self.last_feed_update[symbol] < 1.0:
            return
        
        try:
            self.client.table("mt4_feed").insert({
                "symbol": symbol,
                "price": round(price, 5),
                "equity": round(equity, 2)
            }).execute()
            
            self.last_feed_update[symbol] = time.time()
        except:
            pass  # Silent fail for feed updates
    
    def send_trading_signal(self, symbol, signal_data):
        """
        Send trading signal to ai_oracle table (for app.py display)
        """
        if not self.connected:
            return
        
        try:
            # Prepare payload compatible with your app.py
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
            
            self.last_oracle_update[symbol] = time.time()
            
        except Exception as e:
            logger.error(f"❌ Failed to send signal: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution loop"""
    
    print("\n" + "═" * 70)
    print("🏛️  TITAN V90 ORACLE PRIME - SIMPLIFIED EDITION")
    print("═" * 70)
    print(f"Assets: {', '.join(Config.ASSETS)}")
    print(f"MT4 Path: {Config.MT4_PATH}")
    print("═" * 70 + "\n")
    
    # Validate configuration
    if not Config.validate():
        print("\n❌ Configuration error. Check your .env file")
        return
    
    # Initialize components
    engine = TradingEngine()
    db = DatabaseConnector()
    
    if not db.connected:
        print("\n⚠️  Running without database connection (check SUPABASE credentials)")
    
    logger.info("🚀 System online - Processing market data...")
    
    # Main loop
    iteration = 0
    
    while True:
        iteration += 1
        cycle_start = time.time()
        
        for symbol in Config.ASSETS:
            # Read MT4 JSON file
            file_path = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            
            if not os.path.exists(file_path):
                continue
            
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read().strip()
                
                if not content:
                    continue
                
                data = json.loads(content)
                
                # Extract data
                bid = float(data.get('bid', 0))
                ask = float(data.get('ask', 0))
                equity = float(data.get('equity', 10000))
                
                # Calculate mid price
                if bid > 0 and ask > 0:
                    price = (bid + ask) / 2.0
                else:
                    price = float(data.get('price', 0))
                
                # Validate price
                if not SafeMath.validate_price(price):
                    continue
                
                # Send price feed to database (for charts)
                db.send_price_feed(symbol, price, equity)
                
                # Add to analysis buffer
                engine.add_tick(symbol, price)
                
                # Analyze market
                result = engine.analyze(symbol, price)
                
                # Handle result
                if result['status'] == 'WARMUP':
                    # Still collecting data
                    if iteration % 50 == 0:  # Log every 50 iterations
                        print(f"⏳ {symbol}: {result['progress']} - {result['message']}", end='\r')
                
                elif result['status'] == 'SIGNAL':
                    # NEW TRADING SIGNAL!
                    print(f"\n🎯 {symbol} {result['signal']} @ ${price:.2f}")
                    print(f"   SL: ${result['stop_loss']:.2f} | TP: ${result['take_profit']:.2f}")
                    print(f"   Reason: {result['reason']} | Confidence: {result['confidence']}%")
                    
                    # Send to database
                    db.send_trading_signal(symbol, result)
                
                elif result['status'] == 'SCANNING':
                    # Market scanning
                    if iteration % 100 == 0:  # Log every 100 iterations
                        print(f"👁️  {symbol}: ${price:.2f} | {result['message']}", end='\r')
            
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        # Performance log every 1000 iterations
        if iteration % 1000 == 0:
            logger.info(f"💓 System healthy | Iteration {iteration}")
        
        # CPU throttle (don't burn the processor)
        elapsed = time.time() - cycle_start
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 System shutdown requested")
        print("✅ TITAN V90 stopped gracefully\n")
    except Exception as e:
        logger.critical(f"💀 Fatal error: {e}", exc_info=True)
