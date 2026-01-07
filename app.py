import time
import json
import os
import glob
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import numpy as np
import yfinance as yf
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential

# ================= CONFIGURATION MANAGEMENT =================
@dataclass
class Config:
    """Centralized configuration with environment variable support"""
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://gkffitfxqhxifibfwsfx.supabase.co")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG")
    MT4_PATH: str = os.getenv("MT4_PATH", r"C:\Users\dcbat\AppData\Roaming\MetaQuotes\Terminal\B8925BF731C22E88F33C7A8D7CD3190E\MQL4\Files")
    
    CACHE_TTL_MINUTES: int = 30
    MAX_CACHE_SIZE: int = 50
    POLL_INTERVAL_SEC: float = 1.0
    LOG_LEVEL: str = "INFO"
    
    SYMBOL_MAP: Dict[str, str] = None
    MACRO_TICKERS: Dict[str, str] = None
    
    def __post_init__(self):
        if not self.SUPABASE_KEY or self.SUPABASE_KEY == "YOUR_KEY_HERE":
            raise ValueError("SUPABASE_KEY must be set. Use environment variable or update default value.")
            
        self.SYMBOL_MAP = {
            "XAUUSD": "GC=F", "GOLD": "GC=F", 
            "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
            "US500": "ES=F", "SPX": "^GSPC", 
            "XAGUSD": "SI=F", "EURUSD": "EURUSD=X"
        }
        
        self.MACRO_TICKERS = {
            "DXY": "DX-Y.NYB", 
            "US10Y": "^TNX", 
            "VIX": "^VIX"
        }

# ================= LOGGING SETUP =================
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging with rotation"""
    logger = logging.getLogger("TitanEngine")
    logger.setLevel(getattr(logging, level))
    
    # Console handler with UTF-8 encoding for Windows
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    # Force UTF-8 encoding on Windows
    if hasattr(console.stream, 'reconfigure'):
        try:
            console.stream.reconfigure(encoding='utf-8')
        except:
            pass
    
    logger.addHandler(console)
    
    # File handler for debugging
    try:
        file_handler = logging.FileHandler('titan_engine.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        ))
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create log file: {e}")
    
    return logger

# ================= DATA MODELS =================
@dataclass
class MarketSignal:
    """Structured signal output"""
    symbol: str
    current_price: float
    recommendation: str  # BUY, SELL, WAIT
    confidence_score: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    prob_buy: float
    prob_sell: float
    market_regime: str
    details: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database insertion"""
        return asdict(self)

# ================= ENHANCED CACHE WITH LRU =================
class MarketDataCache:
    """Intelligent cache with automatic eviction and incremental updates"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cache: Dict[str, Dict] = {}
        self.macro_cache: Optional[Dict] = None
        self.last_macro_update = datetime.min
        self.update_interval = timedelta(minutes=config.CACHE_TTL_MINUTES)
    
    def _evict_old_entries(self):
        """LRU eviction to prevent memory bloat"""
        if len(self.cache) > self.config.MAX_CACHE_SIZE:
            oldest = sorted(self.cache.items(), key=lambda x: x[1]['timestamp'])[0][0]
            del self.cache[oldest]
            self.logger.debug(f"Evicted cache entry: {oldest}")
    
    def _validate_dataframe(self, df: pd.DataFrame, name: str) -> bool:
        """Validate downloaded data"""
        if df is None or df.empty:
            self.logger.warning(f"Empty dataframe received: {name}")
            return False
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Missing required columns in {name}")
            return False
        
        return True
    
    def _clean_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Safely clean and normalize dataframe"""
        if df is None or df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Remove NaN rows
        df = df.dropna()
        
        return df if not df.empty else None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _download_with_retry(self, ticker: str, **kwargs) -> Optional[pd.DataFrame]:
        """Download with exponential backoff retry"""
        try:
            df = yf.download(ticker, progress=False, auto_adjust=False, timeout=10, **kwargs)
            return self._clean_dataframe(df)
        except Exception as e:
            self.logger.warning(f"Download attempt failed for {ticker}: {str(e)[:100]}")
            raise
    
    def get_macro_context(self) -> Optional[Dict[str, Any]]:
        """Fetch macro indicators with caching"""
        now = datetime.now()
        
        if self.macro_cache and (now - self.last_macro_update) < self.update_interval:
            return self.macro_cache
        
        try:
            tickers = list(self.config.MACRO_TICKERS.values())
            df = self._download_with_retry(tickers, period="5d", interval="1d")
            
            if df is None or not self._validate_dataframe(df, "macro"):
                return self.macro_cache  # Return stale cache on failure
            
            # Extract close prices only
            close_df = df['Close'] if 'Close' in df.columns else df
            
            self.macro_cache = self._process_macro(close_df)
            self.last_macro_update = now
            self.logger.info("Macro context updated successfully")
            
        except Exception as e:
            self.logger.error(f"Macro context update failed: {e}")
        
        return self.macro_cache
    
    def _process_macro(self, df: pd.DataFrame) -> Optional[Dict]:
        """Process macro indicators safely"""
        try:
            ctx = {}
            
            dxy_col = self.config.MACRO_TICKERS['DXY']
            vix_col = self.config.MACRO_TICKERS['VIX']
            tnx_col = self.config.MACRO_TICKERS['US10Y']
            
            # USD strength
            if dxy_col in df.columns and len(df[dxy_col]) >= 2:
                dxy_curr = df[dxy_col].iloc[-1]
                dxy_prev = df[dxy_col].iloc[-2]
                ctx['USD_TREND'] = "BULL" if dxy_curr > dxy_prev else "BEAR"
                ctx['DXY_VALUE'] = float(dxy_curr)
            
            # Risk sentiment
            if vix_col in df.columns:
                vix_curr = df[vix_col].iloc[-1]
                ctx['RISK_ON'] = bool(vix_curr < 20)
                ctx['VIX_VALUE'] = float(vix_curr)
            
            # Rates direction
            if tnx_col in df.columns and len(df[tnx_col]) >= 2:
                tnx_curr = df[tnx_col].iloc[-1]
                tnx_prev = df[tnx_col].iloc[-2]
                ctx['RATES_TREND'] = "BULL" if tnx_curr > tnx_prev else "BEAR"
                ctx['US10Y_VALUE'] = float(tnx_curr)
            
            return ctx if ctx else None
            
        except Exception as e:
            self.logger.error(f"Macro processing error: {e}")
            return None
    
    def get_asset_history(self, yf_symbol: str) -> Optional[Dict]:
        """Fetch asset history with intelligent caching"""
        now = datetime.now()
        
        # Check cache validity
        if yf_symbol in self.cache:
            cached = self.cache[yf_symbol]
            if (now - cached['timestamp']) < self.update_interval:
                return cached
        
        try:
            self.logger.debug(f"Downloading fresh data for {yf_symbol}")
            
            # Download only necessary data with error handling per timeframe
            daily = hourly = minute = None
            
            try:
                daily = self._download_with_retry(yf_symbol, period="1y", interval="1d")
            except Exception as e:
                self.logger.warning(f"Daily data failed for {yf_symbol}: {str(e)[:50]}")
            
            try:
                hourly = self._download_with_retry(yf_symbol, period="60d", interval="60m")
            except Exception as e:
                self.logger.warning(f"Hourly data failed for {yf_symbol}: {str(e)[:50]}")
            
            try:
                minute = self._download_with_retry(yf_symbol, period="5d", interval="15m")
            except Exception as e:
                self.logger.warning(f"Minute data failed for {yf_symbol}: {str(e)[:50]}")
            
            # Require at least daily and hourly data
            if daily is None or hourly is None:
                self.logger.error(f"Insufficient data for {yf_symbol}, using stale cache")
                return self.cache.get(yf_symbol)
            
            # Use hourly as fallback if minute data fails
            if minute is None:
                self.logger.warning(f"Using hourly data as minute fallback for {yf_symbol}")
                minute = hourly
            
            # Store in cache
            self.cache[yf_symbol] = {
                'daily': daily,
                'hourly': hourly,
                'minute': minute,
                'timestamp': now
            }
            
            self._evict_old_entries()
            self.logger.info(f"Cache updated: {yf_symbol}")
            
            return self.cache[yf_symbol]
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {yf_symbol}: {e}")
            return self.cache.get(yf_symbol)  # Return stale cache on failure

# ================= ENHANCED TITAN ENGINE =================
class TitanEngine:
    """Core analysis engine with robust signal generation"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_manager = MarketDataCache(config, logger)
    
    @staticmethod
    def _safe_indicator(func, *args, default=0.0):
        """Wrapper for safe indicator calculation"""
        try:
            result = func(*args)
            return default if pd.isna(result) or np.isinf(result) else result
        except:
            return default
    
    def calculate_indicators(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Calculate technical indicators without modifying original dataframe"""
        try:
            # Work on copy to avoid warnings
            df = df.copy()
            close = df['Close'].copy()
            
            # EMAs
            ema_21 = self._safe_indicator(lambda: close.ewm(span=21, adjust=False).mean().iloc[-1])
            ema_50 = self._safe_indicator(lambda: close.ewm(span=50, adjust=False).mean().iloc[-1])
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = self._safe_indicator(lambda: (100 - (100 / (1 + rs))).iloc[-1], default=50.0)
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = self._safe_indicator(lambda: tr.rolling(14).mean().iloc[-1], default=current_price * 0.01)
            
            return {
                'ema_21': ema_21,
                'ema_50': ema_50,
                'rsi': rsi,
                'atr': max(atr, current_price * 0.005)  # Minimum 0.5% ATR
            }
            
        except Exception as e:
            self.logger.error(f"Indicator calculation failed: {e}")
            return {
                'ema_21': current_price,
                'ema_50': current_price,
                'rsi': 50.0,
                'atr': current_price * 0.01
            }
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """Volume-weighted average price"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
            return float(vwap.iloc[-1])
        except:
            return float(df['Close'].iloc[-1])
    
    def determine_regime(self, df: pd.DataFrame, current_price: float, ema_diff: float) -> str:
        """Classify market regime"""
        try:
            mean = df['Close'].rolling(50).mean().iloc[-1]
            std = df['Close'].rolling(50).std().iloc[-1]
            z_score = abs((current_price - mean) / (std + 1e-9))
            
            if z_score > 2.5:
                return "VOLATILE"
            elif abs(ema_diff) > current_price * 0.002:
                return "TRENDING"
            else:
                return "RANGING"
        except:
            return "UNKNOWN"
    
    def analyze(self, symbol: str, live_price: float) -> Optional[MarketSignal]:
        """Generate trading signal with comprehensive analysis"""
        
        yf_symbol = self.config.SYMBOL_MAP.get(symbol, symbol)
        
        # Fetch data
        history = self.data_manager.get_asset_history(yf_symbol)
        macro = self.data_manager.get_macro_context()
        
        if not history:
            self.logger.warning(f"No historical data for {symbol}")
            return None
        
        try:
            daily = history['daily']
            hourly = history['hourly']
            minute = history['minute']
            
            # === TREND ANALYSIS ===
            sma_200 = daily['Close'].rolling(200).mean().iloc[-1]
            trend_long = "BULL" if live_price > sma_200 else "BEAR"
            
            # === INTRADAY ANALYSIS ===
            vwap = self.calculate_vwap(hourly)
            indicators = self.calculate_indicators(minute, live_price)
            
            ema_21 = indicators['ema_21']
            ema_50 = indicators['ema_50']
            rsi = indicators['rsi']
            atr = indicators['atr']
            
            # === SCORING SYSTEM ===
            score_bull = 0
            score_bear = 0
            
            # Long-term trend (20 points)
            if trend_long == "BULL":
                score_bull += 20
            else:
                score_bear += 20
            
            # VWAP position (25 points)
            if live_price > vwap:
                score_bull += 25
            else:
                score_bear += 25
            
            # EMA crossover (25 points)
            if ema_21 > ema_50:
                score_bull += 25
            else:
                score_bear += 25
            
            # RSI filter (10 points)
            if 30 < rsi < 70:
                score_bull += 5
                score_bear += 5
            elif rsi < 30:
                score_bull += 10
            elif rsi > 70:
                score_bear += 10
            
            # Macro context (20 points)
            if macro:
                if symbol in ["XAUUSD", "GOLD"]:
                    if macro.get('USD_TREND') == "BEAR":
                        score_bull += 20
                    else:
                        score_bear += 20
                elif symbol in ["BTCUSD", "US500", "SPX"]:
                    if macro.get('RISK_ON'):
                        score_bull += 20
                    else:
                        score_bear += 20
            
            # === SIGNAL GENERATION ===
            signal = "WAIT"
            final_score = 50
            detail = f"Trend: {trend_long} | RSI: {rsi:.1f} | VWAP: {vwap:.2f}"
            
            if score_bull >= 70 and rsi < 70:
                signal = "BUY"
                final_score = score_bull
                detail = f"Bullish Confluence | {detail}"
            elif score_bear >= 70 and rsi > 30:
                signal = "SELL"
                final_score = score_bear
                detail = f"Bearish Confluence | {detail}"
            
            # === RISK MANAGEMENT ===
            regime = self.determine_regime(hourly, live_price, ema_21 - ema_50)
            
            # Dynamic SL/TP based on regime
            atr_multiplier_sl = 2.0 if regime != "VOLATILE" else 2.5
            atr_multiplier_tp = 3.0 if regime != "VOLATILE" else 3.5
            
            if signal == "BUY":
                sl = live_price - (atr * atr_multiplier_sl)
                tp = live_price + (atr * atr_multiplier_tp)
            elif signal == "SELL":
                sl = live_price + (atr * atr_multiplier_sl)
                tp = live_price - (atr * atr_multiplier_tp)
            else:
                sl = tp = live_price
            
            rr = round(abs(tp - live_price) / abs(live_price - sl), 2) if sl != live_price else 0
            
            # === BUILD SIGNAL ===
            return MarketSignal(
                symbol=symbol,
                current_price=round(live_price, 2),
                recommendation=signal,
                confidence_score=final_score,
                entry_price=round(live_price, 2),
                stop_loss=round(sl, 2),
                take_profit=round(tp, 2),
                risk_reward=rr,
                prob_buy=final_score if signal == "BUY" else (100 - final_score if signal == "SELL" else 50),
                prob_sell=final_score if signal == "SELL" else (100 - final_score if signal == "BUY" else 50),
                market_regime=regime,
                details=detail
            )
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)
            return None

# ================= DATABASE MANAGER =================
class DatabaseManager:
    """Handle all database operations with retry logic"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self._connect()
    
    def _connect(self):
        """Initialize Supabase client"""
        try:
            self.client = create_client(self.config.SUPABASE_URL, self.config.SUPABASE_KEY)
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def insert_feed(self, symbol: str, price: float, equity: float):
        """Insert MT4 feed data"""
        try:
            self.client.table("mt4_feed").insert({
                "symbol": symbol,
                "price": price,
                "equity": equity
            }).execute()
        except Exception as e:
            self.logger.error(f"Feed insert failed: {e}")
            raise
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    def insert_signal(self, signal: MarketSignal):
        """Insert AI signal"""
        try:
            self.client.table("ai_oracle").insert(signal.to_dict()).execute()
        except Exception as e:
            self.logger.error(f"Signal insert failed: {e}")
            raise

# ================= MAIN ORCHESTRATOR =================
def main():
    """Main execution loop with graceful error handling"""
    
    # Initialize
    config = Config()
    logger = setup_logging(config.LOG_LEVEL)
    
    logger.info("="*60)
    logger.info("TITAN V7.0 OPTIMUM - PRODUCTION ENGINE")
    logger.info("="*60)
    logger.info(f"Watch Folder: {config.MT4_PATH}")
    logger.info(f"Poll Interval: {config.POLL_INTERVAL_SEC}s")
    logger.info(f"Cache TTL: {config.CACHE_TTL_MINUTES}min")
    logger.info("="*60)
    
    # Initialize components
    try:
        db = DatabaseManager(config, logger)
        engine = TitanEngine(config, logger)
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return
    
    # Preload cache
    logger.info("Preloading market data cache...")
    engine.data_manager.get_macro_context()
    logger.info("Cache preloaded successfully")
    
    # File tracking
    last_files_mod = {}
    spinner = ["|", "/", "-", "\\"]
    idx = 0
    
    while True:
        try:
            # Check directory existence
            if not os.path.exists(config.MT4_PATH):
                logger.warning(f"Directory not found: {config.MT4_PATH}")
                time.sleep(5)
                continue
            
            # Scan for data files
            files = glob.glob(os.path.join(config.MT4_PATH, "*_data.json"))
            activity = False
            
            for file_path in files:
                try:
                    current_mod = os.path.getmtime(file_path)
                    
                    # Skip if not modified
                    if current_mod == last_files_mod.get(file_path, 0):
                        continue
                    
                    last_files_mod[file_path] = current_mod
                    activity = True
                    
                    # Read and parse file
                    with open(file_path, "r", encoding='utf-8-sig') as f:
                        content = f.read().strip()
                    
                    if not content:
                        continue
                    
                    data = json.loads(content)
                    
                    # Extract and clean data
                    symbol = data.get("symbol", "").replace(".m", "").replace(".pro", "")
                    price = float(data.get("price", 0))
                    equity = float(data.get("equity", 0))
                    
                    if not symbol or price <= 0:
                        logger.warning(f"Invalid data in {file_path}")
                        continue
                    
                    # Store feed
                    db.insert_feed(symbol, price, equity)
                    logger.info(f"[{symbol}] Price: ${price:.2f}")
                    
                    # Generate signal
                    signal = engine.analyze(symbol, price)
                    
                    if signal:
                        try:
                            db.insert_signal(signal)
                            
                            if signal.recommendation != "WAIT":
                                logger.info(
                                    f">>> {signal.recommendation} | "
                                    f"Confidence: {signal.confidence_score}% | "
                                    f"R:R {signal.risk_reward}:1 | "
                                    f"Regime: {signal.market_regime}"
                                )
                            else:
                                logger.info("WAIT - No clear setup")
                        except Exception as e:
                            logger.error(f"Failed to store signal for {symbol}: {str(e)[:100]}")
                    else:
                        logger.info("Building historical context...")
                
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in {file_path}")
                except Exception as e:
                    logger.error(f"Processing error for {file_path}: {e}")
            
            # Idle indicator
            if not activity:
                idx = (idx + 1) % len(spinner)
                print(f"\r{spinner[idx]} Engine running...", end="", flush=True)
            
            time.sleep(config.POLL_INTERVAL_SEC)
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            break
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}", exc_info=True)
            time.sleep(5)
    
    logger.info("TITAN Engine stopped gracefully")

if __name__ == "__main__":
    main()
