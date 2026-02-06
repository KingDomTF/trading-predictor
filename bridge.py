import os, sys, time, json, logging, numpy as np
from collections import deque
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    MT4_PATH = os.getenv("MT4_PATH", "").rstrip(os.sep)
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    
    # Parametri per rilevamento istituzionali
    TICK_BUFFER_SIZE = 200
    MIN_TICKS = 100
    SIGNAL_COOLDOWN = 300  # 5 minuti tra segnali

class InstitutionalDetector:
    """
    Rileva accumulo/distribuzione istituzionale tramite:
    1. Order Flow Imbalance (OFI)
    2. Volume Absorption Analysis
    3. Wyckoff Patterns
    """
    
    def __init__(self):
        self.price_buffer = {s: deque(maxlen=Config.TICK_BUFFER_SIZE) for s in Config.ASSETS}
        self.bid_buffer = {s: deque(maxlen=Config.TICK_BUFFER_SIZE) for s in Config.ASSETS}
        self.ask_buffer = {s: deque(maxlen=Config.TICK_BUFFER_SIZE) for s in Config.ASSETS}
        self.volume_buffer = {s: deque(maxlen=Config.TICK_BUFFER_SIZE) for s in Config.ASSETS}
        self.last_signal_time = {s: 0 for s in Config.ASSETS}
        
        # Stati di accumulo/distribuzione
        self.phase_tracker = {s: {'phase': 'NEUTRAL', 'strength': 0, 'duration': 0} for s in Config.ASSETS}
        
    def update_buffers(self, symbol, bid, ask, volume):
        """Aggiorna i buffer con nuovi dati tick"""
        mid_price = (bid + ask) / 2
        self.price_buffer[symbol].append(mid_price)
        self.bid_buffer[symbol].append(bid)
        self.ask_buffer[symbol].append(ask)
        self.volume_buffer[symbol].append(volume)
    
    def calculate_order_flow_imbalance(self, symbol):
        """
        Calcola OFI (Order Flow Imbalance) secondo formula accademica:
        OFI = Δ(volume_bid) - Δ(volume_ask)
        
        Imbalance positivo = pressione d'acquisto (accumulo)
        Imbalance negativo = pressione di vendita (distribuzione)
        """
        if len(self.bid_buffer[symbol]) < 2:
            return 0
        
        bids = list(self.bid_buffer[symbol])
        asks = list(self.ask_buffer[symbol])
        volumes = list(self.volume_buffer[symbol])
        
        # Calcola differenze di volume pesate per movimento bid/ask
        ofi_values = []
        for i in range(1, len(bids)):
            bid_delta = bids[i] - bids[i-1]
            ask_delta = asks[i] - asks[i-1]
            vol = volumes[i]
            
            # Logica OFI: se bid sale e ask rimane = buying pressure
            if bid_delta > 0 and ask_delta >= 0:
                ofi_values.append(vol)  # Buy pressure
            elif bid_delta <= 0 and ask_delta < 0:
                ofi_values.append(-vol)  # Sell pressure
            else:
                ofi_values.append(0)
        
        return np.mean(ofi_values[-30:]) if ofi_values else 0  # Media ultimi 30 tick
    
    def detect_volume_absorption(self, symbol):
        """
        Rileva ABSORPTION: molto volume senza movimento di prezzo
        Segnale chiave: istituzionali assorbono ordini retail
        """
        if len(self.price_buffer[symbol]) < 50:
            return False, 0
        
        prices = np.array(list(self.price_buffer[symbol])[-50:])
        volumes = np.array(list(self.volume_buffer[symbol])[-50:])
        
        # Calcola volatilità prezzo vs volume
        price_range = np.ptp(prices[-20:])  # Range ultimi 20 tick
        avg_volume = np.mean(volumes[-20:])
        prev_avg_volume = np.mean(volumes[-50:-20])
        
        # ABSORPTION = alto volume + bassa volatilità
        is_high_volume = avg_volume > prev_avg_volume * 1.5
        is_low_volatility = price_range < np.std(prices) * 0.8
        
        absorption_strength = (avg_volume / prev_avg_volume) if is_high_volume and is_low_volatility else 0
        
        return (is_high_volume and is_low_volatility), absorption_strength
    
    def detect_spring_or_upthrust(self, symbol):
        """
        SPRING (Wyckoff): falsa rottura al ribasso -> poi recupero = BUY setup
        UPTHRUST: falsa rottura al rialzo -> poi crollo = SELL setup
        """
        if len(self.price_buffer[symbol]) < 100:
            return None, 0
        
        prices = np.array(list(self.price_buffer[symbol]))
        volumes = np.array(list(self.volume_buffer[symbol]))
        
        # Trova range consolidamento (ultimi 60-80 tick)
        consolidation = prices[-80:-20]
        support = np.percentile(consolidation, 10)
        resistance = np.percentile(consolidation, 90)
        
        recent_prices = prices[-20:]
        recent_volumes = volumes[-20:]
        
        # SPRING: va sotto support poi recupera con volume
        spring_detected = False
        if np.min(recent_prices[:10]) < support:  # Rompe sotto
            if recent_prices[-1] > support and np.mean(recent_volumes) > np.mean(volumes[-80:-20]) * 1.3:
                spring_detected = True
                return "SPRING", 90  # Alta confidenza
        
        # UPTHRUST: va sopra resistance poi crolla con volume
        upthrust_detected = False
        if np.max(recent_prices[:10]) > resistance:  # Rompe sopra
            if recent_prices[-1] < resistance and np.mean(recent_volumes) > np.mean(volumes[-80:-20]) * 1.3:
                upthrust_detected = True
                return "UPTHRUST", 90
        
        return None, 0
    
    def detect_wyckoff_phase(self, symbol):
        """
        Identifica fase Wyckoff attuale:
        - ACCUMULATION: volume alto + range stretto + OFI positivo
        - DISTRIBUTION: volume alto + range stretto + OFI negativo
        - MARKUP/MARKDOWN: trend chiaro con volume
        """
        if len(self.price_buffer[symbol]) < Config.MIN_TICKS:
            return "NEUTRAL", 0
        
        prices = np.array(list(self.price_buffer[symbol]))
        volumes = np.array(list(self.volume_buffer[symbol]))
        
        # Calcola metriche
        price_range = np.ptp(prices[-50:])
        volatility = np.std(prices[-50:])
        avg_volume = np.mean(volumes[-50:])
        ofi = self.calculate_order_flow_imbalance(symbol)
        
        # Trend check
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices[-50:])
        
        is_consolidating = price_range < volatility * 1.5
        is_high_volume = avg_volume > np.mean(volumes) * 1.2
        
        # ACCUMULATION
        if is_consolidating and is_high_volume and ofi > 0 and ma_20 >= ma_50 * 0.998:
            strength = min(int((ofi / max(abs(ofi), 1)) * 100), 95)
            return "ACCUMULATION", strength
        
        # DISTRIBUTION
        elif is_consolidating and is_high_volume and ofi < 0 and ma_20 <= ma_50 * 1.002:
            strength = min(int((abs(ofi) / max(abs(ofi), 1)) * 100), 95)
            return "DISTRIBUTION", strength
        
        # MARKUP (uptrend)
        elif ma_20 > ma_50 * 1.005 and ofi > 0:
            return "MARKUP", 70
        
        # MARKDOWN (downtrend)
        elif ma_20 < ma_50 * 0.995 and ofi < 0:
            return "MARKDOWN", 70
        
        return "NEUTRAL", 50
    
    def analyze(self, symbol, bid, ask, volume):
        """
        Analisi principale che combina tutti i rilevatori
        """
        self.update_buffers(symbol, bid, ask, volume)
        
        if len(self.price_buffer[symbol]) < Config.MIN_TICKS:
            return None
        
        # Cooldown check
        if time.time() - self.last_signal_time[symbol] < Config.SIGNAL_COOLDOWN:
            return None
        
        mid_price = (bid + ask) / 2
        
        # 1. Calcola OFI
        ofi = self.calculate_order_flow_imbalance(symbol)
        
        # 2. Rileva absorption
        is_absorbing, absorption_strength = self.detect_volume_absorption(symbol)
        
        # 3. Rileva Spring/Upthrust
        wyckoff_event, event_conf = self.detect_spring_or_upthrust(symbol)
        
        # 4. Identifica fase Wyckoff
        phase, phase_strength = self.detect_wyckoff_phase(symbol)
        
        # Update phase tracker
        self.phase_tracker[symbol]['phase'] = phase
        self.phase_tracker[symbol]['strength'] = phase_strength
        
        signal = None
        confidence = 0
        reason = ""
        
        # SEGNALI BUY
        if wyckoff_event == "SPRING":
            signal = "BUY"
            confidence = event_conf
            reason = f"Wyckoff SPRING detected | Phase: {phase} ({phase_strength}%)"
        
        elif phase == "ACCUMULATION" and phase_strength > 75 and ofi > 0:
            if is_absorbing and absorption_strength > 1.5:
                signal = "BUY"
                confidence = min(85 + int(absorption_strength * 5), 95)
                reason = f"Institutional ACCUMULATION | OFI: {ofi:.2f} | Absorption: {absorption_strength:.1f}x"
        
        # SEGNALI SELL
        elif wyckoff_event == "UPTHRUST":
            signal = "SELL"
            confidence = event_conf
            reason = f"Wyckoff UPTHRUST detected | Phase: {phase} ({phase_strength}%)"
        
        elif phase == "DISTRIBUTION" and phase_strength > 75 and ofi < 0:
            if is_absorbing and absorption_strength > 1.5:
                signal = "SELL"
                confidence = min(85 + int(absorption_strength * 5), 95)
                reason = f"Institutional DISTRIBUTION | OFI: {ofi:.2f} | Absorption: {absorption_strength:.1f}x"
        
        if signal:
            self.last_signal_time[symbol] = time.time()
            
            # Calcola Stop Loss e Take Profit intelligenti
            prices = np.array(list(self.price_buffer[symbol]))
            atr = np.mean(np.abs(np.diff(prices[-50:]))) * 50  # ATR semplificato
            
            if signal == "BUY":
                sl = mid_price - (atr * 1.5)
                tp = mid_price + (atr * 2.5)
            else:  # SELL
                sl = mid_price + (atr * 1.5)
                tp = mid_price - (atr * 2.5)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'phase': phase,
                'ofi': round(ofi, 4),
                'entry': mid_price,
                'sl': round(sl, 2),
                'tp': round(tp, 2)
            }
        
        return None

def main():
    """Main loop con gestione errori migliorata"""
    
    # Verifica configurazione
    if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
        logger.error("❌ SUPABASE_URL e SUPABASE_KEY devono essere impostati nel file .env")
        sys.exit(1)
    
    if not Config.MT4_PATH or not os.path.isdir(Config.MT4_PATH):
        logger.error(f"❌ MT4_PATH non valido: {Config.MT4_PATH}")
        logger.error("Imposta MT4_PATH nel file .env (es: C:\\Users\\TuoNome\\AppData\\Roaming\\MetaQuotes\\Terminal\\XXXXX\\MQL4\\Files)")
        sys.exit(1)
    
    try:
        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        logger.info("✅ Connesso a Supabase")
    except Exception as e:
        logger.error(f"❌ Errore connessione Supabase: {e}")
        sys.exit(1)
    
    detector = InstitutionalDetector()
    logger.info("🚀 TITAN INSTITUTIONAL WHALE RADAR ONLINE...")
    logger.info(f"📂 Monitoring: {Config.MT4_PATH}")
    logger.info(f"💹 Assets: {', '.join(Config.ASSETS)}")
    logger.info("="*60)
    
    consecutive_errors = {s: 0 for s in Config.ASSETS}
    
    while True:
        for symbol in Config.ASSETS:
            data_file = os.path.join(Config.MT4_PATH, f"{symbol}_data.json")
            
            if not os.path.exists(data_file):
                if consecutive_errors[symbol] == 0:  # Log solo primo errore
                    logger.warning(f"⚠️  {symbol}: File non trovato - {data_file}")
                consecutive_errors[symbol] += 1
                continue
            
            try:
                # Leggi dati MT4
                with open(data_file, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                bid = float(data.get('bid', 0))
                ask = float(data.get('ask', 0))
                volume = float(data.get('volume', 1))  # Default 1 se non disponibile
                
                if bid <= 0 or ask <= 0:
                    logger.warning(f"⚠️  {symbol}: Dati invalidi (bid={bid}, ask={ask})")
                    continue
                
                consecutive_errors[symbol] = 0  # Reset errori
                mid_price = (bid + ask) / 2
                
                # PRICE_HISTORY DISABILITATO - non necessario per segnali
                # Se vuoi riattivarlo, crea prima la tabella su Supabase
                
                # Analisi istituzionale
                result = detector.analyze(symbol, bid, ask, volume)
                
                if result:
                    # Segnale rilevato!
                    logger.info("="*60)
                    logger.info(f"🎯 {symbol} | {result['signal']} SIGNAL | Conf: {result['confidence']}%")
                    logger.info(f"📊 Phase: {result['phase']} | OFI: {result['ofi']}")
                    logger.info(f"💰 Entry: ${result['entry']:.2f} | SL: ${result['sl']:.2f} | TP: ${result['tp']:.2f}")
                    logger.info(f"📝 {result['reason']}")
                    logger.info("="*60)
                    
                    # Salva su Supabase
                    try:
                        supabase.table("trading_signals").insert({
                            "symbol": symbol,
                            "recommendation": result['signal'],
                            "current_price": result['entry'],
                            "entry_price": result['entry'],
                            "stop_loss": result['sl'],
                            "take_profit": result['tp'],
                            "confidence_score": result['confidence'],
                            "details": result['reason']
                        }).execute()
                        logger.info(f"✅ Segnale salvato su database")
                    except Exception as e:
                        logger.error(f"❌ Errore salvataggio segnale: {e}")
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ {symbol}: Errore parsing JSON - {e}")
                consecutive_errors[symbol] += 1
            except Exception as e:
                logger.error(f"❌ {symbol}: Errore - {e}")
                consecutive_errors[symbol] += 1
            
            # Se troppi errori consecutivi, alert
            if consecutive_errors[symbol] > 10:
                logger.error(f"🔴 {symbol}: Troppi errori consecutivi ({consecutive_errors[symbol]}). Verificare MT4 EA.")
                consecutive_errors[symbol] = 0  # Reset per evitare spam
        
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Arresto TITAN Whale Radar...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"💥 Errore critico: {e}")
        sys.exit(1)
