import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
import warnings
import json
import time
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== CONFIGURAZIONE ATOMICA (FIX PERCORSO) ====================
# La 'r' all'inizio è fondamentale: dice a Python di ignorare i caratteri speciali di Windows
MT4_DIRECTORY = r"C:\Users\dcbat\AppData\Roaming\MetaQuotes\Terminal\B8925BF731C22E88F33C7A8D7CD3190E\MQL4\Files"

# ==================== MT4 BRIDGE SYSTEM ====================
class MT4Bridge:
    """
    Gestisce comunicazione bidirezionale con MT4.
    Versione 'Bulletproof' con gestione automatica percorsi e file fantasma.
    """
    
    def __init__(self, bridge_folder=MT4_DIRECTORY):
        self.bridge_folder = Path(bridge_folder)
        
        # 1. FORZATURA ESISTENZA CARTELLA
        if not self.bridge_folder.exists():
            # Tentativo di creare il percorso se manca (o se è una sottocartella)
            try:
                self.bridge_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                st.error(f"❌ CRITICAL ERROR: Impossibile accedere al percorso: {self.bridge_folder}")
                st.error(f"Dettaglio: {e}")
                st.stop()

        # [cite_start]2. MAPPING FILE (Match esatto con MQL4 [cite: 26, 27])
        self.signals_file = self.bridge_folder / "signals.json"
        self.status_file = self.bridge_folder / "status.json"
        self.trades_file = self.bridge_folder / "trades.json"
        self.heartbeat_file = self.bridge_folder / "heartbeat.json"
        self.live_price_file = self.bridge_folder / "live_price.json"

        # 3. BOOTSTRAPPING (Creazione file vuoti per evitare crash in lettura)
        self._init_phantom_files()

    def _init_phantom_files(self):
        """Genera file JSON vuoti se non esistono ancora"""
        try:
            # Status vuoto
            if not self.status_file.exists():
                with open(self.status_file, 'w') as f: json.dump({}, f)
            # Trades vuoti (lista)
            if not self.trades_file.exists():
                with open(self.trades_file, 'w') as f: json.dump([], f)
            # Heartbeat finto (per non dare errore immediato)
            if not self.heartbeat_file.exists():
                hb = {"timestamp": datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"), "status": "waiting"}
                with open(self.heartbeat_file, 'w') as f: json.dump(hb, f)
        except Exception as e:
            st.warning(f"⚠️ Impossibile inizializzare file fantasma: {e}")

    def send_signal(self, signal_data):
        """Invia segnale a MT4 (Match Struttura Source 4, 147)"""
        try:
            signal = {
                "symbol": signal_data.get("symbol", "XAUUSD"),
                "direction": signal_data.get("direction", "BUY"),
                "entry": float(signal_data.get("entry", 0)),
                "stop_loss": float(signal_data.get("sl", 0)),      # Key MQL4: stop_loss
                "take_profit": float(signal_data.get("tp", 0)),    # Key MQL4: take_profit
                "lot_size": float(signal_data.get("lot_size", 0.01)),
                "probability": float(signal_data.get("probability", 0)),
                "ai_confidence": float(signal_data.get("ai_confidence", 0)),
                "comment": signal_data.get("comment", "AI_Signal"),
                "status": "PENDING"
            }
            
            # Scrittura con encoding utf-8 per sicurezza
            with open(self.signals_file, 'w', encoding='utf-8') as f:
                json.dump(signal, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Errore invio segnale: {e}")
            return False

    def get_status(self):
        """Legge status MT4 in modo sicuro"""
        return self._safe_read_json(self.status_file)

    def get_open_trades(self):
        """Recupera trade aperti"""
        data = self._safe_read_json(self.trades_file)
        return data if isinstance(data, list) else []

    def get_live_price(self):
        """Recupera prezzo live"""
        return self._safe_read_json(self.live_price_file)

    def is_mt4_connected(self):
        """Verifica connessione MT4 (Heartbeat < 15s)"""
        data = self._safe_read_json(self.heartbeat_file)
        if not data: return False
        
        try:
            # Parsing flessibile della data
            ts_str = data.get('timestamp', '')
            # [cite_start]MQL4 format: YYYY.MM.DD HH:MM:SS [cite: 55]
            if '.' in ts_str:
                last_beat = datetime.datetime.strptime(ts_str, "%Y.%m.%d %H:%M:%S")
            else:
                last_beat = datetime.datetime.fromisoformat(ts_str)
                
            age = (datetime.datetime.now() - last_beat).total_seconds()
            return age < 15
        except:
            return False

    def clear_signal(self):
        """Pulisce file segnale"""
        try:
            if self.signals_file.exists():
                self.signals_file.unlink()
            return True
        except:
            return False

    def _safe_read_json(self, filepath):
        """Lettura resiliente per evitare conflitti di I/O con MT4"""
        if not filepath.exists(): return None
        
        # 3 tentativi rapidi in caso di file lock
        for _ in range(3):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content: return None
                    return json.loads(content)
            except json.JSONDecodeError:
                time.sleep(0.05) # Attesa micro se MT4 sta scrivendo
            except Exception:
                return None
        return None
