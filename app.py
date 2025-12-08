import zmq
import json
import time
import datetime
from supabase import create_client

# ==========================================
# CONFIGURAZIONE (DA COMPILARE CON I TUOI DATI)
# ==========================================
# Inserisci qui l'URL e la KEY che hai preso da Supabase (Settings -> API)
SUPABASE_URL = "INSERISCI_QUI_IL_TUO_URL_SUPABASE"
SUPABASE_KEY = "INSERISCI_QUI_LA_TUA_CHIAVE_SUPABASE"

# Porta ZMQ (Deve essere la stessa indicata nell'EA di MT4, di solito PUSH_PORT o PULL_PORT)
# Se usi il mio snippet EA semplice usa 5555. 
# Se usi l'EA completo DWX, controlla l'input "PUSH_PORT" su MT4 (default 32768).
ZMQ_PORT = 5555 

# ==========================================
# INIZIALIZZAZIONE SISTEMA
# ==========================================

def init_bridge():
    print("-" * 50)
    print("🚀 AVVIO SISTEMA BRIDGE: MT4 -> SUPABASE")
    print("-" * 50)

    # 1. Connessione a Supabase
    try:
        print("☁️  Connessione al Cloud Database...", end=" ")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[OK]")
    except Exception as e:
        print(f"\n[ERRORE CRITICO] Impossibile connettersi a Supabase: {e}")
        return None, None

    # 2. Connessione a ZeroMQ (Lato Locale)
    try:
        print(f"🔌 Apertura Porta Locale {ZMQ_PORT}...", end=" ")
        context = zmq.Context()
        socket = context.socket(zmq.PULL) # PULL ascolta quello che MT4 (PUSH) invia
        socket.bind(f"tcp://*:{ZMQ_PORT}")
        print("[OK]")
    except Exception as e:
        print(f"\n[ERRORE CRITICO] Impossibile aprire la porta ZMQ: {e}")
        return None, None

    print("\n✅ SISTEMA PRONTO. In attesa di dati dalla MetaTrader 4...")
    print("(Premi CTRL+C per fermare il ponte)")
    return socket, supabase

# ==========================================
# CICLO PRINCIPALE
# ==========================================

def main():
    socket, supabase = init_bridge()
    if not socket or not supabase:
        return

    msg_count = 0

    while True:
        try:
            # 1. Ricezione Pacchetto (Bloccante - aspetta finché non arriva qualcosa)
            # Usiamo recv_string per vedere il dato grezzo, poi lo convertiamo
            raw_msg = socket.recv_string()
            
            # Parsing JSON
            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                print(f"⚠️ [WARN] Messaggio non JSON ricevuto: {raw_msg}")
                continue

            # 2. Formattazione per Supabase
            # Adattiamo i dati per la tabella 'mt4_feed'
            # Se l'EA invia dati diversi, qui li mappiamo.
            payload = {
                "symbol": data.get("symbol", "Unknown"),
                "price": data.get("price", 0.0),
                "equity": data.get("equity", 0.0),
                "comment": data.get("comment", ""),
                # Timestamp automatico gestito dal DB, ma possiamo forzarlo se serve
            }

            # 3. Invio al Cloud (Supabase)
            # Usiamo 'insert' per tenere lo storico.
            response = supabase.table("mt4_feed").insert(payload).execute()
            
            # Feedback a video
            msg_count += 1
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 📡 Dato #{msg_count} inviato: {payload['symbol']} @ {payload['price']}")

        except zmq.ZMQError as e:
            print(f"⚠️ [ZMQ ERROR] {e}")
            time.sleep(1) # Pausa di sicurezza
            
        except Exception as e:
            # Errore generico (es. internet caduta)
            print(f"❌ [ERRORE] {e}")
            print("Tentativo di riconnessione in 2 secondi...")
            time.sleep(2)

if __name__ == "__main__":
    main()
