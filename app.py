import zmq
import json
import time
import datetime
from supabase import create_client

# ==========================================
# CONFIGURAZIONE
# ==========================================

# 1. INSERISCI QUI IL TUO URL (Lo trovi su Supabase -> Project Settings -> API -> Project URL)
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co" 

# 2. LA TUA CHIAVE (Già inserita)
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

# 3. CONFIGURAZIONE PORTA (Deve essere uguale all'EA su MT4)
ZMQ_PORT = 5555 

# ==========================================
# INIZIALIZZAZIONE SISTEMA
# ==========================================

def init_bridge():
    print("-" * 50)
    print("🚀 AVVIO SISTEMA BRIDGE: MT4 -> SUPABASE")
    print("-" * 50)

    # Controllo che l'utente abbia messo l'URL
    if "INSERISCI" in SUPABASE_URL:
        print("\n❌ [ERRORE] Manca il SUPABASE_URL!")
        print("Vai nel file bridge.py e incolla il Project URL alla riga 11.")
        return None, None

    # 1. Connessione al Cloud (Supabase)
    try:
        print("☁️  Connessione al Cloud Database...", end=" ")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[OK]")
    except Exception as e:
        print(f"\n[ERRORE CRITICO] Connessione Supabase fallita: {e}")
        print("Suggerimento: Controlla che l'URL sia corretto (es. https://xyz.supabase.co)")
        return None, None

    # 2. Connessione a ZeroMQ (Lato Locale)
    try:
        print(f"🔌 Apertura Porta Locale {ZMQ_PORT}...", end=" ")
        context = zmq.Context()
        socket = context.socket(zmq.PULL) # PULL ascolta
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
        input("\nPremi INVIO per chiudere...")
        return

    msg_count = 0

    while True:
        try:
            # 1. Ricezione Pacchetto (Bloccante)
            raw_msg = socket.recv_string()
            
            # Parsing JSON
            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                print(f"⚠️ [WARN] Messaggio non valido ricevuto: {raw_msg}")
                continue

            # 2. Preparazione Payload
            payload = {
                "symbol": data.get("symbol", "Unknown"),
                "price": data.get("price", 0.0),
                "equity": data.get("equity", 0.0),
                "comment": data.get("comment", "")
            }

            # 3. Invio al Cloud
            response = supabase.table("mt4_feed").insert(payload).execute()
            
            # Feedback
            msg_count += 1
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 📡 Dato #{msg_count} inviato: {payload['symbol']} @ {payload['price']}")

        except zmq.ZMQError as e:
            print(f"⚠️ [ZMQ ERROR] {e}")
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ [ERRORE] {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
