# --- ADD THESE IMPORTS ---
import socket
import threading
import queue
import time
from datetime import datetime

# --- INSERT THIS CLASS BEFORE YOUR MAIN FUNCTIONS ---
class MT4Receiver:
    """
    Palantir-Grade Socket Listener for MT4 Telemetry.
    Runs in a background thread to prevent UI blocking.
    """
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port
        self.data_queue = queue.Queue(maxsize=1) # Keep only latest tick
        self.running = False
        self.thread = None
        self.last_data = None

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()

    def _server_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(1)
            print(f"⚡ ALADDIN UPLINK LISTENING ON {self.host}:{self.port}")
            
            while self.running:
                conn, addr = s.accept()
                with conn:
                    print(f"⚡ MT4 CONNECTED: {addr}")
                    while self.running:
                        try:
                            data = conn.recv(1024)
                            if not data: break
                            # Parse Payload: SYMBOL|BID|ASK|TIME
                            text = data.decode('utf-8')
                            parts = text.split('|')
                            if len(parts) >= 3:
                                tick = {
                                    'symbol': parts[0],
                                    'price': float(parts[1]), # Using Bid as price
                                    'ask': float(parts[2]),
                                    'time': parts[3],
                                    'last_update': datetime.now()
                                }
                                # Update Queue (Overwrite old data)
                                if self.data_queue.full():
                                    try: self.data_queue.get_nowait()
                                    except: pass
                                self.data_queue.put(tick)
                        except Exception as e:
                            print(f"Connection Error: {e}")
                            break
                print("⚡ MT4 DISCONNECTED - WAITING FOR RECONNECTION")

    def get_latest(self):
        try:
            self.last_data = self.data_queue.get_nowait()
        except queue.Empty:
            pass
        return self.last_data

# --- MODIFY YOUR MAIN STREAMLIT CODE ---

# 1. Initialize the receiver in Session State (Run once)
if 'mt4_link' not in st.session_state:
    st.session_state.mt4_link = MT4Receiver()
    st.session_state.mt4_link.start()

# 2. Update your get_live_data function to prefer MT4 data
def get_live_data(symbol):
    # Try getting data from MT4 Link first
    mt4_data = st.session_state.mt4_link.get_latest()
    
    # Check if data matches requested symbol and is fresh (< 5 seconds old)
    if mt4_data and mt4_data['symbol'] in symbol: # basic string match
        freshness = (datetime.now() - mt4_data['last_update']).total_seconds()
        if freshness < 5:
            return {
                'price': mt4_data['price'],
                'open': mt4_data['price'], # Approximation for tick data
                'high': mt4_data['price'], 
                'low': mt4_data['price'],
                'volume': 0,
                'source': '⚡ MT4 LIVE LINK'
            }

    # Fallback to original logic (Yahoo/CoinGecko)
    # ... [Insert your original API logic here] ...
    try:
        if symbol == 'BTC-USD': 
            return get_realtime_crypto('BTC-USD')
        # ... etc
        ticker = yf.Ticker(symbol)
        # ...
        return original_data
    except:
        return None

# 3. CRITICAL: THE REAL-TIME LOOP
# Add this at the top of your script (or in a sidebar toggle)
use_live_stream = st.toggle("⚡ ACTIVATE LIVE STREAM (1s)", value=True)

if use_live_stream:
    # This loop keeps the script rerunning to fetch new data from the queue
    time.sleep(1) 
    st.rerun() 
