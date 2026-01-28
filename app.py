import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import time
import os
from dotenv import load_dotenv

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="TITAN V90 Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS DARK MODE FORZATO & STILE CARD ---
st.markdown("""
<style>
    /* Sfondo Generale e Testo */
    .stApp {
        background-color: #0E1117 !important;
        color: #FAFAFA;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30333D;
    }
    
    /* Container Box */
    .metric-container {
        background-color: #1E2127;
        border: 1px solid #30333D;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: auto;
    }

    /* Colori Segnali */
    .buy-signal { color: #27C469 !important; text-shadow: 0 0 15px rgba(39, 196, 105, 0.2); }
    .sell-signal { color: #E74C3C !important; text-shadow: 0 0 15px rgba(231, 76, 60, 0.2); }
    .wait-signal { color: #7F8C8D !important; }

    /* Bordi Colorati per SL/TP */
    .stop-box { border-bottom: 3px solid #E74C3C !important; }
    .target-box { border-bottom: 3px solid #27C469 !important; }
    .entry-box { border-bottom: 3px solid #5865F2 !important; }
    
    /* Rimozione spazi bianchi eccessivi */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { padding-top: 0px; margin-top: 0px; }
</style>
""", unsafe_allow_html=True)

# --- 3. CONNESSIONE DATABASE ---
try:
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # Override con Secrets se siamo su Cloud
    if 'SUPABASE_URL' in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
    if 'SUPABASE_KEY' in st.secrets:
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("🚨 DATABASE ERROR: Credenziali mancanti (.env o Secrets).")
        st.stop()

    @st.cache_resource
    def init_db():
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    supabase = init_db()

except Exception as e:
    st.error(f"Errore Connessione DB: {e}")
    st.stop()

# --- 4. FUNZIONI DI RECUPERO DATI ---
def get_last_signal(symbol):
    try:
        response = supabase.table("ai_oracle")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        if response.data: return response.data[0]
    except: pass
    return None

def get_price_history(symbol):
    try:
        response = supabase.table("mt4_feed")\
            .select("created_at, price")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df.sort_values('created_at')
    except: pass
    return pd.DataFrame()

# --- 5. INTERFACCIA UTENTE ---

# Sidebar
with st.sidebar:
    st.title
