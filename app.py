import streamlit as st
from supabase import create_client
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

# ================= 1. CONFIGURAZIONE PAGINA =================
st.set_page_config(
    page_title="TITAN Oracle",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed" # Nascondiamo la sidebar per il look "Terminal"
)

# ================= 2. STILE CSS (REPLICA ESATTA REACT/TAILWIND) =================
st.markdown("""
<style>
    /* RESET E FONDO */
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* RIMUOVERE PADDING STREAMLIT */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* CARD STYLE (Replica di .bg-gray-900 .border-gray-800) */
    .titan-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 0.75rem; /* rounded-xl */
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* HEADERS */
    .section-title {
        font-size: 1.125rem; /* text-lg */
        font-weight: 700;
        color: #d1d5db; /* text-gray-300 */
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* MAIN TITLE GRADIENT */
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #34d399, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* SIGNAL COLORS */
    .sig-text-buy { color: #10b981; font-size: 3rem; font-weight: 900; }
    .sig-text-sell { color: #ef4444; font-size: 3rem; font-weight: 900; }
    .sig-text-wait { color: #6b7280; font-size: 3rem; font-weight: 900; }

    /* LIVE STATUS DOT */
    .live-dot {
        height: 10px;
        width: 10px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
        box-shadow: 0 0 5px #10b981;
    }

    /* LIVE STATUS CONTAINER */
    .live-status {
        display: flex;
        align-items: center;
        border: 1px solid #10b981;
        padding: 5px 10px;
        border-radius: 15px;
        color: #10b981;
        font-size: 0.875rem;
        font-weight: 600;
    }

    /* ASSET BUTTONS */
    .stButton>button {
        background-color: #1f2937;
        color: #9ca3af;
        border: 1px solid #374151;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #374151;
        color: #ffffff;
    }
    .stButton>button[data-selected="true"] {
        background-color: #059669;
        color: #ffffff;
        border: 1px solid #10b981;
    }
    
    /* PROGRESS BAR CONTAINER */
    .progress-container {
        width: 100%;
        background-color: #374151;
        border-radius: 9999px;
        height: 0.75rem;
        overflow: hidden;
    }
    
    /* PROGRESS BAR FILL */
    .progress-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.5s ease-in-out;
    }
    .progress-buy { background-color: #10b981; }
    .progress-sell { background-color: #ef4444; }
    .progress-wait { background-color: #6b7280; }

    /* CURRENT PRICE VALUE */
    .current-price {
        font-size: 2.5rem;
        font-weight: 900;
        color: #22d3ee;
    }
</style>
""", unsafe_allow_html=True)

# ================= 3. CONNESSIONE DB =================
SUPABASE_URL = "https://gkffitfxqhxifibfwsfx.supabase.co"
SUPABASE_KEY = "sb_secret_s8jLpFKLhX3pNWXg6mBNOw_9HNs6rlG"

@st.cache_resource
def init_db(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_db()

# Initialize session state for selected asset
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = 'XAUUSD'

# ================= 4. INTERFACCIA UTENTE =================

# --- TOP BAR: Header & Live Status ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("""
        <div style="display: flex; align-items: center;">
            <span style="font-size: 2.5rem; margin-right: 10px;">🛡️</span>
            <div>
                <div class="hero-title" style="line-height: 1.2;">TITAN ORACLE</div>
                <div style="color:#9ca3af; font-size:0.875rem;">Institutional-Grade Trading Intelligence</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
            <div class="live-status">
                <span class="live-dot"></span>LIVE
            </div>
        </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# --- ASSET SELECTION BUTTONS ---
assets = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD", "EURUSD", "GBPUSD"]
cols = st.columns(len(assets))
for i, asset in enumerate(assets):
    if cols[i].button(asset, key=asset, use_container_width=True, type="primary" if st.session_state.selected_asset == asset else "secondary"):
        st.session_state.selected_asset = asset

st.write("") # Spacer

# --- MAIN LOOP ---
placeholder = st.empty()

while True:
    try:
        # Fetch Dati Reali for the selected asset
        resp = supabase.table("ai_oracle").select("*").eq("symbol", st.session_state.selected_asset).order("id", desc=True).limit(1).execute()
        
        with placeholder.container():
            if resp.data:
                d = resp.data[0]
                rec = d['recommendation']
                # For demonstration, let's assume "STRONG BUY" if confidence > 75
                conf_score = d.get('prob_buy', 50) if "BUY" in rec else d.get('prob_sell', 50)
                final_rec = "STRONG BUY" if conf_score > 75 and "BUY" in rec else "STRONG SELL" if conf_score > 75 and "SELL" in rec else rec

                sig_color_class = "sig-text-buy" if "BUY" in final_rec else "sig-text-sell" if "SELL" in final_rec else "sig-text-wait"
                progress_class = "progress-buy" if "BUY" in final_rec else "progress-sell" if "SELL" in final_rec else "progress-wait"
                
                # Get last update time
                last_update = datetime.fromisoformat(d['created_at'].replace('Z', '+00:00')).strftime('%H:%M:%S')

                # --- AI SIGNAL CARD ---
                st.markdown(f"""
                <div class="titan-card">
                    <div class="section-title">
                        <span>AI SIGNAL</span>
                        <span style="font-size: 1.5rem;">🧠</span>
                    </div>
                    <div style="text-align:center; padding:2rem 0;">
                         <div class="{sig_color_class}">{final_rec}</div>
                         <div style="color:#6b7280; font-size:1rem; margin-top: 0.5rem;">{st.session_state.selected_asset} • H1</div>
                    </div>
                    <div style="margin-bottom:0.5rem; display:flex; justify-content:space-between; font-size:0.875rem; color:#9ca3af;">
                        <span>ML Confidence</span>
                        <span>{conf_score:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-fill {progress_class}" style="width: {conf_score}%;"></div>
                    </div>
                    <div style="text-align: right; font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">
                        Last Update: {last_update}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # --- CURRENT PRICE CARD ---
                st.markdown(f"""
                <div class="titan-card">
                    <div class="section-title">
                        <span>CURRENT PRICE</span>
                        <span style="font-size: 1.5rem; color: #22d3ee;">◎</span>
                    </div>
                    <div class="current-price">${d['current_price']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.warning(f"Initializing connection to TITAN Core for {st.session_state.selected_asset}...")
                
        time.sleep(1)
        
    except Exception as e:
        # st.error(f"An error occurred: {e}") # Uncomment for debugging
        time.sleep(1)
