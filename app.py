"""
═══════════════════════════════════════════════════════════════════════════════
TITAN V90 DASHBOARD - PREMIUM FRONTEND INTERFACE
═══════════════════════════════════════════════════════════════════════════════
Professional Real-Time Trading Terminal powered by TITAN V90 Backend
Visualizes market data, AI signals, and performance metrics via Streamlit
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# Dependency check & Import
try:
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.error("❌ Missing libraries. Run: pip install supabase python-dotenv plotly pandas")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class AppConfig:
    """Frontend Configuration"""
    PAGE_TITLE = "TITAN Oracle Prime"
    PAGE_ICON = "⚡"
    LAYOUT = "wide"
    
    # --- LISTA ASSET AGGIORNATA ---
    ASSETS = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    
    # Refresh rates
    AUTO_REFRESH_RATE = 5  # Seconds

# Initialize Page
st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout=AppConfig.LAYOUT,
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL STYLING (CSS ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════

def load_custom_css():
    """Injects the Premium Professional & Futuristic Style"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;400;500;600;700&display=swap');
        
        /* === GLOBAL THEME === */
        * { box-sizing: border-box; }
        
        .main { 
            background: #0B0C10;
            background-image: 
                radial-gradient(ellipse at top, rgba(30, 39, 73, 0.3) 0%, transparent 50%),
                radial-gradient(ellipse at bottom, rgba(17, 24, 39, 0.5) 0%, transparent 50%);
            color: #E4E8F0;
            font-family: 'Rajdhani', sans-serif;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }
        
        /* === HEADER === */
        .titan-header {
            background: linear-gradient(135deg, 
                rgba(15, 23, 42, 0.95) 0%, 
                rgba(30, 41, 59, 0.9) 100%);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 24px;
            padding: 3rem 2rem;
            margin-bottom: 2.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .titan-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                linear-gradient(135deg, transparent 0%, rgba(56, 189, 248, 0.03) 50%, transparent 100%);
            pointer-events: none;
        }
        
        .titan-header-content {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 2rem;
        }
        
        .titan-branding {
            flex: 1;
            min-width: 300px;
        }
        
        .titan-title {
            font-family: 'Orbitron', monospace;
            font-size: 3.5rem;
            font-weight: 900;
            letter-spacing: 2px;
            margin: 0;
            background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            filter: drop-shadow(0 0 40px rgba(56, 189, 248, 0.3));
            line-height: 1.1;
        }
        
        .titan-subtitle {
            font-family: 'Space Mono', monospace;
            font-size: 0.95rem;
            color: #94A3B8;
            margin-top: 0.75rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            font-weight: 400;
        }
        
        .titan-meta {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        
        /* === STATUS BADGE === */
        .status-badge {
            display: inline-flex;
            align-items: center;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #10B981;
            letter-spacing: 1px;
            font-family: 'Space Mono', monospace;
            backdrop-filter: blur(10px);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: #10B981;
            border-radius: 50%;
            margin-right: 10px;
            animation: pulse 2s ease-in-out infinite;
            box-shadow: 0 0 12px #10B981;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.1); }
        }
        
        .live-time {
            font-family: 'Space Mono', monospace;
            font-size: 0.85rem;
            color: #64748B;
            padding: 0.6rem 1.2rem;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }

        /* === TABS STYLING === */
        .stTabs {
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.6);
            padding: 0.75rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.1);
            backdrop-filter: blur(20px);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            color: #94A3B8;
            font-weight: 600;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1rem;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(56, 189, 248, 0.1);
            border-color: rgba(56, 189, 248, 0.3);
            color: #E4E8F0;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(129, 140, 248, 0.15) 100%);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #F0F9FF;
            box-shadow: 
                0 4px 20px rgba(56, 189, 248, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        /* === SIGNAL CARDS === */
        .signal-card {
            background: linear-gradient(135deg, 
                rgba(15, 23, 42, 0.95) 0%, 
                rgba(30, 41, 59, 0.9) 100%);
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 
                0 10px 40px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        .signal-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, transparent, currentColor, transparent);
            opacity: 0.8;
        }
        
        .signal-card-buy {
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #10B981;
        }
        
        .signal-card-buy::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at top right, rgba(16, 185, 129, 0.08) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .signal-card-sell {
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #EF4444;
        }
        
        .signal-card-sell::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at top right, rgba(239, 68, 68, 0.08) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .signal-card-wait {
            border: 1px solid rgba(148, 163, 184, 0.2);
            color: #64748B;
            opacity: 0.75;
        }
        
        .signal-card-closed {
            border: 1px solid rgba(100, 116, 139, 0.4);
            color: #94A3B8;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(20, 25, 35, 0.98) 100%);
        }
        
        .signal-card:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        /* === SIGNAL CONTENT === */
        .signal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            position: relative;
            z-index: 1;
        }
        
        .signal-icon {
            font-size: 2.5rem;
            line-height: 1;
            filter: drop-shadow(0 0 20px currentColor);
        }
        
        .signal-type {
            font-family: 'Orbitron', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: 1px;
        }
        
        .signal-symbol {
            font-family: 'Space Mono', monospace;
            font-size: 0.9rem;
            color: #94A3B8;
            letter-spacing: 2px;
            margin-top: 0.25rem;
        }
        
        .price-display {
            font-family: 'Orbitron', monospace;
            font-size: 3.5rem;
            font-weight: 900;
            margin: 1.5rem 0;
            line-height: 1;
            position: relative;
            z-index: 1;
            text-shadow: 0 0 40px currentColor;
        }
        
        .price-label {
            font-size: 0.75rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 600;
        }

        /* === CONFIDENCE BAR === */
        .confidence-container {
            margin: 2rem 0;
            position: relative;
            z-index: 1;
        }
        
        .confidence-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.75rem;
            font-size: 0.85rem;
            color: #94A3B8;
            font-family: 'Space Mono', monospace;
        }
        
        .confidence-bar {
            background: rgba(30, 41, 59, 0.6);
            height: 10px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, currentColor 0%, rgba(255, 255, 255, 0.8) 100%);
            position: relative;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 20px currentColor;
        }
        
        .confidence-fill::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.3) 50%, transparent 100%);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        /* === STATS GRID === */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 2rem 0;
            position: relative;
            z-index: 1;
        }
        
        .stat-box {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
        }
        
        .stat-box:hover {
            background: rgba(30, 41, 59, 0.6);
            border-color: rgba(148, 163, 184, 0.2);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.75rem;
            font-weight: 600;
            font-family: 'Rajdhani', sans-serif;
        }
        
        .stat-value {
            font-family: 'Orbitron', monospace;
            font-size: 1.5rem;
            font-weight: 700;
            color: #E4E8F0;
            line-height: 1;
        }
        
        .stat-value-green { color: #10B981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
        .stat-value-red { color: #EF4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
        .stat-value-blue { color: #38BDF8; text-shadow: 0 0 20px rgba(56, 189, 248, 0.5); }
        .stat-value-purple { color: #818CF8; text-shadow: 0 0 20px rgba(129, 140, 248, 0.5); }

        /* === STRATEGY DETAILS === */
        .strategy-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-left: 3px solid currentColor;
            border-radius: 8px;
            padding: 1.25rem;
            margin-top: 1.5rem;
            position: relative;
            z-index: 1;
            backdrop-filter: blur(10px);
        }
        
        .strategy-label {
            font-size: 0.75rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.5rem;
            font-weight: 600;
            font-family: 'Rajdhani', sans-serif;
        }
        
        .strategy-text {
            color: #94A3B8;
            font-size: 0.9rem;
            line-height: 1.6;
            font-family: 'Space Mono', monospace;
        }

        /* === METRICS CARDS === */
        .metric-card {
            background: linear-gradient(135deg, 
                rgba(15, 23, 42, 0.8) 0%, 
                rgba(30, 41, 59, 0.6) 100%);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(20px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 
                0 4px 20px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 
                0 12px 40px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
            border
