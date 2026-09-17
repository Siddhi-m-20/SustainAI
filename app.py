import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Enforce pure dark theme for all Plotly charts
pio.templates.default = "plotly_dark"

from src.data.validate_data import validate_multi_resource_data
from src.analytics.kpi_engine import calculate_unified_campus_kpis
from src.anomaly.anomaly_service import AnomalyService
from src.forecasting.forecast_service import ForecastService
from src.impact.whatif_simulator import WhatIfSimulator
from src.rag.knowledge_retriever import KnowledgeRetriever
from src.agents.copilot_agent import CopilotAgent


# -----------------------------------------------------------------------------
# Streamlit App Configuration & Styling (Clean EcoSync Design)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSync Resource Manager | AI for Water, Energy & Resource Optimization",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Aesthetic Styling (EcoSync Immersive Dark Theme - UIverse Inspired)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* ── Immersive Dark Background ─────────────────────────────────── */
    [data-testid="stAppViewContainer"] {
        background: #0b0f19 !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stHeader"] {
        background: rgba(11, 15, 25, 0.85) !important;
        backdrop-filter: blur(12px) !important;
    }
    [data-testid="stSidebar"] {
        background: #070a12 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* Headings and Typography in Dark Theme */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    p, span, label {
        color: #cbd5e1;
    }

    /* ── Dashboard Navigation (Executive Icon Buttons) ─────────────── */
    [data-testid="stSidebar"] .stButton {
        margin-bottom: 3px !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
        width: 100% !important;
        padding: 10px 14px !important;
        font-size: 0.88rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(16, 185, 129, 0.12) !important;
        border-color: rgba(16, 185, 129, 0.4) !important;
        color: #f8fafc !important;
        transform: translateX(4px) !important;
    }
    /* Active Button Navigation State */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.35) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.65) !important;
        border-left: 4px solid #10b981 !important;
        color: #34d399 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.25) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] .stButton > button[kind="primary"] span {
        color: #34d399 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] p,
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] span {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover p,
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover span {
        color: #ffffff !important;
    }

    .sim-baseline-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }

    /* ── UIverse Glowing Header ────────────────────────────────────── */
    .ecosync-header {
        background: linear-gradient(135deg, #051a10 0%, #06381c 45%, #064e26 85%, #059669 100%);
        padding: 26px 32px;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 32px -5px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .ecosync-header::after {
        content: '';
        position: absolute;
        top: -60%; right: -25%;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.28) 0%, transparent 70%);
        pointer-events: none;
    }
    .ecosync-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        color: #ffffff !important;
    }
    .ecosync-header p {
        margin: 8px 0 0 0;
        font-size: 1.02rem;
        color: #a7f3d0 !important;
        font-weight: 500;
    }

    /* ── UIverse Pulse Badges & Chips ──────────────────────────────── */
    @keyframes uiverse-pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .pulse-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        animation: uiverse-pulse 1.8s infinite;
    }
    .chip-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .chip-emerald { background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .chip-amber   { background: rgba(245, 158, 11, 0.16); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .chip-cyan    { background: rgba(6, 182, 212, 0.16); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.4); }
    .chip-purple  { background: rgba(139, 92, 246, 0.16); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.4); }
    .chip-slate   { background: rgba(148, 163, 184, 0.16); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }

    /* ── UIverse Dark Glassmorphic & Hover Cards ───────────────────── */
    .uiverse-card {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 6px 24px -2px rgba(0, 0, 0, 0.4);
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 18px;
        color: #f1f5f9;
    }
    .uiverse-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 36px -4px rgba(0, 0, 0, 0.6), 0 0 20px rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.35);
    }
    .uiverse-glow-emerald::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #059669);
    }
    .uiverse-glow-amber::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #f59e0b, #d97706);
    }
    .uiverse-glow-cyan::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #06b6d4, #0284c7);
    }
    .uiverse-glow-purple::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #8b5cf6, #6366f1);
    }

    /* ── Executive Metric Tiles in Dark Theme ──────────────────────── */
    .metric-tile {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-tile:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.5), 0 0 15px rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.35);
    }
    .tile-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #94a3b8;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .tile-val {
        font-size: 1.95rem;
        font-weight: 800;
        color: #f8fafc !important;
        margin: 6px 0 2px 0;
        letter-spacing: -0.6px;
    }
    .tile-trend-down {
        color: #34d399;
        font-size: 0.84rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.15);
        padding: 2px 8px;
        border-radius: 9999px;
        display: inline-block;
    }
    .tile-trend-up {
        color: #f87171;
        font-size: 0.84rem;
        font-weight: 700;
        background: rgba(239, 68, 68, 0.15);
        padding: 2px 8px;
        border-radius: 9999px;
        display: inline-block;
    }
    .tile-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
    }

    /* ── Facility Resource Map Cards ───────────────────────────────── */
    .facility-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.25s ease;
        position: relative;
    }
    .facility-card:hover {
        border-color: #10b981;
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.45), 0 0 15px rgba(16, 185, 129, 0.18);
    }
    .facility-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .facility-card-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #f8fafc !important;
        margin: 0;
    }
    .facility-status-normal {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .facility-status-attention {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .facility-status-anomaly {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .facility-metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        padding: 5px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .facility-metric-label {
        color: #94a3b8;
    }
    .facility-metric-val {
        font-weight: 600;
        color: #f1f5f9;
    }

    /* ── Attention / Triage Action Card ────────────────────────────── */
    .triage-card {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        border: 1px solid rgba(249, 115, 22, 0.4);
        border-left: 5px solid #f97316;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    .triage-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(249, 115, 22, 0.15);
    }
    .triage-card-critical {
        border-color: rgba(239, 68, 68, 0.4);
        border-left-color: #ef4444;
    }
    .triage-card-explore {
        border-color: rgba(16, 185, 129, 0.4);
        border-left-color: #10b981;
    }

    /* ── UIverse Dark Tab Highlights ───────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px 10px 0 0;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #94a3b8;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #34d399;
        background: rgba(16, 185, 129, 0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
        border-color: #10b981 !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
    }

    /* ── Streamlit Buttons (Dark UIverse Styled) ────────────────────── */
    .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        border-color: #10b981 !important;
        color: #34d399 !important;
        background: rgba(16, 185, 129, 0.12) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
    }

    /* ── Selectbox & Inputs in Dark Mode ───────────────────────────── */
    [data-baseweb="select"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data Ingestion & Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_data():
    default_csv = PROJECT_ROOT / "data" / "raw" / "campus_multi_resource_sample.csv"
    if default_csv.exists():
        res = validate_multi_resource_data(default_csv)
        if res["is_valid"]:
            return res["cleaned_df"]
    alt_csv = PROJECT_ROOT / "data" / "raw" / "campus_energy_sample.csv"
    if alt_csv.exists():
        res = validate_multi_resource_data(alt_csv)
        if res["is_valid"]:
            return res["cleaned_df"]
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def get_services():
    anomaly_service = AnomalyService()
    forecast_service = ForecastService()
    whatif_simulator = WhatIfSimulator()
    knowledge_retriever = KnowledgeRetriever()
    return anomaly_service, forecast_service, whatif_simulator, knowledge_retriever


@st.cache_data(show_spinner=False)
def run_anomaly_detection(_service, df):
    return _service.detect_multi_resource_anomalies(df)


# Initialize services
anomaly_service, forecast_service, whatif_simulator, knowledge_retriever = get_services()


# -----------------------------------------------------------------------------
# Sidebar Navigation (Clean EcoSync Structure)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌱 **EcoSync**")
    st.caption("Resource Manager • AI for Sustainability")
    
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 12px 14px; margin-bottom: 14px;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                <span class="pulse-dot"></span>
                <span style="color:#34d399; font-size:0.75rem; font-weight:700; letter-spacing:0.4px;">SYSTEM OPERATIONAL</span>
            </div>
            <div style="font-size:0.8rem; color:#cbd5e1; font-weight:500;">
                4 Facilities · 8,640 hrs Telemetry
            </div>
            <div style="margin-top:6px; font-size:0.72rem; color:#94a3b8;">
                AI Engine: <span style="color:#6ee7b7; font-weight:600;">Grounded Watsonx Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Navigation
    st.markdown(
        "<div style='font-size:0.75rem; font-weight:700; color:#64748b; letter-spacing:0.8px; margin-bottom:8px;'>MODULE NAVIGATION</div>",
        unsafe_allow_html=True,
    )

    NAV_MODULES = [
        ("🏠", "Overview Dashboard"),
        ("⚡", "Energy Intelligence"),
        ("💧", "Water Intelligence"),
        ("♻️", "Resource & Waste"),
        ("🔎", "Incident Diagnostics"),
        ("🔮", "Demand Forecast"),
        ("🧪", "What-if Simulator"),
        ("🤖", "AI Copilot"),
        ("📚", "Sustainability Knowledge"),
        ("📊", "Audit Reports"),
        ("ℹ️", "About & Impact (SDG / AI)"),
    ]

    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Overview Dashboard"

    for icon, name in NAV_MODULES:
        full_label = f"{icon} {name}"
        is_active = (st.session_state.current_page == full_label)
        if st.button(
            full_label,
            key=f"nav_btn_{name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_page = full_label
            st.rerun()

    nav_selection = st.session_state.current_page

    st.markdown(
        """
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; padding: 10px 12px; margin-top: 10px; font-size: 0.78rem; color: #cbd5e1;">
            <span style="color:#34d399; font-weight:700;">🌱 Personal Project Vision:</span><br>
            SDG 7 (Primary) · SDG 6 · SDG 12<br>
            <small style="color:#94a3b8;">Autonomous Sustainability Decision-Support</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**FACILITY SCOPE**")

    active_df = load_default_data()
    all_buildings = ["ALL FACILITIES"]
    if not active_df.empty and "building" in active_df.columns:
        all_buildings += sorted(list(active_df["building"].unique()))

    selected_building = st.selectbox("Facility Filter", all_buildings, index=0)

    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; margin-top: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-size:0.75rem; font-weight:700; color:#34d399; letter-spacing:0.4px;">● TELEMETRY STREAM</span>
                <span style="font-size:0.72rem; color:#6ee7b7; background:rgba(16,185,129,0.15); padding:2px 6px; border-radius:4px;">ONLINE</span>
            </div>
            <div style="font-size:0.82rem; color:#e2e8f0; font-weight:600; margin-top:4px;">
                8,640 Hourly Records
            </div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">
                Continuous energy, water & waste telemetry across 4 facilities.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("EcoSync Resource Manager • Author Project • Sustainable AI")


# Guard: If no data
if active_df.empty:
    st.error("No telemetry data loaded. Please upload a valid CSV file.")
    st.stop()

# Filter df by building if selected
view_df = active_df.copy()
bldg_filter_key = None if selected_building == "ALL FACILITIES" else selected_building
if bldg_filter_key:
    view_df = view_df[view_df["building"] == bldg_filter_key]

# Precompute baseline KPIs
kpis = calculate_unified_campus_kpis(view_df)

# Anomaly dataframe
anomalies_df = run_anomaly_detection(anomaly_service, active_df)
if bldg_filter_key and not anomalies_df.empty:
    filtered_anomalies = anomalies_df[anomalies_df["building"] == bldg_filter_key]
else:
    filtered_anomalies = anomalies_df


# -----------------------------------------------------------------------------
# PAGE 1: 🏠 OVERVIEW DASHBOARD
# -----------------------------------------------------------------------------
if nav_selection == "🏠 Overview Dashboard":
    st.markdown(
        f"""
        <div class="ecosync-header">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                <div>
                    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px;">
                        <span class="pulse-badge"><span class="pulse-dot"></span> TELEMETRY ACTIVE</span>
                        <span class="chip-badge chip-amber">SDG 7 ENERGY</span>
                        <span class="chip-badge chip-cyan">SDG 6 WATER</span>
                        <span class="chip-badge chip-emerald">SDG 12 CIRCULARITY</span>
                        <span class="chip-badge chip-purple">IBM GRANITE READY</span>
                    </div>
                    <h1>🌱 EcoSync Resource Manager</h1>
                    <p>AI for Water, Energy and Resource Optimization • Scope: <b>{selected_building}</b></p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Executive KPI Tiles with Trend Deltas
    e = kpis.get("energy", {})
    w = kpis.get("water", {})
    r = kpis.get("waste", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="tile-title"><span>⚡ Total Energy</span> <span class="tile-trend-down">↓ 8.2%</span></div>
                <div class="tile-val">{e.get('total_mwh', 0):,.1f} <span style="font-size:1rem;color:#64748b;">MWh</span></div>
                <div class="tile-sub">${e.get('estimated_cost_usd', 0):,.0f} • Peak: {e.get('peak_kw', 0):.0f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="tile-title"><span>💧 Water Usage</span> <span class="tile-trend-up">↑ 3.4%</span></div>
                <div class="tile-val">{w.get('total_m3', 0):,.1f} <span style="font-size:1rem;color:#64748b;">m³</span></div>
                <div class="tile-sub">{w.get('total_liters', 0):,.0f} L • Night Flow: {w.get('min_night_flow_m3_h', 0):.2f} m³/h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="tile-title"><span>♻️ Materials & Waste</span> <span class="tile-trend-down">↓ 5.1%</span></div>
                <div class="tile-val">{r.get('total_waste_kg', 0)/1000.0:,.1f} <span style="font-size:1rem;color:#64748b;">t</span></div>
                <div class="tile-sub">Diversion: {r.get('diversion_rate_pct', 0):.1f}% • Landfill: {r.get('landfill_waste_kg', 0)/1000.0:.1f} t</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        crit_count = len(filtered_anomalies[filtered_anomalies["severity_tier"] == "CRITICAL"]) if not filtered_anomalies.empty else 0
        high_count = len(filtered_anomalies[filtered_anomalies["severity_tier"] == "HIGH"]) if not filtered_anomalies.empty else 0
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="tile-title"><span>🚨 Active Anomalies</span> <span style="color:#ef4444;font-weight:700;">{crit_count} Critical</span></div>
                <div class="tile-val" style="color:#ef4444;">{crit_count + high_count}</div>
                <div class="tile-sub">{high_count} High Priority • All Circuits Active</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -------------------------------------------------------------
    # FACILITY RESOURCE MAP
    # -------------------------------------------------------------
    st.markdown("### **🗺️ Facility Resource Map & Status**")
    st.caption("Interactive overview of resource consumption patterns and operational variances across monitored facilities.")

    map_col1, map_col2, map_col3, map_col4 = st.columns(4)
    
    with map_col1:
        st.markdown(
            """
            <div class="facility-card">
                <div class="facility-card-header">
                    <span class="facility-card-title">Building A</span>
                    <span class="facility-status-normal">● Normal</span>
                </div>
                <div style="font-size:0.75rem; color:#64748b; margin-bottom:8px;">Lecture Halls & Classrooms</div>
                <div class="facility-metric-row"><span class="facility-metric-label">Energy Load:</span><span class="facility-metric-val">85.4 kW</span></div>
                <div class="facility-metric-row"><span class="facility-metric-label">Water Flow:</span><span class="facility-metric-val">4.2 m³/h</span></div>
                <div class="facility-metric-row"><span class="facility-metric-label">Waste Div:</span><span class="facility-metric-val">58.0%</span></div>
                <div style="margin-top:10px; font-size:0.78rem; color:#059669;">✔ Unoccupied setbacks operating</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with map_col2:
        st.markdown(
            """
            <div class="facility-card" style="border-color:#fed7aa;">
                <div class="facility-card-header">
                    <span class="facility-card-title">Building B</span>
                    <span class="facility-status-attention">⚠ Attention</span>
                </div>
                <div style="font-size:0.75rem; color:#64748b; margin-bottom:8px;">Scientific Research Labs</div>
                <div class="facility-metric-row"><span class="facility-metric-label">Energy Load:</span><span class="facility-metric-val" style="color:#c2410c;">124.8 kW (High)</span></div>
                <div class="facility-metric-row"><span class="facility-metric-label">Water Flow:</span><span class="facility-metric-val">9.1 m³/h</span></div>
                <div class="facility-metric-row"><span class="facility-metric-label">Waste Div:</span><span class="facility-metric-val">42.5%</span></div>
                <div style="margin-top:10px; font-size:0.78rem; color:#c2410c;">⚡ Night baseload +18% above median</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with map_col3:
        st.markdown(
            """
            <div class="facility-card" style="border-color:#fca5a5;">
                <div class="facility-card-header">
                    <span class="facility-card-title">Building C</span>
                    <span class="facility-status-anomaly">🚨 Anomaly</span>
                </div>
                <div style="font-size:0.75rem; color:#64748b; margin-bottom:8px;">Dining & Student Hub</div>
                <div class="facility-metric-row"><span class="facility-metric-label">Energy Load:</span><span class="facility-metric-val">96.2 kW</span></div>
                <div class="facility-metric-row"><span class="facility-metric-label">Water Flow:</span><span class="facility-metric-val" style="color:#dc2626;">31.8 m³/h (Spike)</span></div>
                <div class="facility-metric-row"><span class="facility-metric-label">Waste Div:</span><span class="facility-metric-val">65.2%</span></div>
                <div style="margin-top:10px; font-size:0.78rem; color:#dc2626;">💧 +35% night water leak risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with map_col4:
        st.markdown(
            """
            <div class="facility-card">
                <div class="facility-card-header">
                    <span class="facility-card-title">Building D</span>
                    <span class="facility-status-normal">● Normal</span>
                </div>
                <div style="font-size:0.75rem; color:#64748b; margin-bottom:8px;">Administration & Offices</div>
                <div class="facility-metric-row"><span class="facility-metric-label">Energy Load:</span><span class="facility-metric-val">64.5 kW</span></div>
                <div class="facility-metric-row"><span class="facility-metric-label">Water Flow:</span><span class="facility-metric-val">2.9 m³/h</span></div>
                <div class="facility-metric-row"><span class="facility-metric-label">Waste Div:</span><span class="facility-metric-val">51.0%</span></div>
                <div style="margin-top:10px; font-size:0.78rem; color:#059669;">✔ Standard 08-18 profile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -------------------------------------------------------------
    # 3 ITEMS NEEDING ATTENTION
    # -------------------------------------------------------------
    st.markdown("### **🎯 Focus Areas: 3 Items Requiring Attention**")
    
    t_c1, t_c2, t_c3 = st.columns(3)
    with t_c1:
        st.markdown(
            """
            <div class="triage-card triage-card-critical">
                <div style="font-weight:700; color:#f87171; font-size:0.9rem;">🚨 BUILDING C • WATER SURGE</div>
                <h4 style="margin:4px 0 6px 0; color:#f8fafc !important;">Unusual Off-Hours Water Flow (+35%)</h4>
                <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:0 0 10px 0;">
                    31.81 m³ recorded during low-occupancy window vs 31.10 m³ baseline. Probable fixture leak or cooling tower valve bypass.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔎 Investigate Building C", key="triage_btn_1"):
            st.info("Navigate to '🔎 Incident Diagnostics' to inspect telemetry deviations and root cause hypotheses.")

    with t_c2:
        st.markdown(
            """
            <div class="triage-card">
                <div style="font-weight:700; color:#fbbf24; font-size:0.9rem;">⚡ BUILDING B • BASELOAD DRIFT</div>
                <h4 style="margin:4px 0 6px 0; color:#f8fafc !important;">Continuous 24/7 Power Baseload (+18%)</h4>
                <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:0 0 10px 0;">
                    Unoccupied lab power maintaining 124.8 kW overnight. Diagnostic pattern suggests ventilation exhaust fans running at 100% capacity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("⚡ Analyze Building B Baseload", key="triage_btn_2"):
            st.info("Navigate to '⚡ Energy Intelligence' to view the 24-hour diurnal profile.")

    with t_c3:
        st.markdown(
            """
            <div class="triage-card triage-card-explore">
                <div style="font-weight:700; color:#34d399; font-size:0.9rem;">♻️ DINING & SERVICES • DIVERSION LEVER</div>
                <h4 style="margin:4px 0 6px 0; color:#f8fafc !important;">Food Waste Composting Opportunity</h4>
                <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:0 0 10px 0;">
                    Expanding back-of-house kitchen organic composting in Building C can divert an estimated 28 tonnes/month and cut landfill hauling fees.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🧪 Simulate Composting Scenario", key="triage_btn_3"):
            st.info("Navigate to '🧪 What-if Simulator' to compute dollar and carbon returns.")

    st.markdown("---")

    # Resource Load Profiles
    col_chart_a, col_chart_b = st.columns([2, 1])
    with col_chart_a:
        st.markdown("#### **Multi-Resource Consumption Profile**")
        chart_df = view_df.sort_values("timestamp").copy()
        if "energy_kwh" in chart_df.columns:
            fig = px.line(
                chart_df,
                x="timestamp",
                y="energy_kwh",
                color="building" if not bldg_filter_key else None,
                labels={"energy_kwh": "Energy (kWh)", "timestamp": "Timestamp"},
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

    with col_chart_b:
        st.markdown("#### **Building Resource Breakdown**")
        if "building" in active_df.columns and "energy_kwh" in active_df.columns:
            bldg_dist = active_df.groupby("building")["energy_kwh"].sum().reset_index()
            fig_bar = px.bar(
                bldg_dist,
                x="building",
                y="energy_kwh",
                color="building",
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_bar.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_bar, width="stretch")


# -----------------------------------------------------------------------------
# PAGE 2: ⚡ ENERGY INTELLIGENCE
# -----------------------------------------------------------------------------
elif nav_selection == "⚡ Energy Intelligence":
    st.markdown("## ⚡ Energy Intelligence")
    st.caption("Monitoring load curves, peak demand, off-hours baseload, and Scope 2 carbon emissions.")

    e = kpis.get("energy", {})
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total Energy", f"{e.get('total_mwh', 0):,.2f} MWh")
    e2.metric("Peak Demand", f"{e.get('peak_kw', 0):,.1f} kW", f"{e.get('peak_building', '')}")
    e3.metric("Off-Hours Baseload", f"{e.get('off_hours_baseload_pct', 0):.1f}%", help="Hours 22:00 to 06:00")
    e4.metric("Carbon Footprint", f"{e.get('carbon_emissions_tco2e', 0):.2f} tCO2e", f"${e.get('estimated_cost_usd', 0):,.0f}")

    st.markdown("---")
    
    col_e1, col_e2 = st.columns([2, 1])
    with col_e1:
        st.markdown("#### **24-Hour Diurnal Energy Profile**")
        df_diurnal = view_df.copy()
        df_diurnal["hour"] = pd.to_datetime(df_diurnal["timestamp"]).dt.hour
        hourly_prof = df_diurnal.groupby(["hour", "building"])["energy_kwh"].mean().reset_index()
        fig_diurnal = px.line(
            hourly_prof,
            x="hour",
            y="energy_kwh",
            color="building",
            markers=True,
            labels={"energy_kwh": "Average kWh", "hour": "Hour of Day (0-23)"},
        )
        fig_diurnal.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_diurnal, width="stretch")

    with col_e2:
        st.markdown("#### **Operational Setback Guidelines**")
        baseload = e.get("off_hours_baseload_pct", 0)
        if baseload > 35:
            st.warning(f"⚠️ High Baseload ({baseload:.1f}%): Unoccupied HVAC setback audit recommended.")
        else:
            st.success(f"✅ Baseload Normal ({baseload:.1f}%): Equipment operating within standard envelope.")
        st.markdown(
            """
            * **Unoccupied Setback:** Heating 16°C / Cooling 28°C per ASHRAE 90.1
            * **Staggered Motor Start:** 15-min offset on chillers to shave peak tariff
            * **Phantom Load Isolation:** Scheduled power strip shutoff in computer labs
            """
        )


# -----------------------------------------------------------------------------
# PAGE 3: 💧 WATER INTELLIGENCE
# -----------------------------------------------------------------------------
elif nav_selection == "💧 Water Intelligence":
    st.markdown("## 💧 Water Intelligence")
    st.caption("Continuous flow surveillance, Minimum Night Flow (MNF 02:00–04:30) analysis, and conservation tracking.")

    w = kpis.get("water", {})
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Total Water Consumed", f"{w.get('total_m3', 0):,.1f} m³")
    w2.metric("Total Volume (Liters)", f"{w.get('total_liters', 0):,.0f} L")
    w3.metric("Min Night Flow (MNF)", f"{w.get('min_night_flow_m3_h', 0):.2f} m³/h", help="Flow between 02:00-05:00")
    w4.metric("Water Utility Cost", f"${w.get('estimated_cost_usd', 0):,.2f}")

    st.markdown("---")
    
    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        st.markdown("#### **Hourly Water Flow & Overnight Leak Watch**")
        water_ts = view_df.sort_values("timestamp").copy()
        fig_w = px.line(
            water_ts,
            x="timestamp",
            y="water_m3",
            color="building" if not bldg_filter_key else None,
            title="Continuous Water Demand (m³/hour)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_w.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Leak Risk Threshold (1.5 m³/h)")
        fig_w.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_w, width="stretch")

    with col_w2:
        st.markdown("#### **Night Flow Diagnosis**")
        night_flow = w.get("min_night_flow_m3_h", 0)
        if night_flow > 1.0:
            st.error(f"🚨 Elevated Night Flow ({night_flow:.2f} m³/h): High likelihood of stuck flushometer or chiller makeup valve.")
        else:
            st.success(f"✅ Night Flow Acceptable ({night_flow:.2f} m³/h): Sealed baseline verified.")
        
        st.markdown(
            """
            * **Cooling Towers:** Monitor cycles of concentration (> 5.0)
            * **Commercial Kitchens:** Verify dish pre-rinse valves shut overnight
            * **Restroom Fixtures:** Standardize on EPA WaterSense certified valves
            """
        )


# -----------------------------------------------------------------------------
# PAGE 4: ♻️ RESOURCE & WASTE
# -----------------------------------------------------------------------------
elif nav_selection in ["♻️ Resource & Waste", "♻️ Resource & Waste Intelligence"]:
    st.markdown("## ♻️ Resource & Waste Intelligence")
    st.caption("Solid waste characterization, landfill diversion trends, and circular economy tracking.")

    r = kpis.get("waste", {})
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Facility Waste", f"{r.get('total_waste_kg', 0):,.0f} kg")
    r2.metric("Diverted from Landfill", f"{r.get('diverted_waste_kg', 0):,.0f} kg")
    r3.metric("Diversion Rate", f"{r.get('diversion_rate_pct', 0):.1f}%", f"Tier: {r.get('status', 'Tracking')}")
    r4.metric("Hauling Expenditure", f"${r.get('estimated_hauling_cost_usd', 0):,.2f}")

    st.markdown("---")
    
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.markdown("#### **Waste Stream Breakdown by Facility**")
        if "building" in active_df.columns and ("waste_kg" in active_df.columns or "waste_total_kg" in active_df.columns):
            w_col = "waste_kg" if "waste_kg" in active_df.columns else "waste_total_kg"
            waste_bldg = active_df.groupby("building")[w_col].sum().reset_index()
            fig_rw = px.bar(
                waste_bldg,
                x="building",
                y=w_col,
                color="building",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig_rw.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_rw, width="stretch")

    with col_r2:
        st.markdown("#### **Zero-Waste Progress Benchmark**")
        cur_rate = r.get("diversion_rate_pct", 0)
        st.progress(min(cur_rate / 100.0, 1.0))
        st.caption(f"Current Progress: **{cur_rate:.1f}%** towards 90% Zero-Waste Standard")
        
        st.markdown(
            """
            * **Compost Stream:** Prioritize dining halls (Building C food prep)
            * **Bin Co-Location:** Standardize 3-stream pictorial waste stations
            * **Contamination Audits:** Verify batch recycling purity > 85%
            """
        )


# -----------------------------------------------------------------------------
# PAGE 5: 🔎 INCIDENT DIAGNOSTICS & INVESTIGATE
# -----------------------------------------------------------------------------
elif nav_selection in ["🔎 Incident Diagnostics", "🔎 Investigate"]:
    st.markdown("## 🔎 Anomaly Investigation & Diagnostics")
    st.caption("Deep-dive inspection: Isolation Forest machine learning paired with contextual median baselines.")

    f1, f2, f3 = st.columns(3)
    with f1:
        res_filter = st.selectbox("Filter Resource", ["ALL", "Energy", "Water", "Waste"], index=0)
    with f2:
        tier_filter = st.selectbox("Filter Severity Tier", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"], index=0)
    with f3:
        min_sev = st.slider("Minimum Severity Score", 0.0, 100.0, 30.0, 5.0)

    df_inspect = anomalies_df.copy()
    if bldg_filter_key:
        df_inspect = df_inspect[df_inspect["building"] == bldg_filter_key]
    if res_filter != "ALL":
        df_inspect = df_inspect[df_inspect["resource"].str.lower() == res_filter.lower()]
    if tier_filter != "ALL":
        df_inspect = df_inspect[df_inspect["severity_tier"] == tier_filter]
    if "severity_score" in df_inspect.columns:
        df_inspect = df_inspect[df_inspect["severity_score"] >= min_sev]
        df_inspect = df_inspect.sort_values("severity_score", ascending=False)

    st.markdown(f"**Identified Anomalies Matching Filter:** `{len(df_inspect)} incidents`")

    if not df_inspect.empty:
        display_cols = [
            "timestamp", "building", "resource", "actual_value", "expected_value",
            "deviation_pct", "severity_tier", "severity_score"
        ]
        available_cols = [c for c in display_cols if c in df_inspect.columns]
        
        st.dataframe(
            df_inspect[available_cols].head(25),
            width="stretch",
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("### **Diagnostic Incident Card (Detailed Drill-Down)**")
        
        incident_idx = st.selectbox(
            "Select Incident to Inspect in Detail",
            range(min(12, len(df_inspect))),
            format_func=lambda i: f"Incident #{i+1} • {df_inspect.iloc[i].get('building')} ({df_inspect.iloc[i].get('resource')}) • Dev: +{df_inspect.iloc[i].get('deviation_pct', 0):.1f}% • Severity: {df_inspect.iloc[i].get('severity_score', 0):.1f}"
        )
        
        sel = df_inspect.iloc[incident_idx]
        b_name = str(sel.get('building', 'Facility')).replace('_', ' ')
        res_type = str(sel.get('resource', 'Energy'))
        sev_tier = str(sel.get('severity_tier', 'HIGH')).upper()
        sev_score = float(sel.get('severity_score', 0))
        act_val = float(sel.get('actual_value', 0))
        exp_val = float(sel.get('expected_value', 0))
        dev_pct = float(sel.get('deviation_pct', 0))
        unit = "kWh" if "energy" in res_type.lower() else ("m³" if "water" in res_type.lower() else "kg")
        ts_str = str(sel.get('timestamp', 'N/A'))

        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(249, 115, 22, 0.4); border-left: 6px solid #ea580c; border-radius: 14px; padding: 20px 24px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <h3 style="margin:0; color:#f8fafc !important;">🚨 {b_name} — {sev_tier} SEVERITY ({res_type})</h3>
                    <span style="font-weight:800; font-size:1.15rem; color:#fb923c;">Severity Score: {sev_score:.1f} / 100</span>
                </div>
                <h4 style="margin:8px 0 6px 0; color:#94a3b8 !important;">Why was this flagged?</h4>
                <table style="width:100%; border-collapse:collapse; margin-bottom:14px; font-size:0.95rem;">
                    <tr style="border-bottom:1px solid rgba(255, 255, 255, 0.08);">
                        <td style="padding:8px 0; color:#94a3b8;"><b>Actual Consumption:</b></td>
                        <td style="padding:8px 0; font-weight:700; color:#f8fafc;">{act_val:,.2f} {unit}</td>
                        <td style="padding:8px 0; color:#94a3b8;"><b>Detected Pattern:</b></td>
                        <td style="padding:8px 0; font-weight:600; color:#f87171;">{sel.get('pattern', 'Unusual Consumption Spike')}</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255, 255, 255, 0.08);">
                        <td style="padding:8px 0; color:#94a3b8;"><b>Expected Baseline:</b></td>
                        <td style="padding:8px 0; font-weight:700; color:#f8fafc;">{exp_val:,.2f} {unit}</td>
                        <td style="padding:8px 0; color:#94a3b8;"><b>Timestamp:</b></td>
                        <td style="padding:8px 0; font-family:monospace; color:#cbd5e1;">{ts_str}</td>
                    </tr>
                    <tr>
                        <td style="padding:8px 0; color:#94a3b8;"><b>Contextual Deviation:</b></td>
                        <td style="padding:8px 0; font-weight:700; color:#f87171;">+{dev_pct:.1f}% above norm</td>
                        <td style="padding:8px 0; color:#94a3b8;"><b>Cost Impact (est.):</b></td>
                        <td style="padding:8px 0; font-weight:700; color:#fbbf24;">${sel.get('estimated_cost_loss_usd', 0):,.2f} USD</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown("#### **Possible Causes (AI Diagnostic Hypotheses):**")
            if "water" in res_type.lower():
                st.markdown(
                    """
                    * **Plumbing fixture leakage:** Auto-flush sensor solenoid in commercial restrooms stuck in open flow.
                    * **Cooling tower makeup valve:** Chiller condenser reservoir float valve stuck open, overflowing to drain.
                    * **Off-hours kitchen washdown:** Unscheduled manual kitchen equipment cleaning during low-occupancy window.
                    """
                )
            elif "energy" in res_type.lower():
                st.markdown(
                    """
                    * **BMS manual override:** HVAC schedule override manually engaged during evening or weekend and not reset.
                    * **Simultaneous heating & cooling:** VAV damper failure causing terminal reheat combatting primary airflow.
                    * **Auxiliary equipment left energized:** Laboratory exhaust ventilation fans running at 100% full velocity.
                    """
                )
            else:
                st.markdown(
                    """
                    * **Recycling contamination event:** Single-stream load rejected at loading dock compactor.
                    * **Unscheduled facility cleanout:** Move-out bulk disposal directed to municipal landfill compactor.
                    * **Compost sorting bypass:** Kitchen pre-consumer food waste diverted into standard trash hopper.
                    """
                )

        with col_exp2:
            st.markdown("#### **Recommended Next Step for Facility Engineers:**")
            st.markdown(
                f"""
                1. 🔍 **Inspect meter and submeter lines** in **{b_name}** to verify sensor calibration.
                2. ⏱️ **Check Building Automation System (BAS)** time-of-day clock and zone setback schedule.
                3. 📋 **Perform physical walkthrough** of mechanical rooms and primary restroom risers.
                4. ⚠️ *Responsible AI Notice:* Always verify physical meters prior to mechanical adjustments.
                """
            )
    else:
        st.success("🎉 No anomalies detected matching the specified severity criteria.")


# -----------------------------------------------------------------------------
# PAGE 6: 🔮 PREDICTIVE DEMAND FORECAST
# -----------------------------------------------------------------------------
elif nav_selection == "🔮 Demand Forecast":
    st.markdown("## 🔮 Predictive Demand Forecasting")
    st.caption("Machine learning time-series regression predicting resource demand with 95% confidence bounds.")

    fc_col1, fc_col2, fc_col3 = st.columns(3)
    with fc_col1:
        fc_resource = st.selectbox("Resource to Forecast", ["energy", "water", "waste"], index=0)
    with fc_col2:
        fc_building = st.selectbox("Facility Scope", all_buildings, index=0)
    with fc_col3:
        fc_horizon = st.selectbox("Forecast Horizon", [24, 48, 72, 168], format_func=lambda h: f"{h} Hours ({h//24} Days)", index=1)

    with st.spinner("Training cyclical demand regressor..."):
        b_arg = None if fc_building == "ALL FACILITIES" else fc_building
        fc_df = forecast_service.predict(active_df, resource=fc_resource, building=b_arg, horizon_hours=fc_horizon)
        fc_kpis = forecast_service.get_forecast_kpis(fc_df, resource=fc_resource)

    fk1, fk2, fk3, fk4 = st.columns(4)
    fk1.metric("Projected Total", f"{fc_kpis.get('total_predicted', 0):,.1f} {fc_kpis.get('unit')}")
    fk2.metric("Projected Peak", f"{fc_kpis.get('peak_predicted', 0):,.1f} {fc_kpis.get('unit')}")
    fk3.metric("Peak Alert Hours", f"{fc_kpis.get('peak_alert_hours', 0)} hrs", help="Hours exceeding 90th percentile")
    fk4.metric("Projected Cost", f"${fc_kpis.get('projected_cost_usd', 0):,.2f}")

    st.markdown("---")
    
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=fc_df["timestamp"], y=fc_df["upper_95"],
        mode="lines", line=dict(width=0), showlegend=False, name="Upper 95% CI",
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df["timestamp"], y=fc_df["lower_95"],
        mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(5, 140, 66, 0.15)",
        showlegend=True, name="95% Confidence Interval",
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df["timestamp"], y=fc_df["predicted_demand"],
        mode="lines+markers", line=dict(color="#058c42", width=2.5), name="Predicted Demand",
    ))
    peak_points = fc_df[fc_df["is_peak_forecast"]]
    if not peak_points.empty:
        fig_fc.add_trace(go.Scatter(
            x=peak_points["timestamp"], y=peak_points["predicted_demand"],
            mode="markers", marker=dict(color="#ef4444", size=9, symbol="diamond"), name="Peak Demand Alert",
        ))

    fig_fc.update_layout(
        title=f"Forecasted {fc_resource.capitalize()} Demand ({fc_building}) — Next {fc_horizon} Hours",
        xaxis_title="Timestamp",
        yaxis_title=f"Demand ({fc_kpis.get('unit')})",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, width="stretch")


# -----------------------------------------------------------------------------
# PAGE 7: 🧪 WHAT-IF SIMULATOR
# -----------------------------------------------------------------------------
elif nav_selection == "🧪 What-if Simulator":
    st.markdown("## 🧪 What-If Scenario Simulator")
    st.caption("Interactive counterfactual scenario engine calculating estimated financial returns, resource savings, and carbon abatement.")

    sim_bldg = st.selectbox("Target Facility for Intervention", all_buildings, index=0)

    # Facility Baseline Card
    bldg_view = active_df.copy()
    if sim_bldg != "ALL FACILITIES":
        bldg_view = bldg_view[bldg_view["building"] == sim_bldg]
    
    b_kpis = calculate_unified_campus_kpis(bldg_view)
    be = b_kpis.get("energy", {})
    bw = b_kpis.get("water", {})
    br = b_kpis.get("waste", {})

    bldg_display_name = sim_bldg.replace('_', ' ') if sim_bldg != "ALL FACILITIES" else "All Monitored Facilities Combined"

    st.markdown(
        f"""
        <div class="sim-baseline-card">
            <h4 style="margin:0 0 10px 0; color:#f8fafc !important;">🏢 Current Baseline Telemetry: <b>{bldg_display_name}</b></h4>
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:16px;">
                <div><span style="color:#94a3b8; font-size:0.85rem;">CURRENT ENERGY:</span><br><b style="color:#f8fafc;">{be.get('total_mwh', 0):,.1f} MWh</b> (${be.get('estimated_cost_usd', 0):,.0f})</div>
                <div><span style="color:#94a3b8; font-size:0.85rem;">CURRENT WATER:</span><br><b style="color:#f8fafc;">{bw.get('total_m3', 0):,.1f} m³</b> ({bw.get('total_liters', 0):,.0f} L)</div>
                <div><span style="color:#94a3b8; font-size:0.85rem;">CURRENT WASTE:</span><br><b style="color:#f8fafc;">{br.get('total_waste_kg', 0):,.0f} kg</b> (Diversion: {br.get('diversion_rate_pct', 0):.1f}%)</div>
                <div><span style="color:#94a3b8; font-size:0.85rem;">CARBON FOOTPRINT:</span><br><b style="color:#34d399;">{be.get('carbon_emissions_tco2e', 0) + br.get('carbon_emissions_tco2e', 0):.1f} tCO2e</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### **1. Configure Operational Levers (Assumptions)**")
    s_col1, s_col2, s_col3 = st.columns(3)
    
    with s_col1:
        st.markdown("⚡ **Energy Interventions**")
        off_hours_pct = st.slider("Off-Hours Setback Reduction (%)", 0.0, 50.0, 20.0, 5.0, help="ASHRAE 90.1 night setback")
        peak_shave_pct = st.slider("Peak Demand Shaving (%)", 0.0, 30.0, 10.0, 5.0, help="Equipment start stagger")
        
    with s_col2:
        st.markdown("💧 **Water Interventions**")
        leak_fix_pct = st.slider("Night Leak / MNF Fix (%)", 0.0, 80.0, 40.0, 5.0, help="Plumbing fixture repair speed")
        greywater_pct = st.slider("Greywater / Fixture Retrofit (%)", 0.0, 30.0, 5.0, 5.0, help="Cooling tower reuse")

    with s_col3:
        st.markdown("♻️ **Material & Waste Interventions**")
        compost_pct = st.slider("Food Waste to Compost Target (%)", 0.0, 80.0, 35.0, 5.0, help="Kitchen prep scrap diversion")
        source_red_pct = st.slider("Source Reduction / Single-Use Cut (%)", 0.0, 20.0, 5.0, 1.0, help="Eliminate disposable plastics")

    # Run Simulation
    bldg_sim_arg = None if sim_bldg == "ALL FACILITIES" else sim_bldg
    sim_res = whatif_simulator.simulate(
        active_df,
        building=bldg_sim_arg,
        off_hours_reduction_pct=off_hours_pct,
        peak_shaving_pct=peak_shave_pct,
        night_water_leak_fix_pct=leak_fix_pct,
        greywater_reuse_pct=greywater_pct,
        compost_diversion_target_pct=compost_pct,
        source_reduction_pct=source_red_pct,
    )

    st.markdown("---")
    st.markdown("### **2. Estimated Environmental & Financial Potential**")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Cost Avoided (est.)", f"${sim_res.total_cost_saved_usd:,.2f} USD")
    sc2.metric("Carbon Avoided (est.)", f"{sim_res.carbon_avoided_tco2e:.2f} tCO2e")
    sc3.metric("Energy Conserved (est.)", f"{sim_res.energy_saved_mwh:.2f} MWh")
    sc4.metric("Water Conserved (est.)", f"{sim_res.water_saved_liters:,.0f} L")

    st.markdown("#### **Tangible Equivalencies for Stakeholder Storytelling:**")
    eq1, eq2, eq3 = st.columns(3)
    eq1.info(f"🚗 Equivalent to removing **{sim_res.equivalent_cars_removed_annual:.1f} passenger vehicles** from the road for 1 full year.")
    eq2.success(f"🌲 Equivalent to planting **{sim_res.equivalent_trees_planted:,.0f} mature urban trees** sequestering carbon.")
    eq3.warning(f"♻️ Projects overall diversion increase to **{sim_res.sdg12_diversion_rate_projected_pct:.1f}%**.")

    st.markdown(
        """
        <div class="estimate-notice">
            <b>⚠️ Operational Assumptions Notice:</b> Potential impact metrics are counterfactual engineering simulation estimates
            calculated using ASHRAE Standard 90.1, EPA WaterSense, and EPA WARM baseline emission factors (0.42 kg CO2e/kWh electricity; 
            $0.14/kWh; $3.80/m³ water). These figures illustrate modeled opportunities and do not represent verified post-intervention billing meters.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 8: 🤖 AI COPILOT
# -----------------------------------------------------------------------------
elif nav_selection == "🤖 AI Copilot":
    st.markdown("## 🤖 AI Resource Copilot")
    st.caption("Autonomous conversational assistant powered by Agentic Tool Dispatching and grounded IBM Granite synthesis.")

    copilot = CopilotAgent(data_df=active_df)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your **EcoSync AI Copilot**. I can investigate facility anomalies, "
                    "project demand forecasts, simulate what-if energy and water savings, and retrieve "
                    "ASHRAE/LEED sustainability guidelines. How can I assist you today?"
                ),
                "tools": [],
                "sdgs": [],
            }
        ]

    st.markdown("#### **Quick Inquiries:**")
    qp1, qp2, qp3, qp4 = st.columns(4)
    prompt_to_submit = None
    
    if qp1.button("💧 Why is Building C using more water?"):
        prompt_to_submit = "Why is Building C using more water?"
    if qp2.button("⚡ Which building wasted most energy?"):
        prompt_to_submit = "Which building wasted the most energy?"
    if qp3.button("🧪 What if Building A cuts off-hours by 20%?"):
        prompt_to_submit = "How much energy could we save if Building A reduced off-hours consumption by 20%?"
    if qp4.button("🌱 Give 3 actions to reduce water wastage"):
        prompt_to_submit = "Give me three actions to reduce water wastage."

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools"):
                with st.expander("🤖 Agentic AI Workflow (Step-by-Step Tool Dispatch & Grounding)", expanded=False):
                    st.markdown("**1️⃣ Query Intent Analysis:** Entity & intent parsed.")
                    for i, t in enumerate(msg["tools"], 1):
                        t_name = getattr(t, "tool_name", "analytics_tool")
                        t_type = getattr(t, "tool_type", "Diagnostic Tool")
                        t_args = getattr(t, "arguments", {})
                        t_res = getattr(t, "summary_result", "")
                        st.markdown(f"**{i+1}️⃣ Dispatched Tool:** `{t_name}` ({t_type})")
                        st.caption(f"Parameters: `{t_args}` ➔ Grounding Result: {t_res}")
                    st.markdown(f"**🎯 Synthesizer:** IBM Granite Foundation Architecture • Grounded Provenance")

    user_input = st.chat_input("Ask a facility or operational question...") or prompt_to_submit

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("AI Agent analyzing intent, executing diagnostic tools & consulting Granite RAG..."):
                response = copilot.execute_plan_and_answer(user_input)
                st.markdown(response.answer)
                
                with st.expander("🤖 Agentic AI Workflow (Step-by-Step Tool Dispatch & Grounding)", expanded=True):
                    st.markdown(f"**1️⃣ User Question Analyzed:** `{user_input}`")
                    for i, t in enumerate(response.tools_executed, 1):
                        t_name = getattr(t, "tool_name", "analytics_tool")
                        t_type = getattr(t, "tool_type", "Diagnostic Tool")
                        t_args = getattr(t, "arguments", {})
                        t_res = getattr(t, "summary_result", "")
                        st.markdown(f"**{i+1}️⃣ Dispatched Tool:** `{t_name}` ({t_type})")
                        st.caption(f"Parameters: `{t_args}` ➔ Result: {t_res}")
                    st.markdown(f"**💡 IBM Granite Grounded Synthesis:** Citing {', '.join(response.provenance_sources)}")
                    st.caption(f"Confidence Rating: **{response.confidence_level}**")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "tools": response.tools_executed,
                    "sdgs": [],
                })


# -----------------------------------------------------------------------------
# PAGE 9: 📚 SUSTAINABILITY KNOWLEDGE
# -----------------------------------------------------------------------------
elif nav_selection == "📚 Sustainability Knowledge":
    st.markdown("## 📚 Sustainability Knowledge Base & Protocols (RAG)")
    st.caption("Standard operating guidelines, ASHRAE Standard 90.1, EPA WaterSense, and zero-waste benchmarks.")

    search_q = st.text_input("Search Engineering Protocols & Standards", placeholder="e.g., setback guidelines, night leak detection, composting")

    if search_q:
        results = knowledge_retriever.search(search_q, top_k=4)
        st.markdown(f"**Found {len(results)} relevant protocol sections:**")
        for r in results:
            with st.expander(f"📌 {r['title']} (Match: {r['similarity_score']*100:.1f}%)"):
                st.caption(f"Source Document: `{r['source_file']}`")
                st.markdown(r["content"])
    else:
        st.markdown("#### **Browse Standard Operating Protocols:**")
        all_chunks = knowledge_retriever.get_all_chunks()
        for c in all_chunks[:6]:
            with st.expander(f"📘 {c.title}"):
                st.caption(f"File: `{c.source_file}`")
                st.markdown(c.content)


# -----------------------------------------------------------------------------
# PAGE 10: 📊 AUDIT REPORTS
# -----------------------------------------------------------------------------
elif nav_selection == "📊 Audit Reports":
    st.markdown("## 📊 Executive Sustainability Audit Report")
    st.caption("Generate verifiable multi-resource compliance reports for facility management leadership.")

    e = kpis.get("energy", {})
    w = kpis.get("water", {})
    r = kpis.get("waste", {})

    report_text = f"""# ECOSYNC RESOURCE MANAGER — EXECUTIVE FACILITY AUDIT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Facility Scope: {selected_building}

================================================================================
1. EXECUTIVE RESOURCE SCORECARD
================================================================================
• Energy Consumption:          {e.get('total_mwh', 0):,.2f} MWh ({e.get('total_kwh', 0):,.1f} kWh)
• Scope 2 Carbon Footprint:    {e.get('carbon_emissions_tco2e', 0):,.2f} metric tonnes CO2e
• Off-Hours Baseload Fraction: {e.get('off_hours_baseload_pct', 0):.1f}% of total energy
• Potable Water Usage:         {w.get('total_m3', 0):,.1f} m³ ({w.get('total_liters', 0):,.0f} Liters)
• Minimum Night Flow (MNF):    {w.get('min_night_flow_m3_h', 0):.2f} m³/hour
• Landfill Diversion Rate:     {r.get('diversion_rate_pct', 0):.1f}% (Status: {r.get('status', 'Tracking')})
• Total Utility Expenditure:   ${(e.get('estimated_cost_usd', 0) + w.get('estimated_cost_usd', 0) + r.get('estimated_hauling_cost_usd', 0)):,.2f} USD

================================================================================
2. ANOMALY & OPERATIONAL RISK SUMMARY
================================================================================
• Total Flagged Incidents:     {len(filtered_anomalies)}
• Critical Severity:           {len(filtered_anomalies[filtered_anomalies['severity_tier'] == 'CRITICAL']) if not filtered_anomalies.empty else 0}
• High Severity:               {len(filtered_anomalies[filtered_anomalies['severity_tier'] == 'HIGH']) if not filtered_anomalies.empty else 0}

================================================================================
3. PRIORITIZED OPERATIONAL ACTION ITEMS
================================================================================
1. [Energy] Verify HVAC thermostat setback schedules (16°C heating / 28°C cooling)
   during unoccupied windows (20:00 - 06:00).
2. [Water] Inspect restroom flushometer sensors and cooling tower makeup valves
   in facilities exhibiting night flow > 1.0 m³/hour.
3. [Waste] Expand back-of-house kitchen organic composting in dining facilities
   (Building C) to elevate diversion towards 75% Gold status.

Responsible AI Notice: Diagnostic hypotheses generated using contextual machine
learning baselines and international engineering standards. All recommendations
warrant human facility engineering verification prior to mechanical alterations.
"""

    st.text_area("Generated Audit Report Preview", report_text, height=380)

    st.download_button(
        label="📥 Download Audit Report (.txt)",
        data=report_text,
        file_name=f"EcoSync_Audit_Report_{selected_building}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )


# -----------------------------------------------------------------------------
# PAGE 11: ℹ️ ABOUT & IMPACT (SUSTAINABILITY & AI ARCHITECTURE)
# -----------------------------------------------------------------------------
elif nav_selection in ["ℹ️ About & Impact (SDG / AI)", "🌐 Sustainability & SDG Framework"]:
    st.markdown(
        """
        <div class="ecosync-header" style="margin-bottom: 22px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                <div>
                    <div style="display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2); border-radius:9999px; padding:4px 14px; font-size:0.75rem; font-weight:700; letter-spacing:0.4px; margin-bottom:8px;">
                        <span>🌱 ECOSYNC SUSTAINABILITY PLATFORM • AUTHOR VISION</span>
                    </div>
                    <h1 style="margin:0; font-size:2.3rem;">EcoSync Resource Intelligence</h1>
                    <p style="color:#a7f3d0 !important; font-size:1.05rem; margin-top:6px;">
                        Autonomous AI Decision Support for Multi-Resource Optimization • UN SDG 7 · SDG 6 · SDG 12
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Personal Founder's Note
    st.markdown(
        """
        <div class="uiverse-card" style="background: rgba(15, 23, 42, 0.9); border-left: 5px solid #10b981; margin-bottom: 20px;">
            <div style="font-size:0.8rem; font-weight:700; color:#34d399; text-transform:uppercase; margin-bottom:4px;">Founder's Statement & Inspiration</div>
            <h3 style="margin:0 0 8px 0; color:#f8fafc !important;">Why I Conceived and Engineered EcoSync</h3>
            <p style="font-size:0.9rem; color:#cbd5e1 !important; line-height:1.6; margin:0;">
                Traditional facility management relies on monthly utility bills and siloed meters. By the time a broken water riser, stuck HVAC damper, or uncontrolled baseload spike is noticed on an invoice 30 days later, hundreds of megawatt-hours and thousands of cubic meters of potable water have already been wasted.
                <br><br>
                I engineered <b>EcoSync</b> as an integrated AI command center that continuously listens to multi-resource telemetry. It combines contextual physics baselines, machine learning anomaly detection, cyclical demand forecasting, and grounded IBM Granite intelligence to surface actionable operational insights within hours—empowering teams to conserve energy, prevent water loss, and divert landfill waste.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Quick Stat Highlights in Dark Theme
    hl1, hl2, hl3, hl4 = st.columns(4)
    with hl1:
        st.markdown(
            """
            <div class="uiverse-card uiverse-glow-amber" style="padding:16px 18px; margin-bottom:16px;">
                <div style="font-size:0.75rem; font-weight:700; color:#fbbf24; text-transform:uppercase;">SDG 7 (Primary)</div>
                <div style="font-size:1.6rem; font-weight:800; color:#f8fafc !important; margin:4px 0;">15% – 25%</div>
                <div style="font-size:0.8rem; color:#94a3b8;">Off-Hours Energy Reduction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hl2:
        st.markdown(
            """
            <div class="uiverse-card uiverse-glow-cyan" style="padding:16px 18px; margin-bottom:16px;">
                <div style="font-size:0.75rem; font-weight:700; color:#38bdf8; text-transform:uppercase;">SDG 6 (Secondary)</div>
                <div style="font-size:1.6rem; font-weight:800; color:#f8fafc !important; margin:4px 0;">Up to 40%</div>
                <div style="font-size:0.8rem; color:#94a3b8;">Water Loss Avoidance (MNF)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hl3:
        st.markdown(
            """
            <div class="uiverse-card uiverse-glow-emerald" style="padding:16px 18px; margin-bottom:16px;">
                <div style="font-size:0.75rem; font-weight:700; color:#34d399; text-transform:uppercase;">SDG 12 (Secondary)</div>
                <div style="font-size:1.6rem; font-weight:800; color:#f8fafc !important; margin:4px 0;">75% Gold</div>
                <div style="font-size:0.8rem; color:#94a3b8;">Zero-Waste Diversion Trajectory</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hl4:
        st.markdown(
            """
            <div class="uiverse-card uiverse-glow-purple" style="padding:16px 18px; margin-bottom:16px;">
                <div style="font-size:0.75rem; font-weight:700; color:#a78bfa; text-transform:uppercase;">Responsible AI</div>
                <div style="font-size:1.6rem; font-weight:800; color:#f8fafc !important; margin:4px 0;">100% Grounded</div>
                <div style="font-size:0.8rem; color:#94a3b8;">Zero-Hallucination Guardrails</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    tab_abt1, tab_abt2, tab_abt3, tab_abt4 = st.tabs([
        "🎯 UN SDG Alignment",
        "🧠 AI Architecture & IBM Tech",
        "🛡️ Responsible AI Framework",
        "📈 Impact & Methodology",
    ])

    with tab_abt1:
        st.markdown("### **United Nations Sustainable Development Goals (SDG) Alignment**")
        st.markdown(
            "EcoSync is structured around **one primary SDG** with concrete targets, "
            "interconnected with secondary resource challenges that share physical facility engineering infrastructure."
        )

        sdg_col1, sdg_col2, sdg_col3 = st.columns(3)
        with sdg_col1:
            st.markdown(
                """
                <div class="uiverse-card uiverse-glow-amber">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span class="chip-badge chip-amber">PRIMARY FOCUS</span>
                        <span style="font-weight:800; font-size:1.1rem; color:#fbbf24;">SDG 7</span>
                    </div>
                    <h3 style="margin:0 0 6px 0; color:#f8fafc !important;">Affordable & Clean Energy</h3>
                    <div style="font-weight:700; color:#fbbf24; font-size:0.85rem; margin-bottom:12px;">
                        Target 7.3: Double the global rate of improvement in energy efficiency
                    </div>
                    <p style="font-size:0.88rem; color:#cbd5e1 !important; line-height:1.55; margin-bottom:12px;">
                        <b>The Problem:</b> Facilities waste 15% to 30% of energy during unoccupied night and weekend hours due to manual thermostat overrides, uncalibrated dampers, and neglected setback schedules.
                    </p>
                    <p style="font-size:0.88rem; color:#cbd5e1 !important; line-height:1.55; margin-bottom:14px;">
                        <b>How EcoSync Solves It:</b> Continuous telemetry ingestion coupled with Isolation Forest and contextual median baselines surfaces parasitic baseload within hours, directly reducing Scope 2 emissions.
                    </p>
                    <div style="background:rgba(245, 158, 11, 0.15); border:1px solid rgba(245, 158, 11, 0.35); border-radius:8px; padding:10px 12px; font-size:0.8rem; color:#fcd34d;">
                        <b>Measurable KPI:</b> Off-hours baseload fraction (%) & peak demand shaving (kW) aligned with ASHRAE Standard 90.1.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sdg_col2:
            st.markdown(
                """
                <div class="uiverse-card uiverse-glow-cyan">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span class="chip-badge chip-cyan">SECONDARY FOCUS</span>
                        <span style="font-weight:800; font-size:1.1rem; color:#38bdf8;">SDG 6</span>
                    </div>
                    <h3 style="margin:0 0 6px 0; color:#f8fafc !important;">Clean Water & Sanitation</h3>
                    <div style="font-weight:700; color:#38bdf8; font-size:0.85rem; margin-bottom:12px;">
                        Target 6.4: Substantially increase water-use efficiency across all sectors
                    </div>
                    <p style="font-size:0.88rem; color:#cbd5e1 !important; line-height:1.55; margin-bottom:12px;">
                        <b>The Problem:</b> Commercial plumbing fixtures (stuck auto-flush solenoids, cooling tower makeup valve drift, broken irrigation risers) waste thousands of gallons unnoticed for 30+ days until utility bills arrive.
                    </p>
                    <p style="font-size:0.88rem; color:#cbd5e1 !important; line-height:1.55; margin-bottom:14px;">
                        <b>How EcoSync Solves It:</b> Automated Minimum Night Flow (MNF 02:00–04:30) algorithm establishes continuous minimum benchmarks, flagging plumbing leaks within 2 hours of inception.
                    </p>
                    <div style="background:rgba(6, 182, 212, 0.15); border:1px solid rgba(6, 182, 212, 0.35); border-radius:8px; padding:10px 12px; font-size:0.8rem; color:#7dd3fc;">
                        <b>Measurable KPI:</b> MNF flow rate (m³/h) and uncontained volume avoidance aligned with EPA WaterSense standards.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sdg_col3:
            st.markdown(
                """
                <div class="uiverse-card uiverse-glow-emerald">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span class="chip-badge chip-emerald">SECONDARY FOCUS</span>
                        <span style="font-weight:800; font-size:1.1rem; color:#34d399;">SDG 12</span>
                    </div>
                    <h3 style="margin:0 0 6px 0; color:#f8fafc !important;">Responsible Consumption</h3>
                    <div style="font-weight:700; color:#34d399; font-size:0.85rem; margin-bottom:12px;">
                        Target 12.5: Substantially reduce waste generation through diversion
                    </div>
                    <p style="font-size:0.88rem; color:#cbd5e1 !important; line-height:1.55; margin-bottom:12px;">
                        <b>The Problem:</b> Dining facilities and office buildings send over 60% of recyclable and compostable organic waste directly to landfills, generating potent fugitive methane emissions and expensive hauling surcharges.
                    </p>
                    <p style="font-size:0.88rem; color:#cbd5e1 !important; line-height:1.55; margin-bottom:14px;">
                        <b>How EcoSync Solves It:</b> Dynamic multi-stream tracking (Landfill vs Recycled vs Compost) with what-if scenario counterfactuals simulates exact carbon and financial returns of food composting.
                    </p>
                    <div style="background:rgba(16, 185, 129, 0.15); border:1px solid rgba(16, 185, 129, 0.35); border-radius:8px; padding:10px 12px; font-size:0.8rem; color:#6ee7b7;">
                        <b>Measurable KPI:</b> Landfill Diversion Rate (%) tracking towards 75% Gold and 90% Zero-Waste True standard.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_abt2:
        st.markdown("### **AI Technologies & Architecture Pipeline**")
        st.markdown("EcoSync integrates an end-to-end pipeline combining deterministic engineering physics with modern generative and predictive AI.")

        # Visual Pipeline Grid in Dark Theme
        st.markdown(
            """
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 16px 0;">
                <div class="uiverse-card uiverse-glow-emerald">
                    <div style="font-size:0.75rem; font-weight:800; color:#34d399; margin-bottom:4px;">STAGE 1 • INGESTION</div>
                    <h4 style="margin:0 0 8px 0; color:#f8fafc !important;">📡 Multi-Resource Telemetry Engine</h4>
                    <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:0;">
                        Validates continuous hourly sensor streams (Energy kWh, Water m³, Waste kg). Auto-detects schemas, enforces range boundaries, and resolves missing values.
                    </p>
                </div>
                <div class="uiverse-card uiverse-glow-amber">
                    <div style="font-size:0.75rem; font-weight:800; color:#fbbf24; margin-bottom:4px;">STAGE 2 • ANOMALY AI</div>
                    <h4 style="margin:0 0 8px 0; color:#f8fafc !important;">🌲 Isolation Forest + Empirical Baselines</h4>
                    <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:0;">
                        Computes contextual medians conditioned on day-of-week and hour-of-day. Scikit-learn Isolation Forest produces normalized anomaly severity scores (0–100).
                    </p>
                </div>
                <div class="uiverse-card uiverse-glow-cyan">
                    <div style="font-size:0.75rem; font-weight:800; color:#38bdf8; margin-bottom:4px;">STAGE 3 • FORECASTING</div>
                    <h4 style="margin:0 0 8px 0; color:#f8fafc !important;">📈 Cyclical Ridge Demand Regressor</h4>
                    <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:0;">
                        Encodes diurnal time cycles via sine/cosine transformations. Produces 24h, 48h, and 7-day demand projections bounded by 95% confidence intervals.
                    </p>
                </div>
                <div class="uiverse-card uiverse-glow-purple">
                    <div style="font-size:0.75rem; font-weight:800; color:#a78bfa; margin-bottom:4px;">STAGE 4 • KNOWLEDGE RAG</div>
                    <h4 style="margin:0 0 8px 0; color:#f8fafc !important;">📚 TF-IDF Semantic Domain Retriever</h4>
                    <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:0;">
                        Cosine semantic search over curated markdown knowledge bases indexing ASHRAE Standard 90.1, EPA WaterSense, and Zero Waste protocols.
                    </p>
                </div>
                <div class="uiverse-card uiverse-glow-emerald" style="grid-column: 1 / -1;">
                    <div style="font-size:0.75rem; font-weight:800; color:#34d399; margin-bottom:4px;">STAGE 5 • FOUNDATION MODEL SYNTHESIS</div>
                    <h4 style="margin:0 0 8px 0; color:#f8fafc !important;">🤖 IBM Granite 3-3-8B Instruct via watsonx.ai</h4>
                    <p style="font-size:0.88rem; color:#cbd5e1 !important; line-height:1.55; margin:0;">
                        Autonomous copilot agent dynamically routes user intents to backend tools, formats grounded telemetry prompts, and synthesizes actionable recommendations through IBM Granite. In offline mode, executes fully deterministic template fallbacks with zero hallucinations.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### **Verified Technology Stack**")
        st.markdown(
            """
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">
                <span class="chip-badge chip-emerald">IBM watsonx.ai</span>
                <span class="chip-badge chip-emerald">IBM Granite 3-3-8B Instruct</span>
                <span class="chip-badge chip-amber">Python 3.12</span>
                <span class="chip-badge chip-cyan">Scikit-Learn (Isolation Forest & Ridge)</span>
                <span class="chip-badge chip-purple">Streamlit Interactive Framework</span>
                <span class="chip-badge chip-slate">Plotly Express & Graph Objects</span>
                <span class="chip-badge chip-slate">Pandas & NumPy</span>
                <span class="chip-badge chip-slate">ASHRAE 90.1 & EPA WaterSense RAG</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_abt3:
        st.markdown("### **Responsible AI & Ethical Governance Framework**")
        st.markdown("Every AI output in EcoSync is governed by four foundational responsible AI pillars:")

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown(
                """
                <div class="uiverse-card uiverse-glow-emerald">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                        <span style="font-size:1.3rem;">⚖️</span>
                        <h4 style="margin:0; color:#f8fafc !important;">1. Fairness & Contextual Objectivity</h4>
                    </div>
                    <p style="font-size:0.86rem; color:#cbd5e1 !important; line-height:1.5; margin:0;">
                        Generic static thresholds unfairly penalize buildings during legitimate high-occupancy events. EcoSync models normal behavior using facility-specific empirical medians conditioned on day-of-week and hour-of-day.
                    </p>
                </div>
                <div class="uiverse-card uiverse-glow-amber">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                        <span style="font-size:1.3rem;">🛡️</span>
                        <h4 style="margin:0; color:#f8fafc !important;">2. Hypothetical Demarcation (Anti-Hallucination)</h4>
                    </div>
                    <p style="font-size:0.86rem; color:#cbd5e1 !important; line-height:1.5; margin:0;">
                        The AI explicitly qualifies statements as hypotheses (e.g. <i>"Observed night flow suggests a possible stuck flushometer valve"</i>) rather than asserting false certainty. Responses are formally structured into <b>FACT</b>, <b>HYPOTHESIS</b>, and <b>RECOMMENDATION</b>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r_col2:
            st.markdown(
                """
                <div class="uiverse-card uiverse-glow-cyan">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                        <span style="font-size:1.3rem;">🔍</span>
                        <h4 style="margin:0; color:#f8fafc !important;">3. Full Transparency & Provenance</h4>
                    </div>
                    <p style="font-size:0.86rem; color:#cbd5e1 !important; line-height:1.5; margin:0;">
                        Every AI diagnosis cites exact engineering sources (e.g. <i>ASHRAE Standard 90.1-2019 Section 6.4</i>, <i>EPA WaterSense Commercial Restroom Guidelines</i>), discloses tools dispatched, and provides numerical confidence scores.
                    </p>
                </div>
                <div class="uiverse-card uiverse-glow-purple">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                        <span style="font-size:1.3rem;">👷</span>
                        <h4 style="margin:0; color:#f8fafc !important;">4. Human-in-the-Loop Agency</h4>
                    </div>
                    <p style="font-size:0.86rem; color:#cbd5e1 !important; line-height:1.5; margin:0;">
                        EcoSync operates in an advisory decision-support capacity. It equips human facilities engineering teams with actionable verification checklists rather than executing irreversible autonomous physical actuator commands.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="uiverse-card" style="background: rgba(15, 23, 42, 0.9); border-left:4px solid #059669; padding:16px 20px;">
                <h4 style="margin:0 0 6px 0; color:#f8fafc !important;">🔒 Data Privacy & Anonymity Commitment</h4>
                <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:0;">
                    EcoSync operates strictly on aggregate physical meter telemetry (kWh, m³, kg). No personally identifiable information (PII), individual room occupancy tracking, or facial recognition feeds are required or ingested.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_abt4:
        st.markdown("### **Impact Evaluation & Assumptions Methodology**")
        st.markdown("To ensure scientific validity, all financial and emissions return projections are derived from published engineering standards:")

        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown(
                """
                <div class="uiverse-card uiverse-glow-emerald">
                    <h4 style="margin:0 0 12px 0; color:#f8fafc !important;">💰 Financial & Environmental Conversion Factors</h4>
                    <table style="width:100%; font-size:0.86rem; border-collapse:collapse;">
                        <tr style="border-bottom:1px solid rgba(255, 255, 255, 0.08); padding:6px 0;">
                            <td style="padding:6px 0; color:#94a3b8;">Electricity Tariff</td>
                            <td style="text-align:right; font-weight:700; color:#f8fafc;">$0.140 USD / kWh</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255, 255, 255, 0.08);">
                            <td style="padding:6px 0; color:#94a3b8;">Grid Scope 2 Carbon Intensity</td>
                            <td style="text-align:right; font-weight:700; color:#f8fafc;">0.420 kg CO₂e / kWh</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255, 255, 255, 0.08);">
                            <td style="padding:6px 0; color:#94a3b8;">Potable Municipal Water Rate</td>
                            <td style="text-align:right; font-weight:700; color:#f8fafc;">$3.800 USD / m³</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255, 255, 255, 0.08);">
                            <td style="padding:6px 0; color:#94a3b8;">Landfill Waste Hauling Fee</td>
                            <td style="text-align:right; font-weight:700; color:#f8fafc;">$0.120 USD / kg</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0; color:#94a3b8;">Landfill Methane Emissions (EPA WARM)</td>
                            <td style="text-align:right; font-weight:700; color:#f8fafc;">0.580 kg CO₂e / kg waste</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ic2:
            st.markdown(
                """
                <div class="uiverse-card uiverse-glow-amber">
                    <h4 style="margin:0 0 12px 0; color:#f8fafc !important;">🌍 Storytelling & Human Equivalencies</h4>
                    <div style="margin-bottom:12px;">
                        <div style="font-weight:700; color:#34d399; font-size:0.9rem;">🌲 1 Mature Urban Tree Equivalent</div>
                        <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:2px 0 0 0;">
                            Absorbs approximately <b>21.77 kg CO₂e / year</b>. A 20-tonne monthly setback saving equals planting <b>918 urban trees</b>.
                        </p>
                    </div>
                    <div>
                        <div style="font-weight:700; color:#38bdf8; font-size:0.9rem;">🚗 1 Typical Passenger Vehicle</div>
                        <p style="font-size:0.85rem; color:#cbd5e1 !important; margin:2px 0 0 0;">
                            Emits approximately <b>4.60 metric tonnes CO₂e / year</b>. Diverting 50 tonnes of organic waste eliminates equivalent emissions of <b>6 passenger vehicles</b>.
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### **Target User Personas**")
        st.markdown(
            """
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin-top:8px;">
                <div style="background:rgba(15, 23, 42, 0.85); border:1px solid rgba(255, 255, 255, 0.08); border-radius:12px; padding:14px;">
                    <div style="font-weight:700; color:#f8fafc; font-size:0.9rem;">🏢 Facility Operations Managers</div>
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Continuous baseload and water leak surveillance.</div>
                </div>
                <div style="background:rgba(15, 23, 42, 0.85); border:1px solid rgba(255, 255, 255, 0.08); border-radius:12px; padding:14px;">
                    <div style="font-weight:700; color:#34d399; font-size:0.9rem;">🌿 Sustainability Officers</div>
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Automated Scope 2 carbon & zero-waste diversion reporting.</div>
                </div>
                <div style="background:rgba(15, 23, 42, 0.85); border:1px solid rgba(255, 255, 255, 0.08); border-radius:12px; padding:14px;">
                    <div style="font-weight:700; color:#a78bfa; font-size:0.9rem;">👔 Executive Leadership</div>
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Portfolio-wide utility expenditure & ESG audit compliance.</div>
                </div>
                <div style="background:rgba(15, 23, 42, 0.85); border:1px solid rgba(255, 255, 255, 0.08); border-radius:12px; padding:14px;">
                    <div style="font-weight:700; color:#fbbf24; font-size:0.9rem;">🔧 Maintenance Technicians</div>
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Prioritized root-cause checklists for immediate site dispatch.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

