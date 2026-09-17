"""src/ui/styles.py — EcoSync design system CSS (Immersive Dark Theme)."""


def get_styles() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}
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

h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}
p, span, label {
    color: #cbd5e1;
}

/* ── Streamlit native element overrides ───────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 6px;
    border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px 10px 0 0;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 10px 22px;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #34d399;
    background: rgba(16, 185, 129, 0.08);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    border-color: #10b981 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
}
.stChatMessage {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    margin-bottom: 8px;
    color: #f1f5f9;
}
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    background: rgba(255, 255, 255, 0.05) !important;
    color: #f1f5f9 !important;
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
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(15, 23, 42, 0.8) !important;
    color: #f8fafc !important;
    font-size: 0.9rem;
}
.stMetric {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 16px;
}
.stExpander {
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(15, 23, 42, 0.7) !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.65rem;
    font-weight: 800;
    color: #f8fafc !important;
}

/* ── Page header ──────────────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #051a10 0%, #06381c 45%, #064e26 85%, #059669 100%);
    padding: 24px 30px;
    border-radius: 16px;
    color: #ffffff;
    margin-bottom: 22px;
    box-shadow: 0 10px 32px -5px rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.12);
}
.page-header h1 { margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.5px; color: #ffffff !important; }
.page-header p  { margin: 6px 0 0; font-size: 0.95rem; color: #a7f3d0 !important; }

/* ── Metric cards ─────────────────────────────────────────────── */
.kpi-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(0, 0, 0, 0.5); border-color: rgba(16, 185, 129, 0.35); }
.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #94a3b8;
    font-weight: 700;
    margin-bottom: 4px;
}
.kpi-value { font-size: 1.75rem; font-weight: 800; color: #f8fafc !important; line-height: 1.15; }
.kpi-sub   { font-size: 0.82rem; color: #34d399; font-weight: 600; margin-top: 4px; }
.kpi-alert { color: #f87171 !important; }
.kpi-warn  { color: #fbbf24 !important; }

/* ── Story / pipeline banner ──────────────────────────────────── */
.pipeline-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 20px;
    font-size: 0.84rem;
    font-weight: 600;
    color: #e2e8f0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}
.pipeline-step { display: flex; align-items: center; gap: 8px; }
.pipeline-num {
    background: #10b981;
    color: #0b0f19;
    font-weight: 800;
    width: 22px; height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    flex-shrink: 0;
}
.pipeline-arrow { color: #64748b; font-size: 1.1rem; }

/* ── General content cards ────────────────────────────────────── */
.content-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
    color: #cbd5e1;
}
.content-card h4 { color: #f8fafc !important; }
.accent-card {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-left: 4px solid #10b981;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 14px;
    color: #e2e8f0;
}
.warning-card {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #fde68a;
}
.danger-card {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-left: 5px solid #ef4444;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 14px;
    color: #fecdd3;
}

/* ── Incident / investigate cards ────────────────────────────── */
.incident-card {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-left: 5px solid #ef4444;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    color: #f1f5f9;
}
.incident-high  { border-left-color: #f97316; border-color: rgba(249, 115, 22, 0.4); }
.incident-med   { border-left-color: #f59e0b; border-color: rgba(245, 158, 11, 0.4); }

/* ── Evidence labels ─────────────────────────────────────────── */
.label-fact       { display:inline-block; padding:3px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; background:rgba(59, 130, 246, 0.2); color:#93c5fd; border:1px solid rgba(59, 130, 246, 0.4); margin-right:4px; }
.label-hypothesis { display:inline-block; padding:3px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; background:rgba(245, 158, 11, 0.2); color:#fde047; border:1px solid rgba(245, 158, 11, 0.4); margin-right:4px; }
.label-recommend  { display:inline-block; padding:3px 10px; border-radius:4px; font-size:0.75rem; font-weight:700; background:rgba(16, 185, 129, 0.2); color:#6ee7b7; border:1px solid rgba(16, 185, 129, 0.4); margin-right:4px; }

/* ── AI response box ─────────────────────────────────────────── */
.ai-response {
    background: rgba(15, 23, 42, 0.85);
    border-left: 3px solid #10b981;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    font-size: 0.94rem;
    line-height: 1.65;
    margin: 12px 0;
    color: #f1f5f9;
    border-top: 1px solid rgba(255,255,255,0.06);
    border-right: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.tool-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #cbd5e1;
    margin: 3px 4px;
}
.tool-chip.done { background: rgba(16, 185, 129, 0.18); border-color: rgba(16, 185, 129, 0.4); color: #34d399; }

/* ── Provider badge ──────────────────────────────────────────── */
.provider-ibm      { background:rgba(30, 58, 138, 0.35); color:#93c5fd; border:1px solid rgba(147, 197, 253, 0.4); border-radius:6px; padding:4px 10px; font-size:0.78rem; font-weight:700; }
.provider-fallback { background:rgba(255, 255, 255, 0.06); color:#cbd5e1; border:1px solid rgba(255, 255, 255, 0.12); border-radius:6px; padding:4px 10px; font-size:0.78rem; font-weight:600; }

/* ── Simulation estimate notice ──────────────────────────────── */
.estimate-notice {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-left: 4px solid #f59e0b;
    color: #fde68a;
    padding: 12px 16px;
    border-radius: 0 10px 10px 0;
    font-size: 0.84rem;
    margin-top: 12px;
}

/* ── Section separator labels ────────────────────────────────── */
.section-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 700;
    color: #94a3b8;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 6px;
}

/* ── SDG badges ──────────────────────────────────────────────── */
.sdg-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    margin: 3px 4px;
    color: #fff;
}
.sdg-7  { background: #d97706; }
.sdg-6  { background: #0284c7; }
.sdg-12 { background: #7c3aed; }

/* ── Sidebar brand ────────────────────────────────────────────── */
.sidebar-brand {
    padding: 4px 0 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 14px;
}
.sidebar-brand h2 {
    margin: 0;
    font-size: 1.3rem;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.4px;
}
.sidebar-brand p {
    margin: 4px 0 0;
    font-size: 0.82rem;
    color: #94a3b8 !important;
}
</style>
"""
