"""src/ui/components.py — Shared card/badge helper functions."""
import streamlit as st
from typing import Optional


def kpi_card(label: str, value: str, sub: str = "", alert: bool = False, warn: bool = False) -> str:
    sub_class = "kpi-alert" if alert else ("kpi-warn" if warn else "kpi-sub")
    return f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="{sub_class}">{sub}</div>
</div>"""


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def pipeline_banner() -> None:
    st.markdown(
        """
<div class="pipeline-banner">
  <div class="pipeline-step"><span class="pipeline-num">1</span><b>Detect</b></div>
  <span class="pipeline-arrow">›</span>
  <div class="pipeline-step"><span class="pipeline-num">2</span><b>Investigate</b></div>
  <span class="pipeline-arrow">›</span>
  <div class="pipeline-step"><span class="pipeline-num">3</span><b>Explain</b></div>
  <span class="pipeline-arrow">›</span>
  <div class="pipeline-step"><span class="pipeline-num">4</span><b>Recommend</b></div>
  <span class="pipeline-arrow">›</span>
  <div class="pipeline-step"><span class="pipeline-num">5</span><b>Simulate</b></div>
</div>""",
        unsafe_allow_html=True,
    )


def estimate_notice(msg: str = "All values are modelled estimates based on historical telemetry baselines. Verify with facility engineering before capital decisions.") -> None:
    st.markdown(f'<div class="estimate-notice">⚠️ {msg}</div>', unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def provider_badge(status: str) -> str:
    if "IBM" in status:
        return f'<span class="provider-ibm">🤖 {status}</span>'
    return f'<span class="provider-fallback">🔧 {status}</span>'


def severity_color(tier: str) -> str:
    return {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#f59e0b"}.get(tier.upper(), "#64748b")


def fact_label() -> str:
    return '<span class="label-fact">FACT</span>'


def hypothesis_label() -> str:
    return '<span class="label-hypothesis">HYPOTHESIS</span>'


def recommend_label() -> str:
    return '<span class="label-recommend">RECOMMENDATION</span>'
