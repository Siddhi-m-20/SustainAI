"""src/ui/pages/overview.py — Home / Overview page."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui.components import kpi_card, page_header, pipeline_banner


def render(view_df: pd.DataFrame, kpis: dict, filtered_anomalies: pd.DataFrame, building: str) -> None:
    subtitle = f"Facility scope: <b>{building}</b>" if building != "ALL" else "All facilities"
    page_header("EcoSync Resource Intelligence", subtitle)
    pipeline_banner()

    e = kpis.get("energy", {})
    w = kpis.get("water", {})
    r = kpis.get("waste", {})

    crit_count = 0
    high_count = 0
    if not filtered_anomalies.empty and "severity_tier" in filtered_anomalies.columns:
        crit_count = int((filtered_anomalies["severity_tier"] == "CRITICAL").sum())
        high_count  = int((filtered_anomalies["severity_tier"] == "HIGH").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card("Total Energy", f"{e.get('total_mwh', 0):,.1f} MWh",
                     f"${e.get('estimated_cost_usd', 0):,.0f} · {e.get('carbon_emissions_tco2e', 0):.1f} tCO₂e"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card("Total Water", f"{w.get('total_m3', 0):,.0f} m³",
                     f"Night flow: {w.get('min_night_flow_m3_h', 0):.2f} m³/h"),
            unsafe_allow_html=True,
        )
    with c3:
        div = r.get("diversion_rate_pct", 0)
        st.markdown(
            kpi_card("Waste Diversion", f"{div:.1f}%",
                     f"{r.get('total_waste_kg', 0):,.0f} kg total · {r.get('status', 'Tracking')}"),
            unsafe_allow_html=True,
        )
    with c4:
        alert_sub = f"{crit_count} Critical · {high_count} High"
        st.markdown(
            kpi_card("Active Alerts", str(crit_count + high_count), alert_sub,
                     alert=(crit_count > 0), warn=(high_count > 0 and crit_count == 0)),
            unsafe_allow_html=True,
        )

    # ── Connected nexus narrative ────────────────────────────────────────────────
    st.markdown(
        """
<div class="content-card" style="border-left:4px solid #00a86b;">
<b>Connected Resource Nexus</b> &nbsp;·&nbsp; Energy ↔ Water ↔ Waste<br>
<small style="color:#475569;line-height:1.6;">
Resources are interdependent: dishwashing in <b>Building C</b> drives both water and heating-energy consumption.
<b>Building B</b> labs maintain a continuous 24/7 electrical baseload requiring dedicated chilled-water loops.
Fixing off-hours water leaks reduces booster-pump electricity. Diverting kitchen waste cuts landfill transport emissions.
</small>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("#### Multi-Resource Telemetry")
    t1, t2 = st.tabs(["Load Curves", "Facility Distribution"])

    with t1:
        if "energy_kwh" in view_df.columns and "timestamp" in view_df.columns:
            chart_df = view_df.sort_values("timestamp")
            color_col = "building" if (building == "ALL" and "building" in chart_df.columns) else None
            fig = px.line(
                chart_df, x="timestamp", y="energy_kwh", color=color_col,
                labels={"energy_kwh": "Energy (kWh)", "timestamp": ""},
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                              hovermode="x unified", legend_title_text="Building",
                              plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            fig.update_xaxes(gridcolor="#f1f5f9")
            fig.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        col_a, col_b = st.columns(2)
        with col_a:
            if "energy_kwh" in view_df.columns and "building" in view_df.columns:
                bldg_e = view_df.groupby("building")["energy_kwh"].sum().reset_index()
                fig2 = px.bar(bldg_e, x="building", y="energy_kwh",
                              color="building", labels={"energy_kwh": "Energy (kWh)"},
                              color_discrete_sequence=px.colors.qualitative.Prism)
                fig2.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10),
                                   showlegend=False, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
                fig2.update_xaxes(gridcolor="#f1f5f9")
                fig2.update_yaxes(gridcolor="#f1f5f9", title_text="kWh")
                st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            if "water_m3" in view_df.columns and "building" in view_df.columns:
                bldg_w = view_df.groupby("building")["water_m3"].sum().reset_index()
                fig3 = px.pie(bldg_w, names="building", values="water_m3",
                              hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig3.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig3, use_container_width=True)

    # ── Latest critical alert (quick summary) ────────────────────────────────
    if crit_count > 0 and not filtered_anomalies.empty:
        top_a = filtered_anomalies[filtered_anomalies["severity_tier"] == "CRITICAL"].iloc[0]
        res   = top_a.get("resource", "unknown")
        bldg_a = str(top_a.get("building", "")).replace("_", " ")
        dev   = top_a.get("deviation_pct", 0)
        st.markdown(
            f'<div class="danger-card">🚨 <b>Critical Alert:</b> {res.capitalize()} anomaly in '
            f'<b>{bldg_a}</b> — +{dev:.0f}% above baseline. '
            f'<a href="#" style="color:#ef4444;">→ Open Investigate</a></div>',
            unsafe_allow_html=True,
        )
