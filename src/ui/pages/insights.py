"""src/ui/pages/insights.py — Insights page (Energy / Water / Waste tabs)."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui.components import kpi_card, page_header, section_label


def render(active_df: pd.DataFrame, view_df: pd.DataFrame, kpis: dict, building: str) -> None:
    page_header("Resource Insights", "Interactive analytics across energy, water, and waste streams")

    tab_e, tab_w, tab_r = st.tabs(["⚡  Energy", "💧  Water", "♻️  Waste"])

    # ── ENERGY ────────────────────────────────────────────────────────────────
    with tab_e:
        e = kpis.get("energy", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("Total Consumption", f"{e.get('total_mwh', 0):,.2f} MWh",
                                 f"{e.get('total_kwh', 0):,.0f} kWh"), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Peak Demand", f"{e.get('peak_kw', 0):,.1f} kW",
                                 f"Peak: {e.get('peak_building', 'N/A')}"), unsafe_allow_html=True)
        with c3:
            ob = e.get("off_hours_baseload_pct", 0)
            st.markdown(kpi_card("Off-Hours Baseload", f"{ob:.1f}%",
                                 "of total energy in unoccupied hours",
                                 warn=(ob > 20)), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Carbon Footprint", f"{e.get('carbon_emissions_tco2e', 0):.2f} tCO₂e",
                                 f"${e.get('estimated_cost_usd', 0):,.0f} estimated cost"), unsafe_allow_html=True)

        if "energy_kwh" in view_df.columns and "timestamp" in view_df.columns:
            color_col = "building" if (building == "ALL" and "building" in view_df.columns) else None
            df_s = view_df.sort_values("timestamp")
            fig = px.line(df_s, x="timestamp", y="energy_kwh", color=color_col,
                          labels={"energy_kwh": "Energy (kWh)", "timestamp": ""},
                          color_discrete_sequence=px.colors.qualitative.Prism)
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10),
                              hovermode="x unified", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                              title="Hourly Energy Demand Profile")
            fig.update_xaxes(gridcolor="#f1f5f9")
            fig.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)

        # building breakdown
        if "energy_kwh" in active_df.columns and "building" in active_df.columns:
            bldg_e = active_df.groupby("building")["energy_kwh"].sum().sort_values(ascending=False).reset_index()
            fig2 = px.bar(bldg_e, x="building", y="energy_kwh", color="building",
                          labels={"energy_kwh": "Total kWh"},
                          color_discrete_sequence=px.colors.qualitative.Safe)
            fig2.update_layout(height=260, showlegend=False, margin=dict(l=10,r=10,t=30,b=10),
                               plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                               title="Total Energy by Facility")
            fig2.update_xaxes(gridcolor="#f1f5f9")
            fig2.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            '<div class="accent-card"><b>Off-Hours Efficiency Note:</b> ASHRAE 90.1 recommends '
            'automated HVAC setbacks (16°C heating / 28°C cooling) during unoccupied periods (20:00–06:00). '
            f'Current off-hours baseload is <b>{e.get("off_hours_baseload_pct", 0):.1f}%</b> of total consumption.</div>',
            unsafe_allow_html=True,
        )

    # ── WATER ─────────────────────────────────────────────────────────────────
    with tab_w:
        w = kpis.get("water", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("Total Consumption", f"{w.get('total_m3', 0):,.1f} m³",
                                 f"{w.get('total_liters', 0):,.0f} litres"), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Peak Flow", f"{w.get('peak_m3_h', 0):.2f} m³/h",
                                 "Maximum hourly flow rate"), unsafe_allow_html=True)
        with c3:
            mnf = w.get("min_night_flow_m3_h", 0)
            st.markdown(kpi_card("Min Night Flow", f"{mnf:.3f} m³/h",
                                 "02:00–05:00 average (leak indicator)",
                                 warn=(mnf > 0.15)), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Water Cost", f"${w.get('estimated_cost_usd', 0):,.0f}",
                                 "Estimated utility expenditure"), unsafe_allow_html=True)

        if "water_m3" in view_df.columns and "timestamp" in view_df.columns:
            color_col = "building" if (building == "ALL" and "building" in view_df.columns) else None
            df_s = view_df.sort_values("timestamp")
            fig = px.line(df_s, x="timestamp", y="water_m3", color=color_col,
                          labels={"water_m3": "Water (m³)", "timestamp": ""},
                          color_discrete_sequence=px.colors.qualitative.Vivid)
            # Add MNF threshold
            fig.add_hline(y=0.15, line_dash="dash", line_color="#f59e0b",
                          annotation_text="MNF Threshold 0.15 m³/h", annotation_position="top right")
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10),
                              hovermode="x unified", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                              title="Hourly Water Flow Profile")
            fig.update_xaxes(gridcolor="#f1f5f9")
            fig.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)

        if "water_m3" in active_df.columns and "building" in active_df.columns:
            bldg_w = active_df.groupby("building")["water_m3"].sum().reset_index()
            fig2 = px.pie(bldg_w, names="building", values="water_m3", hole=0.42,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(height=260, margin=dict(l=10,r=10,t=30,b=10),
                               title="Water Distribution by Facility")
            st.plotly_chart(fig2, use_container_width=True)

        mnf = w.get("min_night_flow_m3_h", 0)
        if mnf > 0.15:
            st.markdown(
                f'<div class="warning-card">⚠️ <b>Leak Risk:</b> Minimum night flow is '
                f'<b>{mnf:.3f} m³/h</b>, above the 0.15 m³/h benchmark. '
                'MNF above threshold during 02:00–05:00 is a strong indicator of uncontained plumbing flow.</div>',
                unsafe_allow_html=True,
            )

    # ── WASTE ─────────────────────────────────────────────────────────────────
    with tab_r:
        r = kpis.get("waste", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("Total Waste", f"{r.get('total_waste_kg', 0):,.0f} kg",
                                 f"{r.get('total_waste_tonnes', 0):.2f} tonnes"), unsafe_allow_html=True)
        with c2:
            div = r.get("diversion_rate_pct", 0)
            st.markdown(kpi_card("Diversion Rate", f"{div:.1f}%",
                                 r.get("status", "Tracking"),
                                 alert=(div < 40), warn=(40 <= div < 60)), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("Diverted from Landfill", f"{r.get('diverted_kg', 0):,.0f} kg",
                                 "Composted or recycled"), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Landfill Cost", f"${r.get('estimated_hauling_cost_usd', 0):,.0f}",
                                 f"{r.get('landfill_kg', 0):,.0f} kg to landfill"), unsafe_allow_html=True)

        if "building" in active_df.columns:
            waste_cols = [c for c in ["waste_kg", "waste_total_kg"] if c in active_df.columns]
            if waste_cols:
                wcol = waste_cols[0]
                bldg_r = active_df.groupby("building")[wcol].sum().reset_index()
                fig = px.bar(bldg_r, x="building", y=wcol, color="building",
                             labels={wcol: "Waste (kg)"},
                             color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(height=260, showlegend=False, margin=dict(l=10,r=10,t=30,b=10),
                                  plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                                  title="Total Waste by Facility")
                fig.update_xaxes(gridcolor="#f1f5f9")
                fig.update_yaxes(gridcolor="#f1f5f9")
                st.plotly_chart(fig, use_container_width=True)

        # Diversion tier guide
        div = r.get("diversion_rate_pct", 0)
        tier_html = ""
        for label, lo, hi, col in [("Bronze", 0, 40, "#cd7f32"), ("Silver", 40, 60, "#9ca3af"), ("Gold", 60, 100, "#d97706")]:
            active = "font-weight:700;" if lo <= div < hi else "opacity:0.5;"
            tier_html += f'<span style="padding:3px 10px;border-radius:4px;background:#f1f5f9;color:{col};{active}margin-right:6px;">{label} {lo}–{hi}%</span>'
        st.markdown(
            f'<div class="content-card">Diversion tier: {tier_html}<br>'
            f'<small style="color:#64748b;">Current rate: <b>{div:.1f}%</b> — target 75% (Gold) per facility sustainability plan.</small></div>',
            unsafe_allow_html=True,
        )
