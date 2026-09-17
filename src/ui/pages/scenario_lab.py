"""src/ui/pages/scenario_lab.py — What-If Scenario Lab page."""
import pandas as pd
import streamlit as st

from src.ui.components import page_header, kpi_card, estimate_notice


def render(active_df: pd.DataFrame, whatif_simulator, building: str) -> None:
    page_header("Scenario Lab", "Model the impact of operational changes on cost, carbon, and resource usage")

    st.markdown(
        '<div class="warning-card">⚠️ <b>Modelled Estimates Only</b> — All values are projections '
        'based on historical telemetry baselines. Actual results depend on building systems, occupancy, '
        'and implementation quality. Review assumptions with facility engineering before decisions.</div>',
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    all_bldgs = ["ALL"] + sorted(active_df["building"].unique().tolist()) if "building" in active_df.columns else ["ALL"]
    default_idx = all_bldgs.index(building) if building in all_bldgs else 0

    st.markdown("#### Simulation Scope")
    bldg_sel = st.selectbox("Facility", all_bldgs, index=default_idx, key="sim_bldg")

    st.markdown("---")
    c_e, c_w, c_r = st.columns(3)

    with c_e:
        st.markdown('<div class="section-label">Energy Levers</div>', unsafe_allow_html=True)
        off_hours_pct = st.slider("Off-hours HVAC setback %", 0, 50, 20, step=5,
                                  help="Reduction in energy use during unoccupied hours (20:00–06:00)", key="sim_oh")
        peak_shave_pct = st.slider("Peak demand shaving %", 0, 30, 10, step=5,
                                   help="Demand-side management via equipment stagger", key="sim_ps")

    with c_w:
        st.markdown('<div class="section-label">Water Levers</div>', unsafe_allow_html=True)
        leak_fix_pct = st.slider("Night-flow leak remediation %", 0, 100, 40, step=10,
                                 help="Percentage of minimum night flow anomaly addressed", key="sim_lf")
        greywater_pct = st.slider("Greywater / cooling reuse %", 0, 50, 15, step=5,
                                  help="Cooling tower condensate and fixture grey-water reuse", key="sim_gw")

    with c_r:
        st.markdown('<div class="section-label">Waste Levers</div>', unsafe_allow_html=True)
        compost_pct = st.slider("Organic composting increase %", 0, 80, 30, step=10,
                                help="Additional food waste redirected to compost", key="sim_cp")
        source_red_pct = st.slider("Source reduction %", 0, 30, 10, step=5,
                                   help="Upstream waste prevention (procurement, packaging)", key="sim_sr")

    # ── Run simulation ────────────────────────────────────────────────────────
    with st.spinner("Calculating scenario impact…"):
        try:
            bldg_param = None if bldg_sel == "ALL" else bldg_sel
            result = whatif_simulator.simulate(
                active_df,
                building=bldg_param,
                off_hours_reduction_pct=off_hours_pct,
                peak_shaving_pct=peak_shave_pct,
                night_water_leak_fix_pct=leak_fix_pct,
                greywater_reuse_pct=greywater_pct,
                compost_diversion_target_pct=compost_pct,
                source_reduction_pct=source_red_pct,
            )
            sim = result.to_dict() if hasattr(result, "to_dict") else {}
        except Exception as e:
            st.error(f"Simulation error: {e}")
            return

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Projected Impact")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(
            kpi_card("Total Cost Saved", f"${sim.get('total_cost_saved_usd', 0):,.0f}",
                     "USD / year (estimated)"),
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            kpi_card("Carbon Avoided", f"{sim.get('carbon_avoided_tco2e', 0):.2f}",
                     "metric tonnes CO₂e"),
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            kpi_card("Energy Conserved", f"{sim.get('energy_saved_kwh', 0):,.0f}",
                     f"kWh ({sim.get('energy_saved_mwh', 0):.1f} MWh)"),
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            kpi_card("Water Conserved", f"{sim.get('water_saved_m3', 0):.1f}",
                     f"m³ ({sim.get('water_saved_liters', 0):,.0f} L)"),
            unsafe_allow_html=True,
        )

    r5, r6, r7 = st.columns(3)
    with r5:
        st.markdown(
            kpi_card("Waste Diverted", f"{sim.get('waste_diverted_kg', 0):,.0f}",
                     f"kg · new diversion rate: {sim.get('sdg12_diversion_rate_projected_pct', 0):.1f}%"),
            unsafe_allow_html=True,
        )
    with r6:
        st.markdown(
            kpi_card("Cars Removed (equiv.)", f"{sim.get('equivalent_cars_removed_annual', 0):.1f}",
                     "passenger vehicles / year"),
            unsafe_allow_html=True,
        )
    with r7:
        st.markdown(
            kpi_card("Trees Planted (equiv.)", f"{sim.get('equivalent_trees_planted', 0):,.0f}",
                     "urban tree equivalents"),
            unsafe_allow_html=True,
        )

    # Assumptions
    with st.expander("Methodology & Assumptions", expanded=False):
        st.markdown(f"""
**Cost factors:**
- Electricity: $0.14/kWh | Water: $3.80/m³ | Landfill hauling: $0.12/kg

**Emissions factors:**
- Grid: 0.420 kg CO₂e/kWh | Landfill: 0.580 kg CO₂e/kg waste

**Carbon equivalencies:**
- Passenger vehicle: 4.6 tCO₂e/year | Urban tree: 21.77 kg CO₂e/year

**Scenario applied to:** {bldg_sel}

**Off-hours definition:** 20:00–06:00 weekdays, all hours weekends.

*These are modelled estimates. Results are linear approximations from the historical baseline and do not account for diminishing returns, equipment constraints, or seasonal variation.*
""")

    estimate_notice()
