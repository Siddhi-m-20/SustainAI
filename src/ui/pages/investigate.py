"""src/ui/pages/investigate.py — Anomaly investigation page."""
import pandas as pd
import streamlit as st

from src.ui.components import (
    page_header, section_label, fact_label, hypothesis_label, recommend_label, severity_color
)


def render(active_df: pd.DataFrame, anomalies_df: pd.DataFrame,
           filtered_anomalies: pd.DataFrame, knowledge_retriever, building: str) -> None:
    page_header("Investigate", "Anomaly detection, root-cause analysis, and evidence-grounded diagnostics")

    if filtered_anomalies.empty:
        st.info("No anomalies detected for the current scope. All resources are tracking within normal bounds.")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        resources = ["ALL"] + sorted(filtered_anomalies["resource"].dropna().unique().tolist()) if "resource" in filtered_anomalies.columns else ["ALL"]
        res_filter = st.selectbox("Resource", resources, index=0, key="inv_res")
    with col_f2:
        tiers = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
        tier_filter = st.selectbox("Severity Tier", tiers, index=0, key="inv_tier")
    with col_f3:
        min_score = st.slider("Min Severity Score", 0, 100, 0, step=5, key="inv_score")

    display_df = filtered_anomalies.copy()
    if res_filter != "ALL" and "resource" in display_df.columns:
        display_df = display_df[display_df["resource"].str.lower() == res_filter.lower()]
    if tier_filter != "ALL" and "severity_tier" in display_df.columns:
        display_df = display_df[display_df["severity_tier"] == tier_filter]
    if "severity_score" in display_df.columns:
        display_df = display_df[display_df["severity_score"] >= min_score]
        display_df = display_df.sort_values("severity_score", ascending=False)

    st.markdown(f"**{len(display_df)} incidents** matching filters")

    # ── Anomaly table ─────────────────────────────────────────────────────────
    if display_df.empty:
        st.info("No incidents match the current filters.")
        return

    table_cols = ["building", "resource", "severity_tier", "severity_score",
                  "actual_value", "expected_value", "deviation_pct", "timestamp"]
    show_cols = [c for c in table_cols if c in display_df.columns]
    styled = display_df[show_cols].head(50).copy()
    if "timestamp" in styled.columns:
        styled["timestamp"] = styled["timestamp"].astype(str).str[:19]
    if "severity_score" in styled.columns:
        styled["severity_score"] = styled["severity_score"].round(1)
    if "deviation_pct" in styled.columns:
        styled["deviation_pct"] = styled["deviation_pct"].round(1)

    st.dataframe(styled, use_container_width=True, height=200)

    # ── Drill-down ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Incident Detail")
    idx_options = list(range(min(20, len(display_df))))
    labels = []
    for i in idx_options:
        row = display_df.iloc[i]
        labels.append(f"#{i+1} — {str(row.get('building','')).replace('_',' ')} | "
                      f"{str(row.get('resource',''))} | {str(row.get('severity_tier',''))} "
                      f"(score {float(row.get('severity_score',0)):.1f})")

    sel = st.selectbox("Select incident to investigate:", labels, key="inv_sel")
    sel_idx = int(sel.split("#")[1].split(" ")[0]) - 1
    row = display_df.iloc[sel_idx]

    bldg_name = str(row.get("building", "")).replace("_", " ")
    res_name  = str(row.get("resource", "")).capitalize()
    actual    = float(row.get("actual_value", 0))
    expected  = float(row.get("expected_value", 0))
    dev_pct   = float(row.get("deviation_pct", 0))
    sev_score = float(row.get("severity_score", 0))
    sev_tier  = str(row.get("severity_tier", "UNKNOWN"))
    pattern   = str(row.get("pattern", "Unclassified"))
    ts        = str(row.get("timestamp", ""))[:19]
    unit      = "kWh" if "energy" in res_name.lower() else ("m³" if "water" in res_name.lower() else "kg")
    col_hex   = severity_color(sev_tier)

    # Incident header card
    st.markdown(
        f'<div class="incident-card" style="border-left-color:{col_hex};">'
        f'<b style="color:{col_hex};">{sev_tier}</b> &nbsp;·&nbsp; Score: {sev_score:.1f}/100 &nbsp;·&nbsp; '
        f'{bldg_name} &nbsp;·&nbsp; {res_name}'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"{fact_label()} **Telemetry Reading**", unsafe_allow_html=True)
        st.markdown(
            f'<div class="content-card">'
            f'<b>Observed:</b> {actual:,.3f} {unit}<br>'
            f'<b>Expected baseline:</b> {expected:,.3f} {unit}<br>'
            f'<b>Deviation:</b> +{dev_pct:.1f}% above norm<br>'
            f'<b>Pattern:</b> {pattern}<br>'
            f'<b>Timestamp:</b> {ts}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(f"{hypothesis_label()} **Diagnostic Explanation**", unsafe_allow_html=True)
        if "water" in res_name.lower():
            hyp = ("Consumption during low-occupancy hours is consistent with uncontained plumbing flow — "
                   "possible failed flushometer solenoid, cooling tower makeup overflow, or fixture leak. "
                   "<i>This is a machine-learning hypothesis, not a confirmed diagnosis.</i>")
        elif "energy" in res_name.lower():
            hyp = ("Power draw outside scheduled hours is consistent with an un-reverted BMS manual override, "
                   "auxiliary HVAC chiller cycling, or lab exhaust fans at full speed. "
                   "<i>This is a machine-learning hypothesis, not a confirmed diagnosis.</i>")
        else:
            hyp = ("Abnormal landfill mass is consistent with single-stream contamination batch rejection "
                   "or unscheduled cleanouts bypassing sorting stations. "
                   "<i>This is a machine-learning hypothesis, not a confirmed diagnosis.</i>")
        st.markdown(f'<div class="content-card">{hyp}</div>', unsafe_allow_html=True)

    # ── Knowledge guidance ────────────────────────────────────────────────────
    st.markdown(f"{recommend_label()} **Knowledge-Base Guidance**", unsafe_allow_html=True)
    rag_results = knowledge_retriever.search(f"{res_name.lower()} anomaly baseline", top_k=2)
    if rag_results:
        for doc in rag_results:
            with st.expander(f"📖 {doc['title']} — *{doc['source_file']}*", expanded=False):
                st.markdown(doc["content"])
    else:
        st.info("No matching guidance found in knowledge base.")

    # ── Next steps ────────────────────────────────────────────────────────────
    st.markdown(f"{recommend_label()} **Recommended Next Steps**", unsafe_allow_html=True)
    if "water" in res_name.lower():
        steps = [
            "Dispatch technician to inspect main submeter isolation valves and mechanical rooms.",
            "Perform acoustic leak verification on supply risers.",
            "Check automated irrigation timers for schedule conflicts.",
        ]
    elif "energy" in res_name.lower():
        steps = [
            "Verify BAS time-of-day clock and zone damper schedules.",
            "Audit plug loads and computer lab shut-off automations.",
            "Confirm laboratory fume hood sashes are closed in research zones.",
        ]
    else:
        steps = [
            "Audit loading dock compactors for sorting compliance.",
            "Reinforce pictorial bin signage in dining and common areas.",
            "Review recent cleanout schedules for unplanned events.",
        ]
    st.markdown(
        '<div class="accent-card"><b>Human verification required before physical intervention.</b><ol>'
        + "".join(f"<li>{s}</li>" for s in steps)
        + "</ol></div>",
        unsafe_allow_html=True,
    )
