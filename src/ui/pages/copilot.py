"""src/ui/pages/copilot.py — AI Copilot chat page."""
import streamlit as st
import pandas as pd

from src.ui.components import page_header, provider_badge


_TOOL_LABELS = {
    "simulate_whatif":   ("📊 Ran what-if simulation", "Counterfactual engine"),
    "detect_anomalies":  ("🔍 Analysed anomaly data", "Isolation Forest detector"),
    "run_forecast":      ("📈 Generated demand forecast", "Ridge regression model"),
    "search_guidelines": ("📚 Retrieved knowledge context", "RAG knowledge base"),
    "query_kpis":        ("🔢 Fetched facility KPIs", "KPI engine"),
}

_QUICK_PROMPTS = [
    ("💧 Water anomaly", "Why is Building C using more water?"),
    ("⚡ Energy issue",  "Which building has unusual energy consumption?"),
    ("🧪 What-if",       "How much energy could we save if Building A reduced off-hours by 20%?"),
    ("🌱 Best actions",  "Give me three actions to reduce resource waste across facilities."),
]


def render(active_df: pd.DataFrame, copilot_agent) -> None:
    page_header("AI Copilot", "Ask questions about facility resources — grounded in real telemetry and sustainability guidelines")

    # Provider badge
    pstatus = copilot_agent.granite.provider_status()
    st.markdown(
        f'Powered by: {provider_badge(pstatus)} &nbsp; '
        f'<small style="color:#94a3b8;">Intel grounded in live telemetry + RAG knowledge base</small>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Quick prompts
    st.markdown("**Quick questions:**")
    qcols = st.columns(len(_QUICK_PROMPTS))
    prompt_to_submit = None
    for col, (label, prompt) in zip(qcols, _QUICK_PROMPTS):
        if col.button(label, use_container_width=True):
            prompt_to_submit = prompt

    st.markdown("---")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm the **EcoSync AI Copilot**. I can investigate facility anomalies, "
                    "forecast demand, simulate what-if scenarios, and retrieve sustainability guidelines. "
                    "Ask me anything about your facility resources."
                ),
                "tools": [],
                "provider": pstatus,
            }
        ]

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_tool_log(msg.get("tools", []), msg.get("provider", ""))

    # Input
    user_input = st.chat_input("Ask about energy, water, waste, anomalies, forecasts…") or prompt_to_submit

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "tools": []})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analysing intent, running diagnostics…"):
                response = copilot_agent.execute_plan_and_answer(user_input)

            st.markdown(response.answer)
            _render_tool_log(response.tools_executed, response.provider)

            # Follow-up suggestions
            if response.suggested_followups:
                st.markdown("**Suggested follow-ups:**")
                fu_cols = st.columns(min(3, len(response.suggested_followups)))
                for col, fu in zip(fu_cols, response.suggested_followups[:3]):
                    col.caption(f"→ {fu}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "tools": response.tools_executed,
                "provider": response.provider,
            })


def _render_tool_log(tools: list, provider: str) -> None:
    if not tools:
        return
    with st.expander("How this was generated", expanded=False):
        for t in tools:
            name = getattr(t, "tool_name", "") if hasattr(t, "tool_name") else t.get("tool_name", "")
            label, source = _TOOL_LABELS.get(name, (f"🔧 {name}", "analytics engine"))
            st.markdown(f'<span class="tool-chip done">✓ {label}</span>', unsafe_allow_html=True)
        if provider:
            badge = provider_badge(provider)
            st.markdown(f"&nbsp;{badge} &nbsp;<small style='color:#94a3b8;'>synthesised the final response</small>",
                        unsafe_allow_html=True)
