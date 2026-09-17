"""src/ui/pages/about.py — About / Impact page."""
import streamlit as st

from src.ui.components import page_header


def render(provider_status: str) -> None:
    page_header("About & Impact", "Project purpose, SDG alignment, AI architecture, and responsible AI principles")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "SDG Alignment", "AI Architecture", "Responsible AI"])

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
## EcoSync Resource Intelligence Platform
*Conceived, Architected & Engineered by the Author*

### Founder's Statement & Personal Vision
Commercial and institutional facilities consume 30–40% of all grid electricity and millions of gallons of water, yet silent operational waste routinely continues for weeks unnoticed. A faulty cooling tower makeup valve or an overridden HVAC night setback wastes thousands of dollars and metric tonnes of carbon before anyone opens a monthly utility bill.

I engineered **EcoSync** from the ground up as an unified operational command center. By coupling continuous multi-resource telemetry (energy, water, waste) with contextual physics baselines, machine learning anomaly detection, cyclical demand forecasting, and grounded IBM Granite intelligence, EcoSync bridges the gap between raw meter data and proactive facility interventions.

### The Core Problem It Solves
Fragmented legacy building management systems create resource silos. Energy managers don't monitor water leaks, plumbing teams don't track chiller pump electricity, and waste diversion is calculated months after the fact. EcoSync unites these critical resource flows into a unified nexus.

### The Solution
EcoSync applies machine learning to continuous multi-resource telemetry, surfaces anomalies
within hours, grounds AI-generated explanations in real data and engineering standards,
and helps facility teams prioritise interventions before they become expensive problems.

### Target Users
- **Facility Operations Managers** — identifying and fixing resource waste
- **Sustainability Officers** — tracking SDG progress and reporting
- **Executive Leadership** — visibility into operational efficiency and carbon metrics
- **Maintenance Teams** — prioritised work orders from AI-detected anomalies

### Expected Impact
| Intervention | Estimated Annual Saving |
|---|---|
| Off-hours HVAC setbacks | 15–25% energy reduction |
| Night-flow leak detection | Up to 40% water loss reduction |
| Organic waste diversion | 25–35% landfill reduction |
| Demand peak shaving | 5–15% demand charge reduction |
""")

    # ── SDG ALIGNMENT ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### UN Sustainable Development Goal Alignment")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="content-card">'
                '<span class="sdg-badge sdg-7">SDG 7</span>'
                '<h4>Affordable & Clean Energy</h4>'
                '<b>Target 7.3:</b> Double global rate of improvement in energy efficiency.<br><br>'
                'EcoSync monitors energy KPIs, detects off-hours baseload waste, forecasts '
                'peak demand, and simulates HVAC setback savings aligned to ASHRAE 90.1.'
                '</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="content-card">'
                '<span class="sdg-badge sdg-6">SDG 6</span>'
                '<h4>Clean Water & Sanitation</h4>'
                '<b>Target 6.4:</b> Substantially increase water-use efficiency.<br><br>'
                'EcoSync monitors minimum night flow (MNF), detects plumbing anomalies, '
                'and provides remediation guidance aligned to EPA WaterSense protocols.'
                '</div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div class="content-card">'
                '<span class="sdg-badge sdg-12">SDG 12</span>'
                '<h4>Responsible Consumption</h4>'
                '<b>Target 12.5:</b> Substantially reduce waste generation through prevention.<br><br>'
                'EcoSync tracks diversion rates, simulates composting and source reduction '
                'scenarios, and provides facility-level waste stream analysis.'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── AI ARCHITECTURE ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("### AI & Technical Architecture")

        st.markdown("""
```
User Question
      │
      ▼
 CopilotAgent — Intent Classification & Entity Extraction
      │
      ├─ Tool: detect_anomalies  →  Isolation Forest (scikit-learn)
      ├─ Tool: run_forecast       →  Ridge Regression (scikit-learn)
      ├─ Tool: simulate_whatif   →  Counterfactual simulation engine
      ├─ Tool: query_kpis        →  KPI calculation engine
      └─ Tool: search_guidelines →  TF-IDF RAG (local knowledge base)
      │
      ▼
 GraniteClient.synthesize(prompt, context)
      │
      ├─ IBM credentials set? ──YES──► ibm_watsonx_ai.ModelInference
      │                                ibm/granite-3-3-8b-instruct
      │                                Responsible AI system prompt
      │
      └────────────────────── NO ───► Local template fallback
      │
      ▼
 AgentResponse → Streamlit UI
```

#### Components

| Component | Technology | Purpose |
|---|---|---|
| Anomaly detection | Isolation Forest (sklearn) | Detect resource usage anomalies |
| Forecasting | Ridge Regression (sklearn) | 24–168h demand projections |
| KPI engine | pandas / NumPy | Energy, water, waste KPIs |
| What-if simulator | Custom Python engine | Counterfactual impact modelling |
| RAG retrieval | TF-IDF cosine similarity | Retrieve ASHRAE/EPA/SDG guidance |
| LLM synthesis | IBM Granite via watsonx.ai | Grounded natural-language responses |
| Fallback synthesis | Python templates | Used when IBM unavailable |
""")

        # Current provider status
        if "IBM" in provider_status:
            st.success(f"🤖 Active AI provider: **{provider_status}**")
        else:
            st.info(f"🔧 Active AI provider: **{provider_status}** (set WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID to enable IBM Granite)")

        st.markdown("""
#### System Design Highlights
- **End-to-End Multi-Resource Pipeline**: Conceived and built to ingest, validate, and compute unified KPIs across energy, water, and circular waste streams.
- **Contextual Physics & ML Guardrails**: Combines empirical median profiles conditioned on day and hour with Isolation Forest anomaly scoring.
- **Enterprise Foundation Model Integration**: Native connector to IBM Granite 3-3-8B Instruct via watsonx.ai with strict FACT / HYPOTHESIS / RECOMMENDATION output governance.
- **Full Offline Autonomy**: Grounded local fallback templates ensure continuous operation with zero hallucinations even when cloud APIs are unconfigured.
""")

    # ── RESPONSIBLE AI ────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Responsible AI Principles")
        st.markdown("""
#### Transparency
Every AI response explicitly shows which tools were executed, what data was used,
and which knowledge sources were retrieved. Users always see the evidence chain.

#### Grounded Generation
IBM Granite is instructed to use only the supplied telemetry data and retrieved
knowledge context. The system prompt explicitly forbids fabricating numbers.

#### FACT / HYPOTHESIS / RECOMMENDATION Labelling
AI-generated responses distinguish between:
- **FACT** — directly measured telemetry values
- **HYPOTHESIS** — machine-learning diagnostic explanation (not confirmed)
- **RECOMMENDATION** — suggested action pending human verification

#### Human Verification Required
EcoSync is a **decision-support tool**, not an autonomous infrastructure controller.
All recommendations for physical interventions (checking valves, adjusting BMS settings,
dispatching technicians) explicitly require human facility engineering verification.

#### No Autonomous Physical Control
EcoSync has no integration with building control systems. It cannot issue commands
to HVAC, plumbing, or electrical systems. All outputs are advisory only.

#### Data Privacy
Only aggregate facility telemetry (hourly consumption by building) is processed.
No personal, occupancy, or individual-level data is used or transmitted.

#### Uncertainty Communication
Forecast and simulation outputs are explicitly labelled as estimates. Confidence
intervals are shown on all forecasts. Assumptions are documented and transparent.

#### Fallback Behaviour
When IBM Granite is unavailable, the system falls back to deterministic template
responses — clearly labelled "Local grounded fallback" in the provider badge.
The UI never falsely claims IBM Granite is active.
""")
