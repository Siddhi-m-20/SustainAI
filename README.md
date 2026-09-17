# EcoSync Resource Intelligence Platform

> AI-powered decision support for campus energy, water, and waste optimisation

---

## Overview

EcoSync is a multi-resource sustainability intelligence platform for campus and
commercial facilities. It continuously monitors energy, water, and waste telemetry,
detects anomalies using machine learning, forecasts demand, simulates operational
interventions, and provides grounded AI-generated recommendations.

The AI Copilot synthesises answers from live telemetry data, machine-learning outputs,
and retrieved sustainability guidelines — powered by **IBM Granite** via watsonx.ai
when credentials are available, with a transparent local fallback when not.

---

## The Problem

Facilities account for 30–40% of institutional energy consumption. Critical resource
waste — overnight HVAC overruns, plumbing leaks, landfill-bound organic material —
goes undetected for weeks due to fragmented monitoring and manual reporting.

## The Solution

EcoSync applies Isolation Forest anomaly detection to continuous multi-resource
telemetry, surfaces anomalies within hours, and grounds AI explanations in real
data and engineering standards — helping facility teams prioritise interventions
before they become expensive problems.

---

## SDG Alignment

| SDG | Target | EcoSync Feature |
|-----|--------|----------------|
| **SDG 7** Affordable & Clean Energy | 7.3: Double energy efficiency improvement rate | Energy monitoring, off-hours baseload detection, peak demand forecasting |
| **SDG 6** Clean Water & Sanitation | 6.4: Increase water-use efficiency | MNF night-flow surveillance, leak anomaly detection |
| **SDG 12** Responsible Consumption | 12.5: Reduce waste generation | Diversion rate tracking, waste anomaly detection, scenario simulation |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
│  Overview · Insights · Investigate · Forecast · AI Copilot  │
│  Scenario Lab · Knowledge · Reports · About/Impact          │
└───────────────────────────┬─────────────────────────────────┘
                            │
         ┌──────────────────┼───────────────────┐
         ▼                  ▼                   ▼
  CopilotAgent        Analytics Engine    Services
  (agents/)           (analytics/         (anomaly/
  - Intent routing     kpi_engine.py)      forecasting/
  - Tool dispatch    (data/               impact/
  - IBM Granite       validate_data.py)    rag/)
    synthesis
         │
         ▼
  GraniteClient (ibm/watsonx_client.py)
  ├── IBM mode: ibm_watsonx_ai → ibm/granite-3-3-8b-instruct
  └── Fallback: deterministic template renderer
```

---

## AI Components

| Component | Technology | Purpose |
|---|---|---|
| Anomaly detection | Isolation Forest (scikit-learn) | Detect resource anomalies |
| Forecasting | Ridge Regression (scikit-learn) | 24–168h demand projections |
| KPI engine | pandas/NumPy | Energy, water, waste metrics |
| What-if simulation | Custom Python engine | Counterfactual impact modelling |
| RAG retrieval | TF-IDF cosine similarity | Retrieve ASHRAE/EPA/SDG guidance |
| LLM synthesis | IBM Granite via watsonx.ai | Grounded natural-language responses |
| Fallback synthesis | Python templates | Used when IBM unavailable |

---

## IBM Granite Integration

The `GraniteClient` (`src/ibm/watsonx_client.py`) wraps `ibm_watsonx_ai`:

- **Model**: `ibm/granite-3-3-8b-instruct` (verified from IBM watsonx.ai documentation)
- **SDK**: `ibm-watsonx-ai>=1.1.3`
- **Auth**: IBM Cloud API key → IAM token (managed by SDK)
- **Prompt**: Every call includes a Responsible AI system instruction (FACT/HYPOTHESIS/RECOMMENDATION labelling, no fabrication, human verification required)
- **Fallback**: Silent fallback to templates on any credential/network failure
- **Provider badge**: Sidebar always shows `"IBM Granite / watsonx.ai"` or `"Local grounded fallback"` — never falsely claims IBM is active

See [`docs/bob/ibm-architecture.md`](docs/bob/ibm-architecture.md) for full details.

---

## Setup

### Prerequisites
- Python 3.10+
- Virtual environment recommended

### Install
```bash
git clone <repo>
cd SustainAI
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Environment Variables
```bash
cp .env.example .env
# Edit .env with your IBM credentials (optional — app works without them)
```

| Variable | Required | Description |
|---|---|---|
| `WATSONX_APIKEY` | For IBM mode | IBM Cloud API key |
| `WATSONX_URL` | For IBM mode | e.g. `https://us-south.ml.cloud.ibm.com` |
| `WATSONX_PROJECT_ID` | For IBM mode | watsonx.ai project UUID |
| `GRANITE_MODEL_ID` | Optional | Default: `ibm/granite-3-3-8b-instruct` |

Without IBM credentials, the app runs fully in **Local grounded fallback** mode.

---

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Testing

```bash
# All tests
python -m pytest tests/ -v

# IBM integration tests only (no credentials needed)
python -m pytest tests/test_ibm_integration.py -v

# Existing service tests
python -m pytest tests/test_ecosync_services.py -v
python -m pytest tests/run_tests.py -v
```

**Expected result:** All tests pass in offline mode (no IBM credentials required).

---

## Application Pages

| Page | Description |
|---|---|
| **Overview** | Executive KPI dashboard, pipeline story, campus nexus narrative |
| **Insights** | Interactive Energy / Water / Waste analytics (tabbed) |
| **Investigate** | Anomaly drill-down with FACT/HYPOTHESIS/RECOMMENDATION labelling |
| **Forecast** | 24–168h demand forecast with confidence intervals |
| **AI Copilot** | IBM Granite-powered chat with tool transparency |
| **Scenario Lab** | What-if simulator with real-time impact calculations |
| **Knowledge** | Search ASHRAE / EPA / SDG guidelines with provenance |
| **Reports** | Generate and download sustainability audit reports |
| **About / Impact** | SDG alignment, architecture, responsible AI, founder vision |

---

## Responsible AI

- **Transparency**: Every copilot response shows tools executed and knowledge sources retrieved
- **Grounded generation**: Granite receives only actual telemetry data — no fabrication
- **Uncertainty labelling**: Responses distinguish FACT, HYPOTHESIS, and RECOMMENDATION
- **Human verification**: All recommendations for physical actions require human verification
- **No autonomous control**: EcoSync cannot issue commands to building systems
- **Honest provider status**: UI never claims IBM Granite is active when running in fallback mode

---

## Project Structure

```
├── app.py                          # Thin shell: config, caching, sidebar, routing
├── src/
│   ├── ibm/
│   │   └── watsonx_client.py      # GraniteClient — IBM integration layer
│   ├── agents/
│   │   └── copilot_agent.py       # Agentic routing + Granite synthesis wiring
│   ├── rag/
│   │   └── knowledge_retriever.py # TF-IDF RAG over local knowledge base
│   ├── anomaly/                   # Isolation Forest anomaly detection
│   ├── forecasting/               # Ridge regression demand forecasting
│   ├── analytics/                 # KPI calculation engine
│   ├── impact/                    # Counterfactual what-if simulator
│   ├── data/                      # CSV validation + data generation
│   └── ui/
│       ├── styles.py              # CSS design system
│       ├── components.py          # Shared UI helpers
│       └── pages/                 # One module per page
├── knowledge_base/                # ASHRAE, EPA, SDG markdown docs
├── tests/                         # Unit + integration + IBM tests
├── docs/bob/                      # IBM architecture + implementation plan
├── .env.example                   # Credential template
└── requirements.txt               # Dependencies (clean one-per-line format)
```

---

- [Streamlit](https://streamlit.io) — Web application framework
- [scikit-learn](https://scikit-learn.org) — ML models (Isolation Forest, Ridge, TF-IDF)
- [Plotly](https://plotly.com) — Interactive visualisations
- [ibm-watsonx-ai](https://ibm.github.io/watsonx-ai-python-sdk/) — IBM Granite 3-3-8B foundation model inference

---

*EcoSync Resource Intelligence Platform · Grounded AI for Sustainable Facility Operations*
