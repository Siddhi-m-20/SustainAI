# EcoSync System Architecture

## 1. Architectural Overview

The **EcoSync Resource Intelligence Platform** is designed as a unified, decoupled decision-support architecture for facility resource management. It addresses the fundamental fragmentation between energy, water, and circular waste systems by processing synchronized multi-resource telemetry through an end-to-end analytics and machine learning pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STREAMLIT INTERACTION UI                         │
│   Overview · Energy · Water · Waste · Incident Diagnostics · Forecast       │
│   What-If Simulator · Copilot · Knowledge Base · Audit Reports · About      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AGENTIC COPILOT DISPATCHER                         │
│                           (src/agents/copilot_agent.py)                     │
│               - Intent Classification & Entity Extraction                   │
│               - Grounded Prompt Construction (Anti-Hallucination)           │
│               - Provider Fallback State Management                          │
└──────────────┬───────────────────────┬───────────────────────────────┬──────┘
               │                       │                               │
               ▼                       ▼                               ▼
┌─────────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐
│   ML ANOMALY SERVICE    │ │  FORECAST SERVICE   │ │    WHAT-IF SIMULATOR    │
│(src/anomaly/service.py) │ │(src/forecasting/)   │ │(src/impact/simulator.py)│
│- Contextual Median Norm │ │- Cyclical Transform │ │- Counterfactual Engine  │
│- Isolation Forest Model │ │- Ridge Regressor    │ │- ASHRAE 90.1 Setbacks   │
│- Severity Scoring (0-100)│ │- 95% Confidence Band│ │- WaterSense / EPA WARM  │
└──────────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────────┘
               │                       │                       │
               └───────────────────────┼───────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-RESOURCE KPI ENGINE                         │
│                          (src/analytics/kpi_engine.py)                      │
│        - Baseload Fraction (%)           - Scope 2 Carbon Emissions (tCO₂e) │
│        - Minimum Night Flow (MNF, m³/h)  - Landfill Diversion Rate (%)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TELEMETRY VALIDATION LAYER                         │
│                         (src/data/validate_data.py)                         │
│         - Hourly Multi-Facility Continuous Sensor Ingestion                 │
│         - Out-of-Range Cleaning, Schema Normalization, Null Handling        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Subsystems

### 2.1 Telemetry Validation & Ingestion Layer
* **Location:** [`src/data/validate_data.py`](../../src/data/validate_data.py)
* **Function:** Validates continuous telemetry across four core resource metrics:
  * Energy consumption ($kWh$)
  * Water volume ($m^3$)
  * Solid waste generation and diversion ($kg$)
* **Schema Contract:** Enforces timestamps, building IDs, numeric types, and nonnegative constraints. Rejects or interpolates physical sensor anomalies (e.g., negative flows, transmission dropouts).

### 2.2 Physics-Informed Anomaly Service
* **Location:** [`src/anomaly/anomaly_service.py`](../../src/anomaly/anomaly_service.py)
* **Dual-Mechanism Detection:**
  1. **Empirical Contextual Medians:** Computes standard diurnal envelopes conditioned on `(building, day_of_week, hour_of_day)` to account for legitimate occupancy cycles.
  2. **Scikit-Learn Isolation Forest:** Unsupervised tree ensemble modeling multivariate deviations to detect compound anomalies.
* **Severity Scoring:** Normalizes multi-resource deviations onto an enterprise scale ($0–100$), categorizing events into `NORMAL`, `MEDIUM`, `HIGH`, and `CRITICAL`.

### 2.3 Cyclical Demand Forecasting
* **Location:** [`src/forecasting/forecast_service.py`](../../src/forecasting/forecast_service.py)
* **Model:** L2-regularized Ridge Regression with cyclical Fourier features:
  $$\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
* **Output:** $24$-hour, $48$-hour, and $7$-day lookahead forecasts with $95\%$ empirical confidence intervals.

### 2.4 Counterfactual What-If Simulator
* **Location:** [`src/impact/whatif_simulator.py`](../../src/impact/whatif_simulator.py)
* **Purpose:** Computes forward projections for operational interventions:
  * Off-hours HVAC setback adjustments (ASHRAE Standard 90.1)
  * Night water leak repairs & fixture retrofits (EPA WaterSense)
  * Kitchen food waste composting & single-use reduction (EPA WARM model)
* **Outputs:** Dollar savings ($\text{USD}$), resource volumes ($MWh, m^3, t$), and avoided Scope 2 greenhouse gas emissions ($tCO_2e$).

### 2.5 Retrieval-Augmented Generation (RAG)
* **Location:** [`src/rag/knowledge_retriever.py`](../../src/rag/knowledge_retriever.py)
* **Knowledge Store:** Curated engineering markdown documents in [`knowledge_base/`](../../knowledge_base/):
  * `energy_efficiency_standards.md` (ASHRAE 90.1-2019)
  * `water_conservation_protocols.md` (EPA WaterSense commercial guidelines)
  * `waste_diversion_circularity.md` (TRUE Zero Waste / EPA WARM)
  * `sdg_alignment_framework.md` (UN SDG 7, 6, and 12)
* **Retrieval Engine:** TF-IDF vectorization with cosine similarity scoring, extracting top-$k$ domain guidelines.

### 2.6 Enterprise Foundation Model Integration
* **Location:** [`src/ibm/watsonx_client.py`](../../src/ibm/watsonx_client.py)
* **Model:** `ibm/granite-3-3-8b-instruct` via `ibm-watsonx-ai` Python SDK.
* **Fallback Design:** If cloud credentials are unconfigured or connectivity drops, the system automatically falls back to deterministic local grounded templates. The active AI state is explicitly declared in the UI.
