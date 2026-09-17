# EcoSync Resource Manager — Full Implementation Plan

## Top-Level Overview

**Goal:** Upgrade EcoSync Resource Manager from a demo-quality Streamlit dashboard
into a polished, production-grade sustainability intelligence platform with genuine
IBM Granite integration via watsonx.ai, a redesigned premium UI, and a comprehensive
test suite — without discarding any working functionality.

**Non-goals:** Do NOT replace anomaly detection (Isolation Forest), forecasting
(Ridge regression), KPI calculations, what-if simulation engine, or data validation.
These modules are correct and tested.

**Implementation order:**
1. Clean environment/dependency layer
2. IBM Granite integration (`src/ibm/`)
3. Wire IBM into CopilotAgent
4. Full `app.py` redesign (premium UI + new information architecture)
5. Test suite
6. Documentation

---

## Findings from Repository Inspection

### Architecture as-is

| Module | Technology | Status |
|--------|-----------|--------|
| Data validation | pandas + custom schema | Working |
| KPI engine | pandas / NumPy | Working |
| Anomaly detection | sklearn Isolation Forest | Working |
| Forecasting | sklearn Ridge/RF | Working |
| What-if simulator | custom Python | Working |
| RAG retrieval | sklearn TF-IDF | Working |
| Agent/copilot | deterministic templates | Working — but no LLM |
| LLM backend | **None** | Missing |
| IBM integration | **None** | Missing |
| Provider status | **Always claims IBM** | Incorrect |
| UI | Streamlit 1.62.0 | Works but basic, SDG-heavy nav |

### IBM/Granite gap
All IBM references are string literals. `ibm-watsonx-ai` is absent from
`requirements.txt`. No `os.getenv` calls anywhere. No `.env` file.
The copilot produces responses entirely via Python f-string templates.

### UI gap
Navigation exposes SDG labels ("⚡ Energy Intelligence", "💧 Water Intelligence",
"♻️ Resource/Waste Intelligence") as separate pages for the same functionality.
SDG badges appear on every card label throughout the UI.
No "About / Impact" page. No premium design treatment.

### Performance note
`CopilotAgent` is re-instantiated on every Streamlit interaction (line 777 of
`app.py`). `AnomalyService`, `ForecastService`, `WhatIfSimulator`,
`KnowledgeRetriever` are correctly cached via `@st.cache_resource`.

### Deprecation note
`st.plotly_chart(..., width="stretch")` — the `width` kwarg is deprecated in
Streamlit 1.x; should be replaced with `use_container_width=True`.

---

## Sub-Task 1 — Dependency & Environment Layer

**Status:** `[ ] pending`

**Intent:** Fix `requirements.txt` encoding, add `ibm-watsonx-ai`, and create
`.env.example` as the contract for credentials.

**Expected Outcomes:**
- `requirements.txt` is plain one-package-per-line UTF-8 text, preserving all
  existing packages, adding `ibm-watsonx-ai>=1.1.3`
- `.env.example` documents all IBM env vars with descriptions
- No other file changed in this sub-task

**Todo List:**
1. Rewrite `requirements.txt` in standard one-package-per-line format.
   Preserve every existing package and version exactly. Append:
   ```
   ibm-watsonx-ai>=1.1.3
   ```
   Place it after `requests==2.34.2`.
2. Create `.env.example` at project root:
   ```dotenv
   # IBM watsonx.ai credentials — copy to .env and fill in values
   # All three are required to enable IBM Granite mode.
   WATSONX_APIKEY=your-ibm-cloud-api-key
   WATSONX_URL=https://us-south.ml.cloud.ibm.com
   WATSONX_PROJECT_ID=your-watsonx-project-uuid

   # Optional overrides (defaults shown)
   GRANITE_MODEL_ID=ibm/granite-3-3-8b-instruct
   GRANITE_MAX_NEW_TOKENS=600
   GRANITE_TEMPERATURE=0.2
   ```

**Relevant Context:**
- Current `requirements.txt` is space-separated (file encoding artifact). Rewrite
  fully clean — do not just append to the corrupted file.
- `python-dotenv==1.2.3` is already present; `.env.example` is the documented
  template; `.env` is already git-ignored.
- Official confirmed model ID from IBM docs: `ibm/granite-3-3-8b-instruct`
- Verified SDK name: `ibm-watsonx-ai` (not the older `ibm-generative-ai`)

---

## Sub-Task 2 — IBM Granite Integration Layer (`src/ibm/`)

**Status:** `[ ] pending`

**Intent:** Create a clean, self-contained IBM integration module that the rest
of the codebase can call with a single import. Provides real Granite inference
when credentials are available and transparent template fallback when not.

**Expected Outcomes:**
- `src/ibm/__init__.py` exists (empty package marker)
- `src/ibm/watsonx_client.py` contains `GraniteClient` class
- `GraniteClient()` instantiates cleanly with no env vars set (fallback mode)
- `GraniteClient()` instantiates correctly when all three IBM vars are set
- `synthesize(prompt, context)` always returns a non-empty string
- `is_available()` returns correct boolean
- `provider_status()` returns exact strings:
  `"IBM Granite / watsonx.ai"` or `"Local grounded fallback"`

**Todo List:**

1. Create `src/ibm/__init__.py` — empty.

2. Create `src/ibm/watsonx_client.py`:

   **Imports:**
   ```python
   import os
   import logging
   from pathlib import Path
   from typing import Dict, Any
   from dotenv import load_dotenv
   ```

   **Credential loading** (module-level, called at `__init__`):
   ```python
   load_dotenv()  # reads .env if present; harmless if absent
   _APIKEY  = os.getenv("WATSONX_APIKEY", "").strip()
   _URL     = os.getenv("WATSONX_URL", "").strip()
   _PROJ    = os.getenv("WATSONX_PROJECT_ID", "").strip()
   _MODEL   = os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-3-8b-instruct").strip()
   _MAX_TOK = int(os.getenv("GRANITE_MAX_NEW_TOKENS", "600"))
   _TEMP    = float(os.getenv("GRANITE_TEMPERATURE", "0.2"))
   ```

   **`GraniteClient` class:**

   a. `__init__`: set `self._available = False`, call `self._init_client()`

   b. `_init_client()`:
      - If `_APIKEY`, `_URL`, `_PROJ` all non-empty:
        ```python
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.foundation_models.utils.enums import GenTextParamsMetaNames as GenParams
        credentials = Credentials(url=_URL, api_key=_APIKEY)
        self._model = ModelInference(
            model_id=_MODEL,
            credentials=credentials,
            project_id=_PROJ,
            params={
                GenParams.MAX_NEW_TOKENS: _MAX_TOK,
                GenParams.TEMPERATURE: _TEMP,
                GenParams.STOP_SEQUENCES: ["###END"],
            }
        )
        self._available = True
        ```
      - Wrap in `try/except Exception as e:` → log warning, set `_available=False`

   c. `is_available() -> bool`: return `self._available`

   d. `provider_status() -> str`:
      return `"IBM Granite / watsonx.ai"` if available else `"Local grounded fallback"`

   e. `synthesize(prompt: str, context: Dict[str, Any]) -> str`:
      - If `self._available`:
        ```python
        try:
            result = self._model.generate_text(prompt=prompt)
            text = result.strip() if isinstance(result, str) else ""
            if text:
                return text
        except Exception as e:
            logging.warning(f"[GraniteClient] IBM inference failed: {e}")
        ```
      - Fall through to `return self._fallback_render(context)`

   f. `_fallback_render(context: Dict[str, Any]) -> str`:
      - Reads scenario key from context (`context["scenario"]`) which is one of:
        `"whatif"`, `"anomaly"`, `"forecast"`, `"general"`
      - Contains the **verbatim** existing template strings extracted from
        `src/agents/copilot_agent.py` scenarios A/B/C/D
      - This ensures existing test assertions still pass in fallback mode

   **Prompt construction helper** (used by `CopilotAgent`):

   ```python
   def build_prompt(self, scenario: str, context: Dict[str, Any]) -> str:
       """
       Build a grounded Granite prompt from tool results + RAG context.
       Instructs Granite to:
       - Use only supplied data; cite sources by name
       - Distinguish FACT / HYPOTHESIS / RECOMMENDATION
       - Never confirm a fault without evidence
       - Show assumptions; recommend human verification
       - Avoid fabricated numbers
       """
       ...
   ```

   The prompt template must include:
   - A system instruction block (Responsible AI framing)
   - Structured tool results section (numeric values from context dict)
   - RAG context section (retrieved knowledge chunk titles and content)
   - User question
   - Output format instructions

**Responsible AI safeguards implemented in this layer:**
- All prompts include explicit instruction: "Do not confirm any fault without evidence.
  Label diagnostic explanations as HYPOTHESIS. Always recommend human verification
  before physical interventions."
- `synthesize()` falls back to template on any IBM error — never crashes the app
- No IBM credentials are logged or printed
- Context dict is serialised to plain text before being passed to the model
  (no raw DataFrames, no PII)

**Note on guardrails:** IBM's HAP filter via `moderations` parameter is available
in the SDK. However, since EcoSync operates in a technical sustainability domain
(HVAC, plumbing, waste management) and not a consumer-facing chat context, and
since the responsible-AI framing in the system prompt already constrains outputs,
the HAP filter is optional. It should be documented in `docs/bob/ibm-architecture.md`
but implementation should be conditional on whether the deployed model version supports
the `moderations` field cleanly. If `ModelInference.generate_text` accepts it,
add it; if it raises an exception for this model, omit it without pretending it works.

**Relevant Context:**
- Official pattern from IBM watsonx.ai Python SDK docs:
  ```python
  from ibm_watsonx_ai import Credentials
  from ibm_watsonx_ai.foundation_models import ModelInference
  credentials = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key="...")
  model = ModelInference(model_id="ibm/granite-3-3-8b-instruct",
                         credentials=credentials, project_id="...")
  result = model.generate_text(prompt="...")
  ```
- `generate_text()` returns a plain string in current SDK versions
- `GenTextParamsMetaNames` is the correct enum for generation params

---

## Sub-Task 3 — Wire IBM into CopilotAgent

**Status:** `[ ] pending`

**Intent:** Replace the four f-string template blocks in `execute_plan_and_answer()`
with calls to `GraniteClient.synthesize()`. All tool dispatch, intent routing, SDG
tagging, and `AgentResponse` structure remain identical. Only `answer =` changes.

**Expected Outcomes:**
- `CopilotAgent` instantiates `GraniteClient` once in `__init__`
- Each scenario (A/B/C/D) builds a `context` dict and calls `granite.synthesize()`
- Fallback mode returns identical string to what the templates returned
- All existing `test_ecosync_services.py` assertions still pass
- `AgentResponse.provider_status` field added (string from `GraniteClient.provider_status()`)

**Todo List:**

1. Add import: `from src.ibm.watsonx_client import GraniteClient`

2. In `CopilotAgent.__init__()`, add:
   ```python
   self.granite = GraniteClient()
   ```

3. For **Scenario A (What-If)** — replace the `answer = (f"### What-If...")` block:
   ```python
   context = {
       "scenario": "whatif",
       "building": bldg_label,
       "reduction_pct": reduction,
       "resource": resource,
       "sim_data": {k: v for k, v in sim_data.items()},
       "rag_docs": [{"title": d["title"], "content": d["content"][:300],
                     "source": d["source_file"]} for d in rag_docs],
       "user_query": user_query,
   }
   prompt = self.granite.build_prompt("whatif", context)
   answer = self.granite.synthesize(prompt, context)
   ```

4. For **Scenario B (Anomaly)** — replace `answer = (f"### Diagnostic...")` block:
   ```python
   context = {
       "scenario": "anomaly",
       "building": bldg,
       "resource": resource,
       "anomalies": anomalies[:3],   # top 3 serialisable dicts
       "rag_docs": [{"title": d["title"], "content": d["content"][:300],
                     "source": d["source_file"]} for d in rag_docs],
       "user_query": user_query,
   }
   prompt = self.granite.build_prompt("anomaly", context)
   answer = self.granite.synthesize(prompt, context)
   ```

5. For **Scenario C (Forecast)** — replace `answer = (f"### 48-Hour...")` block:
   ```python
   context = {
       "scenario": "forecast",
       "building": bldg_name,
       "resource": fc_res,
       "fc_summary": {k: v for k, v in fc_summary.items()},
       "user_query": user_query,
   }
   prompt = self.granite.build_prompt("forecast", context)
   answer = self.granite.synthesize(prompt, context)
   ```

6. For **Scenario D (General)** — replace `answer = (f"### Recommended...")` block:
   ```python
   context = {
       "scenario": "general",
       "building": bldg,
       "resource": resource,
       "rag_docs": [{"title": d["title"], "content": d["content"][:300],
                     "source": d["source_file"]} for d in rag_docs],
       "kpis": {
           "energy_mwh": kpis.get("energy", {}).get("total_mwh", 0),
           "water_m3": kpis.get("water", {}).get("total_m3", 0),
           "diversion_pct": kpis.get("waste", {}).get("diversion_rate_pct", 0),
       },
       "user_query": user_query,
   }
   prompt = self.granite.build_prompt("general", context)
   answer = self.granite.synthesize(prompt, context)
   ```

7. Update `AgentResponse` dataclass to add:
   ```python
   provider: str = "Local grounded fallback"
   ```
   Set in `execute_plan_and_answer()`:
   ```python
   return AgentResponse(
       ...,
       provider=self.granite.provider_status(),
   )
   ```

8. Keep `followups`, `provenance`, `sdg_alignment`, `confidence_level` unchanged.

**Relevant Context:**
- Existing test assertions at `tests/test_ecosync_services.py:77,83`:
  - `assertIn("What-If Impact Simulation", res1.answer)` — must appear in fallback
  - `assertIn("Diagnostic Investigation", res2.answer)` — must appear in fallback
  - These strings must be present in `_fallback_render()` for scenario A and B

---

## Sub-Task 4 — Full App Redesign (`app.py`)

**Status:** `[ ] pending`

**Intent:** Rewrite `app.py` with a premium information architecture, removing
SDG-as-navigation-labels, consolidating the three separate resource pages into a
single "Insights" page with tabs, adding "About / Impact", and fixing all
deprecation warnings. All existing backend service calls remain identical.

**Expected Outcomes:**
- App opens without errors or warnings
- Navigation has exactly 9 pages (see below)
- SDG labels appear only in About/Impact page and contextually inside content
- No `st.plotly_chart(... width="stretch")` deprecation warnings
- Provider status badge appears in sidebar
- CopilotAgent instantiated once via `@st.cache_resource` (not re-created per interaction)
- All existing functionality preserved

**New Navigation Structure:**
```
1. Overview
2. Insights          (was: Energy Intelligence + Water Intelligence + Waste Intelligence)
3. Investigate       (existing — improved)
4. Forecast          (existing — improved presentation)
5. AI Copilot        (existing — improved UX + provider badge)
6. Scenario Lab      (was: What-If Simulator)
7. Knowledge         (was: Sustainability Knowledge)
8. Reports           (existing — improved)
9. About / Impact    (new)
```


**Architecture Decision (user-confirmed):**
`app.py` is a thin shell (config, caching, sidebar, routing).
All page content lives in `src/ui/pages/` — one file per page.

New `src/ui/` structure:
```
src/ui/__init__.py
src/ui/styles.py            # get_styles() -> str
src/ui/components.py        # shared card/badge helpers
src/ui/pages/__init__.py
src/ui/pages/overview.py    # def render(view_df, kpis, anomalies, building)
src/ui/pages/insights.py    # def render(active_df, view_df, kpis, building)
src/ui/pages/investigate.py # def render(active_df, anomalies, kr, building)
src/ui/pages/forecast.py    # def render(active_df, forecast_service, building)
src/ui/pages/copilot.py     # def render(active_df, copilot_agent)
src/ui/pages/scenario_lab.py# def render(active_df, whatif_simulator, building)
src/ui/pages/knowledge.py   # def render(knowledge_retriever)
src/ui/pages/reports.py     # def render(active_df, kpis, anomalies, building)
src/ui/pages/about.py       # def render(provider_status)
```

app.py routing pattern:
```python
from src.ui.pages import overview, insights, investigate, forecast
from src.ui.pages import copilot, scenario_lab, knowledge, reports, about
...
if page == "Overview":
    overview.render(view_df, kpis, filtered_anomalies, selected_building)
elif page == "Insights":
    insights.render(active_df, view_df, kpis, selected_building)
# etc.
```


**Todo List:**

### 4.1 — CSS / Design System

Replace the existing `<style>` block (lines 36–206) with a comprehensive design
system. Key design tokens:

```
Primary:   #0a2540  (deep navy — brand authority)
Accent:    #00a86b  (emerald green — sustainability)
Surface:   #f8fafc  (near-white cards)
Border:    #e2e8f0
Text:      #0f172a (primary), #475569 (secondary), #94a3b8 (muted)
Error:     #ef4444
Warning:   #f59e0b
```

Card design: subtle shadow, 12px radius, hover lift (+2px translateY).
Typography: `Inter` or `system-ui` fallback; clear size hierarchy.
Remove all inline SDG badge CSS from sidebar (keep the class definitions for
About page only).

### 4.2 — Sidebar

Replace current sidebar with:
```
[Logo area]  EcoSync
             Resource Intelligence Platform

[Separator]

[Navigation radio — clean labels without SDG prefixes]
  Overview
  Insights
  Investigate
  Forecast
  AI Copilot
  Scenario Lab
  Knowledge
  Reports
  About / Impact

[Separator]

[Data Scope]
  File uploader
  Building selector

[Separator]

[Status area]
  AI Provider: [badge from GraniteClient.provider_status()]
  Data: N rows
  Version: 2.1
```

Provider badge rendering:
```python
_provider = GraniteClient().provider_status()
if "IBM" in _provider:
    st.success(f"🤖 {_provider}")
else:
    st.info(f"🤖 {_provider}")
```

### 4.3 — Page 1: Overview

Keep the Detect→Investigate→Explain→Recommend→Simulate story banner.
Replace SDG-prefixed metric card labels:
- "⚡ SDG 7 • Total Energy" → "Total Energy"
- "💧 SDG 6 • Total Water" → "Total Water"  
- "♻️ SDG 12 • Diversion Rate" → "Waste Diversion"

Add an "AI Insight" card in the fourth column showing the most recent critical
anomaly summary (or "No critical alerts" if none) with a "→ Investigate" link.

Keep the Connected Campus Nexus card but remove SDG prefix from the heading.
Keep the Multi-Resource Telemetry charts.
Fix: `use_container_width=True` instead of `width="stretch"`.

### 4.4 — Page 2: Insights (replaces 3 separate pages)

Single page with three tabs: **Energy** | **Water** | **Resource & Waste**

Each tab shows:
- Resource KPI cards (4 metrics in a row)
- Primary chart (hourly load curve by building)
- Secondary chart (distribution or breakdown)
- Quick insights callout (e.g., off-hours baseload %, night flow status)

This collapses the three current separate pages (`⚡ Energy Intelligence`,
`💧 Water Intelligence`, `♻️ Resource/Waste Intelligence`) into one page without
losing any content.

### 4.5 — Page 3: Investigate (improved)

Keep current anomaly table and drill-down. Improve:
- Add severity colour-coded rows (CRITICAL=red, HIGH=orange, MEDIUM=yellow)
- When an incident is selected, clearly label sections:
  ```
  FACT         — Actual vs expected values, timestamp, deviation %
  HYPOTHESIS   — AI diagnostic explanation (clearly labelled)
  GUIDANCE     — Retrieved knowledge-base passage + source name
  NEXT STEPS   — Recommended human verification actions
  ```
- Remove the "SDG 6/7/12" prefix from sub-headings within this page

### 4.6 — Page 4: Forecast (improved presentation)

Keep `ForecastService` calls unchanged. Improve:
- Controls at top: resource selector, building selector, horizon selector (24/48/72/168h)
- KPI summary row: Total Projected | Peak | Peak Alert Hours | Est. Cost
- Main chart: actual history + predicted + 95% CI bands + peak markers
  (combine into a single Plotly figure with dual traces)
- Fix: `use_container_width=True`
- Add: confidence rating label, methodology note

### 4.7 — Page 5: AI Copilot (improved UX)

Keep `CopilotAgent` call unchanged. Cache `CopilotAgent` via `@st.cache_resource`:
```python
@st.cache_resource
def get_copilot(_df_hash):
    return CopilotAgent(data_df=active_df)
```
(Use a hash of the DataFrame shape/checksum as the cache key to invalidate on
data change.)

Improve UX:
- Add provider badge at top of page: "Powered by: {provider_status}"
- Tool transparency: show concise activity log (not raw code):
  ```
  ✓ Analyzed water anomaly in Building C
  ✓ Retrieved WaterSense guidance (MNF protocols)
  ✓ Generated grounded response with IBM Granite
  ```
  (Map `tool_name` to human-readable labels)
- Response display: show answer, then collapsible "How this was generated" section
  with tool log and provenance — not expanded by default
- Remove the raw Python dict display of tool arguments
- Show `AgentResponse.provider` value in the response footer
- Keep the 4 quick-inquiry buttons but style them as proper action chips

### 4.8 — Page 6: Scenario Lab (was: What-If Simulator)

Rename. Keep all `WhatIfSimulator` calls unchanged. Improve:
- Three columns: SDG 7 controls | SDG 6 controls | SDG 12 controls
  (sliders remain, but cleaner layout)
- Results show in a dashboard-style grid with large impact numbers
- Add "Modeled Estimates" banner at top
- Clearly label: "These are projections, not guarantees. Review assumptions below."
- Show assumptions section (existing ones, just better formatted)

### 4.9 — Page 7: Knowledge

Keep `KnowledgeRetriever` calls unchanged. Improve:
- Search bar at top
- Results shown as clean cards (not raw markdown blocks)
- Show source file name and SDG tag as metadata
- "Browse All" section shows chunks in expandable cards by knowledge domain

### 4.10 — Page 8: Reports

Improve the generated report:
- Add AI Copilot general analysis (call `CopilotAgent` with a standardised
  "Generate sustainability summary" query)
- Better formatted output
- Download button remains
- Add "Responsible AI Notice" section to the generated report

### 4.11 — Page 9: About / Impact (new)

Create a rich About page containing all SDG content, IBM details, and architecture:

**Sections:**
1. **Product Overview** — what EcoSync is, the problem it solves
2. **Target Users** — facility managers, sustainability officers, campus ops
3. **SDG Alignment** — SDG 7, 6, 12 with UN target references, displayed properly
4. **AI Architecture** — pipeline diagram (text-based in Streamlit),
   IBM Granite, RAG, anomaly detection, forecasting
5. **IBM Granite Integration** — model used, provider status, responsible AI
6. **Responsible AI** — transparency, grounded generation, human-in-the-loop,
   data privacy, uncertainty, no autonomous control
7. **Expected Impact** — quantified estimate (% energy savings, leak detection,
   diversion improvement)
8. **Built with IBM BOB** — BOB's role in architecture, coding, testing, documentation

---

## Sub-Task 5 — IBM Integration Tests

**Status:** `[ ] pending`

**Intent:** Create `tests/test_ibm_integration.py` with comprehensive coverage of
the IBM layer. All tests must pass with no real IBM credentials (using mocks).

**Expected Outcomes:**
- All 6 tests pass offline
- Tests cover: credential absence, fallback synthesis, mocked IBM path,
  provider status, context serialization, `EmbeddingsRetriever` not broken

**Todo List:**

Create `tests/test_ibm_integration.py`:

```python
class TestGraniteClientOffline(unittest.TestCase):
    """Tests that run with no IBM credentials."""

    def setUp(self):
        # Ensure env vars are unset for each test
        for var in ["WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID"]:
            os.environ.pop(var, None)
        # Re-import to pick up unset vars
        import importlib
        import src.ibm.watsonx_client as m
        importlib.reload(m)
        from src.ibm.watsonx_client import GraniteClient
        self.client = GraniteClient()

    def test_no_credentials_not_available(self):
        self.assertFalse(self.client.is_available())

    def test_provider_status_fallback(self):
        self.assertEqual(self.client.provider_status(), "Local grounded fallback")

    def test_synthesize_fallback_returns_string(self):
        ctx = {"scenario": "general", "building": "ALL", "resource": "ALL",
               "rag_docs": [], "kpis": {}, "user_query": "test"}
        result = self.client.synthesize("test prompt", ctx)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

    def test_synthesize_whatif_fallback_contains_marker(self):
        ctx = {"scenario": "whatif", "building": "All Campus",
               "reduction_pct": 20.0, "resource": "ALL",
               "sim_data": {"total_cost_saved_usd": 1000.0,
                            "carbon_avoided_tco2e": 2.5,
                            "energy_saved_kwh": 5000.0,
                            "energy_saved_mwh": 5.0,
                            "water_saved_liters": 2000.0,
                            "water_saved_m3": 2.0,
                            "waste_diverted_kg": 100.0,
                            "sdg7_energy_intensity_change_pct": 5.0,
                            "sdg6_water_reduction_pct": 10.0,
                            "sdg12_diversion_rate_projected_pct": 65.0,
                            "equivalent_cars_removed_annual": 1.2,
                            "equivalent_trees_planted": 115},
               "rag_docs": [], "user_query": "save energy"}
        result = self.client.synthesize("test", ctx)
        self.assertIn("What-If Impact Simulation", result)

    def test_synthesize_anomaly_fallback_contains_marker(self):
        ctx = {"scenario": "anomaly", "building": "Building_C",
               "resource": "water", "anomalies": [],
               "rag_docs": [], "user_query": "why water"}
        result = self.client.synthesize("test", ctx)
        self.assertIn("Diagnostic", result)


class TestGraniteClientMocked(unittest.TestCase):
    """Tests that mock the IBM SDK."""

    def test_synthesize_ibm_path_with_mock(self):
        import importlib
        os.environ["WATSONX_APIKEY"] = "test-key"
        os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com"
        os.environ["WATSONX_PROJECT_ID"] = "test-project"
        try:
            with unittest.mock.patch(
                "ibm_watsonx_ai.foundation_models.ModelInference"
            ) as MockModel:
                mock_instance = MockModel.return_value
                mock_instance.generate_text.return_value = "Mocked Granite response"
                import src.ibm.watsonx_client as m
                importlib.reload(m)
                from src.ibm.watsonx_client import GraniteClient
                client = GraniteClient()
                # If IBM SDK not installed, will be in fallback mode — test conditionally
                if client.is_available():
                    result = client.synthesize("test prompt", {"scenario": "general"})
                    self.assertIsInstance(result, str)
        finally:
            for var in ["WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID"]:
                os.environ.pop(var, None)

    def test_build_prompt_returns_nonempty(self):
        from src.ibm.watsonx_client import GraniteClient
        client = GraniteClient()
        ctx = {"scenario": "general", "user_query": "test", "rag_docs": [],
               "building": "ALL", "resource": "ALL",
               "kpis": {"energy_mwh": 100, "water_m3": 50, "diversion_pct": 45}}
        prompt = client.build_prompt("general", ctx)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)
```

Also add `TestGraniteClientMocked` to `tests/run_tests.py` test suite.

**Relevant Context:**
- Tests must not require a real IBM connection
- The `importlib.reload()` pattern is needed because module-level vars are read at
  import time
- If `ibm-watsonx-ai` is not installed in the test environment,
  `_init_client()` will catch the `ImportError` and set `_available=False` gracefully

---

## Sub-Task 6 — Documentation

**Status:** `[ ] pending`

**Intent:** Create `docs/bob/ibm-architecture.md` and update `README.md`.

**Expected Outcomes:**
- `docs/bob/ibm-architecture.md` documents the full IBM integration
- `README.md` accurately describes the project, setup, and IBM integration
- README does not claim IBM Granite is active when running in fallback mode

**Todo List:**

1. Create `docs/bob/ibm-architecture.md`:
   - Overview of IBM integration
   - Component table
   - Environment variable reference table
   - Sequence description (user query → tools → Granite → response)
   - Fallback behaviour description
   - Responsible AI safeguards section
   - Testing section (how to run, mock credentials)
   - Note on HAP guardrails (optional, model-dependent)

2. Update `README.md` to include:
   - Project overview and problem statement
   - Solution architecture
   - AI components (anomaly, forecast, RAG, copilot, IBM Granite)
   - IBM Granite integration section (with honest status note)
   - Environment variables table
   - Setup and run instructions
   - Testing instructions
   - Screenshots note
   - Responsible AI section

---

## Verification / Testing Plan

### Run all tests (offline — no IBM credentials required)

```bash
python -m pytest tests/ -v
```

Expected: all pass (existing + new IBM tests).

Specifically:
- `tests/run_tests.py` — existing unit tests
- `tests/test_ecosync_services.py` — integration tests including copilot
- `tests/test_anomaly_service.py`
- `tests/test_multi_resource.py`
- `tests/test_ibm_integration.py` — new IBM tests (all offline/mocked)

### Start the application

```bash
streamlit run app.py
```

### Manual verification checklist

| Check | Expected |
|-------|----------|
| App opens | No errors, loads Overview |
| Sidebar | 9 navigation items, no SDG labels |
| Provider badge | "Local grounded fallback" (no IBM creds) |
| Overview | KPI cards, charts load, no SDG prefixes on metric labels |
| Insights | 3 tabs (Energy / Water / Waste), all charts render |
| Investigate | Anomaly table loads, drill-down shows FACT/HYPOTHESIS/GUIDANCE sections |
| Forecast | Charts load with CI bands, `use_container_width=True` |
| AI Copilot | Sends query, gets response, shows provider in footer |
| Scenario Lab | Sliders work, results show modeled estimates banner |
| Knowledge | Search works, shows source names |
| Reports | Generates and downloads report |
| About/Impact | Shows SDG content, IBM section, responsible AI |
| No SDG nav labels | Navigation items use task-oriented names |
| No fake IBM claims | UI never says "Powered by IBM Granite" in fallback mode |
| No exposed secrets | No API keys in source code or UI |
| No deprecation warnings | No `width="stretch"` in Streamlit calls |

---

## File Manifest

### New files

| File | Purpose |
|------|---------|
| `src/ibm/__init__.py` | Package marker |
| `src/ibm/watsonx_client.py` | GraniteClient wrapper |
| `tests/test_ibm_integration.py` | IBM layer tests |
| `.env.example` | Credential template |
| `docs/bob/ibm-architecture.md` | IBM integration documentation |

### Modified files

| File | Changes |
|------|---------|
| `requirements.txt` | Rewritten clean + `ibm-watsonx-ai>=1.1.3` added |
| `src/agents/copilot_agent.py` | Import GraniteClient, wire synthesize(), add provider field |
| `app.py` | Full redesign — 9-page structure, premium CSS, provider badge, deprecation fixes |
| `README.md` | Full rewrite |

### Unchanged files (do not touch)

| File | Reason |
|------|--------|
| `src/anomaly/anomaly_service.py` | Working correctly |
| `src/forecasting/forecast_service.py` | Working correctly |
| `src/analytics/kpi_engine.py` | Working correctly |
| `src/impact/whatif_simulator.py` | Working correctly |
| `src/rag/knowledge_retriever.py` | Working correctly |
| `src/data/validate_data.py` | Working correctly |
| `knowledge_base/*.md` | Content is correct |
| `tests/test_ecosync_services.py` | Must pass unchanged |
| `tests/test_anomaly_service.py` | Must pass unchanged |
| `tests/test_multi_resource.py` | Must pass unchanged |

---

## Implementation Order for Agent Mode

The subtasks should be implemented in this exact order to avoid dependency failures:

1. **Sub-Task 1** (requirements + .env.example) — no code dependencies
2. **Sub-Task 2** (`src/ibm/watsonx_client.py`) — needed by Sub-Tasks 3 and 4
3. **Sub-Task 5** (tests) — can write tests as soon as `watsonx_client.py` exists;
   run after Sub-Task 3 to verify copilot wiring
4. **Sub-Task 3** (wire into CopilotAgent) — needs Sub-Task 2
5. **Sub-Task 4** (app.py redesign) — needs Sub-Tasks 2 and 3
6. **Sub-Task 6** (documentation) — last, after everything works

---

*Plan written for EcoSync Resource Manager.*
*Approved for implementation in Agent mode via `start_subtask` per subtask.*
