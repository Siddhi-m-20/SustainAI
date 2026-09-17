# IBM Integration Plan — EcoSync Resource Manager

## Top-Level Overview

**Goal:** Replace the EcoSync copilot's deterministic template-response engine with a real
IBM Granite LLM call through watsonx.ai, while keeping every other module (anomaly detection,
forecasting, KPI engine, what-if simulator, Streamlit UI pages) completely unchanged.

**Scope:**
- New file `src/ibm/watsonx_client.py` — thin wrapper around `ibm_watsonx_ai`
- Modified file `src/agents/copilot_agent.py` — insert Granite synthesis call at one spot
- New file `src/rag/embeddings_retriever.py` — Granite-embedding-based retriever (replaces TF-IDF when IBM credentials are present; TF-IDF fallback remains)
- New file `.env.example` — documents all required env vars
- New file `tests/test_ibm_integration.py` — IBM-specific unit/integration tests
- New file `docs/bob/ibm-architecture.md` — architecture reference doc
- `requirements.txt` — add `ibm-watsonx-ai` (one package, no transitive rewrites)
- `app.py` — surface provider status badge in sidebar only (2-line change)

**Non-goals:** Do NOT modify anomaly, forecast, KPI, simulator, or any UI page except
the two-line sidebar provider badge.

---

## Current IBM-Related Implementation

| Location | Type | Content |
|----------|------|---------|
| `app.py:305` | UI label | "EcoSync v2.0 • Resource Intelligence Platform" |
| `app.py:775` | UI caption | "powered by Agentic Tool Dispatching and IBM Granite synthesis" |
| `app.py:819` | UI markdown | "IBM Granite Foundation Architecture • Grounded Provenance" |
| `app.py:832` | Spinner text | "consulting Granite RAG..." |
| `app.py:845` | Response display | "IBM Granite Grounded Synthesis: Citing {provenance_sources}" |
| `src/agents/copilot_agent.py:39` | Docstring | "powered by IBM Granite" |
| `src/agents/__init__.py:3` | Module doc | "powered by IBM Granite and grounded telemetry" |
| `README.md` | Badge + docs | IBM Granite badge, feature description |

**Verdict:** All IBM references are documentation and UI labels only. There is no
`ibm-watsonx-ai` package in `requirements.txt`, no `Credentials` / `ModelInference` import
anywhere, and no `.env` file with IBM credentials. The copilot produces responses through
deterministic Python string templates — not LLM inference.

---

## Missing IBM Components

1. **`ibm-watsonx-ai` Python SDK** — not in `requirements.txt`
2. **`GraniteClient` wrapper** — no file under `src/ibm/`
3. **Credential / config layer** — no `.env` file, no `os.getenv` calls
4. **LLM synthesis call** — `execute_plan_and_answer` never calls an LLM; it builds
   strings from templates
5. **Embedding-based RAG** — retrieval uses scikit-learn TF-IDF; Granite embedding
   endpoint (`ibm/slate-125m-english-rtrvr`) is never invoked
6. **Provider status indicator** — no runtime check or UI badge showing
   "IBM Granite / watsonx.ai" vs "Local fallback"
7. **AI guardrails (HAP)** — the `moderations` field is never passed to any inference
   call, so harmful-content filtering is absent
8. **IBM integration tests** — `tests/` has no file that mocks or calls the watsonx API
9. **Architecture documentation** — `docs/bob/` is empty

---

## Proposed IBM-Only Architecture

The integration is a **thin vertical slice** inserted into the one point where the
copilot currently emits a string: `execute_plan_and_answer()`.

```
User Query
    │
    ▼
CopilotAgent.execute_plan_and_answer()
    │  (unchanged: intent routing, entity extraction, tool dispatch)
    ▼
Tool results assembled into a structured context dict
    │
    ▼  ◄── NEW: GraniteClient.synthesize(prompt, context)
GraniteClient (src/ibm/watsonx_client.py)
    │
    ├─ IBM credentials present? ─── YES ──► ModelInference.generate_text()
    │                                       ibm/granite-3-3-8b-instruct
    │                                       + HAP guardrails
    │
    └─────────────────────────── NO ───► TemplateRenderer.render(context)
                                          (existing string-template logic,
                                           moved to a helper, not deleted)
    │
    ▼
AgentResponse (unchanged struct)
    │
    ▼
Streamlit UI (unchanged)
    sidebar badge: "IBM Granite / watsonx.ai" or "Local fallback"
```

### RAG Extension (Granite Embeddings)

```
KnowledgeRetriever (src/rag/knowledge_retriever.py)  — unchanged
     │
     └── called by tool_search_guidelines()  — unchanged

EmbeddingsRetriever (src/rag/embeddings_retriever.py)  — NEW (optional upgrade)
     │
     ├─ IBM credentials present? ─── YES ──► Embeddings.embed_documents()
     │                                       ibm/slate-125m-english-rtrvr
     │                                       cosine similarity search
     │
     └─────────────────────────── NO ───► delegates to KnowledgeRetriever
```

The `EmbeddingsRetriever` is a **drop-in enhancement**; `tool_search_guidelines()` can
optionally swap to it without changing the existing `KnowledgeRetriever` at all.

---

## Required IBM Services / Packages

| Item | Value |
|------|-------|
| **Python package** | `ibm-watsonx-ai>=1.1.3` (current stable; published on PyPI) |
| **LLM model ID** | `ibm/granite-3-3-8b-instruct` |
| **Embedding model ID** | `ibm/slate-125m-english-rtrvr` |
| **IBM Cloud service** | watsonx.ai (region: `us-south`, `eu-de`, etc.) |
| **Auth method** | IBM Cloud API key → IAM token (managed by SDK) |
| **guardrails** | HAP filter via `moderations` field in `generate_text()` |

**No additional packages are needed.** `ibm-watsonx-ai` bundles its own auth and HTTP
client. No LangChain, no vector DB, no extra embedding library.

---

## Required Environment Variables

```dotenv
# ── IBM watsonx.ai credentials ──────────────────────────────────────────────
WATSONX_APIKEY=          # IBM Cloud API key (required for IBM provider)
WATSONX_URL=             # e.g. https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=      # watsonx.ai project UUID

# ── IBM Granite model selection ──────────────────────────────────────────────
GRANITE_MODEL_ID=ibm/granite-3-3-8b-instruct   # override if needed
GRANITE_MAX_NEW_TOKENS=512
GRANITE_TEMPERATURE=0.2                         # low for grounded factual output

# ── Responsible AI ───────────────────────────────────────────────────────────
GRANITE_HAP_THRESHOLD=0.5    # HAP filter threshold 0.0–1.0 (0.5 = balanced)
```

All variables are **optional at runtime**. When any of the three required credentials
(`WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID`) is absent or empty, the system
silently falls back to the existing template-based responses with no error.

---

## Minimal Files That Need Modification

### New files (4)

| File | Purpose |
|------|---------|
| `src/ibm/__init__.py` | Package marker |
| `src/ibm/watsonx_client.py` | `GraniteClient` — credentials loading, `synthesize()`, fallback logic, HAP guardrails |
| `src/rag/embeddings_retriever.py` | `EmbeddingsRetriever` — Granite-embedding-based search, delegates to `KnowledgeRetriever` when IBM unavailable |
| `tests/test_ibm_integration.py` | IBM layer tests: credential loading, fallback, mock inference, guardrails field, provider status |

### Modified files (3)

| File | Change | Lines affected |
|------|--------|---------------|
| `requirements.txt` | Add `ibm-watsonx-ai>=1.1.3` | 1 line appended |
| `src/agents/copilot_agent.py` | Import `GraniteClient`; replace final string-template block at end of each scenario with `self.granite.synthesize(context)` | ~20 lines changed across 4 scenarios |
| `app.py` | In sidebar section (~line 300): read `GraniteClient.provider_status()` and render badge | 2–4 lines |

### Documentation files (2)

| File | Purpose |
|------|---------|
| `.env.example` | Lists all env vars with descriptions and safe example values |
| `docs/bob/ibm-architecture.md` | Architecture reference, sequence diagram, env var table |

---

## Sub-Tasks

---

### Sub-Task 1 — IBM Package & Environment Config

**Status:** `[ ] pending`

**Intent:**
Add `ibm-watsonx-ai` to the dependency list and create `.env.example` so the project
can be installed with IBM support and developers know what credentials to provide.

**Expected Outcomes:**
- `requirements.txt` contains `ibm-watsonx-ai>=1.1.3`
- `.env.example` exists at project root with all six env vars documented
- No other file is changed

**Todo List:**
1. Append `ibm-watsonx-ai>=1.1.3` to `requirements.txt` (after the existing `requests` line)
2. Create `.env.example` with documented env vars and safe placeholder values

**Relevant Context:**
- Current `requirements.txt` is space-separated (unusual encoding); append without reformatting
- `python-dotenv==1.2.3` is already in `requirements.txt`; `.env.example` simply documents
  what goes in `.env` which is already git-ignored

---

### Sub-Task 2 — GraniteClient Wrapper (`src/ibm/watsonx_client.py`)

**Status:** `[ ] pending`

**Intent:**
Create the IBM integration layer that the rest of the codebase will call. It must:
- Load credentials from env vars via `python-dotenv`
- Expose `is_available() -> bool` (True when all three creds are set and reachable)
- Expose `synthesize(prompt: str, context: dict) -> str` which calls
  `ModelInference.generate_text()` with HAP guardrails and returns the Granite response
- Fall back to a deterministic template render when IBM is unavailable
- Expose `provider_status() -> str` returning `"IBM Granite / watsonx.ai"` or
  `"Local fallback"`

**Expected Outcomes:**
- `src/ibm/__init__.py` exists
- `src/ibm/watsonx_client.py` exists with `GraniteClient` class
- Class instantiates cleanly with no credentials (fallback mode, no exception)
- Class instantiates correctly when all env vars are set
- `synthesize()` always returns a non-empty string regardless of credential state

**Todo List:**
1. Create `src/ibm/__init__.py` (empty package marker)
2. Create `src/ibm/watsonx_client.py`:
   a. Load `.env` with `python_dotenv.load_dotenv()` at module import
   b. Read `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID`,
      `GRANITE_MODEL_ID`, `GRANITE_MAX_NEW_TOKENS`, `GRANITE_TEMPERATURE`,
      `GRANITE_HAP_THRESHOLD` from `os.getenv`
   c. `_build_client()` — creates `ibm_watsonx_ai.Credentials` and `ModelInference`
      only when all three required vars are non-empty; catches any exception and
      sets `self._available = False`
   d. `is_available() -> bool`
   e. `provider_status() -> str`
   f. `synthesize(prompt: str, context: dict) -> str`:
      - If available: call `ModelInference.generate_text(prompt, moderations=hap_config)`;
        return generated text
      - If unavailable or exception: call `_fallback_render(context)` and return result
   g. `_fallback_render(context: dict) -> str` — accepts the structured context dict
      that will be passed from `copilot_agent.py` and produces the same markdown string
      that the current template logic produces (the existing template logic will be
      moved here verbatim, not rewritten)

**Relevant Context:**
- Official SDK pattern from docs:
  ```python
  from ibm_watsonx_ai import Credentials
  from ibm_watsonx_ai.foundation_models import ModelInference
  credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_APIKEY)
  model = ModelInference(model_id=GRANITE_MODEL_ID, credentials=credentials,
                         project_id=WATSONX_PROJECT_ID)
  result = model.generate_text(prompt, moderations={"hap": {"input": {"enabled": True,
            "threshold": HAP_THRESHOLD}, "output": {"enabled": True,
            "threshold": HAP_THRESHOLD}}})
  ```
- HAP `moderations` field documented in watsonx.ai SDK; threshold 0.5 is balanced default
- `model_id` must be `"ibm/granite-3-3-8b-instruct"` (confirmed available in SDK docs)

---

### Sub-Task 3 — Wire GraniteClient into CopilotAgent

**Status:** `[ ] pending`

**Intent:**
Replace the four string-template response blocks in `execute_plan_and_answer()` with calls
to `GraniteClient.synthesize()`. The tool dispatch, intent routing, SDG tagging, and
`AgentResponse` structure remain identical. Only the final `answer` string generation
changes.

**Expected Outcomes:**
- `CopilotAgent.__init__()` instantiates `GraniteClient()` as `self.granite`
- Each of the four scenario blocks (A/B/C/D) assembles a `context` dict and calls
  `self.granite.synthesize(prompt, context)` to produce `answer`
- The fallback renders the identical markdown that the current templates produce
- All existing tests in `test_ecosync_services.py` still pass unchanged

**Todo List:**
1. Import `GraniteClient` at top of `src/agents/copilot_agent.py`
2. Add `self.granite = GraniteClient()` in `CopilotAgent.__init__()`
3. For each scenario (A, B, C, D):
   a. Assemble a `context` dict containing all data already computed (sim_data,
      anomalies, fc_summary, rag_docs, kpis, bldg_label, resource, pct_value, etc.)
   b. Build a natural-language `prompt` string that describes the scenario and
      instructs Granite to produce a grounded sustainability advisory answer
   c. Replace the `answer = (f"### ...")` block with:
      `answer = self.granite.synthesize(prompt, context)`
   d. Keep `followups`, `provenance`, `sdg_alignment`, `confidence_level` unchanged
4. The fallback path in `GraniteClient._fallback_render()` must contain the verbatim
   current template strings so that existing test assertions
   (`assertIn("What-If Impact Simulation", res1.answer)`) still pass

**Relevant Context:**
- Four scenario blocks are at lines 191–396 of `src/agents/copilot_agent.py`
- Test assertions in `tests/test_ecosync_services.py:76–83` check for specific strings
  ("What-If Impact Simulation", "Diagnostic Investigation") — these must remain in the
  fallback template and ideally in the Granite prompt so they appear in LLM output too
- The `context` dict should be serialisable (no DataFrames; pass pre-computed scalars)

---

### Sub-Task 4 — Granite Embeddings Retriever (Optional Enhancement)

**Status:** `[ ] pending`

**Intent:**
Create `src/rag/embeddings_retriever.py` that uses Granite's embedding endpoint
(`ibm/slate-125m-english-rtrvr`) when IBM credentials are available, for higher-quality
semantic retrieval than TF-IDF. Falls back to the existing `KnowledgeRetriever` when
unavailable. This is a non-breaking enhancement; `tool_search_guidelines()` can adopt it
without removing any existing code.

**Expected Outcomes:**
- `EmbeddingsRetriever` class exists and is importable
- When credentials are absent, `EmbeddingsRetriever.search()` delegates entirely to
  `KnowledgeRetriever.search()` — identical output
- When credentials are present, returns the same dict structure as `KnowledgeRetriever`
  so callers need no changes
- `KnowledgeRetriever` is NOT modified

**Todo List:**
1. Create `src/rag/embeddings_retriever.py` with `EmbeddingsRetriever` class:
   a. Constructor: instantiate `GraniteClient` for credential check; instantiate
      `KnowledgeRetriever` as fallback
   b. `search(query, top_k=3) -> List[Dict[str, Any]]`:
      - If `GraniteClient.is_available()`: embed query via
        `ibm_watsonx_ai.foundation_models.embeddings.Embeddings.embed_query(query)`;
        embed all chunks; return top-k by cosine similarity in the same dict format
      - Else: return `self._fallback.search(query, top_k)`
2. Update `CopilotAgent.tool_search_guidelines()` to use `EmbeddingsRetriever` instead
   of `KnowledgeRetriever` (one import swap, return type unchanged)

**Relevant Context:**
- SDK embeddings API: `Embeddings(model_id="ibm/slate-125m-english-rtrvr",
  credentials=credentials, project_id=project_id).embed_query(text)`
- The existing `KnowledgeChunk` list (already loaded by `KnowledgeRetriever`) can be
  reused; `EmbeddingsRetriever` wraps it and re-embeds chunks on first use
- Chunk count is small (≈20 chunks from 4 markdown files) so in-memory embedding is fine

---

### Sub-Task 5 — Provider Status Badge in Streamlit Sidebar

**Status:** `[ ] pending`

**Intent:**
Show the user which AI provider is active. A single `st.caption()` or `st.info()` in the
sidebar sidebar section of `app.py` reads `GraniteClient().provider_status()` and displays
either "🟢 IBM Granite / watsonx.ai" or "🟡 Local fallback (template)".

**Expected Outcomes:**
- Sidebar always shows the provider badge on app load
- No other page or UI section is changed
- `CopilotAgent` and all service instantiations remain in their existing locations

**Todo List:**
1. Locate the sidebar block in `app.py` (around line 290–310)
2. Import `GraniteClient` at the top of `app.py` (alongside existing imports)
3. Add `_provider = GraniteClient().provider_status()` in the sidebar block
4. Render `st.caption(f"AI Provider: {_provider}")` immediately after the existing
   `st.caption("EcoSync v2.0 • Resource Intelligence Platform")` line

**Relevant Context:**
- `app.py:305` already has a caption in the sidebar; the new badge goes on the next line
- `GraniteClient()` must be instantiated cheaply (no HTTP call at constructor time;
  only credential check via `os.getenv`)

---

### Sub-Task 6 — IBM Integration Tests

**Status:** `[ ] pending`

**Intent:**
Create `tests/test_ibm_integration.py` that tests the IBM layer in isolation without
requiring real IBM credentials. Uses `unittest.mock` to patch `ModelInference` so tests
run in CI with no network access.

**Expected Outcomes:**
- All tests pass with no credentials set (fallback mode)
- All tests pass with mocked credentials (IBM mode)
- Tests cover: credential loading, `is_available()`, `provider_status()`, `synthesize()`
  IBM path, `synthesize()` fallback path, HAP guardrails field presence,
  `EmbeddingsRetriever` fallback delegation

**Todo List:**
1. Create `tests/test_ibm_integration.py` with class `TestIBMIntegration(unittest.TestCase)`:
   a. `test_no_credentials_fallback()` — unset all WATSONX env vars; assert
      `GraniteClient().is_available() == False` and `provider_status() == "Local fallback"`
   b. `test_synthesize_fallback_returns_string()` — no credentials; call
      `synthesize("test prompt", {})` and assert returns non-empty string
   c. `test_synthesize_ibm_path()` — mock `ModelInference.generate_text` to return
      `{"results": [{"generated_text": "test output"}]}`; set env vars; call `synthesize()`
      and assert returns "test output"
   d. `test_hap_guardrails_present()` — patch `generate_text`; capture kwargs; assert
      `"moderations"` key is in call kwargs and contains `"hap"`
   e. `test_provider_status_ibm()` — mock credentials valid; assert
      `provider_status() == "IBM Granite / watsonx.ai"`
   f. `test_embeddings_retriever_fallback()` — no credentials; assert
      `EmbeddingsRetriever().search("energy leak", top_k=2)` returns same count as
      `KnowledgeRetriever().search("energy leak", top_k=2)`

**Relevant Context:**
- Use `unittest.mock.patch` and `os.environ` manipulation within each test
- Pattern: `with unittest.mock.patch("ibm_watsonx_ai.foundation_models.ModelInference") as mock_mi:`
- Tests must be addable to `tests/run_tests.py` test suite

---

### Sub-Task 7 — Documentation

**Status:** `[ ] pending`

**Intent:**
Write two documentation files that satisfy the plan requirement for `docs/bob/` coverage
and provide a clear reference for anyone operating or extending the IBM integration.

**Expected Outcomes:**
- `docs/bob/ibm-architecture.md` exists with architecture description, env var reference
  table, and sequence description of the IBM inference path
- `.env.example` exists at project root (created in Sub-Task 1, finalised here)
- `README.md` is NOT modified

**Todo List:**
1. Write `docs/bob/ibm-architecture.md` covering:
   - Overview paragraph
   - Component table (GraniteClient, EmbeddingsRetriever, CopilotAgent integration point)
   - Environment variable reference table (all 7 vars, required/optional, default)
   - Sequence description: user query → tool dispatch → Granite synthesis → response
   - Fallback behaviour description
   - Responsible AI section: HAP filter config, human-in-the-loop note
   - Testing section: how to run IBM tests, how to mock credentials

---

## Verification / Testing Plan

### Offline (no IBM credentials)

```
python -m pytest tests/test_ibm_integration.py -v
python -m pytest tests/test_ecosync_services.py -v   # must still pass unchanged
python -m pytest tests/run_tests.py -v
```

All tests must pass. `GraniteClient.is_available()` returns `False`. Copilot responses
are produced by the fallback template and contain the same assertions as today.

### Online (IBM credentials in `.env`)

```
# Set WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID in .env
python -m pytest tests/test_ibm_integration.py -v -k "ibm_path"
streamlit run app.py
# Verify sidebar shows "🟢 IBM Granite / watsonx.ai"
# Send a query in AI Copilot; verify response is Granite-generated prose
```

### Provider status verification

- With credentials: sidebar badge → "🟢 IBM Granite / watsonx.ai"
- Without credentials: sidebar badge → "🟡 Local fallback (template)"
- Remove `WATSONX_APIKEY` from `.env` at runtime: badge should show fallback on reload

### Responsible AI verification

- Inspect network call (or test mock): `moderations` dict with `"hap"` key must be
  present in every `generate_text()` call
- Verify `GRANITE_HAP_THRESHOLD` env var is read and passed to the filter

### Regression check

- All pages of the Streamlit app render without error in both IBM and fallback modes
- `tests/test_anomaly_service.py`, `tests/test_multi_resource.py`,
  `tests/test_ecosync_services.py` all pass with zero changes

---

*Plan written for EcoSync Resource Manager — IBM integration layer only.*
*Do not implement until this plan is approved.*
