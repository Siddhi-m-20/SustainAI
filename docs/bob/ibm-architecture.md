# EcoSync — IBM Granite Integration Architecture

## Overview

EcoSync integrates IBM Granite (`ibm/granite-3-3-8b-instruct`) via the
`ibm_watsonx_ai` Python SDK to provide grounded natural-language responses in the AI
Copilot. When IBM credentials are absent or unavailable, the system transparently falls
back to deterministic template responses — the UI never falsely claims IBM Granite is
active.

---

## Component Table

| Component | File | Purpose |
|---|---|---|
| `GraniteClient` | `src/ibm/watsonx_client.py` | Credential loading, model init, synthesis, fallback |
| `CopilotAgent` | `src/agents/copilot_agent.py` | Tool dispatch, intent routing, Granite wiring |
| `KnowledgeRetriever` | `src/rag/knowledge_retriever.py` | TF-IDF RAG over local knowledge base |
| Provider badge | `app.py` sidebar | Displays IBM/fallback status in real time |
| IBM tests | `tests/test_ibm_integration.py` | Offline + mocked IBM test suite |

---

## Environment Variable Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `WATSONX_APIKEY` | Yes (IBM mode) | — | IBM Cloud API key |
| `WATSONX_URL` | Yes (IBM mode) | — | watsonx.ai endpoint, e.g. `https://us-south.ml.cloud.ibm.com` |
| `WATSONX_PROJECT_ID` | Yes (IBM mode) | — | watsonx.ai project UUID |
| `GRANITE_MODEL_ID` | No | `ibm/granite-3-3-8b-instruct` | Granite model identifier |
| `GRANITE_MAX_NEW_TOKENS` | No | `600` | Max tokens per response |
| `GRANITE_TEMPERATURE` | No | `0.2` | Generation temperature (low = factual) |

Copy `.env.example` to `.env` and fill in the three required values to enable IBM mode.

---

## Inference Flow

```
User Question
      │
      ▼
CopilotAgent.execute_plan_and_answer(user_query)
      │
      ├── _extract_building / _extract_resource / _extract_percentage
      │
      ├── Scenario A (what-if):   tool_simulate_whatif + tool_search_guidelines
      ├── Scenario B (anomaly):   tool_detect_anomalies + tool_search_guidelines
      ├── Scenario C (forecast):  tool_run_forecast
      └── Scenario D (general):   tool_search_guidelines + tool_query_kpis
      │
      ▼
context dict assembled (serialisable scalars + RAG passages)
      │
      ▼
GraniteClient.build_prompt(scenario, context)
  → Responsible AI system instruction
  → Structured tool results
  → Retrieved knowledge chunks (title + 400 chars + source name)
  → User question + output format instructions
      │
      ▼
GraniteClient.synthesize(prompt, context)
      │
      ├── _available == True?
      │     YES → ModelInference.generate_text(prompt=prompt)
      │           Stop sequences: ["###END"]
      │           Returns generated text
      │
      └── _available == False  or  IBM error
            → _fallback_render(context)
               Returns deterministic template (identical to pre-IBM behaviour)
      │
      ▼
AgentResponse.answer (always a non-empty string)
AgentResponse.provider ("IBM Granite / watsonx.ai" or "Local grounded fallback")
      │
      ▼
Streamlit UI → provider badge in sidebar
```

---

## Responsible AI Safeguards

### System Prompt (always injected)
Every Granite call includes an explicit system instruction that:
1. Requires using only supplied data — no fabricated numbers
2. Labels outputs as FACT, HYPOTHESIS, or RECOMMENDATION
3. Forbids confirming any physical fault without evidence
4. Requires citing knowledge source names
5. Recommends human verification for all physical interventions

### Fallback Behaviour
Any network failure, SDK exception, or missing credential causes immediate fallback
to the deterministic template renderer. The app never crashes due to IBM unavailability.

### No PII / No DataFrames
Context dicts passed to IBM contain only pre-computed scalars and truncated text
excerpts. Raw DataFrames, timestamps, and building names are converted to plain text
before leaving this module.

### HAP Guardrails
IBM's HAP (Hate, Abuse, Profanity) filter is available via the `moderations` field
of `ModelInference.generate_text()`. For the EcoSync domain (technical facility
management), the responsible AI system prompt already constrains outputs appropriately.
If operating in a public-facing or consumer context, add:
```python
result = model.generate_text(
    prompt=prompt,
    moderations={"hap": {"input": {"enabled": True, "threshold": 0.5},
                         "output": {"enabled": True, "threshold": 0.5}}}
)
```

---

## Running IBM Integration Tests

```bash
# All tests (no credentials needed — fully mocked)
python -m pytest tests/test_ibm_integration.py -v

# Specific test classes
python -m pytest tests/test_ibm_integration.py::TestGraniteClientOffline -v
python -m pytest tests/test_ibm_integration.py::TestGraniteClientMocked -v
python -m pytest tests/test_ibm_integration.py::TestCopilotAgentWithGranite -v

# Full test suite
python -m pytest tests/ -v
```

### Mocking credentials in tests
```python
import os, importlib
os.environ["WATSONX_APIKEY"] = "test-key"
os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WATSONX_PROJECT_ID"] = "test-project"

import src.ibm.watsonx_client as m
importlib.reload(m)

from unittest.mock import MagicMock, patch
with patch("ibm_watsonx_ai.foundation_models.ModelInference") as MockMI:
    MockMI.return_value.generate_text.return_value = "Mocked response"
    importlib.reload(m)
    from src.ibm.watsonx_client import GraniteClient
    client = GraniteClient()
```

---

## Built with IBM BOB

IBM BOB (AI coding assistant) contributed to this integration:

- **Repository inspection**: Read all source files before making any changes
- **Architecture design**: Designed the thin injection-point pattern that avoids
  modifying anomaly/forecast/KPI modules
- **IBM documentation lookup**: Used `search_ibm_docs` against official watsonx docs
  to verify `ibm/granite-3-3-8b-instruct` model ID, SDK import paths, and `Credentials`
  constructor signature
- **Implementation**: Wrote `GraniteClient`, updated `CopilotAgent`, created all UI
  page modules
- **Testing**: Wrote 21 IBM-specific tests covering offline fallback, mocked IBM path,
  responsible AI prompt content, and CopilotAgent integration
- **Documentation**: Created this architecture document and updated README
