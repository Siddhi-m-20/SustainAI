"""
src/ibm/watsonx_client.py
─────────────────────────
IBM Granite integration layer for EcoSync Resource Manager.

Provides GraniteClient — a thin wrapper around ibm_watsonx_ai that:
  • Loads credentials exclusively from environment variables (never hard-coded)
  • Calls ibm/granite-3-3-8b-instruct via watsonx.ai when credentials are present
  • Silently falls back to deterministic template rendering when IBM is unavailable
  • Exposes provider_status() so the UI never falsely claims IBM is active
  • Applies a Responsible AI system prompt to every Granite call

No secrets are logged.  Context dicts are serialised to plain text
before leaving this module — no DataFrames or PII reach the model.
"""

import os
import logging
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load .env if present (harmless if absent)
load_dotenv()

logger = logging.getLogger(__name__)

# ── Read credentials from environment ─────────────────────────────────────────
_APIKEY: str = os.getenv("WATSONX_APIKEY", "").strip()
_URL: str = os.getenv("WATSONX_URL", "").strip()
_PROJECT: str = os.getenv("WATSONX_PROJECT_ID", "").strip()
_MODEL: str = os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-3-8b-instruct").strip()
_MAX_TOK: int = int(os.getenv("GRANITE_MAX_NEW_TOKENS", "600"))
_TEMPERATURE: float = float(os.getenv("GRANITE_TEMPERATURE", "0.2"))

# ── Responsible AI system prompt injected into every Granite call ──────────────
_SYSTEM_PROMPT = """You are EcoSync AI, a grounded sustainability advisor for campus facility managers.

IMPORTANT RULES — follow these exactly:
1. Use ONLY the data and context provided in this prompt. Do not fabricate numbers.
2. Clearly label sections as FACT, HYPOTHESIS, or RECOMMENDATION.
3. Never confirm a physical fault (leak, equipment failure) as certain — always label as HYPOTHESIS and recommend human verification.
4. When citing knowledge sources, use the exact source name provided (e.g. "energy_efficiency_standards.md").
5. Show your assumptions explicitly when making calculations or projections.
6. Recommend human facility engineering verification before any physical intervention.
7. Keep the response concise, structured with markdown headers, and professional.
8. Do not repeat the raw data back verbatim — synthesise it into insights.
"""


class GraniteClient:
    """
    IBM Granite inference client with transparent local fallback.

    Usage:
        client = GraniteClient()
        if client.is_available():
            # IBM Granite will be used
        status = client.provider_status()  # "IBM Granite / watsonx.ai" or "Local grounded fallback"
        answer = client.synthesize(prompt, context)  # always returns a string
    """

    def __init__(self) -> None:
        self._available: bool = False
        self._model: Optional[Any] = None
        self._init_client()

    def _init_client(self) -> None:
        """Attempt to build a ModelInference object. Sets _available=False on any failure."""
        if not (_APIKEY and _URL and _PROJECT):
            logger.debug("[GraniteClient] IBM credentials not set — running in fallback mode.")
            return
        try:
            from ibm_watsonx_ai import Credentials  # type: ignore
            from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # type: ignore

            credentials = Credentials(url=_URL, api_key=_APIKEY)
            params = {
                GenParams.MAX_NEW_TOKENS: _MAX_TOK,
                GenParams.TEMPERATURE: _TEMPERATURE,
                GenParams.STOP_SEQUENCES: ["###END", "---END---"],
            }
            self._model = ModelInference(
                model_id=_MODEL,
                credentials=credentials,
                project_id=_PROJECT,
                params=params,
            )
            self._available = True
            logger.info(f"[GraniteClient] Connected to IBM watsonx.ai — model: {_MODEL}")
        except ImportError:
            logger.warning("[GraniteClient] ibm-watsonx-ai not installed — using fallback mode.")
        except Exception as exc:
            logger.warning(f"[GraniteClient] Could not connect to IBM watsonx.ai: {exc}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True when IBM credentials are valid and the SDK is installed."""
        return self._available

    def provider_status(self) -> str:
        """Returns the display string for the UI provider badge."""
        if self._available:
            return "IBM Granite / watsonx.ai"
        return "Local grounded fallback"

    def build_prompt(self, scenario: str, context: Dict[str, Any]) -> str:
        """
        Build a grounded Granite prompt from tool results + RAG context.

        The prompt contains:
          - A Responsible AI system instruction block
          - Structured tool results (serialised scalars, no DataFrames)
          - Retrieved RAG knowledge chunks (titles + first 300 chars)
          - The user's original question
          - Output format instructions
        """
        user_query = context.get("user_query", "")
        building = context.get("building", "All Facilities")
        resource = context.get("resource", "ALL")
        rag_docs = context.get("rag_docs", [])

        # Build RAG context block
        rag_block = ""
        if rag_docs:
            rag_block = "\n## Retrieved Knowledge Context\n"
            for doc in rag_docs:
                rag_block += (
                    f"\n**Source: {doc.get('source', 'knowledge_base')}**\n"
                    f"*{doc.get('title', '')}*\n"
                    f"{doc.get('content', '')[:400]}\n"
                )
        else:
            rag_block = "\n## Retrieved Knowledge Context\nNo specific guidelines retrieved for this query.\n"

        # Build scenario-specific data block
        data_block = self._build_data_block(scenario, context)

        prompt = (
            f"{_SYSTEM_PROMPT}\n"
            f"---\n"
            f"## Query\nUser question: {user_query}\n"
            f"Facility scope: {building} | Resource focus: {resource}\n"
            f"{data_block}"
            f"{rag_block}\n"
            f"---\n"
            f"## Instructions\n"
            f"Provide a concise professional advisory response using the data above.\n"
            f"Structure: one markdown heading, then FACT / HYPOTHESIS / RECOMMENDATION sections as applicable.\n"
            f"End with 'Recommended next steps for the facility team.'\n"
            f"###END"
        )
        return prompt

    def synthesize(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Generate a grounded response.

        Tries IBM Granite first; falls back to template renderer on any error.
        Always returns a non-empty string.
        """
        if self._available and self._model is not None:
            try:
                result = self._model.generate_text(prompt=prompt)
                text = result.strip() if isinstance(result, str) else ""
                # Strip any trailing stop sequences
                for stop in ["###END", "---END---"]:
                    if text.endswith(stop):
                        text = text[: -len(stop)].strip()
                if text:
                    return text
            except Exception as exc:
                logger.warning(f"[GraniteClient] IBM inference failed, using fallback: {exc}")

        return self._fallback_render(context)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_data_block(self, scenario: str, context: Dict[str, Any]) -> str:
        """Serialise tool result data into a readable text block for the prompt."""
        if scenario == "whatif":
            sim = context.get("sim_data", {})
            return (
                f"\n## Simulation Results\n"
                f"- Reduction applied: {context.get('reduction_pct', 20):.0f}%\n"
                f"- Total cost saved: ${sim.get('total_cost_saved_usd', 0):,.2f} USD\n"
                f"- Energy conserved: {sim.get('energy_saved_kwh', 0):,.1f} kWh ({sim.get('energy_saved_mwh', 0):.2f} MWh)\n"
                f"- Water conserved: {sim.get('water_saved_liters', 0):,.0f} L ({sim.get('water_saved_m3', 0):.1f} m³)\n"
                f"- Waste diverted: {sim.get('waste_diverted_kg', 0):,.1f} kg\n"
                f"- Carbon avoided: {sim.get('carbon_avoided_tco2e', 0):.2f} tCO2e\n"
                f"- Equivalent cars removed: {sim.get('equivalent_cars_removed_annual', 0):.1f}\n"
                f"- Equivalent trees planted: {sim.get('equivalent_trees_planted', 0):,.0f}\n"
                f"- Energy intensity change: {sim.get('sdg7_energy_intensity_change_pct', 0)}%\n"
                f"- Water reduction: {sim.get('sdg6_water_reduction_pct', 0)}%\n"
                f"- Projected diversion rate: {sim.get('sdg12_diversion_rate_projected_pct', 0)}%\n"
            )
        elif scenario == "anomaly":
            anomalies = context.get("anomalies", [])
            if anomalies:
                top = anomalies[0]
                return (
                    f"\n## Anomaly Detection Results\n"
                    f"- Top anomaly building: {top.get('building', 'Unknown')}\n"
                    f"- Resource: {top.get('resource', context.get('resource', 'Unknown'))}\n"
                    f"- Actual value: {top.get('actual_value', 0):.2f}\n"
                    f"- Expected baseline: {top.get('expected_value', 0):.2f}\n"
                    f"- Deviation: {top.get('deviation_pct', 0):.1f}%\n"
                    f"- Severity score: {top.get('severity_score', 0):.1f}/100\n"
                    f"- Severity tier: {top.get('severity_tier', 'UNKNOWN')}\n"
                    f"- Detected pattern: {top.get('pattern', 'Unclassified')}\n"
                    f"- Timestamp: {top.get('timestamp', 'Recent period')}\n"
                    f"- Total anomalies detected: {len(anomalies)}\n"
                )
            else:
                return (
                    f"\n## Anomaly Detection Results\n"
                    f"No anomalies above severity threshold detected for this scope.\n"
                )
        elif scenario == "forecast":
            fc = context.get("fc_summary", {})
            return (
                f"\n## Forecast Results (48-Hour Horizon)\n"
                f"- Resource: {context.get('resource', 'energy')}\n"
                f"- Total projected: {fc.get('total_predicted', 0):,.1f} {fc.get('unit', '')}\n"
                f"- Anticipated peak: {fc.get('peak_predicted', 0):,.1f} {fc.get('unit', '')}\n"
                f"- Peak time: {fc.get('peak_forecast_time', 'N/A')}\n"
                f"- Peak alert hours: {fc.get('peak_alert_hours', 0)}\n"
                f"- Projected cost: ${fc.get('projected_cost_usd', 0):,.2f} USD\n"
            )
        else:  # general
            kpis = context.get("kpis", {})
            return (
                f"\n## Current Campus KPIs\n"
                f"- Total energy: {kpis.get('energy_mwh', 0):,.2f} MWh\n"
                f"- Total water: {kpis.get('water_m3', 0):,.1f} m³\n"
                f"- Waste diversion rate: {kpis.get('diversion_pct', 0):.1f}%\n"
            )

    def _fallback_render(self, context: Dict[str, Any]) -> str:
        """
        Deterministic template renderer used when IBM is unavailable.
        Produces responses matching existing test assertions.
        """
        scenario = context.get("scenario", "general")
        building = context.get("building", "All Facilities")
        resource = context.get("resource", "ALL")
        bldg_label = building.replace("_", " ") if building != "ALL" else "All Campus Facilities"

        if scenario == "whatif":
            return self._fallback_whatif(context, bldg_label, resource)
        elif scenario == "anomaly":
            return self._fallback_anomaly(context, building, resource)
        elif scenario == "forecast":
            return self._fallback_forecast(context)
        else:
            return self._fallback_general()

    def _fallback_whatif(self, context: Dict[str, Any], bldg_label: str, resource: str) -> str:
        reduction = context.get("reduction_pct", 20.0)
        sim = context.get("sim_data", {})
        answer = (
            f"### What-If Impact Simulation for {bldg_label}\n\n"
            f"Based on historical baseline telemetry and a **{reduction:.0f}% operational reduction**:\n\n"
            f"* **Financial Savings:** Estimated **${sim.get('total_cost_saved_usd', 0):,.2f} USD** in avoided utility expenditures.\n"
        )
        if resource in ["energy", "ALL"]:
            answer += (
                f"* **Energy Conserved:** **{sim.get('energy_saved_kwh', 0):,.1f} kWh** "
                f"({sim.get('energy_saved_mwh', 0):.2f} MWh), reducing energy intensity by "
                f"**{sim.get('sdg7_energy_intensity_change_pct', 0)}%**.\n"
            )
        if resource in ["water", "ALL"]:
            answer += (
                f"* **Water Conserved:** **{sim.get('water_saved_liters', 0):,.0f} Liters** "
                f"({sim.get('water_saved_m3', 0):.1f} m³), a "
                f"**{sim.get('sdg6_water_reduction_pct', 0)}%** reduction.\n"
            )
        if resource in ["waste", "ALL"]:
            answer += (
                f"* **Waste Diverted:** **{sim.get('waste_diverted_kg', 0):,.1f} kg** redirected from landfill, "
                f"raising diversion to **{sim.get('sdg12_diversion_rate_projected_pct', 0)}%**.\n\n"
            )
        answer += (
            f"\n**Environmental Decarbonization:**\n"
            f"* Avoids **{sim.get('carbon_avoided_tco2e', 0):.2f} metric tonnes CO2e**.\n"
            f"* Equivalent to removing **{sim.get('equivalent_cars_removed_annual', 0):.1f} passenger vehicles** "
            f"for a year or planting **{sim.get('equivalent_trees_planted', 0):,.0f} urban trees**.\n\n"
            f"**Possible Operational Path:**\n"
            f"Implement automated thermostat setbacks during unoccupied hours (20:00–06:00) as recommended "
            f"in ASHRAE Standard 90.1, and verify BMS scheduler overrides.\n\n"
            f"*ASSUMPTION: Savings calculated from historical baseline using linear reduction model. "
            f"Actual results depend on building systems, occupancy patterns, and implementation quality. "
            f"Recommend human engineering review before committing to capital investment decisions.*"
        )
        return answer

    def _fallback_anomaly(self, context: Dict[str, Any], building: str, resource: str) -> str:
        anomalies = context.get("anomalies", [])
        if anomalies:
            top = anomalies[0]
            bldg_name = top.get("building", building).replace("_", " ")
            res_name = top.get("resource", resource).capitalize()
            actual = top.get("actual_value", 0.0)
            expected = top.get("expected_value", 0.0)
            dev_pct = top.get("deviation_pct", 0.0)
            sev = top.get("severity_score", 0.0)
            unit = "kWh" if "energy" in res_name.lower() else ("m³" if "water" in res_name.lower() else "kg")
            answer = (
                f"### Diagnostic Investigation: {bldg_name} ({res_name})\n\n"
                f"**FACT — Telemetry Reading:**\n"
                f"EcoSync detected a **{top.get('severity_tier', 'HIGH')} Severity Anomaly** (Score: {sev:.1f}/100):\n\n"
                f"* Observed: `{actual:,.2f} {unit}`\n"
                f"* Expected baseline: `{expected:,.2f} {unit}`\n"
                f"* Deviation: `+{dev_pct:.1f}%` above seasonal/schedule norm\n"
                f"* Timestamp: `{top.get('timestamp', 'Recent Period')}`\n\n"
                f"**HYPOTHESIS — AI Diagnostic Explanation:**\n"
            )
            if "water" in res_name.lower():
                answer += (
                    "Elevated consumption during low-occupancy hours is consistent with an uncontained plumbing flow. "
                    "Engineering benchmarks (EPA WaterSense / MNF analysis) suggest possible: failed flushometer solenoid, "
                    "cooling tower makeup overflow, or fixture leak. *This is a hypothesis — not a confirmed diagnosis.*\n\n"
                    "**RECOMMENDATION — Human Verification Required:**\n"
                    "1. Dispatch a technician to inspect main submeter isolation valves and mechanical rooms.\n"
                    "2. Perform acoustic leak verification on supply risers.\n"
                    "3. Check automated irrigation timers."
                )
            elif "energy" in res_name.lower():
                answer += (
                    "Substantial power draw outside scheduled operating hours is consistent with an un-reverted BMS "
                    "manual override, auxiliary HVAC chiller cycling, or lab exhaust fans at full capacity. "
                    "*This is a hypothesis — not a confirmed diagnosis.*\n\n"
                    "**RECOMMENDATION — Human Verification Required:**\n"
                    "1. Verify BAS time-of-day clock and zone damper schedules.\n"
                    "2. Audit plug loads and computer lab shut-off automations.\n"
                    "3. Confirm laboratory fume hood sashes are closed."
                )
            else:
                answer += (
                    "Abnormal surge in residual landfill mass is consistent with single-stream contamination batch "
                    "rejection or unscheduled cleanouts bypassing sorting stations. "
                    "*This is a hypothesis — not a confirmed diagnosis.*\n\n"
                    "**RECOMMENDATION — Human Verification Required:**\n"
                    "1. Audit loading dock compactors for sorting compliance.\n"
                    "2. Reinforce pictorial bin signage in dining and common areas."
                )
        else:
            answer = (
                f"### Diagnostic Status\n\n"
                f"**FACT:** No critical anomalies (severity > 35) are currently flagged for the selected scope.\n\n"
                f"Consumption is tracking within normal 2-sigma contextual confidence bounds."
            )
        return answer

    def _fallback_forecast(self, context: Dict[str, Any]) -> str:
        fc = context.get("fc_summary", {})
        building = context.get("building", "Campus Wide")
        resource = context.get("resource", "energy")
        return (
            f"### 48-Hour Demand Forecast for {building} ({resource.capitalize()})\n\n"
            f"**FACT — Model Projection:**\n"
            f"* **Total Projected:** **{fc.get('total_predicted', 0):,.1f} {fc.get('unit', '')}**\n"
            f"* **Anticipated Peak:** **{fc.get('peak_predicted', 0):,.1f} {fc.get('unit', '')}**\n"
            f"* **Peak Time:** `{fc.get('peak_forecast_time', 'N/A')}`\n"
            f"* **Peak Alert Hours:** **{fc.get('peak_alert_hours', 0)}** hours above 90th percentile\n"
            f"* **Projected Cost:** **${fc.get('projected_cost_usd', 0):,.2f} USD**\n\n"
            f"**RECOMMENDATION:**\n"
            f"Stagger large equipment start times 30 minutes before projected peak hours to flatten load spikes "
            f"and prevent utility demand penalties.\n\n"
            f"*ASSUMPTION: Forecast based on cyclical demand patterns from historical data. "
            f"Actual demand may vary due to weather, occupancy changes, or equipment events.*"
        )

    def _fallback_general(self) -> str:
        return (
            f"### Recommended Priority Actions for Resource Optimisation\n\n"
            f"Grounded in campus benchmarks and sustainability frameworks:\n\n"
            f"1. **Eliminate Off-Hours Energy Load:**\n"
            f"   * Enforce automatic HVAC unoccupied setbacks (16°C heating / 28°C cooling) per ASHRAE 90.1.\n"
            f"   * Potential saving: 15–25% of off-peak energy expenditure.\n\n"
            f"2. **Night-Flow Water Surveillance:**\n"
            f"   * Monitor 02:00–04:30 AM baseline flow (< 0.15 m³/h). Flags leaks within 2 hours.\n"
            f"   * Potential saving: Up to 40% of non-essential water loss.\n\n"
            f"3. **Targeted Organics Diversion:**\n"
            f"   * Focus organic collection in dining buildings (>40% of campus organic volume).\n"
            f"   * Potential: raise diversion towards 60% Silver standard.\n\n"
            f"*RECOMMENDATION: Verify all actions with facility engineering staff before implementation.*"
        )
