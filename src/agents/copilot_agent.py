import os
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from src.analytics.kpi_engine import calculate_unified_campus_kpis
from src.anomaly.anomaly_service import AnomalyService
from src.forecasting.forecast_service import ForecastService
from src.impact.whatif_simulator import WhatIfSimulator
from src.rag.knowledge_retriever import KnowledgeRetriever
from src.ibm.watsonx_client import GraniteClient


@dataclass
class ToolExecutionRecord:
    tool_name: str
    arguments: Dict[str, Any]
    summary_result: str
    tool_type: str = "Diagnostic Analytics"
    grounding_source: str = "Telemetry Engine"


@dataclass
class AgentResponse:
    user_query: str
    answer: str
    tools_executed: List[ToolExecutionRecord]
    sdg_alignment: List[str]
    confidence_level: str
    provenance_sources: List[str]
    suggested_followups: List[str]
    provider: str = "Local grounded fallback"


class CopilotAgent:
    """
    EcoSync Agentic AI Copilot.
    Routes natural language facility queries to specialised diagnostic tools,
    retrieves institutional domain knowledge via RAG, and synthesises grounded,
    responsible recommendations through IBM Granite (with local fallback).
    """

    def __init__(self, data_df: Optional[pd.DataFrame] = None):
        self.data_df = data_df
        self.anomaly_service = AnomalyService()
        self.forecast_service = ForecastService()
        self.whatif_simulator = WhatIfSimulator()
        self.knowledge_retriever = KnowledgeRetriever()
        self.granite = GraniteClient()

        # Cache anomaly results if dataset is provided
        self.cached_anomalies: Optional[pd.DataFrame] = None
        if self.data_df is not None:
            self._precompute_anomalies()

    def set_data(self, df: pd.DataFrame) -> None:
        self.data_df = df
        self._precompute_anomalies()

    def _precompute_anomalies(self) -> None:
        try:
            if self.data_df is not None and not self.data_df.empty:
                self.cached_anomalies = self.anomaly_service.detect_multi_resource_anomalies(self.data_df)
        except Exception as e:
            print(f"[CopilotAgent] Anomaly precomputation note: {e}")
            self.cached_anomalies = None

    # ------------------------------------------------------------------
    # Tool Definitions
    # ------------------------------------------------------------------
    def tool_query_kpis(self, resource: str = "ALL", building: str = "ALL") -> Dict[str, Any]:
        """Tool: Aggregates current consumption, cost, emissions, and baseloads."""
        if self.data_df is None or self.data_df.empty:
            return {"error": "No campus dataset loaded."}

        df_subset = self.data_df.copy()
        if building != "ALL" and "building" in df_subset.columns:
            df_subset = df_subset[df_subset["building"] == building]

        return calculate_unified_campus_kpis(df_subset)

    def tool_detect_anomalies(self, resource: str = "ALL", building: str = "ALL", min_severity: float = 30.0) -> List[Dict[str, Any]]:
        """Tool: Identifies active deviations, actual vs baseline, and severity."""
        if self.cached_anomalies is None:
            if self.data_df is not None:
                self._precompute_anomalies()
            if self.cached_anomalies is None:
                return []

        anoms = self.cached_anomalies.copy()
        if building != "ALL" and "building" in anoms.columns:
            anoms = anoms[anoms["building"] == building]
        if resource != "ALL" and "resource" in anoms.columns:
            anoms = anoms[anoms["resource"].str.lower() == resource.lower()]
        if "severity_score" in anoms.columns:
            anoms = anoms[anoms["severity_score"] >= min_severity]
            anoms = anoms.sort_values("severity_score", ascending=False)

        top_anoms = anoms.head(5).to_dict(orient="records")
        # Serialise timestamps to string
        for a in top_anoms:
            if "timestamp" in a:
                a["timestamp"] = str(a["timestamp"])
        return top_anoms

    def tool_run_forecast(self, resource: str = "energy", building: str = "ALL", horizon_hours: int = 24) -> Dict[str, Any]:
        """Tool: Predicts future demand and highlights anticipated peak periods."""
        if self.data_df is None or self.data_df.empty:
            return {"error": "No campus dataset loaded."}

        bldg_param = None if building == "ALL" else building
        fc_df = self.forecast_service.predict(self.data_df, resource=resource, building=bldg_param, horizon_hours=horizon_hours)
        summary = self.forecast_service.get_forecast_kpis(fc_df, resource=resource)
        return summary

    def tool_simulate_whatif(
        self,
        building: str = "ALL",
        off_hours_reduction_pct: float = 20.0,
        leak_fix_pct: float = 40.0,
        compost_diversion_pct: float = 30.0,
    ) -> Dict[str, Any]:
        """Tool: Calculates cost and carbon savings for hypothetical changes."""
        if self.data_df is None or self.data_df.empty:
            return {"error": "No campus dataset loaded."}

        sim_res = self.whatif_simulator.simulate(
            self.data_df,
            building=building,
            off_hours_reduction_pct=off_hours_reduction_pct,
            night_water_leak_fix_pct=leak_fix_pct,
            compost_diversion_target_pct=compost_diversion_pct,
        )
        return sim_res.to_dict()

    def tool_search_guidelines(self, query: str) -> List[Dict[str, Any]]:
        """Tool: Retrieves ASHRAE, LEED, EPA, and SDG engineering guidelines via RAG."""
        return self.knowledge_retriever.search(query, top_k=2)

    # ------------------------------------------------------------------
    # Agentic Intent Analysis & Tool Selection
    # ------------------------------------------------------------------
    def _extract_building(self, query: str) -> str:
        q_upper = query.upper()
        for letter in ["A", "B", "C", "D"]:
            if f"BUILDING {letter}" in q_upper or f"BUILDING_{letter}" in q_upper:
                return f"Building_{letter}"
        return "ALL"

    def _extract_resource(self, query: str) -> str:
        q_lower = query.lower()
        if "water" in q_lower or "leak" in q_lower or "plumb" in q_lower:
            return "water"
        elif "waste" in q_lower or "compost" in q_lower or "recycl" in q_lower or "trash" in q_lower:
            return "waste"
        elif "energy" in q_lower or "kwh" in q_lower or "power" in q_lower or "electric" in q_lower:
            return "energy"
        return "ALL"

    def _extract_percentage(self, query: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", query)
        if match:
            return float(match.group(1))
        return None

    def execute_plan_and_answer(self, user_query: str) -> AgentResponse:
        """
        Agentic Execution Pipeline:
        1. Classify user question intent; extract target entities (building, resource, numbers).
        2. Decide and execute the optimal set of diagnostic tools.
        3. Retrieve engineering knowledge chunks for responsible grounding.
        4. Synthesise a structured, evidence-backed answer via IBM Granite (or fallback).
        """
        tools_executed: List[ToolExecutionRecord] = []
        bldg = self._extract_building(user_query)
        resource = self._extract_resource(user_query)
        pct_value = self._extract_percentage(user_query)
        q_lower = user_query.lower()

        sdg_alignment = []
        if resource == "energy" or "energy" in q_lower or "power" in q_lower or "kwh" in q_lower:
            sdg_alignment.append("SDG 7: Affordable & Clean Energy")
        if resource == "water" or "water" in q_lower or "leak" in q_lower:
            sdg_alignment.append("SDG 6: Clean Water & Sanitation")
        if resource == "waste" or "waste" in q_lower or "recycle" in q_lower or "compost" in q_lower:
            sdg_alignment.append("SDG 12: Responsible Consumption")
        if not sdg_alignment:
            sdg_alignment = ["SDG 7: Clean Energy", "SDG 6: Clean Water", "SDG 12: Responsible Consumption"]

        # ------------------------------------------------------------------
        # Scenario A: What-If / Savings Inquiry
        # ------------------------------------------------------------------
        if any(w in q_lower for w in ["how much", "save", "saving", "reduce", "what if", "if building"]):
            reduction = pct_value if pct_value is not None else 20.0
            sim_data = self.tool_simulate_whatif(
                building=bldg,
                off_hours_reduction_pct=reduction if resource in ["energy", "ALL"] else 0.0,
                leak_fix_pct=reduction if resource in ["water", "ALL"] else 0.0,
                compost_diversion_pct=reduction if resource in ["waste", "ALL"] else 0.0,
            )
            tools_executed.append(ToolExecutionRecord(
                tool_name="simulate_whatif",
                arguments={"building": bldg, "reduction_pct": reduction, "resource": resource},
                summary_result=f"Total cost saved: ${sim_data.get('total_cost_saved_usd', 0):,.2f} | Carbon avoided: {sim_data.get('carbon_avoided_tco2e', 0)} tCO2e"
            ))

            rag_docs = self.tool_search_guidelines(f"setback efficiency {resource}")
            tools_executed.append(ToolExecutionRecord(
                tool_name="search_guidelines",
                arguments={"query": f"setback efficiency {resource}"},
                summary_result=f"Found {len(rag_docs)} standard operational protocols."
            ))

            bldg_label = bldg.replace("_", " ") if bldg != "ALL" else "All Campus Facilities"
            context = {
                "scenario": "whatif",
                "building": bldg_label,
                "reduction_pct": reduction,
                "resource": resource,
                "sim_data": {k: v for k, v in sim_data.items()},
                "rag_docs": [
                    {"title": d["title"], "content": d["content"][:300], "source": d["source_file"]}
                    for d in rag_docs
                ],
                "user_query": user_query,
            }
            prompt = self.granite.build_prompt("whatif", context)
            answer = self.granite.synthesize(prompt, context)

            followups = [
                f"What is the forecasted demand for {bldg_label} next week?",
                "Which building used the most energy this period?",
                f"Show active water anomalies in {bldg_label}.",
            ]
            provenance = ["Counterfactual Simulation Engine", "ASHRAE 90.1 / EPA WARM Model"]

        # ------------------------------------------------------------------
        # Scenario B: "Why" / Anomaly / Investigation Inquiry
        # ------------------------------------------------------------------
        elif any(w in q_lower for w in ["why", "anomaly", "unusual", "spike", "leak", "high", "issue", "wasted", "waste most"]):
            anomalies = self.tool_detect_anomalies(resource=resource, building=bldg, min_severity=35.0)
            tools_executed.append(ToolExecutionRecord(
                tool_name="detect_anomalies",
                arguments={"resource": resource, "building": bldg, "min_severity": 35.0},
                summary_result=f"Detected {len(anomalies)} prioritised anomalies."
            ))

            rag_docs = self.tool_search_guidelines(f"{resource} anomaly leak baseline {bldg}")
            tools_executed.append(ToolExecutionRecord(
                tool_name="search_guidelines",
                arguments={"query": f"{resource} anomaly leak baseline {bldg}"},
                summary_result=f"Retrieved {len(rag_docs)} institutional diagnostic protocols."
            ))

            context = {
                "scenario": "anomaly",
                "building": bldg,
                "resource": resource,
                "anomalies": anomalies[:3],  # top 3, already serialisable dicts
                "rag_docs": [
                    {"title": d["title"], "content": d["content"][:300], "source": d["source_file"]}
                    for d in rag_docs
                ],
                "user_query": user_query,
            }
            prompt = self.granite.build_prompt("anomaly", context)
            answer = self.granite.synthesize(prompt, context)

            followups = [
                f"How much could we save if {bldg.replace('_', ' ')} fixes this issue?",
                "Forecast next 48-hour demand for this facility.",
                "Search sustainability guidelines for cooling tower optimisation.",
            ]
            provenance = ["Isolation Forest Multi-Resource Service", "Contextual Baseline Engine", "Campus Standard Operating Guidelines"]

        # ------------------------------------------------------------------
        # Scenario C: Forecast / Future Demand Inquiry
        # ------------------------------------------------------------------
        elif any(w in q_lower for w in ["forecast", "predict", "future", "tomorrow", "next week", "demand"]):
            fc_res = resource if resource != "ALL" else "energy"
            fc_summary = self.tool_run_forecast(resource=fc_res, building=bldg, horizon_hours=48)
            tools_executed.append(ToolExecutionRecord(
                tool_name="run_forecast",
                arguments={"resource": fc_res, "building": bldg, "horizon_hours": 48},
                summary_result=f"Total projected: {fc_summary.get('total_predicted')} {fc_summary.get('unit')} | Peak: {fc_summary.get('peak_predicted')} {fc_summary.get('unit')}"
            ))

            bldg_name = bldg.replace("_", " ") if bldg != "ALL" else "Campus Wide"
            context = {
                "scenario": "forecast",
                "building": bldg_name,
                "resource": fc_res,
                "fc_summary": {k: v for k, v in fc_summary.items()},
                "user_query": user_query,
            }
            prompt = self.granite.build_prompt("forecast", context)
            answer = self.granite.synthesize(prompt, context)

            followups = [
                f"Simulate a 15% peak shaving impact for {bldg_name}.",
                "Check if any buildings have current energy anomalies.",
                "Review ASHRAE peak demand shaving protocols.",
            ]
            provenance = ["Ridge Cyclical Demand Regressor", "Campus Utility Rate Schedule"]

        # ------------------------------------------------------------------
        # Scenario D: Actions / Best Practices / General Copilot
        # ------------------------------------------------------------------
        else:
            rag_docs = self.tool_search_guidelines(user_query)
            tools_executed.append(ToolExecutionRecord(
                tool_name="search_guidelines",
                arguments={"query": user_query},
                summary_result=f"Retrieved {len(rag_docs)} standard sustainability references."
            ))

            kpis = self.tool_query_kpis(resource=resource, building=bldg)
            tools_executed.append(ToolExecutionRecord(
                tool_name="query_kpis",
                arguments={"resource": resource, "building": bldg},
                summary_result="Extracted aggregate multi-resource KPIs."
            ))

            context = {
                "scenario": "general",
                "building": bldg,
                "resource": resource,
                "rag_docs": [
                    {"title": d["title"], "content": d["content"][:300], "source": d["source_file"]}
                    for d in rag_docs
                ],
                "kpis": {
                    "energy_mwh": kpis.get("energy", {}).get("total_mwh", 0),
                    "water_m3": kpis.get("water", {}).get("total_m3", 0),
                    "diversion_pct": kpis.get("waste", {}).get("diversion_rate_pct", 0),
                },
                "user_query": user_query,
            }
            prompt = self.granite.build_prompt("general", context)
            answer = self.granite.synthesize(prompt, context)

            followups = [
                "How much energy can Building C save by cutting night load?",
                "Investigate active water anomalies across campus.",
                "Run a 7-day demand forecast for electricity.",
            ]
            provenance = ["ASHRAE Standard 90.1", "EPA WaterSense Protocols", "UN SDG Target Framework"]

        return AgentResponse(
            user_query=user_query,
            answer=answer,
            tools_executed=tools_executed,
            sdg_alignment=sdg_alignment,
            confidence_level="High (Telemetry Grounded)",
            provenance_sources=provenance,
            suggested_followups=followups,
            provider=self.granite.provider_status(),
        )
