"""Startup validation script — run to verify all modules load correctly."""
import sys
from pathlib import Path
# Add project root (parent of tests/) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.styles import get_styles
from src.ui.components import provider_badge
from src.ui.pages import overview, insights, investigate
from src.ui.pages import forecast as fp, copilot, scenario_lab, knowledge, reports, about
from src.ibm.watsonx_client import GraniteClient
from src.agents.copilot_agent import CopilotAgent
from src.data.validate_data import validate_multi_resource_data
from src.analytics.kpi_engine import calculate_unified_campus_kpis
from src.anomaly.anomaly_service import AnomalyService
from src.forecasting.forecast_service import ForecastService
from src.impact.whatif_simulator import WhatIfSimulator
from src.rag.knowledge_retriever import KnowledgeRetriever

sample = Path("data/raw/campus_multi_resource_sample.csv")
val = validate_multi_resource_data(sample)
assert val["is_valid"], "Data validation failed"
df = val["cleaned_df"]
print(f"Data: {len(df)} rows")

g = GraniteClient()
kr = KnowledgeRetriever()
ag = CopilotAgent(data_df=df)
kpis = calculate_unified_campus_kpis(df)

print(f"GraniteClient: {g.provider_status()}")
print(f"KnowledgeRetriever: {len(kr.get_all_chunks())} chunks")
print(f"KPIs energy: {kpis['energy']['total_mwh']:.1f} MWh")

resp = ag.execute_plan_and_answer("Why is Building C using more water?")
assert "Diagnostic" in resp.answer
assert resp.provider == "Local grounded fallback"
print(f"Copilot: provider={resp.provider}, answer_len={len(resp.answer)}")

styles = get_styles()
assert "<style>" in styles
print("Styles: OK")

print("\nAll startup checks PASSED")
