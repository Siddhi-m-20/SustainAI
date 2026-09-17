import unittest
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate_data import validate_multi_resource_data
from src.forecasting.forecast_service import ForecastService
from src.impact.whatif_simulator import WhatIfSimulator
from src.rag.knowledge_retriever import KnowledgeRetriever
from src.agents.copilot_agent import CopilotAgent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "data" / "raw" / "campus_multi_resource_sample.csv"


class TestEcoSyncServices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        val_res = validate_multi_resource_data(SAMPLE_DATA)
        assert val_res["is_valid"], "Sample data failed validation"
        cls.df = val_res["cleaned_df"]

    def test_forecast_service(self):
        service = ForecastService()
        fc = service.predict(self.df, resource="energy", building="Building_A", horizon_hours=24)
        
        self.assertEqual(len(fc), 24)
        self.assertIn("predicted_demand", fc.columns)
        self.assertIn("lower_95", fc.columns)
        self.assertIn("upper_95", fc.columns)
        self.assertTrue((fc["predicted_demand"] >= 0).all())
        self.assertTrue((fc["upper_95"] >= fc["predicted_demand"]).all())

        kpis = service.get_forecast_kpis(fc, resource="energy")
        self.assertGreater(kpis["total_predicted"], 0)
        self.assertEqual(kpis["unit"], "kWh")

    def test_whatif_simulator(self):
        simulator = WhatIfSimulator()
        result = simulator.simulate(
            self.df,
            building="Building_A",
            off_hours_reduction_pct=25.0,
            night_water_leak_fix_pct=50.0,
            compost_diversion_target_pct=30.0,
        )

        self.assertGreater(result.total_cost_saved_usd, 0)
        self.assertGreater(result.carbon_avoided_tco2e, 0)
        self.assertGreater(result.energy_saved_kwh, 0)
        self.assertGreater(result.water_saved_liters, 0)
        self.assertGreater(result.equivalent_trees_planted, 0)

    def test_knowledge_retriever(self):
        retriever = KnowledgeRetriever()
        chunks = retriever.get_all_chunks()
        self.assertGreaterEqual(len(chunks), 4)

        # Test semantic search
        results = retriever.search("unoccupied temperature setbacks ASHRAE", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("SDG 7", results[0]["sdg_tag"])

        water_res = retriever.search("minimum night flow leak detection", top_k=1)
        self.assertGreater(len(water_res), 0)
        self.assertIn("SDG 6", water_res[0]["sdg_tag"])

    def test_copilot_agent(self):
        agent = CopilotAgent(data_df=self.df)

        # Test what-if question
        res1 = agent.execute_plan_and_answer("How much energy could we save if Building A reduced off-hours consumption by 20%?")
        self.assertIn("What-If Impact Simulation", res1.answer)
        self.assertTrue(any(t.tool_name == "simulate_whatif" for t in res1.tools_executed))

        # Test anomaly investigation question
        res2 = agent.execute_plan_and_answer("Why is Building C using more water?")
        self.assertIn("Diagnostic Investigation", res2.answer)
        self.assertTrue(any(t.tool_name == "detect_anomalies" for t in res2.tools_executed))


if __name__ == "__main__":
    unittest.main()
