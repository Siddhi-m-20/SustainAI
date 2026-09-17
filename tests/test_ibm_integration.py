"""
tests/test_ibm_integration.py
─────────────────────────────
IBM integration layer tests for EcoSync Resource Manager.

All tests run OFFLINE — no real IBM credentials required.
Mock credentials and mocked SDK responses are used where needed.

Tests cover:
  • No-credential fallback mode
  • provider_status() strings
  • synthesize() fallback — returns string, contains expected markers
  • build_prompt() — returns non-empty string
  • Mocked IBM inference path (when SDK is importable)
  • Data block serialisation for each scenario
"""

import os
import sys
import importlib
import unittest
import unittest.mock
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _unset_ibm_env():
    """Remove all IBM env vars so each test starts clean."""
    for var in ["WATSONX_APIKEY", "WATSONX_URL", "WATSONX_PROJECT_ID",
                "GRANITE_MODEL_ID", "GRANITE_MAX_NEW_TOKENS", "GRANITE_TEMPERATURE"]:
        os.environ.pop(var, None)


def _reload_client_module():
    """Reload the watsonx_client module to pick up env var changes."""
    import src.ibm.watsonx_client as m
    importlib.reload(m)
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Offline Tests — no credentials set
# ──────────────────────────────────────────────────────────────────────────────

class TestGraniteClientOffline(unittest.TestCase):
    """All tests run with no IBM credentials in the environment."""

    def setUp(self):
        _unset_ibm_env()
        m = _reload_client_module()
        from src.ibm.watsonx_client import GraniteClient
        self.GraniteClient = GraniteClient
        self.client = GraniteClient()

    def test_no_credentials_not_available(self):
        """is_available() must be False when no credentials are set."""
        self.assertFalse(self.client.is_available())

    def test_provider_status_is_fallback(self):
        """provider_status() must return the exact fallback string."""
        self.assertEqual(self.client.provider_status(), "Local grounded fallback")

    def test_synthesize_returns_string(self):
        """synthesize() must always return a non-empty string."""
        ctx = {
            "scenario": "general",
            "building": "ALL",
            "resource": "ALL",
            "rag_docs": [],
            "kpis": {"energy_mwh": 100, "water_m3": 50, "diversion_pct": 45},
            "user_query": "What should we prioritise?",
        }
        result = self.client.synthesize("test prompt", ctx)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)

    def test_whatif_fallback_contains_required_marker(self):
        """Fallback whatif response must contain 'What-If Impact Simulation' for test compatibility."""
        sim_data = {
            "total_cost_saved_usd": 1500.0,
            "carbon_avoided_tco2e": 3.2,
            "energy_saved_kwh": 8000.0,
            "energy_saved_mwh": 8.0,
            "water_saved_liters": 3000.0,
            "water_saved_m3": 3.0,
            "waste_diverted_kg": 120.0,
            "sdg7_energy_intensity_change_pct": 5.0,
            "sdg6_water_reduction_pct": 10.0,
            "sdg12_diversion_rate_projected_pct": 65.0,
            "equivalent_cars_removed_annual": 1.5,
            "equivalent_trees_planted": 147,
        }
        ctx = {
            "scenario": "whatif",
            "building": "All Campus Facilities",
            "reduction_pct": 20.0,
            "resource": "ALL",
            "sim_data": sim_data,
            "rag_docs": [],
            "user_query": "How much can we save?",
        }
        result = self.client.synthesize("test", ctx)
        self.assertIn("What-If Impact Simulation", result)

    def test_anomaly_fallback_contains_required_marker(self):
        """Fallback anomaly response must contain 'Diagnostic Investigation'."""
        ctx = {
            "scenario": "anomaly",
            "building": "Building_C",
            "resource": "water",
            "anomalies": [
                {
                    "building": "Building_C",
                    "resource": "water",
                    "actual_value": 2.5,
                    "expected_value": 0.8,
                    "deviation_pct": 212.5,
                    "severity_score": 78.3,
                    "severity_tier": "CRITICAL",
                    "pattern": "Off-Hours Excess",
                    "timestamp": "2024-01-15 03:00:00",
                }
            ],
            "rag_docs": [],
            "user_query": "Why is Building C using more water?",
        }
        result = self.client.synthesize("test", ctx)
        self.assertIn("Diagnostic Investigation", result)

    def test_anomaly_fallback_no_anomalies(self):
        """Fallback with empty anomalies list must not crash."""
        ctx = {
            "scenario": "anomaly",
            "building": "ALL",
            "resource": "energy",
            "anomalies": [],
            "rag_docs": [],
            "user_query": "Any energy issues?",
        }
        result = self.client.synthesize("test", ctx)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

    def test_forecast_fallback_returns_string(self):
        """Fallback forecast response must be a non-empty string."""
        ctx = {
            "scenario": "forecast",
            "building": "Campus Wide",
            "resource": "energy",
            "fc_summary": {
                "total_predicted": 4800.0,
                "peak_predicted": 210.5,
                "unit": "kWh",
                "peak_forecast_time": "2024-01-16 14:00",
                "peak_alert_hours": 6,
                "projected_cost_usd": 672.0,
            },
            "user_query": "What is the forecast?",
        }
        result = self.client.synthesize("test", ctx)
        self.assertIsInstance(result, str)
        self.assertIn("Forecast", result)

    def test_build_prompt_returns_nonempty_string(self):
        """build_prompt() must return a string with content for all scenarios."""
        for scenario in ["whatif", "anomaly", "forecast", "general"]:
            ctx = {
                "scenario": scenario,
                "user_query": "test query",
                "rag_docs": [{"title": "Test", "content": "Test content", "source": "test.md"}],
                "building": "Building_A",
                "resource": "energy",
                "sim_data": {"total_cost_saved_usd": 0},
                "anomalies": [],
                "fc_summary": {},
                "kpis": {"energy_mwh": 0, "water_m3": 0, "diversion_pct": 0},
                "reduction_pct": 20.0,
            }
            prompt = self.client.build_prompt(scenario, ctx)
            self.assertIsInstance(prompt, str, f"Failed for scenario: {scenario}")
            self.assertGreater(len(prompt), 50, f"Prompt too short for scenario: {scenario}")
            self.assertIn("test query", prompt)

    def test_build_prompt_includes_rag_source(self):
        """build_prompt() must include the RAG source name in the prompt."""
        ctx = {
            "scenario": "general",
            "user_query": "energy efficiency tips",
            "rag_docs": [
                {
                    "title": "ASHRAE 90.1 Setbacks",
                    "content": "Unoccupied setback temperatures should be...",
                    "source": "energy_efficiency_standards.md",
                }
            ],
            "building": "ALL",
            "resource": "energy",
            "kpis": {},
        }
        prompt = self.client.build_prompt("general", ctx)
        self.assertIn("energy_efficiency_standards.md", prompt)

    def test_build_prompt_responsible_ai_instruction_present(self):
        """Every prompt must contain the responsible AI instruction."""
        ctx = {
            "scenario": "general",
            "user_query": "test",
            "rag_docs": [],
            "kpis": {},
        }
        prompt = self.client.build_prompt("general", ctx)
        self.assertIn("HYPOTHESIS", prompt)
        self.assertIn("human verification", prompt.lower())


# ──────────────────────────────────────────────────────────────────────────────
# Mocked IBM path tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGraniteClientMocked(unittest.TestCase):
    """Tests that simulate the IBM SDK being present and functional."""

    def setUp(self):
        _unset_ibm_env()

    def tearDown(self):
        _unset_ibm_env()

    def test_synthesize_returns_fallback_when_sdk_absent(self):
        """When SDK is not installed, synthesize() returns fallback template."""
        # This is the normal offline case — SDK not installed
        from src.ibm.watsonx_client import GraniteClient
        client = GraniteClient()
        ctx = {"scenario": "general", "user_query": "test", "rag_docs": [], "kpis": {}}
        result = client.synthesize("test prompt", ctx)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

    def test_synthesize_falls_back_on_exception(self):
        """synthesize() must fall back gracefully if generate_text raises."""
        from src.ibm.watsonx_client import GraniteClient
        client = GraniteClient()
        # Manually force _available=True with a mock model that raises
        mock_model = unittest.mock.MagicMock()
        mock_model.generate_text.side_effect = RuntimeError("Connection timeout")
        client._available = True
        client._model = mock_model

        ctx = {"scenario": "general", "user_query": "test", "rag_docs": [], "kpis": {}}
        result = client.synthesize("test prompt", ctx)
        # Must not raise; must return fallback string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        mock_model.generate_text.assert_called_once()

    def test_synthesize_uses_model_when_available(self):
        """synthesize() must call generate_text and return its result when available."""
        from src.ibm.watsonx_client import GraniteClient
        client = GraniteClient()
        mock_model = unittest.mock.MagicMock()
        mock_model.generate_text.return_value = "Granite grounded response about energy."
        client._available = True
        client._model = mock_model

        ctx = {"scenario": "general", "user_query": "test", "rag_docs": [], "kpis": {}}
        result = client.synthesize("test prompt", ctx)
        self.assertEqual(result, "Granite grounded response about energy.")
        mock_model.generate_text.assert_called_once_with(prompt="test prompt")

    def test_empty_generate_text_falls_back(self):
        """If generate_text returns empty string, fallback must be used."""
        from src.ibm.watsonx_client import GraniteClient
        client = GraniteClient()
        mock_model = unittest.mock.MagicMock()
        mock_model.generate_text.return_value = "   "  # whitespace only
        client._available = True
        client._model = mock_model

        ctx = {"scenario": "general", "user_query": "test", "rag_docs": [], "kpis": {}}
        result = client.synthesize("test prompt", ctx)
        # Should fall back to template
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 10)

    def test_provider_status_when_available(self):
        """provider_status() must return 'IBM Granite / watsonx.ai' when _available."""
        from src.ibm.watsonx_client import GraniteClient
        client = GraniteClient()
        client._available = True
        self.assertEqual(client.provider_status(), "IBM Granite / watsonx.ai")

    def test_is_available_reflects_state(self):
        """is_available() must reflect the _available attribute."""
        from src.ibm.watsonx_client import GraniteClient
        client = GraniteClient()
        client._available = False
        self.assertFalse(client.is_available())
        client._available = True
        self.assertTrue(client.is_available())


# ──────────────────────────────────────────────────────────────────────────────
# CopilotAgent integration with GraniteClient
# ──────────────────────────────────────────────────────────────────────────────

class TestCopilotAgentWithGranite(unittest.TestCase):
    """Verify CopilotAgent correctly uses GraniteClient and falls back gracefully."""

    @classmethod
    def setUpClass(cls):
        from src.data.validate_data import validate_multi_resource_data
        sample = PROJECT_ROOT / "data" / "raw" / "campus_multi_resource_sample.csv"
        if sample.exists():
            val = validate_multi_resource_data(sample)
            cls.df = val["cleaned_df"] if val["is_valid"] else None
        else:
            cls.df = None

    def setUp(self):
        _unset_ibm_env()

    def test_copilot_agent_has_granite_attribute(self):
        """CopilotAgent must expose a .granite attribute."""
        from src.agents.copilot_agent import CopilotAgent
        agent = CopilotAgent()
        self.assertTrue(hasattr(agent, "granite"))

    def test_copilot_agent_response_has_provider_field(self):
        """AgentResponse must include a provider field."""
        if self.df is None:
            self.skipTest("Sample data not available")
        from src.agents.copilot_agent import CopilotAgent
        agent = CopilotAgent(data_df=self.df)
        resp = agent.execute_plan_and_answer("What should we focus on?")
        self.assertTrue(hasattr(resp, "provider"))

    def test_copilot_whatif_scenario_fallback(self):
        """What-if query must route to whatif scenario and return expected marker."""
        if self.df is None:
            self.skipTest("Sample data not available")
        from src.agents.copilot_agent import CopilotAgent
        agent = CopilotAgent(data_df=self.df)
        resp = agent.execute_plan_and_answer(
            "How much energy could we save if Building A reduced off-hours by 20%?"
        )
        self.assertIn("What-If Impact Simulation", resp.answer)
        self.assertTrue(any(t.tool_name == "simulate_whatif" for t in resp.tools_executed))

    def test_copilot_anomaly_scenario_fallback(self):
        """Anomaly investigation query must route to anomaly scenario."""
        if self.df is None:
            self.skipTest("Sample data not available")
        from src.agents.copilot_agent import CopilotAgent
        agent = CopilotAgent(data_df=self.df)
        resp = agent.execute_plan_and_answer("Why is Building C using more water?")
        self.assertIn("Diagnostic", resp.answer)
        self.assertTrue(any(t.tool_name == "detect_anomalies" for t in resp.tools_executed))

    def test_copilot_provider_status_is_fallback_offline(self):
        """Provider status must be 'Local grounded fallback' with no IBM credentials."""
        if self.df is None:
            self.skipTest("Sample data not available")
        from src.agents.copilot_agent import CopilotAgent
        agent = CopilotAgent(data_df=self.df)
        resp = agent.execute_plan_and_answer("What is the current energy status?")
        self.assertEqual(resp.provider, "Local grounded fallback")


if __name__ == "__main__":
    unittest.main(verbosity=2)
