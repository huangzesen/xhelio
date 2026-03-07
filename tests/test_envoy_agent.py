"""
Tests for agent.envoy_agent — EnvoyAgent class.

Tests the mission-specific sub-agent without requiring a Gemini API key.
Verifies prompt construction, structural behavior, and tool filtering.

Run with: python -m pytest tests/test_envoy_agent.py
"""

import pytest
from knowledge.catalog import SPACECRAFT
from knowledge.prompt_builder import build_envoy_prompt
from agent.tools import get_tool_schemas
from agent.agent_registry import ENVOY_TOOLS


class TestBuildEnvoyPromptForAgent:
    """Verify that build_envoy_prompt produces usable prompts for EnvoyAgent."""

    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        """Block network calls so build_envoy_prompt uses only local cache."""
        monkeypatch.setattr(
            "knowledge.metadata_client.requests.get",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("blocked")),
        )

    def test_representative_missions_produce_prompt(self):
        """Quick check on representative CDAWeb + PPI missions.

        The full all-missions test lives in test_prompt_builder.py and is
        marked @pytest.mark.slow.
        """
        for sc_id in ("PSP", "ACE", "CASSINI_PPI"):
            prompt = build_envoy_prompt(sc_id)
            assert isinstance(prompt, str)
            assert len(prompt) > 50

    def test_psp_prompt_is_focused(self):
        prompt = build_envoy_prompt("PSP")
        assert "Parker Solar Probe" in prompt
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in prompt
        # Should NOT mention other missions
        assert "AC_H2_MFI" not in prompt
        assert "OMNI_HRO_1MIN" not in prompt

    def test_ace_prompt_is_focused(self):
        prompt = build_envoy_prompt("ACE")
        # ACE name may be "ACE" or "Advanced Composition Explorer"
        assert "ACE" in prompt
        assert "AC_H2_MFI" in prompt
        assert "PSP_FLD_L2_MAG_RTN_1MIN" not in prompt

    def test_prompt_contains_data_specialist_identity(self):
        prompt = build_envoy_prompt("PSP")
        assert "data specialist agent" in prompt.lower()

    def test_prompt_directs_to_list_parameters(self):
        prompt = build_envoy_prompt("PSP")
        assert "list_parameters" in prompt

    def test_ppi_mission_produces_prompt(self):
        """PPI missions (ppi group) should produce a valid prompt."""
        prompt = build_envoy_prompt("JUNO_PPI")
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_spice_mission_uses_spice_prompt(self):
        """SPICE mission (spice group) should use the SPICE-specific prompt."""
        prompt = build_envoy_prompt("SPICE")
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        # SPICE prompt should mention ephemeris/trajectory concepts
        assert "ephemeri" in prompt.lower() or "spice" in prompt.lower()

    def test_invalid_mission_raises(self):
        with pytest.raises(KeyError):
            build_envoy_prompt("NONEXISTENT")


class TestEnvoyAgentToolFiltering:
    """Verify mission sub-agents do NOT have plotting tools."""

    PLOTTING_TOOLS = {
        "plot_data",
        "change_time_range",
        "export_plot",
        "get_plot_info",
        "plot_computed_data",
    }

    def test_envoy_tools_exclude_plotting(self):
        envoy_tools = get_tool_schemas(names=ENVOY_TOOLS)
        names = {t["name"] for t in envoy_tools}
        assert names.isdisjoint(self.PLOTTING_TOOLS), (
            f"Mission sub-agents should not have plotting tools, found: {names & self.PLOTTING_TOOLS}"
        )

    def test_envoy_tools_include_fetch(self):
        envoy_tools = get_tool_schemas(names=ENVOY_TOOLS)
        names = {t["name"] for t in envoy_tools}
        assert "fetch_data" in names
        assert "list_fetched_data" in names

    def test_envoy_tools_exclude_compute(self):
        """EnvoyAgent no longer has compute tools — those moved to DataOpsAgent."""
        envoy_tools = get_tool_schemas(names=ENVOY_TOOLS)
        names = {t["name"] for t in envoy_tools}
        assert "custom_operation" not in names
        assert "describe_data" not in names
        assert "save_data" not in names

    def test_envoy_tools_include_discovery(self):
        envoy_tools = get_tool_schemas(names=ENVOY_TOOLS)
        names = {t["name"] for t in envoy_tools}
        assert "search_datasets" in names
        assert "list_parameters" in names
        assert "get_dataset_docs" in names

    def test_envoy_tools_include_conversation(self):
        envoy_tools = get_tool_schemas(names=ENVOY_TOOLS)
        names = {t["name"] for t in envoy_tools}
        assert "ask_clarification" in names


class TestEnvoyAgentImport:
    """Verify EnvoyAgent can be imported and has expected Agent interface."""

    def test_import(self):
        from agent.envoy_agent import EnvoyAgent

        assert EnvoyAgent is not None

    def test_class_has_send(self):
        from agent.envoy_agent import EnvoyAgent

        assert hasattr(EnvoyAgent, "send")
        assert callable(getattr(EnvoyAgent, "send"))

    def test_class_has_get_token_usage(self):
        from agent.envoy_agent import EnvoyAgent

        assert hasattr(EnvoyAgent, "get_token_usage")
        assert callable(getattr(EnvoyAgent, "get_token_usage"))

    def test_class_has_start_stop_status(self):
        from agent.envoy_agent import EnvoyAgent

        assert hasattr(EnvoyAgent, "start")
        assert hasattr(EnvoyAgent, "stop")
        assert hasattr(EnvoyAgent, "status")
