"""Tests for tool_registry.json validation and envoy kind registry."""

import json
from pathlib import Path

import pytest

from agent import tools
from agent.agent_registry import (
    AGENT_CALL_REGISTRY,
    _REGISTRY,
)
from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY


def get_tool_names() -> set[str]:
    """Get all tool names from tools.py TOOLS list."""
    return {t["name"] for t in tools.TOOLS}


class TestToolRegistryJson:
    """Validation tests for tool_registry.json."""

    def test_json_loads(self):
        """tool_registry.json is valid JSON."""
        path = Path(__file__).parent.parent / "agent" / "tool_registry.json"
        data = json.loads(path.read_text())
        assert data["version"] == 1

    def test_agents_section_exists(self):
        """Agents section is present and non-empty."""
        assert "agents" in _REGISTRY
        assert len(_REGISTRY["agents"]) >= 7  # orchestrator, viz_plotly, viz_mpl, viz_jsx, dataops, planner, data_io

    def test_all_agent_call_tools_exist(self):
        """Every tool name referenced in agents.*.call must exist in tools.py TOOLS.

        This catches typos and stale references.
        """
        known = get_tool_names()
        for agent_name, cfg in _REGISTRY["agents"].items():
            for tool_name in cfg["call"]:
                assert tool_name in known, (
                    f"agents.{agent_name}.call references unknown tool '{tool_name}'"
                )

    def test_all_agent_informed_tools_exist(self):
        """Every tool name in agents.*.informed must exist in tools.py."""
        known = get_tool_names()
        for agent_name, cfg in _REGISTRY["agents"].items():
            for tool_name in cfg["informed"]:
                assert tool_name in known, (
                    f"agents.{agent_name}.informed references unknown tool '{tool_name}'"
                )

    def test_agent_call_registry_matches_json(self):
        """AGENT_CALL_REGISTRY should contain entries for all agents in JSON."""
        for agent_name in _REGISTRY["agents"]:
            ctx_key = f"ctx:{agent_name}"
            assert ctx_key in AGENT_CALL_REGISTRY, (
                f"AGENT_CALL_REGISTRY missing '{ctx_key}'"
            )

    def test_envoy_call_registry_from_kinds(self):
        """ctx:envoy should exist and contain kind-derived tools."""
        envoy_tools = AGENT_CALL_REGISTRY.get("ctx:envoy")
        assert envoy_tools is not None
        assert "browse_parameters" in envoy_tools
        assert "fetch_data_cdaweb" in envoy_tools
        assert "fetch_data_ppi" in envoy_tools
        assert "list_fetched_data" in envoy_tools

    def test_derived_constants_not_empty(self):
        """Derived Python constants are non-empty."""
        assert AGENT_CALL_REGISTRY

    def test_orchestrator_has_permission_and_package_tools(self):
        """Orchestrator call list includes permission and package management tools."""
        orch_tools = _REGISTRY["agents"]["orchestrator"]["call"]
        assert "ask_user_permission" in orch_tools
        assert "install_package" in orch_tools
        assert "manage_sandbox_packages" in orch_tools

    def test_envoy_sections_removed_from_json(self):
        """Old envoy sections should no longer exist in tool_registry.json."""
        assert "envoy_groups" not in _REGISTRY
        assert "envoy_group_assignments" not in _REGISTRY
        assert "envoy_default_group" not in _REGISTRY
        assert "envoy_informed" not in _REGISTRY
        assert "envoy" not in _REGISTRY.get("agents", {})


class TestEnvoyKindRegistry:
    """Tests for the envoy kind registry (replaces old group tests)."""

    def test_kind_registry_resolves_cdaweb(self):
        """Default missions resolve to 'cdaweb' kind."""
        assert ENVOY_KIND_REGISTRY.get_kind("ACE") == "cdaweb"
        assert ENVOY_KIND_REGISTRY.get_kind("WIND") == "cdaweb"

    def test_kind_registry_resolves_ppi(self):
        """PPI missions resolve to 'ppi' kind."""
        assert ENVOY_KIND_REGISTRY.get_kind("JUNO_PPI") == "ppi"
        assert ENVOY_KIND_REGISTRY.get_kind("VOYAGER1_PPI") == "ppi"
        assert ENVOY_KIND_REGISTRY.get_kind("GALILEO") == "ppi"

    def test_kind_registry_resolves_spice(self):
        """SPICE resolves to 'spice' kind."""
        assert ENVOY_KIND_REGISTRY.get_kind("SPICE") == "spice"

    def test_cdaweb_tools(self):
        """CDAWeb envoys get fetch_data_cdaweb + browse_parameters."""
        names = ENVOY_KIND_REGISTRY.get_tool_names("ACE")
        assert "fetch_data_cdaweb" in names
        assert "browse_parameters" in names
        assert "ask_clarification" in names

    def test_ppi_tools(self):
        """PPI envoys get fetch_data_ppi + browse_parameters."""
        names = ENVOY_KIND_REGISTRY.get_tool_names("JUNO_PPI")
        assert "fetch_data_ppi" in names
        assert "browse_parameters" in names

    def test_cdaweb_and_ppi_have_different_fetch_tools(self):
        """CDAWeb and PPI envoys should have kind-specific fetch tools."""
        cdaweb_names = set(ENVOY_KIND_REGISTRY.get_tool_names("ACE"))
        ppi_names = set(ENVOY_KIND_REGISTRY.get_tool_names("JUNO_PPI"))
        assert "fetch_data_cdaweb" in cdaweb_names
        assert "fetch_data_cdaweb" not in ppi_names
        assert "fetch_data_ppi" in ppi_names
        assert "fetch_data_ppi" not in cdaweb_names

    def test_ppi_missions_registered(self):
        """All expected PPI missions should resolve to 'ppi' kind."""
        from agent.envoy_kinds.registry import MISSION_KINDS
        ppi_missions = [m for m, k in MISSION_KINDS.items() if k == "ppi"]
        assert len(ppi_missions) >= 17, (
            f"Expected >= 17 PPI missions, got {len(ppi_missions)}: {ppi_missions}"
        )

    def test_register_spice_tools_updates_call_registry(self):
        """register_spice_tools should add tools to ORCHESTRATOR_TOOLS and call registry."""
        from agent.agent_registry import register_spice_tools, ORCHESTRATOR_TOOLS

        fake = ["_test_spice_probe"]
        register_spice_tools(fake)

        assert "_test_spice_probe" in ORCHESTRATOR_TOOLS
        assert "_test_spice_probe" in AGENT_CALL_REGISTRY["ctx:orchestrator"]

        # Cleanup
        ORCHESTRATOR_TOOLS.remove("_test_spice_probe")
        from agent.envoy_agent import EnvoyAgent
        EnvoyAgent._PARALLEL_SAFE_TOOLS.discard("_test_spice_probe")
