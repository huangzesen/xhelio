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
        assert data["version"] == 2

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

    def test_envoy_call_registry_exists(self):
        """ctx:envoy should exist (may be empty when no kind modules exist)."""
        assert "ctx:envoy" in AGENT_CALL_REGISTRY

    def test_derived_constants_not_empty(self):
        """Derived Python constants are non-empty."""
        assert AGENT_CALL_REGISTRY

    def test_orchestrator_has_permission_tool(self):
        """Orchestrator call list includes permission tool."""
        orch_tools = _REGISTRY["agents"]["orchestrator"]["call"]
        assert "xhelio__ask_user_permission" in orch_tools
        # Package management moved to sub-agents
        assert "xhelio__install_package" not in orch_tools
        assert "xhelio__manage_sandbox_packages" not in orch_tools

    def test_subagents_have_package_management(self):
        """DataOps and DataIO have manage_sandbox_packages."""
        assert "xhelio__manage_sandbox_packages" in _REGISTRY["agents"]["dataops"]["call"]
        assert "xhelio__manage_sandbox_packages" in _REGISTRY["agents"]["data_io"]["call"]
        # Orchestrator sees it as informed
        assert "xhelio__manage_sandbox_packages" in _REGISTRY["agents"]["orchestrator"]["informed"]

    def test_envoy_sections_removed_from_json(self):
        """Old envoy sections should no longer exist in tool_registry.json."""
        assert "envoy_groups" not in _REGISTRY
        assert "envoy_group_assignments" not in _REGISTRY
        assert "envoy_default_group" not in _REGISTRY
        assert "envoy_informed" not in _REGISTRY
        assert "envoy" not in _REGISTRY.get("agents", {})


class TestToolNamespacing:
    """Tests for xhelio__ tool name prefix convention."""

    def test_all_xhelio_tools_have_prefix(self):
        """Every xhelio-native tool name starts with 'xhelio__'."""
        from agent.tool_handlers import TOOL_REGISTRY

        for name in TOOL_REGISTRY:
            # Envoy-namespaced tools use kind:tool_name format
            if ":" in name:
                continue
            assert name.startswith("xhelio__"), (
                f"Tool {name!r} missing xhelio__ prefix"
            )

    def test_tool_schemas_have_prefix(self):
        """Every tool schema in tools.py has xhelio__ prefix."""
        for tool in tools.TOOLS:
            name = tool["name"]
            assert name.startswith("xhelio__"), (
                f"tools.py schema {name!r} missing xhelio__ prefix"
            )

    def test_registry_json_call_tools_have_prefix(self):
        """Every tool in tool_registry.json call lists has xhelio__ prefix."""
        for agent_name, cfg in _REGISTRY["agents"].items():
            for tool_name in cfg["call"]:
                if ":" in tool_name:
                    continue
                assert tool_name.startswith("xhelio__"), (
                    f"agents.{agent_name}.call: {tool_name!r} missing prefix"
                )

    def test_registry_json_informed_tools_have_prefix(self):
        """Every tool in tool_registry.json informed lists has xhelio__ prefix."""
        for agent_name, cfg in _REGISTRY["agents"].items():
            for tool_name in cfg["informed"]:
                if ":" in tool_name:
                    continue
                assert tool_name.startswith("xhelio__"), (
                    f"agents.{agent_name}.informed: {tool_name!r} missing prefix"
                )


class TestToolPermissions:
    """Tests for the tool permission registry."""

    def test_permissions_exist_for_all_agents(self):
        """Every agent should have a permissions entry for 'assets'."""
        for agent_name, cfg in _REGISTRY["agents"].items():
            perms = cfg.get("permissions", {})
            assert "xhelio__assets" in perms, f"{agent_name} missing assets permissions"

    def test_orchestrator_has_all_actions(self):
        perms = _REGISTRY["agents"]["orchestrator"]["permissions"]["xhelio__assets"]
        assert set(perms) == {"list", "status", "restore_plot"}

    def test_viz_agents_list_only(self):
        for agent in ("viz_plotly", "viz_mpl", "viz_jsx"):
            perms = _REGISTRY["agents"][agent]["permissions"]["xhelio__assets"]
            assert perms == ["list"], f"{agent} should only have list"

    def test_dataops_has_list_and_status(self):
        perms = _REGISTRY["agents"]["dataops"]["permissions"]["xhelio__assets"]
        assert set(perms) == {"list", "status"}

    def test_permissions_loaded_into_python(self):
        """AGENT_PERMISSIONS should be populated from JSON."""
        from agent.agent_registry import AGENT_PERMISSIONS
        assert "ctx:orchestrator" in AGENT_PERMISSIONS
        assert AGENT_PERMISSIONS["ctx:viz_plotly"]["xhelio__assets"] == ["list"]

    def test_schema_filtering(self):
        """get_tool_schemas_for_agent should filter enum by permissions."""
        from agent.tools import get_tool_schemas_for_agent
        viz_schemas = get_tool_schemas_for_agent(["xhelio__assets"], "ctx:viz_plotly")
        action_enum = viz_schemas[0]["parameters"]["properties"]["action"]["enum"]
        assert action_enum == ["list"]

        orch_schemas = get_tool_schemas_for_agent(["xhelio__assets"], "ctx:orchestrator")
        action_enum = orch_schemas[0]["parameters"]["properties"]["action"]["enum"]
        assert set(action_enum) == {"list", "status", "restore_plot"}
