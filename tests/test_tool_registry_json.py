"""Tests for tool_registry.json validation and regression."""

import json
from pathlib import Path

import pytest

from agent import tools
from agent.agent_registry import (
    AGENT_CALL_REGISTRY,
    ENVOY_BASE_TOOLS,
    ENVOY_TOOL_REGISTRY,
    _REGISTRY,
)


def get_tool_names() -> set[str]:
    """Get all tool names from tools.py TOOLS list."""
    return {t["name"] for t in tools.TOOLS}


class TestToolRegistryJson:
    """Validation tests for tool_registry.json."""

    def test_json_loads(self):
        """JSON file loads successfully."""
        assert _REGISTRY is not None
        assert _REGISTRY["version"] == 1

    def test_version_is_one(self):
        """Registry version must be 1."""
        assert _REGISTRY["version"] == 1

    def test_agents_have_call_and_informed(self):
        """Every agent has 'call' and 'informed' lists, optional 'think' list."""
        for name, cfg in _REGISTRY["agents"].items():
            assert "call" in cfg, f"Agent '{name}' missing 'call'"
            assert "informed" in cfg, f"Agent '{name}' missing 'informed'"
            assert isinstance(cfg["call"], list), f"Agent '{name}' 'call' not a list"
            assert isinstance(cfg["informed"], list), (
                f"Agent '{name}' 'informed' not a list"
            )
            if "think" in cfg:
                assert isinstance(cfg["think"], list), (
                    f"Agent '{name}' 'think' not a list"
                )

    def test_tools_exist_in_tools_py(self):
        """Every tool name in JSON exists in tools.py TOOLS list."""
        tool_names = get_tool_names()
        missing = []

        for agent_name, cfg in _REGISTRY["agents"].items():
            for tool in cfg.get("call", []):
                if tool not in tool_names:
                    missing.append(f"agent '{agent_name}' call: {tool}")
            for tool in cfg.get("think", []):
                if tool not in tool_names:
                    missing.append(f"agent '{agent_name}' think: {tool}")
            for tool in cfg.get("informed", []):
                if tool not in tool_names:
                    missing.append(f"agent '{agent_name}' informed: {tool}")

        for group, tools_list in _REGISTRY["envoy_groups"].items():
            for tool in tools_list:
                if tool not in tool_names:
                    missing.append(f"envoy_groups '{group}': {tool}")

        assert not missing, f"Tools in JSON not found in tools.py: {missing}"

    def test_no_call_and_informed_overlap(self):
        """No tool appears in both call and informed for the same agent."""
        overlaps = []
        for agent_name, cfg in _REGISTRY["agents"].items():
            call_set = set(cfg.get("call", []))
            informed_set = set(cfg.get("informed", []))
            overlap = call_set & informed_set
            if overlap:
                overlaps.append(f"agent '{agent_name}': {overlap}")
        assert not overlaps, f"Tools in both call and informed: {overlaps}"

    def test_envoy_groups_has_base(self):
        """envoy_groups must have a 'base' group."""
        assert "base" in _REGISTRY["envoy_groups"], "envoy_groups missing 'base'"

    def test_envoy_group_assignments_valid(self):
        """All assigned groups in envoy_group_assignments exist."""
        groups = set(_REGISTRY["envoy_groups"].keys())
        for mission, group in _REGISTRY.get("envoy_group_assignments", {}).items():
            assert group in groups, (
                f"Mission '{mission}' assigned to non-existent group '{group}'"
            )

    def test_envoy_default_group_exists(self):
        """Default group exists in envoy_groups."""
        default = _REGISTRY.get("envoy_default_group")
        assert default in _REGISTRY["envoy_groups"], (
            f"Default group '{default}' not found"
        )

    def test_derived_constants_not_empty(self):
        """Derived Python constants are non-empty."""
        assert AGENT_CALL_REGISTRY
        assert ENVOY_BASE_TOOLS

    def test_mission_tool_registry_loads_groups(self):
        """MissionToolRegistry loads groups from JSON."""
        groups = ENVOY_TOOL_REGISTRY._group_tools
        assert "cdaweb" in groups
        assert "base" not in groups  # base is separate

    def test_mission_tool_registry_get_group(self):
        """MissionToolRegistry.get_group returns correct group."""
        assert ENVOY_TOOL_REGISTRY.get_group("ACE") == "cdaweb"
        assert ENVOY_TOOL_REGISTRY.get_group("WIND") == "cdaweb"

    def test_mission_tool_registry_get_tools(self):
        """MissionToolRegistry.get_tools returns base + group tools."""
        ace_tools = ENVOY_TOOL_REGISTRY.get_tools("ACE")
        assert "ask_clarification" in ace_tools
        assert "fetch_data" in ace_tools
