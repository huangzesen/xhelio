"""
Tests for the visualization tool registry.

Run with: python -m pytest tests/test_registry.py -v
"""

import pytest
from rendering.registry import TOOLS, _TOOL_MAP


class TestRegistryStructure:
    """Verify that all registry entries have required fields."""

    def test_all_tools_have_name(self):
        for t in TOOLS:
            assert "name" in t, f"Tool missing 'name': {t}"

    def test_all_tools_have_description(self):
        for t in TOOLS:
            assert "description" in t, f"Tool '{t.get('name', '?')}' missing 'description'"

    def test_all_tools_have_parameters(self):
        for t in TOOLS:
            assert "parameters" in t, f"Tool '{t['name']}' missing 'parameters'"
            assert isinstance(t["parameters"], list)

    def test_no_duplicate_names(self):
        names = [t["name"] for t in TOOLS]
        assert len(names) == len(set(names)), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"

    def test_tool_count(self):
        assert len(TOOLS) == 2  # render_plotly_json, manage_plot

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {"render_plotly_json", "manage_plot"}

    def test_parameters_have_required_fields(self):
        for t in TOOLS:
            for p in t["parameters"]:
                assert "name" in p, f"{t['name']}: param missing 'name'"
                assert "type" in p, f"{t['name']}.{p.get('name', '?')}: missing 'type'"
                assert "required" in p, f"{t['name']}.{p['name']}: missing 'required'"
                assert "description" in p, f"{t['name']}.{p['name']}: missing 'description'"

    def test_render_plotly_json_has_figure_json_param(self):
        t = _TOOL_MAP["render_plotly_json"]
        assert t is not None
        param_names = [p["name"] for p in t["parameters"]]
        assert "figure_json" in param_names
        param = next(p for p in t["parameters"] if p["name"] == "figure_json")
        assert param["required"] is True

    def test_manage_plot_has_action_param(self):
        t = _TOOL_MAP["manage_plot"]
        assert t is not None
        param_names = [p["name"] for p in t["parameters"]]
        assert "action" in param_names
        action_param = next(p for p in t["parameters"] if p["name"] == "action")
        assert action_param["required"] is True
        assert "enum" in action_param

    def test_manage_plot_has_format_param(self):
        t = _TOOL_MAP["manage_plot"]
        assert t is not None
        param_names = [p["name"] for p in t["parameters"]]
        assert "format" in param_names
        fmt_param = next(p for p in t["parameters"] if p["name"] == "format")
        assert fmt_param["required"] is False
        assert fmt_param["default"] == "png"
        assert set(fmt_param["enum"]) == {"png", "pdf"}


class TestToolMap:
    def test_known_tool(self):
        t = _TOOL_MAP.get("render_plotly_json")
        assert t is not None
        assert t["name"] == "render_plotly_json"

    def test_unknown_tool(self):
        assert _TOOL_MAP.get("nonexistent") is None

    def test_removed_tools_not_found(self):
        """Removed legacy tools should not exist in the registry."""
        assert _TOOL_MAP.get("plot_data") is None
        assert _TOOL_MAP.get("style_plot") is None

    def test_all_tools_retrievable(self):
        for t in TOOLS:
            assert _TOOL_MAP.get(t["name"]) is t

    def test_old_methods_not_found(self):
        """Old method names should no longer exist in the registry."""
        for name in ("plot_stored_data", "set_time_range", "export",
                      "get_plot_state", "plot_spectrogram", "reset",
                      "execute_visualization", "custom_visualization"):
            assert _TOOL_MAP.get(name) is None, f"{name} should have been removed"
