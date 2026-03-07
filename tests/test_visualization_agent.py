"""
Tests for the visualization agent tools and helpers.

Run with: python -m pytest tests/test_visualization_agent.py -v
"""

import pytest
from agent.tools import get_tool_schemas
from agent.viz_plotly_agent import _extract_labels_from_instruction
from agent.agent_registry import VIZ_PLOTLY_TOOLS


class TestVizAgentToolFiltering:
    """Test that VizActor gets the right tools."""

    def test_viz_tools_include_render_plotly_json(self):
        tools = get_tool_schemas(names=VIZ_PLOTLY_TOOLS)
        names = {t["name"] for t in tools}
        assert "render_plotly_json" in names

    def test_viz_tools_include_manage_plot(self):
        tools = get_tool_schemas(names=VIZ_PLOTLY_TOOLS)
        names = {t["name"] for t in tools}
        assert "manage_plot" in names

    def test_viz_tools_include_list_fetched_data(self):
        tools = get_tool_schemas(names=VIZ_PLOTLY_TOOLS)
        names = {t["name"] for t in tools}
        assert "list_fetched_data" in names

    def test_viz_tools_include_data_inspection(self):
        """Viz tools should include describe_data and preview_data for data inspection."""
        tools = get_tool_schemas(names=VIZ_PLOTLY_TOOLS)
        names = {t["name"] for t in tools}
        assert "describe_data" in names
        assert "preview_data" in names

    def test_viz_tools_count(self):
        tools = get_tool_schemas(names=VIZ_PLOTLY_TOOLS)
        # render_plotly_json + manage_plot + list_fetched_data + describe_data + preview_data + review_memory + events
        assert len(tools) == 7

    def test_viz_tools_exclude_routing(self):
        tools = get_tool_schemas(names=VIZ_PLOTLY_TOOLS)
        names = {t["name"] for t in tools}
        assert "delegate_to_envoy" not in names
        assert "delegate_to_viz" not in names

    def test_viz_tools_exclude_discovery(self):
        tools = get_tool_schemas(names=VIZ_PLOTLY_TOOLS)
        names = {t["name"] for t in tools}
        assert "search_datasets" not in names
        assert "list_parameters" not in names


class TestExtractLabels:
    """Test _extract_labels_from_instruction helper."""

    def test_extracts_labels_from_store_format(self):
        instruction = (
            "Plot ACE and Wind magnetic field\n\n"
            "Data currently in memory:\n"
            "  - AC_H0_MFI.Magnitude (37800 pts)\n"
            "  - WI_H0_MFI@0.BF1 (10080 pts)"
        )
        labels = _extract_labels_from_instruction(instruction)
        assert labels == ["AC_H0_MFI.Magnitude", "WI_H0_MFI@0.BF1"]

    def test_extracts_multiple_labels(self):
        instruction = (
            "Task instruction\n\n"
            "Data currently in memory:\n"
            "  - A (100 pts)\n"
            "  - B (200 pts)\n"
            "  - C (300 pts)\n"
            "  - D (400 pts)"
        )
        labels = _extract_labels_from_instruction(instruction)
        assert labels == ["A", "B", "C", "D"]

    def test_returns_empty_for_no_labels(self):
        instruction = "Just plot something"
        labels = _extract_labels_from_instruction(instruction)
        assert labels == []

    def test_handles_dots_and_at_signs(self):
        instruction = "  - WI_H0_MFI@0.BGSE (10080 pts)"
        labels = _extract_labels_from_instruction(instruction)
        assert labels == ["WI_H0_MFI@0.BGSE"]
