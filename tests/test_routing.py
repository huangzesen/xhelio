"""
Tests for routing, tool filtering, and the delegate tools.

Tests tool name filtering and LLM-driven routing architecture
without requiring a Gemini API key.

Run with: python -m pytest tests/test_routing.py -v
"""

import pytest
from agent.tools import get_tool_schemas
from agent.agent_registry import (
    ENVOY_TOOLS,
    DATAOPS_TOOLS,
    DATA_IO_TOOLS,
    VIZ_PLOTLY_TOOLS,
    ORCHESTRATOR_TOOLS,
)


class TestToolNameFiltering:
    """Test get_tool_schemas() name filtering."""

    def test_no_filter_returns_all_tools(self):
        all_tools = get_tool_schemas()
        # 28 static tools (after asset decomposition: removed load_file, save_figure, show_figure; added manage_files, manage_figure)
        assert len(all_tools) >= 28
        names = {t["name"] for t in all_tools}
        assert "xhelio__render_plotly_json" in names
        assert "xhelio__manage_plot" in names
        assert "xhelio__assets" in names
        assert "delegate_to_envoy" in names
        assert "delegate_to_viz" in names
        assert "delegate_to_data_ops" in names
        assert "delegate_to_data_io" in names
        assert "xhelio__read_document" in names
        # Removed tools
        assert "plot_data" not in names
        assert "style_plot" not in names
        assert "list_fetched_data" not in names
        assert "manage_session_assets" not in names
        assert "list_assets" not in names

    def test_envoy_tools_empty_when_mcp_only(self):
        """ENVOY_TOOLS is empty when no static envoy kind modules exist."""
        assert len(ENVOY_TOOLS) == 0
        mission_tools = get_tool_schemas(names=ENVOY_TOOLS)
        assert len(mission_tools) == 0

    def test_visualization_tools_only(self):
        viz_tools = get_tool_schemas(names=["xhelio__render_plotly_json", "xhelio__manage_plot"])
        names = {t["name"] for t in viz_tools}
        assert names == {"xhelio__render_plotly_json", "xhelio__manage_plot"}

    def test_visualization_with_extras(self):
        tools = get_tool_schemas(names=VIZ_PLOTLY_TOOLS)
        names = {t["name"] for t in tools}
        assert "xhelio__render_plotly_json" in names
        assert "xhelio__manage_plot" in names
        assert "xhelio__assets" in names
        assert "xhelio__manage_data" in names
        assert "xhelio__review_memory" in names
        assert "xhelio__events" in names

    def test_orchestrator_tools(self):
        orch_tools = get_tool_schemas(names=ORCHESTRATOR_TOOLS)
        names = {t["name"] for t in orch_tools}
        # Should include assets
        assert "xhelio__assets" in names
        # Should include pipeline and events_admin (consolidated)
        assert "xhelio__pipeline" in names
        assert "xhelio__events_admin" in names
        # Should include permission tool (package management moved to sub-agents)
        assert "xhelio__ask_user_permission" in names
        assert "xhelio__manage_sandbox_packages" not in names
        # Should NOT include data_ops (delegated to sub-agents)
        assert "xhelio__run_code" not in names
        # Should NOT include visualization
        assert "plot_data" not in names

    def test_dataops_tools(self):
        dataops_tools = get_tool_schemas(names=DATAOPS_TOOLS)
        names = {t["name"] for t in dataops_tools}
        # Should include compute tools
        assert "xhelio__run_code" in names
        assert "xhelio__manage_data" in names
        # Should include assets
        assert "xhelio__assets" in names
        # Should NOT include fetch (mission-specific)
        assert "fetch_data" not in names
        # Should NOT include store_dataframe (removed)
        assert "store_dataframe" not in names
        # Should NOT include routing or visualization
        assert "delegate_to_envoy" not in names
        assert "plot_data" not in names

    def test_empty_names_returns_nothing(self):
        assert get_tool_schemas(names=[]) == []

    def test_specific_names_returns_only_those(self):
        tools = get_tool_schemas(names=["xhelio__assets"])
        assert len(tools) == 1
        assert tools[0]["name"] == "xhelio__assets"


class TestDelegateToEnvoyTool:
    """Test that the delegate_to_envoy tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_envoy" in names

    def test_tool_not_in_envoy_agent_tools(self):
        names = set(ENVOY_TOOLS)
        assert "delegate_to_envoy" not in names

    def test_tool_requires_envoy_and_request(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_envoy")
        assert "envoy" in tool["parameters"]["properties"]
        assert "request" in tool["parameters"]["properties"]
        assert "envoy" in tool["parameters"]["required"]
        assert "request" in tool["parameters"]["required"]


class TestDelegateToVizTool:
    """Test that the delegate_to_viz tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_viz" in names

    def test_tool_requires_request(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_viz")
        assert "request" in tool["parameters"]["properties"]
        assert "request" in tool["parameters"]["required"]

    def test_tool_has_backend_param(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_viz")
        assert "backend" in tool["parameters"]["properties"]
        assert set(tool["parameters"]["properties"]["backend"]["enum"]) == {
            "plotly",
            "matplotlib",
            "jsx",
        }

    def test_tool_not_in_viz_agent_tools(self):
        names = set(VIZ_PLOTLY_TOOLS)
        assert "delegate_to_viz" not in names


class TestDelegateToDataOpsTool:
    """Test that the delegate_to_data_ops tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_data_ops" in names

    def test_tool_requires_request(self):
        tool = next(
            t for t in get_tool_schemas() if t["name"] == "delegate_to_data_ops"
        )
        assert "request" in tool["parameters"]["properties"]
        assert "request" in tool["parameters"]["required"]

    def test_tool_not_in_envoy_agent_tools(self):
        names = set(ENVOY_TOOLS)
        assert "delegate_to_data_ops" not in names

    def test_tool_not_in_dataops_agent_tools(self):
        names = set(DATAOPS_TOOLS)
        assert "delegate_to_data_ops" not in names


class TestDataIOCategories:
    """Test DataIOAgent tool filtering."""

    def test_data_io_agent_gets_run_code(self):
        tools = get_tool_schemas(names=DATA_IO_TOOLS)
        names = {t["name"] for t in tools}
        assert "xhelio__run_code" in names

    def test_data_io_agent_gets_read_document(self):
        tools = get_tool_schemas(names=DATA_IO_TOOLS)
        names = {t["name"] for t in tools}
        assert "xhelio__read_document" in names

    def test_data_io_agent_gets_assets(self):
        tools = get_tool_schemas(names=DATA_IO_TOOLS)
        names = {t["name"] for t in tools}
        assert "xhelio__assets" in names

    def test_data_io_agent_gets_manage_data(self):
        tools = get_tool_schemas(names=DATA_IO_TOOLS)
        names = {t["name"] for t in tools}
        assert "xhelio__manage_data" in names

    def test_data_io_agent_excludes_fetch(self):
        names = set(DATA_IO_TOOLS)
        assert "fetch_data" not in names

    def test_data_io_agent_excludes_compute(self):
        names = set(DATA_IO_TOOLS)
        assert "custom_operation" not in names
        assert "store_dataframe" not in names
        assert "save_data" not in names

    def test_data_io_agent_excludes_routing(self):
        names = set(DATA_IO_TOOLS)
        assert "delegate_to_envoy" not in names
        assert "delegate_to_viz" not in names

    def test_data_io_agent_excludes_visualization(self):
        names = set(DATA_IO_TOOLS)
        assert "plot_data" not in names
        assert "style_plot" not in names
        assert "xhelio__manage_plot" not in names


class TestDelegateToDataIOTool:
    """Test that the delegate_to_data_io tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_data_io" in names

    def test_tool_requires_request(self):
        tool = next(
            t for t in get_tool_schemas() if t["name"] == "delegate_to_data_io"
        )
        assert "request" in tool["parameters"]["properties"]
        assert "request" in tool["parameters"]["required"]

    def test_tool_has_optional_context(self):
        tool = next(
            t for t in get_tool_schemas() if t["name"] == "delegate_to_data_io"
        )
        assert "context" in tool["parameters"]["properties"]

    def test_tool_not_in_extraction_agent_tools(self):
        names = set(DATA_IO_TOOLS)
        assert "delegate_to_data_io" not in names


class TestDataIOAgentImportAndInterface:
    """Verify DataIOAgent has the correct Agent interface."""

    def test_import(self):
        from agent.data_io_agent import DataIOAgent

        assert DataIOAgent is not None

    def test_send_method_exists_data_io(self):
        from agent.data_io_agent import DataIOAgent

        assert hasattr(DataIOAgent,"send")

    def test_get_token_usage_method_exists(self):
        from agent.data_io_agent import DataIOAgent

        assert hasattr(DataIOAgent,"get_token_usage")
        assert callable(getattr(DataIOAgent,"get_token_usage"))

    def test_start_stop_status_methods_exist(self):
        from agent.data_io_agent import DataIOAgent

        assert hasattr(DataIOAgent,"start")
        assert hasattr(DataIOAgent,"stop")
        assert hasattr(DataIOAgent,"status")


class TestEnvoyAgentImportAndInterface:
    """Verify EnvoyAgent has the correct Agent interface."""

    def test_send_method_exists(self):
        from agent.envoy_agent import EnvoyAgent

        assert hasattr(EnvoyAgent, "send")
        assert callable(getattr(EnvoyAgent, "send"))

    def test_start_stop_status_methods_exist(self):
        from agent.envoy_agent import EnvoyAgent

        assert hasattr(EnvoyAgent, "start")
        assert hasattr(EnvoyAgent, "stop")
        assert hasattr(EnvoyAgent, "status")


class TestDataOpsAgentImportAndInterface:
    """Verify DataOpsAgent has the correct Agent interface."""

    def test_import(self):
        from agent.data_ops_agent import DataOpsAgent

        assert DataOpsAgent is not None

    def test_send_method_exists(self):
        from agent.data_ops_agent import DataOpsAgent

        assert hasattr(DataOpsAgent, "send")
        assert callable(getattr(DataOpsAgent, "send"))

    def test_get_token_usage_method_exists(self):
        from agent.data_ops_agent import DataOpsAgent

        assert hasattr(DataOpsAgent, "get_token_usage")
        assert callable(getattr(DataOpsAgent, "get_token_usage"))

    def test_start_stop_status_methods_exist(self):
        from agent.data_ops_agent import DataOpsAgent

        assert hasattr(DataOpsAgent, "start")
        assert hasattr(DataOpsAgent, "stop")
        assert hasattr(DataOpsAgent, "status")


class TestPlanTool:
    """Test that the plan tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "plan" in names

    def test_tool_requires_action(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "plan")
        assert "action" in tool["parameters"]["properties"]
        assert "action" in tool["parameters"]["required"]

    def test_tool_in_orchestrator_tools(self):
        # plan is a private tool, not in tool_registry.json
        names = {t["name"] for t in get_tool_schemas()}
        assert "plan" in names

    def test_tool_not_in_envoy_agent_tools(self):
        names = set(ENVOY_TOOLS)
        assert "plan" not in names

    def test_tool_not_in_viz_agent_tools(self):
        names = set(VIZ_PLOTLY_TOOLS)
        assert "plan" not in names

    def test_tool_not_in_dataops_agent_tools(self):
        names = set(DATAOPS_TOOLS)
        assert "plan" not in names
