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
    PLANNER_TOOLS,
)


class TestToolNameFiltering:
    """Test get_tool_schemas() name filtering."""

    def test_no_filter_returns_all_tools(self):
        all_tools = get_tool_schemas()
        # 39 static tools (after consolidating 11→3) + SPICE tools registered dynamically from MCP
        assert len(all_tools) >= 39
        names = {t["name"] for t in all_tools}
        assert "render_plotly_json" in names
        assert "manage_plot" in names
        assert "fetch_data" in names
        assert "delegate_to_envoy" in names
        assert "delegate_to_viz" in names
        assert "delegate_to_data_ops" in names
        assert "delegate_to_data_io" in names
        assert "delegate_to_insight" in names
        assert "get_dataset_docs" in names
        assert "read_document" in names
        # Removed tools
        assert "plot_data" not in names
        assert "style_plot" not in names

    def test_mission_tools_exclude_visualization_and_routing(self):
        mission_tools = get_tool_schemas(names=ENVOY_TOOLS)
        names = {t["name"] for t in mission_tools}
        # Should not include visualization tools
        assert "plot_data" not in names
        assert "style_plot" not in names
        assert "manage_plot" not in names
        # Should not include routing tools (no recursive delegation)
        assert "delegate_to_envoy" not in names
        assert "delegate_to_viz" not in names
        assert "delegate_to_data_ops" not in names
        # Should include fetch + discovery tools
        assert "fetch_data" in names
        assert "search_datasets" in names
        assert "list_fetched_data" in names
        assert "ask_clarification" in names
        # Should NOT include compute tools (moved to DataOpsAgent)
        assert "custom_operation" not in names
        assert "describe_data" not in names
        assert "save_data" not in names
        # Should NOT include document tools
        assert "read_document" not in names

    def test_visualization_tools_only(self):
        viz_tools = get_tool_schemas(names=["render_plotly_json", "manage_plot"])
        names = {t["name"] for t in viz_tools}
        assert names == {"render_plotly_json", "manage_plot"}

    def test_visualization_with_extras(self):
        tools = get_tool_schemas(names=VIZ_PLOTLY_TOOLS)
        names = {t["name"] for t in tools}
        assert names == {
            "render_plotly_json",
            "manage_plot",
            "list_fetched_data",
            "review_memory",
            "events",
        }

    def test_orchestrator_tools(self):
        orch_tools = get_tool_schemas(names=ORCHESTRATOR_TOOLS)
        names = {t["name"] for t in orch_tools}
        # Should include routing
        assert "delegate_to_envoy" in names
        assert "delegate_to_viz" in names
        assert "delegate_to_data_ops" in names
        assert "delegate_to_data_io" in names
        assert "request_planning" in names
        # Should include list_fetched_data
        assert "list_fetched_data" in names
        # Should include pipeline and events (consolidated)
        assert "pipeline" in names
        assert "events" in names
        # Should NOT include data_ops (delegated to sub-agents)
        assert "fetch_data" not in names
        assert "custom_operation" not in names
        # Should NOT include visualization
        assert "plot_data" not in names

    def test_dataops_tools(self):
        dataops_tools = get_tool_schemas(names=DATAOPS_TOOLS)
        names = {t["name"] for t in dataops_tools}
        # Should include compute tools
        assert "custom_operation" in names
        assert "describe_data" in names
        # Should include list_fetched_data
        assert "list_fetched_data" in names
        # Should include conversation
        assert "ask_clarification" in names
        # Should NOT include fetch (mission-specific)
        assert "fetch_data" not in names
        # Should NOT include save_data (data_export, orchestrator only)
        assert "save_data" not in names
        # Should NOT include store_dataframe (moved to data_extraction)
        assert "store_dataframe" not in names
        # Should NOT include routing or visualization
        assert "delegate_to_envoy" not in names
        assert "plot_data" not in names

    def test_empty_names_returns_nothing(self):
        assert get_tool_schemas(names=[]) == []

    def test_specific_names_returns_only_those(self):
        tools = get_tool_schemas(names=["fetch_data"])
        assert len(tools) == 1
        assert tools[0]["name"] == "fetch_data"


class TestDelegateToEnvoyTool:
    """Test that the delegate_to_envoy tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_envoy" in names

    def test_tool_not_in_envoy_agent_tools(self):
        names = set(ENVOY_TOOLS)
        assert "delegate_to_envoy" not in names

    def test_tool_requires_mission_id_and_request(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_envoy")
        assert "mission_id" in tool["parameters"]["properties"]
        assert "request" in tool["parameters"]["properties"]
        assert "mission_id" in tool["parameters"]["required"]
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

    def test_data_io_agent_gets_store_dataframe(self):
        tools = get_tool_schemas(names=DATA_IO_TOOLS)
        names = {t["name"] for t in tools}
        assert "store_dataframe" in names

    def test_data_io_agent_gets_read_document(self):
        tools = get_tool_schemas(names=DATA_IO_TOOLS)
        names = {t["name"] for t in tools}
        assert "read_document" in names

    def test_data_io_agent_gets_list_fetched_data(self):
        tools = get_tool_schemas(names=DATA_IO_TOOLS)
        names = {t["name"] for t in tools}
        assert "list_fetched_data" in names

    def test_data_io_agent_gets_ask_clarification(self):
        tools = get_tool_schemas(names=DATA_IO_TOOLS)
        names = {t["name"] for t in tools}
        assert "ask_clarification" in names

    def test_data_io_agent_excludes_fetch(self):
        names = set(DATA_IO_TOOLS)
        assert "fetch_data" not in names

    def test_data_io_agent_excludes_compute(self):
        names = set(DATA_IO_TOOLS)
        assert "custom_operation" not in names
        assert "describe_data" not in names
        assert "save_data" not in names

    def test_data_io_agent_excludes_routing(self):
        names = set(DATA_IO_TOOLS)
        assert "delegate_to_envoy" not in names
        assert "delegate_to_viz" not in names

    def test_data_io_agent_excludes_visualization(self):
        names = set(DATA_IO_TOOLS)
        assert "plot_data" not in names
        assert "style_plot" not in names
        assert "manage_plot" not in names


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


class TestRequestPlanningTool:
    """Test that the request_planning tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "request_planning" in names

    def test_tool_requires_request_reasoning_and_time_start_end(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "request_planning")
        assert "request" in tool["parameters"]["properties"]
        assert "reasoning" in tool["parameters"]["properties"]
        assert "time_start" in tool["parameters"]["properties"]
        assert "time_end" in tool["parameters"]["properties"]
        for field in ["request", "reasoning", "time_start", "time_end"]:
            assert field in tool["parameters"]["required"]

    def test_tool_in_orchestrator_tools(self):
        names = set(ORCHESTRATOR_TOOLS)
        assert "request_planning" in names

    def test_tool_not_in_envoy_agent_tools(self):
        names = set(ENVOY_TOOLS)
        assert "request_planning" not in names

    def test_tool_not_in_viz_agent_tools(self):
        names = set(VIZ_PLOTLY_TOOLS)
        assert "request_planning" not in names

    def test_tool_not_in_dataops_agent_tools(self):
        names = set(DATAOPS_TOOLS)
        assert "request_planning" not in names

    def test_tool_not_in_extraction_agent_tools(self):
        names = set(DATA_IO_TOOLS)
        assert "request_planning" not in names
