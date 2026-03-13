# tests/test_tool_context.py
"""Tests for ToolContext protocol and ReplayContext."""

import pytest
from agent.tool_context import ToolContext, ReplayContext


def test_tool_context_is_protocol():
    """ToolContext should be a runtime-checkable Protocol."""
    import typing
    assert hasattr(ToolContext, '__protocol_attrs__') or issubclass(type(ToolContext), type(typing.Protocol))


def test_replay_context_satisfies_protocol(tmp_path):
    """ReplayContext must satisfy the ToolContext protocol."""
    from data_ops.store import DataStore
    store = DataStore(tmp_path / "data")
    ctx = ReplayContext(store=store)
    assert isinstance(ctx, ToolContext)


def test_replay_context_properties(tmp_path):
    """ReplayContext exposes correct property values."""
    from data_ops.store import DataStore
    store = DataStore(tmp_path / "data")
    ctx = ReplayContext(store=store)
    assert ctx.store is store
    assert ctx.event_bus is None
    assert ctx.dag is None
    assert ctx.session_dir is None
    assert ctx.renderer is not None
    assert ctx.mcp_client is None
    assert ctx.web_mode is True


def test_replay_context_with_session_dir(tmp_path):
    """ReplayContext accepts an optional session_dir."""
    from data_ops.store import DataStore
    store = DataStore(tmp_path / "data")
    ctx = ReplayContext(store=store, session_dir=tmp_path)
    assert ctx.session_dir == tmp_path


def test_codegen_generates_ctx_signature(tmp_path):
    """Generated handler files should use (ctx, tool_args) not (orch, tool_args)."""
    from agent.envoy_kinds.codegen import generate_kind_files

    manifest = {
        "kind": "test_codegen",
        "envoy_id": "test",
        "source": "test",
        "source_type": "test",
        "tools": [
            {
                "name": "my_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}},
                "handler_code": "def my_tool(**kwargs): return {'status': 'success'}",
            }
        ],
    }
    generate_kind_files(tmp_path, manifest)

    handlers_py = (tmp_path / "handlers.py").read_text()
    assert "def handle_my_tool(ctx, tool_args):" in handlers_py
    assert "def handle_my_tool(orch, tool_args):" not in handlers_py


def test_session_context_has_tool_context_properties():
    """SessionContext must expose all ToolContext properties.

    Handlers receive SessionContext (not OrchestratorAgent), so it must
    satisfy the ToolContext protocol structurally.
    """
    import inspect
    from agent.tool_context import ToolContext
    from agent.session_context import SessionContext

    # Get required property names from the protocol
    required = set()
    for name, obj in inspect.getmembers(ToolContext):
        if not name.startswith('_') and isinstance(inspect.getattr_static(ToolContext, name), property):
            required.add(name)

    # Verify SessionContext dataclass has them as fields
    field_names = {f.name for f in SessionContext.__dataclass_fields__.values()}
    for prop_name in required:
        assert prop_name in field_names, (
            f"SessionContext missing ToolContext property: {prop_name}"
        )


def test_make_mcp_handler_returns_callable():
    """make_mcp_handler should return a (ctx, tool_args) -> dict callable."""
    from agent.tool_handlers import make_mcp_handler
    handler = make_mcp_handler("cdaweb", "fetch_data")
    assert callable(handler)


def test_make_mcp_handler_returns_error_without_client(tmp_path):
    """MCP handler should return error when mcp_client is None."""
    from agent.tool_handlers import make_mcp_handler
    from data_ops.store import DataStore

    handler = make_mcp_handler("cdaweb", "fetch_data")
    ctx = ReplayContext(store=DataStore(tmp_path / "data"))
    result = handler(ctx, {"dataset": "AC_H2_MFI"})
    assert result["status"] == "error"
    assert "MCP client not available" in result["error"]


def test_register_envoy_tools_adds_to_registry():
    """register_envoy_tools should add namespaced handlers to TOOL_REGISTRY."""
    from agent.tool_handlers import TOOL_REGISTRY, register_envoy_tools
    from data_ops.dag import PIPELINE_TOOLS

    # Register fake tools
    tools = [
        {"name": "fetch_data", "pipeline_relevant": True},
        {"name": "list_datasets", "pipeline_relevant": False},
    ]
    register_envoy_tools("test_kind", tools)

    assert "test_kind:fetch_data" in TOOL_REGISTRY
    assert "test_kind:list_datasets" in TOOL_REGISTRY
    assert "test_kind:fetch_data" in PIPELINE_TOOLS
    assert "test_kind:list_datasets" not in PIPELINE_TOOLS

    # Cleanup
    del TOOL_REGISTRY["test_kind:fetch_data"]
    del TOOL_REGISTRY["test_kind:list_datasets"]
    PIPELINE_TOOLS.discard("test_kind:fetch_data")
