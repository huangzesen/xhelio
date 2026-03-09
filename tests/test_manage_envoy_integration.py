"""Integration test: create a runtime envoy and verify it works through the full path."""
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.envoy_kinds.registry import EnvoyKindRegistry, _load_kind_module


@pytest.fixture
def temp_kind(tmp_path):
    """Create a temp kind in knowledge/envoys/ and clean up after."""
    from agent.tool_handlers.manage_envoy import _ENVOYS_DIR
    kind_dir = _ENVOYS_DIR / "_test_integration"
    kind_dir.mkdir(parents=True, exist_ok=True)
    (kind_dir / "__init__.py").write_text(
        "TOOLS = [{'name': 'test_tool', 'description': 'test', 'parameters': {'type': 'object', 'properties': {}}}]\n"
        "HANDLERS = {'test_tool': lambda orch, args: {'status': 'success'}}\n"
        "GLOBAL_TOOLS = ['ask_clarification']\n"
    )
    (kind_dir / "handlers.py").write_text("")
    yield kind_dir
    # Cleanup
    if kind_dir.exists():
        shutil.rmtree(kind_dir)
    sys.modules.pop("knowledge.envoys._test_integration", None)


def test_runtime_envoy_full_cycle(temp_kind):
    """Create → register → get_tool_names → get_handler → unregister cycle."""
    registry = EnvoyKindRegistry()

    # Register
    registry.register_mission("_TEST_INT", "_test_integration")
    assert registry.get_kind("_TEST_INT") == "_test_integration"

    # Tool names
    tool_names = registry.get_tool_names("_TEST_INT")
    assert "test_tool" in tool_names
    assert "ask_clarification" in tool_names

    # Handler
    handler = registry.get_handler("test_tool", "_TEST_INT")
    assert handler is not None
    result = handler(None, {})
    assert result["status"] == "success"

    # Unregister
    registry.unregister_mission("_TEST_INT")
    assert registry.get_kind("_TEST_INT") == "cdaweb"  # falls back to default


def test_load_kind_module_for_runtime_kind(temp_kind):
    """_load_kind_module can load runtime-created kinds."""
    mod = _load_kind_module("_test_integration")
    assert mod.__name__ == "knowledge.envoys._test_integration"
    assert len(mod.TOOLS) == 1
    assert mod.TOOLS[0]["name"] == "test_tool"
