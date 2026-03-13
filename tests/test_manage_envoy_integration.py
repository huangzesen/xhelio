"""Integration test: create a runtime envoy and verify it works through the full path."""
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.envoy_kinds.registry import EnvoyKindRegistry, _load_kind_module


@pytest.fixture
def temp_kind(tmp_path):
    """Create a temp kind in knowledge/envoys/ using codegen and clean up after."""
    from agent.tool_handlers.manage_envoy import _ENVOYS_DIR, _DEFAULT_GLOBAL_TOOLS
    from agent.envoy_kinds.codegen import generate_kind_files

    kind_dir = _ENVOYS_DIR / "_test_integration"
    kind_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "kind": "_test_integration",
        "envoy_id": "_TEST_INT",
        "source": "test",
        "source_type": "package",
        "global_tools": ["ask_clarification"],
        "tools": [
            {
                "name": "test_tool",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
                "handler_code": (
                    "def test_tool(**kwargs):\n"
                    "    return {'status': 'success'}\n"
                ),
            }
        ],
    }
    (kind_dir / "kind.json").write_text(json.dumps(manifest, indent=2))
    generate_kind_files(kind_dir, manifest)

    yield kind_dir
    # Cleanup
    if kind_dir.exists():
        shutil.rmtree(kind_dir)
    sys.modules.pop("knowledge.envoys._test_integration", None)
    sys.modules.pop("knowledge.envoys._test_integration.handlers", None)


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
