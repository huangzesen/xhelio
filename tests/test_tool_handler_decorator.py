"""Tests for the @tool_handler decorator."""
import pytest
from agent.tool_handlers.decorator import _HANDLERS, tool_handler, get_all_handlers


def test_decorator_registers_handler():
    saved = dict(_HANDLERS)
    try:
        _HANDLERS.clear()

        @tool_handler("test_xyz_tool")
        def handle_test(orch, args):
            return {"result": "ok"}

        handlers = get_all_handlers()
        assert "test_xyz_tool" in handlers
        assert handlers["test_xyz_tool"] is handle_test
    finally:
        _HANDLERS.clear()
        _HANDLERS.update(saved)


def test_duplicate_raises():
    saved = dict(_HANDLERS)
    try:
        _HANDLERS.clear()

        @tool_handler("dup_tool")
        def handle_first(orch, args):
            return {}

        with pytest.raises(ValueError, match="Duplicate"):
            @tool_handler("dup_tool")
            def handle_second(orch, args):
                return {}
    finally:
        _HANDLERS.clear()
        _HANDLERS.update(saved)


def test_migrated_handler_in_registry():
    """At least one handler should be registered via decorator."""
    from agent.tool_handlers import TOOL_REGISTRY
    # After migration, the decorated handler should be in TOOL_REGISTRY
    assert len(TOOL_REGISTRY) > 0
    # Specifically, the migrated handler should be present
    assert "ask_user_permission" in TOOL_REGISTRY
