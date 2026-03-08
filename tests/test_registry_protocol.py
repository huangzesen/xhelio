"""Tests for the Registry protocol and meta-registry."""

from typing import Any

import pytest

from agent.registry_protocol import (
    Registry,
    _META_REGISTRY,
    _reset_for_testing,
    get_registry,
    list_registries,
    register_registry,
)


class FakeRegistry:
    """A minimal registry that satisfies the Registry protocol."""

    def __init__(self, name: str, description: str, entries: dict[str, Any] | None = None):
        self.name = name
        self.description = description
        self._entries = entries or {}

    def get(self, key: str) -> Any:
        return self._entries.get(key)

    def list_all(self) -> dict[str, Any]:
        return dict(self._entries)


def test_fake_registry_satisfies_protocol():
    """FakeRegistry is recognized as a structural subtype of Registry."""
    fake = FakeRegistry(name="test", description="A test registry")
    assert isinstance(fake, Registry)


def test_register_and_list_registries():
    """Registering a fake registry makes it appear in list_registries()."""
    saved = _reset_for_testing()
    try:
        fake = FakeRegistry(name="catalog", description="Dataset catalog")
        register_registry(fake)
        result = list_registries()
        assert "catalog" in result
        assert result["catalog"] is fake
    finally:
        _META_REGISTRY.clear()
        _META_REGISTRY.update(saved)


def test_get_registry():
    """get_registry returns the registry by name, or None if missing."""
    saved = _reset_for_testing()
    try:
        fake = FakeRegistry(name="tools", description="Tool registry")
        register_registry(fake)
        assert get_registry("tools") is fake
        assert get_registry("nonexistent") is None
    finally:
        _META_REGISTRY.clear()
        _META_REGISTRY.update(saved)


def test_duplicate_registration_raises():
    """Registering two *different* registries with the same name raises ValueError."""
    saved = _reset_for_testing()
    try:
        fake1 = FakeRegistry(name="dup", description="First")
        fake2 = FakeRegistry(name="dup", description="Second")
        register_registry(fake1)
        with pytest.raises(ValueError):
            register_registry(fake2)
    finally:
        _META_REGISTRY.clear()
        _META_REGISTRY.update(saved)


def test_idempotent_registration():
    """Re-registering the same object is a silent no-op (module reload safety)."""
    saved = _reset_for_testing()
    try:
        fake = FakeRegistry(name="idem", description="Idempotent test")
        register_registry(fake)
        register_registry(fake)  # should not raise
        assert get_registry("idem") is fake
    finally:
        _META_REGISTRY.clear()
        _META_REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# Task 2: Truncation and turn_limits as Registry implementations
# ---------------------------------------------------------------------------


def test_truncation_satisfies_protocol():
    from agent.truncation import TEXT_REGISTRY, ITEM_REGISTRY
    assert isinstance(TEXT_REGISTRY, Registry)
    assert TEXT_REGISTRY.name == "truncation.text"
    val = TEXT_REGISTRY.get("console.summary")
    assert isinstance(val, int)
    all_entries = TEXT_REGISTRY.list_all()
    assert "console.summary" in all_entries


def test_truncation_item_registry_satisfies_protocol():
    from agent.truncation import ITEM_REGISTRY
    assert isinstance(ITEM_REGISTRY, Registry)
    assert ITEM_REGISTRY.name == "truncation.items"
    val = ITEM_REGISTRY.get("items.tool_args")
    assert isinstance(val, int)
    all_entries = ITEM_REGISTRY.list_all()
    assert "items.tool_args" in all_entries


def test_turn_limits_satisfies_protocol():
    from agent.turn_limits import TURN_LIMITS_REGISTRY
    assert isinstance(TURN_LIMITS_REGISTRY, Registry)
    assert TURN_LIMITS_REGISTRY.name == "turn_limits"
    val = TURN_LIMITS_REGISTRY.get("orchestrator.max_iterations")
    assert isinstance(val, int)
    all_entries = TURN_LIMITS_REGISTRY.list_all()
    assert "orchestrator.max_iterations" in all_entries


def test_truncation_registered_in_meta():
    all_regs = list_registries()
    assert "truncation.text" in all_regs
    assert "truncation.items" in all_regs


def test_turn_limits_registered_in_meta():
    all_regs = list_registries()
    assert "turn_limits" in all_regs


# ---------------------------------------------------------------------------
# Task 3: Fallback registry as Registry implementation
# ---------------------------------------------------------------------------


def test_fallback_registry_satisfies_protocol():
    from agent.fallback_registry import FALLBACK_REGISTRY
    assert isinstance(FALLBACK_REGISTRY, Registry)
    assert FALLBACK_REGISTRY.name == "fallbacks"
    all_entries = FALLBACK_REGISTRY.list_all()
    assert len(all_entries) > 0


# ---------------------------------------------------------------------------
# Task 4: Tool handlers, tool schemas, event tags as Registry implementations
# ---------------------------------------------------------------------------


def test_tool_handler_registry_satisfies_protocol():
    from agent.tool_handlers import TOOL_HANDLER_REGISTRY
    assert isinstance(TOOL_HANDLER_REGISTRY, Registry)
    assert TOOL_HANDLER_REGISTRY.name == "tools.handlers"
    all_entries = TOOL_HANDLER_REGISTRY.list_all()
    assert len(all_entries) > 0


def test_tool_schema_registry_satisfies_protocol():
    from agent.tools import TOOL_SCHEMA_REGISTRY
    assert isinstance(TOOL_SCHEMA_REGISTRY, Registry)
    assert TOOL_SCHEMA_REGISTRY.name == "tools.schemas"
    all_entries = TOOL_SCHEMA_REGISTRY.list_all()
    assert len(all_entries) > 0


def test_event_tags_registry_satisfies_protocol():
    from agent.event_bus import EVENT_TAGS_REGISTRY
    assert isinstance(EVENT_TAGS_REGISTRY, Registry)
    assert EVENT_TAGS_REGISTRY.name == "events.infrastructure_tags"
    all_entries = EVENT_TAGS_REGISTRY.list_all()
    assert len(all_entries) > 0


# ---------------------------------------------------------------------------
# Task 5: Formatters, observations, sandbox, rendering, providers
# ---------------------------------------------------------------------------


def test_event_formatters_registry():
    from agent.event_formatters import EVENT_FORMATTER_REGISTRY
    assert isinstance(EVENT_FORMATTER_REGISTRY, Registry)
    assert EVENT_FORMATTER_REGISTRY.name == "events.formatters"


def test_observations_registry():
    from agent.observations import OBSERVATION_REGISTRY
    assert isinstance(OBSERVATION_REGISTRY, Registry)
    assert OBSERVATION_REGISTRY.name == "observations"


def test_sandbox_registry_protocol():
    from data_ops.custom_ops import SANDBOX_PROTOCOL_REGISTRY
    assert isinstance(SANDBOX_PROTOCOL_REGISTRY, Registry)
    assert SANDBOX_PROTOCOL_REGISTRY.name == "sandbox.packages"


def test_rendering_registry_protocol():
    from rendering.registry import RENDERING_REGISTRY
    assert isinstance(RENDERING_REGISTRY, Registry)
    assert RENDERING_REGISTRY.name == "rendering.tools"


def test_provider_registry_protocol():
    from config import PROVIDER_REGISTRY
    assert isinstance(PROVIDER_REGISTRY, Registry)
    assert PROVIDER_REGISTRY.name == "llm.providers"


# ---------------------------------------------------------------------------
# Task 6: Agent call registry + full discoverability
# ---------------------------------------------------------------------------


def test_agent_call_registry_protocol():
    from agent.agent_registry import AGENT_CALL_PROTOCOL_REGISTRY
    assert isinstance(AGENT_CALL_PROTOCOL_REGISTRY, Registry)
    assert AGENT_CALL_PROTOCOL_REGISTRY.name == "agents.tool_access"
    all_entries = AGENT_CALL_PROTOCOL_REGISTRY.list_all()
    assert len(all_entries) > 0


def test_all_registries_discoverable():
    """All registries should be discoverable via list_registries()."""
    # Force imports
    import agent.truncation        # noqa: F401
    import agent.turn_limits       # noqa: F401
    import agent.fallback_registry # noqa: F401
    import agent.tool_handlers     # noqa: F401
    import agent.tools             # noqa: F401
    import agent.event_bus         # noqa: F401
    import agent.event_formatters  # noqa: F401
    import agent.observations      # noqa: F401
    import agent.agent_registry    # noqa: F401
    import rendering.registry      # noqa: F401
    import config                  # noqa: F401

    all_regs = list_registries()
    expected = {
        "truncation.text", "truncation.items", "turn_limits", "fallbacks",
        "tools.handlers", "tools.schemas", "events.infrastructure_tags",
        "events.formatters", "observations", "sandbox.packages",
        "rendering.tools", "llm.providers", "agents.tool_access",
    }
    assert expected.issubset(set(all_regs.keys())), (
        f"Missing: {expected - set(all_regs.keys())}"
    )
