"""
Tests for InformedRegistry — mutable, thread-safe, persistable tool log registry.

Run with: python -m pytest tests/test_informed_registry.py -v
"""

import json
import threading

import pytest

from agent.agent_registry import (
    InformedRegistry,
    AGENT_CALL_REGISTRY,
    AGENT_INFORMED_REGISTRY,
    ORCHESTRATOR_TOOLS,
    ORCHESTRATOR_INFORMED_TOOLS,
    ENVOY_TOOLS,
    ENVOY_INFORMED_TOOLS,
    VIZ_PLOTLY_TOOLS,
    VIZ_PLOTLY_INFORMED_TOOLS,
    DATAOPS_TOOLS,
    DATAOPS_INFORMED_TOOLS,
    PLANNER_TOOLS,
    PLANNER_INFORMED_TOOLS,
    _INFORMED_DEFAULTS,
)


class TestInformedRegistryDefaults:
    """Verify initial state matches the old frozen dict behavior."""

    def test_default_state_matches_static(self):
        """A fresh InformedRegistry should match the old frozen AGENT_INFORMED_REGISTRY."""
        reg = InformedRegistry()
        for ctx, call_tools, informed in _INFORMED_DEFAULTS:
            expected = call_tools | frozenset(informed)
            actual = reg.get(ctx)
            assert actual == expected, f"{ctx}: expected {expected}, got {actual}"

    def test_has_expected_keys(self):
        expected_keys = {
            "ctx:orchestrator",
            "ctx:envoy",
            "ctx:viz_plotly",
            "ctx:viz_mpl",
            "ctx:viz_jsx",
            "ctx:dataops",
            "ctx:planner",
            "ctx:extraction",
            "ctx:eureka",
        }
        reg = InformedRegistry()
        assert set(reg.keys()) == expected_keys

    def test_items_returns_all_pairs(self):
        reg = InformedRegistry()
        items = reg.items()
        assert len(items) == 9
        for ctx, tools in items:
            assert isinstance(tools, frozenset)
            assert len(tools) > 0


class TestInformedRegistryAdd:
    def test_add_tool(self):
        reg = InformedRegistry()
        # web_search is not in viz's informed set by default
        assert "web_search" not in reg.get("ctx:viz_plotly")
        added = reg.add("ctx:viz_plotly", "web_search", "want to see search results")
        assert added is True
        assert "web_search" in reg.get("ctx:viz_plotly")

    def test_add_already_present(self):
        reg = InformedRegistry()
        # fetch_data should already be in viz (it's an informed tool)
        assert "fetch_data" in reg.get("ctx:viz_plotly")
        added = reg.add("ctx:viz_plotly", "fetch_data", "redundant add")
        assert added is False

    def test_add_to_nonexistent_ctx(self):
        reg = InformedRegistry()
        added = reg.add("ctx:nonexistent", "fetch_data", "test")
        assert added is False


class TestInformedRegistryDrop:
    def test_drop_informed_tool(self):
        reg = InformedRegistry()
        # custom_operation is an informed-only tool for viz
        assert "custom_operation" in reg.get("ctx:viz_plotly")
        ok, err = reg.drop("ctx:viz_plotly", "custom_operation", "not useful")
        assert ok is True
        assert err == ""
        assert "custom_operation" not in reg.get("ctx:viz_plotly")

    def test_cannot_drop_call_tool(self):
        reg = InformedRegistry()
        # render_plotly_json is a call tool for viz
        ok, err = reg.drop("ctx:viz_plotly", "render_plotly_json", "want to drop")
        assert ok is False
        assert "callable tool" in err

    def test_drop_nonexistent_tool(self):
        reg = InformedRegistry()
        ok, err = reg.drop("ctx:viz_plotly", "totally_fake_tool", "test")
        assert ok is False
        assert "not in informed set" in err

    def test_drop_from_nonexistent_ctx(self):
        reg = InformedRegistry()
        ok, err = reg.drop("ctx:nonexistent", "fetch_data", "test")
        assert ok is False


class TestInformedRegistryChangelog:
    def test_changelog_recorded_on_add(self):
        reg = InformedRegistry()
        reg.add("ctx:viz_plotly", "web_search", "test reason")
        assert len(reg._changelog) == 1
        entry = reg._changelog[0]
        assert entry["action"] == "add"
        assert entry["ctx"] == "ctx:viz_plotly"
        assert entry["tool"] == "web_search"
        assert entry["reasoning"] == "test reason"
        assert "ts" in entry

    def test_changelog_recorded_on_drop(self):
        reg = InformedRegistry()
        reg.drop("ctx:viz_plotly", "custom_operation", "test reason")
        assert len(reg._changelog) == 1
        entry = reg._changelog[0]
        assert entry["action"] == "drop"


class TestInformedRegistrySaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        reg1 = InformedRegistry()
        reg1.add("ctx:viz_plotly", "web_search", "search logs are useful")
        reg1.drop("ctx:viz_plotly", "custom_operation", "too noisy")

        path = tmp_path / "informed_tools.json"
        reg1.save(path)

        # Verify file exists and is valid JSON
        assert path.exists()
        data = json.loads(path.read_text())
        assert "overrides" in data
        assert "changelog" in data
        assert len(data["changelog"]) == 2

        # Load into a fresh registry
        reg2 = InformedRegistry()
        reg2.load(path)

        # web_search should be present (added as override)
        assert "web_search" in reg2.get("ctx:viz_plotly")
        # custom_operation was dropped but load is additive on defaults,
        # so it reappears (load only adds overrides, doesn't remove)
        # This is by design — drop is session-local
        assert "custom_operation" in reg2.get("ctx:viz_plotly")

    def test_save_only_overrides(self, tmp_path):
        """save() should only persist tools beyond the call tools (overrides only)."""
        reg = InformedRegistry()
        reg.add("ctx:envoy", "web_search", "test")

        path = tmp_path / "informed_tools.json"
        reg.save(path)

        data = json.loads(path.read_text())
        overrides = data["overrides"]
        # Mission overrides should include the newly added tool
        mission_overrides = set(overrides.get("ctx:envoy", []))
        # ENVOY_INFORMED_TOOLS is empty, so only web_search is an override
        assert "web_search" in mission_overrides
        # Call tools should NOT be in overrides
        assert "fetch_data" not in mission_overrides

    def test_load_nonexistent_file_is_noop(self, tmp_path):
        reg = InformedRegistry()
        reg.load(tmp_path / "nonexistent.json")
        # Should still have defaults
        assert len(reg.get("ctx:orchestrator")) > 0

    def test_load_invalid_json_is_noop(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json!!!")
        reg = InformedRegistry()
        reg.load(path)
        assert len(reg.get("ctx:orchestrator")) > 0


class TestInformedRegistryThreadSafety:
    def test_concurrent_add_drop(self):
        reg = InformedRegistry()
        errors = []

        def adder():
            try:
                for i in range(50):
                    reg.add("ctx:orchestrator", f"fake_tool_{i}", f"reason_{i}")
            except Exception as e:
                errors.append(e)

        def dropper():
            try:
                for i in range(50):
                    reg.drop("ctx:orchestrator", f"fake_tool_{i}", f"reason_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=adder),
            threading.Thread(target=dropper),
            threading.Thread(target=adder),
            threading.Thread(target=dropper),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_items_and_add(self):
        reg = InformedRegistry()
        errors = []

        def reader():
            try:
                for _ in range(100):
                    items = reg.items()
                    assert len(items) == 9
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100):
                    reg.add("ctx:dataops", f"test_tool_{i}", "concurrent test")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestModuleSingleton:
    """Verify the module-level AGENT_INFORMED_REGISTRY singleton."""

    def test_singleton_is_informed_registry(self):
        assert isinstance(AGENT_INFORMED_REGISTRY, InformedRegistry)

    def test_singleton_has_expected_keys(self):
        expected_keys = {
            "ctx:orchestrator",
            "ctx:envoy",
            "ctx:viz_plotly",
            "ctx:viz_mpl",
            "ctx:viz_jsx",
            "ctx:dataops",
            "ctx:planner",
            "ctx:extraction",
            "ctx:eureka",
        }
        assert set(AGENT_INFORMED_REGISTRY.keys()) == expected_keys

    def test_singleton_informed_is_superset_of_call(self):
        """AGENT_INFORMED_REGISTRY should always be a superset of AGENT_CALL_REGISTRY."""
        for ctx in AGENT_CALL_REGISTRY:
            call_tools = AGENT_CALL_REGISTRY[ctx]
            informed_tools = AGENT_INFORMED_REGISTRY.get(ctx)
            assert call_tools <= informed_tools, (
                f"{ctx}: call tools not subset of informed tools. "
                f"Missing: {call_tools - informed_tools}"
            )
