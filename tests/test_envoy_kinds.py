"""Tests for the envoy kind registry system."""
import pytest
from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY


def test_load_kind_module_uses_knowledge_envoys():
    """Kind modules are loaded from knowledge.envoys, not agent.envoy_kinds."""
    from agent.envoy_kinds.registry import _load_kind_module
    mod = _load_kind_module("cdaweb")
    assert mod.__name__ == "knowledge.envoys.cdaweb"


class TestEnvoyKindRegistry:
    def test_cdaweb_is_default(self):
        assert ENVOY_KIND_REGISTRY.get_kind("ACE") == "cdaweb"
        assert ENVOY_KIND_REGISTRY.get_kind("PSP") == "cdaweb"

    def test_ppi_missions(self):
        assert ENVOY_KIND_REGISTRY.get_kind("JUNO_PPI") == "ppi"
        assert ENVOY_KIND_REGISTRY.get_kind("CASSINI_PPI") == "ppi"

    def test_spice(self):
        assert ENVOY_KIND_REGISTRY.get_kind("SPICE") == "spice"

    def test_cdaweb_tools(self):
        names = ENVOY_KIND_REGISTRY.get_tool_names("ACE")
        assert "browse_parameters" in names
        assert "fetch_data_cdaweb" in names
        assert "fetch_data_ppi" not in names
        assert "list_fetched_data" in names  # global tool

    def test_ppi_tools(self):
        names = ENVOY_KIND_REGISTRY.get_tool_names("JUNO_PPI")
        assert "browse_parameters" in names
        assert "fetch_data_ppi" in names
        assert "fetch_data_cdaweb" not in names

    def test_cdaweb_has_handlers(self):
        handler = ENVOY_KIND_REGISTRY.get_handler("fetch_data_cdaweb", "ACE")
        assert handler is not None

    def test_ppi_has_handlers(self):
        handler = ENVOY_KIND_REGISTRY.get_handler("fetch_data_ppi", "JUNO_PPI")
        assert handler is not None

    def test_browse_parameters_handler_shared(self):
        cdaweb_handler = ENVOY_KIND_REGISTRY.get_handler("browse_parameters", "ACE")
        ppi_handler = ENVOY_KIND_REGISTRY.get_handler("browse_parameters", "JUNO_PPI")
        assert cdaweb_handler is not None
        assert ppi_handler is not None

    def test_global_tool_handler_fallback(self):
        handler = ENVOY_KIND_REGISTRY.get_handler("list_fetched_data", "ACE")
        assert handler is not None

    def test_register_unregister_mission(self):
        ENVOY_KIND_REGISTRY.register_mission("TEST_MISSION", "ppi")
        assert ENVOY_KIND_REGISTRY.get_kind("TEST_MISSION") == "ppi"
        ENVOY_KIND_REGISTRY.unregister_mission("TEST_MISSION")
        assert ENVOY_KIND_REGISTRY.get_kind("TEST_MISSION") == "cdaweb"  # default

    def test_mark_active(self):
        ENVOY_KIND_REGISTRY.clear_active()
        assert ENVOY_KIND_REGISTRY.mark_active("ACE") is True
        assert ENVOY_KIND_REGISTRY.mark_active("ACE") is False  # already active
        ENVOY_KIND_REGISTRY.clear_active()

    def test_function_schemas_return_type(self):
        schemas = ENVOY_KIND_REGISTRY.get_function_schemas("ACE")
        assert isinstance(schemas, list)
        assert len(schemas) > 0
        # Should be FunctionSchema objects
        from agent.llm.base import FunctionSchema
        assert all(isinstance(s, FunctionSchema) for s in schemas)

    def test_function_schemas_include_kind_and_global(self):
        schemas = ENVOY_KIND_REGISTRY.get_function_schemas("ACE")
        names = {s.name for s in schemas}
        assert "fetch_data_cdaweb" in names
        assert "browse_parameters" in names
        assert "list_fetched_data" in names  # global
        assert "ask_clarification" in names  # global

    def test_spice_tools_added_to_all_kinds(self):
        """After add_tools_to_kind for all kinds, all should have the tool."""
        from knowledge.envoys import cdaweb, ppi, spice

        fake_tool = {"name": "_test_spice_all_kinds", "description": "test", "parameters": {"type": "object", "properties": {}}}
        fake_handler = lambda *a: {"status": "ok"}

        for kind in ("cdaweb", "ppi", "spice"):
            ENVOY_KIND_REGISTRY.add_tools_to_kind(kind, [fake_tool], {"_test_spice_all_kinds": fake_handler})

        try:
            assert any(t["name"] == "_test_spice_all_kinds" for t in cdaweb.TOOLS)
            assert any(t["name"] == "_test_spice_all_kinds" for t in ppi.TOOLS)
            assert any(t["name"] == "_test_spice_all_kinds" for t in spice.TOOLS)
        finally:
            # Cleanup
            cdaweb.TOOLS[:] = [t for t in cdaweb.TOOLS if t["name"] != "_test_spice_all_kinds"]
            ppi.TOOLS[:] = [t for t in ppi.TOOLS if t["name"] != "_test_spice_all_kinds"]
            spice.TOOLS[:] = [t for t in spice.TOOLS if t["name"] != "_test_spice_all_kinds"]
            cdaweb.HANDLERS.pop("_test_spice_all_kinds", None)
            ppi.HANDLERS.pop("_test_spice_all_kinds", None)
            spice.HANDLERS.pop("_test_spice_all_kinds", None)

    def test_spice_starts_empty(self):
        names = ENVOY_KIND_REGISTRY.get_tool_names("SPICE")
        # Spice kind has no kind-specific tools initially, only globals
        kind_only = [n for n in names if n not in (
            "ask_clarification", "manage_session_assets", "list_fetched_data",
            "review_memory", "events",
        )]
        assert kind_only == []


class TestDiscoverRuntimeKinds:
    def test_startup_discovers_runtime_kinds(self, tmp_path):
        """Runtime-created kinds are discovered from disk at startup."""
        from agent.envoy_kinds.registry import _discover_runtime_kinds

        # Create a fake runtime kind
        envoys_dir = tmp_path / "envoys"
        kind_dir = envoys_dir / "pfss"
        kind_dir.mkdir(parents=True)
        (envoys_dir / "__init__.py").write_text("")
        (kind_dir / "__init__.py").write_text("TOOLS = []\nHANDLERS = {}\nGLOBAL_TOOLS = []")
        (kind_dir / "pfss.json").write_text('{"id": "PFSS", "name": "PFSS"}')

        result = _discover_runtime_kinds(envoys_dir)
        assert "PFSS" in result
        assert result["PFSS"] == "pfss"

    def test_skips_prebuilt_kinds(self, tmp_path):
        from agent.envoy_kinds.registry import _discover_runtime_kinds

        envoys_dir = tmp_path / "envoys"
        (envoys_dir / "cdaweb").mkdir(parents=True)
        (envoys_dir / "cdaweb" / "__init__.py").write_text("")
        (envoys_dir / "cdaweb" / "cdaweb.json").write_text('{"id": "ACE"}')

        result = _discover_runtime_kinds(envoys_dir)
        assert "ACE" not in result

    def test_skips_dirs_without_init(self, tmp_path):
        from agent.envoy_kinds.registry import _discover_runtime_kinds

        envoys_dir = tmp_path / "envoys"
        (envoys_dir / "noinit").mkdir(parents=True)
        (envoys_dir / "noinit" / "data.json").write_text('{"id": "NOINIT"}')

        result = _discover_runtime_kinds(envoys_dir)
        assert len(result) == 0


class TestPerMissionPermissionGate:
    """Test that the permission gate is per-mission, not flat."""

    def test_cdaweb_envoy_allows_cdaweb_tools(self):
        tools = ENVOY_KIND_REGISTRY.get_tool_names("PSP")
        assert "fetch_data_cdaweb" in tools
        assert "browse_parameters" in tools

    def test_cdaweb_envoy_blocks_ppi_tools(self):
        tools = ENVOY_KIND_REGISTRY.get_tool_names("PSP")
        assert "fetch_data_ppi" not in tools

    def test_ppi_envoy_allows_ppi_tools(self):
        tools = ENVOY_KIND_REGISTRY.get_tool_names("JUNO_PPI")
        assert "fetch_data_ppi" in tools
        assert "browse_parameters" in tools

    def test_ppi_envoy_blocks_cdaweb_tools(self):
        tools = ENVOY_KIND_REGISTRY.get_tool_names("JUNO_PPI")
        assert "fetch_data_cdaweb" not in tools
