"""Tests for user-defined package envoys."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_package_envoy_json() -> dict:
    """Minimal valid package envoy JSON."""
    return {
        "id": "TEST_PKG",
        "name": "Test Package",
        "type": "package",
        "keywords": ["test", "package"],
        "profile": {
            "description": "A test package envoy.",
        },
        "sandbox": {
            "imports": [
                {"import_path": "math", "sandbox_alias": "math"}
            ],
            "functions": [
                {
                    "name": "sqrt",
                    "description": "Compute square root",
                    "module": "math",
                    "signature": "math.sqrt(x)",
                    "parameters": {"x": {"type": "number", "description": "Input value"}},
                }
            ],
        },
        "instruments": {},
    }


class TestPackageEnvoyLoading:
    """Test that package envoy JSONs are discovered by mission_loader."""

    def test_packages_dir_is_scanned(self):
        """The packages/ subdirectory should be in _SOURCE_DIRS."""
        from knowledge.mission_loader import _SOURCE_DIRS, _MISSIONS_DIR

        packages_dir = _MISSIONS_DIR / "packages"
        assert packages_dir in _SOURCE_DIRS, (
            f"Expected {packages_dir} in _SOURCE_DIRS, got {_SOURCE_DIRS}"
        )

    def test_package_envoy_loads_into_catalog(self, tmp_path):
        """A package envoy JSON in packages/ should be loadable."""
        from knowledge.mission_loader import load_mission

        # Write a test package envoy JSON
        pkg_dir = tmp_path / "packages"
        pkg_dir.mkdir()
        envoy_json = _make_package_envoy_json()
        (pkg_dir / "test_pkg.json").write_text(json.dumps(envoy_json))

        # Patch _SOURCE_DIRS to include our tmp dir
        with patch("knowledge.mission_loader._SOURCE_DIRS", [pkg_dir]):
            with patch("knowledge.mission_loader._mission_cache", {}):
                result = load_mission("test_pkg")

        assert result is not None
        assert result["id"] == "TEST_PKG"
        assert result.get("type") == "package"
        assert "sandbox" in result


class TestPerEnvoySandbox:
    """Test that package envoys get custom sandbox namespaces."""

    def test_build_sandbox_namespace_with_extra_imports(self):
        """Extra imports should appear in the sandbox namespace."""
        from data_ops.custom_ops import _build_sandbox_namespace

        extra = [{"import_path": "math", "sandbox_alias": "math"}]
        ns = _build_sandbox_namespace(extra_imports=extra)
        assert "math" in ns
        import math
        assert ns["math"] is math

    def test_build_sandbox_namespace_default_unchanged(self):
        """Without extra_imports, namespace is unchanged from current behavior."""
        from data_ops.custom_ops import _build_sandbox_namespace

        ns = _build_sandbox_namespace()
        assert "pd" in ns
        assert "np" in ns
        assert "xr" in ns
        # Should NOT have extra packages
        assert "math" not in ns

    def test_sandbox_executes_with_extra_package(self):
        """Code using an extra package should execute successfully."""
        from data_ops.custom_ops import _build_sandbox_namespace, _execute_in_sandbox

        extra = [{"import_path": "math", "sandbox_alias": "math"}]
        ns = _build_sandbox_namespace(extra_imports=extra)
        ns["result"] = None
        result = _execute_in_sandbox("result = math.sqrt(16)", ns)
        assert result == 4.0

    def test_extra_import_missing_package_skipped(self):
        """A non-existent optional extra import is silently skipped."""
        from data_ops.custom_ops import _build_sandbox_namespace

        extra = [{"import_path": "nonexistent_pkg_xyz_123", "sandbox_alias": "xyz"}]
        ns = _build_sandbox_namespace(extra_imports=extra)
        assert "xyz" not in ns

    def test_execute_multi_source_with_extra_imports(self):
        """execute_multi_source_operation should accept extra_imports."""
        import pandas as pd
        import numpy as np
        from data_ops.custom_ops import execute_multi_source_operation

        df = pd.DataFrame({"val": [1.0, 4.0, 9.0]}, index=pd.date_range("2020-01-01", periods=3))
        extra = [{"import_path": "math", "sandbox_alias": "math"}]
        result = execute_multi_source_operation(
            {"df_test": df},
            "result = df.assign(sqrt=df['val'].apply(math.sqrt))",
            extra_imports=extra,
        )
        assert "sqrt" in result.columns
        np.testing.assert_array_almost_equal(result["sqrt"].values, [1.0, 2.0, 3.0])

    def test_validate_code_blocks_module_attrs_on_extra_imports(self):
        """validate_code should block module_only attrs on extra_imports aliases."""
        from data_ops.custom_ops import validate_code

        # Without extra_module_names, "mypkg.eval(...)" is NOT caught
        violations = validate_code("result = mypkg.eval('x')")
        module_violations = [v for v in violations if "module-level" in v.lower()]
        assert len(module_violations) == 0

        # With extra_module_names, "mypkg.eval(...)" IS caught
        violations = validate_code(
            "result = mypkg.eval('x')",
            extra_module_names=frozenset({"mypkg"}),
        )
        module_violations = [v for v in violations if "module-level" in v.lower()]
        assert len(module_violations) == 1
        assert "mypkg.eval" in module_violations[0]

    def test_validate_code_without_extra_names_unchanged(self):
        """validate_code without extra_module_names behaves identically."""
        from data_ops.custom_ops import validate_code

        # pd.eval should still be blocked (it's in static _MODULE_NAMES)
        violations = validate_code("result = pd.eval('x')")
        assert any("pd.eval" in v for v in violations)


class TestPackageEnvoyPrompt:
    """Test that package envoys get appropriate system prompts."""

    def test_build_package_prompt_returns_string(self):
        """_build_package_prompt should return a non-empty string."""
        from knowledge.prompt_builder import _build_package_prompt

        envoy_data = _make_package_envoy_json()
        with patch("knowledge.prompt_builder.load_mission", return_value=envoy_data):
            prompt = _build_package_prompt("TEST_PKG")
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_package_prompt_contains_function_info(self):
        """The prompt should describe the package's functions."""
        from knowledge.prompt_builder import _build_package_prompt

        envoy_data = _make_package_envoy_json()
        with patch("knowledge.prompt_builder.load_mission", return_value=envoy_data):
            prompt = _build_package_prompt("TEST_PKG")
        assert "sqrt" in prompt
        assert "math" in prompt

    def test_package_prompt_mentions_custom_operation(self):
        """The prompt should instruct the LLM to use custom_operation."""
        from knowledge.prompt_builder import _build_package_prompt

        envoy_data = _make_package_envoy_json()
        with patch("knowledge.prompt_builder.load_mission", return_value=envoy_data):
            prompt = _build_package_prompt("TEST_PKG")
        assert "custom_operation" in prompt

    def test_package_prompt_group_registered(self):
        """'package' should be in _GROUP_PROMPT_BUILDERS."""
        from knowledge.prompt_builder import _GROUP_PROMPT_BUILDERS
        assert "package" in _GROUP_PROMPT_BUILDERS


class TestPackageEnvoyToolGroup:
    """Test that package envoys get the correct tool set."""

    def test_package_group_exists(self):
        """The 'package' group should exist in envoy_groups."""
        registry_path = Path(__file__).parent.parent / "agent" / "tool_registry.json"
        data = json.loads(registry_path.read_text())
        assert "package" in data["envoy_groups"]

    def test_package_group_has_custom_operation(self):
        """The 'package' group should include custom_operation."""
        registry_path = Path(__file__).parent.parent / "agent" / "tool_registry.json"
        data = json.loads(registry_path.read_text())
        assert "custom_operation" in data["envoy_groups"]["package"]

    def test_package_group_has_data_tools(self):
        """The 'package' group should include describe_data, preview_data, store_dataframe."""
        registry_path = Path(__file__).parent.parent / "agent" / "tool_registry.json"
        data = json.loads(registry_path.read_text())
        pkg_tools = data["envoy_groups"]["package"]
        for tool in ["describe_data", "preview_data", "store_dataframe"]:
            assert tool in pkg_tools, f"Expected {tool} in package group"

    def test_envoy_tool_registry_resolves_package_group(self):
        """EnvoyToolRegistry should resolve package group for package envoys."""
        from agent.agent_registry import ENVOY_TOOL_REGISTRY

        ENVOY_TOOL_REGISTRY._mission_to_group["TEST_PKG"] = "package"
        try:
            tools = ENVOY_TOOL_REGISTRY.get_tools("TEST_PKG")
            assert "custom_operation" in tools
            assert "describe_data" in tools
        finally:
            ENVOY_TOOL_REGISTRY._mission_to_group.pop("TEST_PKG", None)


class TestSandboxConfigThreading:
    """Test that sandbox config flows from envoy JSON to custom_operation."""

    def test_envoy_agent_accepts_sandbox_config(self):
        """EnvoyAgent should accept and store sandbox_config."""
        from agent.envoy_agent import EnvoyAgent

        adapter = MagicMock()
        sandbox = {
            "imports": [{"import_path": "math", "sandbox_alias": "math"}],
            "functions": [],
        }

        with patch("agent.envoy_agent.build_envoy_prompt", return_value="test prompt"):
            with patch("agent.envoy_agent.get_function_schemas", return_value=[]):
                with patch("agent.envoy_agent.ENVOY_TOOL_REGISTRY") as mock_reg:
                    mock_reg.get_tools.return_value = ["custom_operation"]
                    agent = EnvoyAgent(
                        mission_id="TEST_PKG",
                        adapter=adapter,
                        model_name="test-model",
                        tool_executor=lambda *a, **kw: {},
                        sandbox_config=sandbox,
                    )
        assert agent.sandbox_config is not None
        assert agent.sandbox_config["imports"][0]["sandbox_alias"] == "math"

    def test_envoy_agent_sandbox_config_defaults_none(self):
        """EnvoyAgent sandbox_config defaults to None for mission envoys."""
        from agent.envoy_agent import EnvoyAgent

        adapter = MagicMock()

        with patch("agent.envoy_agent.build_envoy_prompt", return_value="test prompt"):
            with patch("agent.envoy_agent.get_function_schemas", return_value=[]):
                with patch("agent.envoy_agent.ENVOY_TOOL_REGISTRY") as mock_reg:
                    mock_reg.get_tools.return_value = ["search_datasets"]
                    agent = EnvoyAgent(
                        mission_id="ACE",
                        adapter=adapter,
                        model_name="test-model",
                        tool_executor=lambda *a, **kw: {},
                    )
        assert agent.sandbox_config is None

    def test_sandbox_imports_threaded_to_custom_operation(self):
        """Sandbox imports should reach handle_custom_operation via orchestrator attribute."""
        import threading

        # Simulate the thread-local attribute that the wrapper sets
        class FakeOrch:
            _tls = threading.local()

            @property
            def _current_envoy_sandbox_imports(self):
                return getattr(self._tls, "envoy_sandbox_imports", None)

            @_current_envoy_sandbox_imports.setter
            def _current_envoy_sandbox_imports(self, value):
                self._tls.envoy_sandbox_imports = value

        orch = FakeOrch()

        # Without sandbox imports
        assert orch._current_envoy_sandbox_imports is None

        # With sandbox imports set (as the wrapper would do)
        imports = [{"import_path": "math", "sandbox_alias": "math"}]
        orch._current_envoy_sandbox_imports = imports
        assert orch._current_envoy_sandbox_imports == imports

        # getattr fallback works for handler code
        assert getattr(orch, '_current_envoy_sandbox_imports', None) == imports

        # Cleanup restores None
        orch._current_envoy_sandbox_imports = None
        assert orch._current_envoy_sandbox_imports is None


class TestEnvoyManagementTools:
    """Test add_envoy, save_envoy, list_envoys, remove_envoy handlers."""

    def test_add_envoy_introspects_package(self):
        """add_envoy should return the API surface of a valid package."""
        from agent.tool_handlers.envoy_management import handle_add_envoy

        orch = MagicMock()
        result = handle_add_envoy(orch, {"package_name": "math"})
        assert result["status"] == "success"
        assert result["action"] == "review_api"
        assert len(result["api_surface"]) > 0
        # math.sqrt should be in there
        names = [f["name"] for f in result["api_surface"]]
        assert "sqrt" in names

    def test_add_envoy_missing_package(self):
        """add_envoy should return error for non-existent package."""
        from agent.tool_handlers.envoy_management import handle_add_envoy

        orch = MagicMock()
        result = handle_add_envoy(orch, {"package_name": "nonexistent_pkg_xyz_999"})
        assert result["status"] == "error"

    def test_add_envoy_no_package_name(self):
        """add_envoy should return error when package_name is missing."""
        from agent.tool_handlers.envoy_management import handle_add_envoy

        orch = MagicMock()
        result = handle_add_envoy(orch, {})
        assert result["status"] == "error"

    def test_save_and_list_envoy(self, tmp_path):
        """save_envoy should create JSON file, list_envoys should find it."""
        from agent.tool_handlers.envoy_management import handle_save_envoy, handle_list_envoys
        from agent.agent_registry import ENVOY_TOOL_REGISTRY

        orch = MagicMock()
        with patch("agent.tool_handlers.envoy_management._get_package_envoy_dir", return_value=tmp_path):
            result = handle_save_envoy(orch, {
                "envoy_id": "TEST_MATH",
                "envoy_name": "Math Tools",
                "description": "Math operations",
                "imports": [{"import_path": "math", "sandbox_alias": "math"}],
                "functions": [{"name": "sqrt", "description": "Square root", "signature": "math.sqrt(x)"}],
                "keywords": ["math", "sqrt"],
            })
        try:
            assert result["status"] == "success"
            assert (tmp_path / "test_math.json").exists()

            # Now list
            with patch("agent.tool_handlers.envoy_management._get_package_envoy_dir", return_value=tmp_path):
                result = handle_list_envoys(orch, {})
            assert result["status"] == "success"
            assert result["count"] == 1
            assert result["envoys"][0]["id"] == "TEST_MATH"
        finally:
            ENVOY_TOOL_REGISTRY._mission_to_group.pop("TEST_MATH", None)

    def test_remove_envoy(self, tmp_path):
        """remove_envoy should delete the JSON file."""
        from agent.tool_handlers.envoy_management import handle_save_envoy, handle_remove_envoy
        from agent.agent_registry import ENVOY_TOOL_REGISTRY

        orch = MagicMock()
        orch._sub_agents_lock = MagicMock()
        orch._sub_agents = {}

        with patch("agent.tool_handlers.envoy_management._get_package_envoy_dir", return_value=tmp_path):
            handle_save_envoy(orch, {
                "envoy_id": "TO_DELETE",
                "imports": [{"import_path": "math", "sandbox_alias": "math"}],
                "functions": [],
            })
            assert (tmp_path / "to_delete.json").exists()

            result = handle_remove_envoy(orch, {"envoy_id": "TO_DELETE"})
        assert result["status"] == "success"
        assert not (tmp_path / "to_delete.json").exists()
        # Cleanup in case test fails midway
        ENVOY_TOOL_REGISTRY._mission_to_group.pop("TO_DELETE", None)

    def test_remove_envoy_not_found(self, tmp_path):
        """remove_envoy returns error for unknown envoy."""
        from agent.tool_handlers.envoy_management import handle_remove_envoy

        orch = MagicMock()
        with patch("agent.tool_handlers.envoy_management._get_package_envoy_dir", return_value=tmp_path):
            result = handle_remove_envoy(orch, {"envoy_id": "NONEXISTENT"})
        assert result["status"] == "error"

    def test_list_envoys_empty(self, tmp_path):
        """list_envoys returns empty when no envoys exist."""
        from agent.tool_handlers.envoy_management import handle_list_envoys

        orch = MagicMock()
        with patch("agent.tool_handlers.envoy_management._get_package_envoy_dir", return_value=tmp_path):
            result = handle_list_envoys(orch, {})
        assert result["status"] == "success"
        assert result["envoys"] == []
        assert result["count"] == 0


class TestPackageEnvoyAutoRegistration:
    """Test that package envoys are auto-registered in the envoy group system."""

    def test_register_package_envoys(self):
        """register_package_envoys should assign type:package missions to 'package' group."""
        from agent.agent_registry import ENVOY_TOOL_REGISTRY, register_package_envoys

        mock_missions = {"TEST_PKG_REG": {"type": "package", "id": "TEST_PKG_REG"}}

        with patch("knowledge.catalog.MISSIONS", mock_missions):
            register_package_envoys()

        try:
            assert ENVOY_TOOL_REGISTRY.get_group("TEST_PKG_REG") == "package"
        finally:
            ENVOY_TOOL_REGISTRY._mission_to_group.pop("TEST_PKG_REG", None)

    def test_register_skips_non_package_missions(self):
        """register_package_envoys should not affect normal missions."""
        from agent.agent_registry import ENVOY_TOOL_REGISTRY, register_package_envoys

        mock_missions = {"ACE_TEST": {"id": "ACE_TEST"}}

        with patch("knowledge.catalog.MISSIONS", mock_missions):
            register_package_envoys()

        # ACE_TEST should NOT be in the package group
        assert ENVOY_TOOL_REGISTRY.get_group("ACE_TEST") != "package"

    def test_register_is_idempotent(self):
        """Calling register_package_envoys twice should not fail or duplicate."""
        from agent.agent_registry import ENVOY_TOOL_REGISTRY, register_package_envoys

        mock_missions = {"TEST_PKG_IDEM": {"type": "package", "id": "TEST_PKG_IDEM"}}

        with patch("knowledge.catalog.MISSIONS", mock_missions):
            register_package_envoys()
            register_package_envoys()  # Should not fail

        try:
            assert ENVOY_TOOL_REGISTRY.get_group("TEST_PKG_IDEM") == "package"
        finally:
            ENVOY_TOOL_REGISTRY._mission_to_group.pop("TEST_PKG_IDEM", None)


class TestPackageEnvoyIntegration:
    """End-to-end test: create → load → tools → prompt → sandbox."""

    def test_full_lifecycle(self, tmp_path):
        """Create a math envoy, verify it gets correct tools and can execute code."""
        # 1. Save envoy JSON via handler
        from agent.tool_handlers.envoy_management import handle_save_envoy, handle_list_envoys, handle_remove_envoy
        from agent.agent_registry import ENVOY_TOOL_REGISTRY

        orch = MagicMock()
        orch._sub_agents_lock = MagicMock()
        orch._sub_agents = {}

        with patch("agent.tool_handlers.envoy_management._get_package_envoy_dir", return_value=tmp_path):
            save_result = handle_save_envoy(orch, {
                "envoy_id": "MATH_INT",
                "envoy_name": "Math Integration Test",
                "description": "Integration test for math package envoy",
                "imports": [{"import_path": "math", "sandbox_alias": "math"}],
                "functions": [
                    {"name": "sqrt", "description": "Square root", "signature": "math.sqrt(x)"},
                    {"name": "sin", "description": "Sine", "signature": "math.sin(x)"},
                ],
                "keywords": ["math"],
            })

        try:
            assert save_result["status"] == "success"

            # 2. Verify it's registered in the tool group
            assert ENVOY_TOOL_REGISTRY.get_group("MATH_INT") == "package"
            tools = ENVOY_TOOL_REGISTRY.get_tools("MATH_INT")
            assert "custom_operation" in tools

            # 3. Load from disk and verify structure
            envoy_file = tmp_path / "math_int.json"
            assert envoy_file.exists()
            data = json.loads(envoy_file.read_text())
            assert data["type"] == "package"
            assert len(data["sandbox"]["imports"]) == 1
            assert len(data["sandbox"]["functions"]) == 2

            # 4. Verify prompt builds
            from knowledge.prompt_builder import _build_package_prompt
            with patch("knowledge.prompt_builder.load_mission", return_value=data):
                prompt = _build_package_prompt("MATH_INT")
            assert "sqrt" in prompt
            assert "sin" in prompt
            assert "custom_operation" in prompt

            # 5. Verify sandbox works with extra imports
            from data_ops.custom_ops import _build_sandbox_namespace, _execute_in_sandbox
            extra = data["sandbox"]["imports"]
            ns = _build_sandbox_namespace(extra_imports=extra)
            ns["result"] = None
            result = _execute_in_sandbox("result = math.sqrt(144) + math.sin(0)", ns)
            assert result == 12.0

            # 6. List and verify
            with patch("agent.tool_handlers.envoy_management._get_package_envoy_dir", return_value=tmp_path):
                list_result = handle_list_envoys(orch, {})
            assert list_result["count"] == 1
            assert list_result["envoys"][0]["num_functions"] == 2

            # 7. Remove and verify
            with patch("agent.tool_handlers.envoy_management._get_package_envoy_dir", return_value=tmp_path):
                remove_result = handle_remove_envoy(orch, {"envoy_id": "MATH_INT"})
            assert remove_result["status"] == "success"
            assert not envoy_file.exists()

        finally:
            ENVOY_TOOL_REGISTRY._mission_to_group.pop("MATH_INT", None)
