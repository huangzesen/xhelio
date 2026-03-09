"""Tests for the generic envoy JSON generator."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestFromMCP:
    """Test from_mcp() generates valid envoy JSON from MCP tool schemas."""

    def test_generates_valid_structure(self, tmp_path):
        from knowledge.generate_envoy_json import from_mcp

        fake_schemas = [
            {"name": "get_position", "description": "Get spacecraft position", "inputSchema": {"type": "object", "properties": {"target": {"type": "string"}}}},
            {"name": "list_missions", "description": "List supported missions", "inputSchema": {"type": "object", "properties": {}}},
        ]
        fake_package_info = {
            "name": "heliospice",
            "version": "0.4.3",
            "doc": "Spacecraft ephemeris made easy.",
        }

        with patch("knowledge.generate_envoy_json._call_llm") as mock_llm:
            mock_llm.return_value = {
                "name": "SPICE Ephemeris",
                "description": "Spacecraft ephemeris via SPICE kernels",
                "keywords": ["spice", "ephemeris"],
                "instruments": {
                    "Ephemeris Tools": {
                        "name": "Ephemeris Tools",
                        "keywords": ["position", "trajectory"],
                        "datasets": {
                            "get_position": {"description": "Get spacecraft position"},
                            "list_missions": {"description": "List supported missions"},
                        }
                    }
                }
            }
            result = from_mcp(
                tool_schemas=fake_schemas,
                package_info=fake_package_info,
                envoy_id="SPICE",
                output_dir=tmp_path,
            )

        assert result.exists()
        data = json.loads(result.read_text())
        assert data["id"] == "SPICE"
        assert "instruments" in data
        assert "profile" in data
        assert data["_generator_version"] == "0.4.3"
        assert data["_generator_source"] == "mcp"

    def test_skips_if_version_matches(self, tmp_path):
        from knowledge.generate_envoy_json import from_mcp

        existing = {"id": "SPICE", "name": "SPICE", "_generator_version": "0.4.3", "instruments": {}, "profile": {}, "keywords": []}
        (tmp_path / "spice.json").write_text(json.dumps(existing))

        result = from_mcp(
            tool_schemas=[],
            package_info={"name": "heliospice", "version": "0.4.3", "doc": ""},
            envoy_id="SPICE",
            output_dir=tmp_path,
        )

        assert result == tmp_path / "spice.json"

    def test_fallback_without_llm(self, tmp_path):
        from knowledge.generate_envoy_json import from_mcp

        fake_schemas = [
            {"name": "get_pos", "description": "Get position"},
        ]

        with patch("knowledge.generate_envoy_json._call_llm") as mock_llm:
            mock_llm.return_value = {}  # LLM failed
            result = from_mcp(
                tool_schemas=fake_schemas,
                package_info={"name": "test", "version": "1.0", "doc": "Test doc"},
                envoy_id="TEST",
                output_dir=tmp_path,
            )

        data = json.loads(result.read_text())
        assert data["id"] == "TEST"
        assert "Tools" in data["instruments"]
        assert "get_pos" in data["instruments"]["Tools"]["datasets"]

    def test_regenerates_on_version_change(self, tmp_path):
        from knowledge.generate_envoy_json import from_mcp

        # Write an existing file with an older version
        existing = {"id": "SPICE", "name": "SPICE", "_generator_version": "0.3.0", "instruments": {}, "profile": {}, "keywords": []}
        (tmp_path / "spice.json").write_text(json.dumps(existing))

        with patch("knowledge.generate_envoy_json._call_llm") as mock_llm:
            mock_llm.return_value = {}
            result = from_mcp(
                tool_schemas=[{"name": "t1", "description": "Tool 1"}],
                package_info={"name": "heliospice", "version": "0.4.3", "doc": ""},
                envoy_id="SPICE",
                output_dir=tmp_path,
            )

        data = json.loads(result.read_text())
        assert data["_generator_version"] == "0.4.3"
        # Should have regenerated with fallback instruments
        assert "Tools" in data["instruments"]

    def test_profile_has_expected_keys(self, tmp_path):
        from knowledge.generate_envoy_json import from_mcp

        with patch("knowledge.generate_envoy_json._call_llm") as mock_llm:
            mock_llm.return_value = {}
            result = from_mcp(
                tool_schemas=[],
                package_info={"name": "test", "version": "1.0", "doc": "My doc"},
                envoy_id="TEST",
                output_dir=tmp_path,
            )

        data = json.loads(result.read_text())
        profile = data["profile"]
        assert "description" in profile
        assert "coordinate_systems" in profile
        assert "typical_cadence" in profile
        assert "data_caveats" in profile
        assert "analysis_patterns" in profile
        assert profile["description"] == "My doc"

    def test_empty_tools_produces_empty_instruments(self, tmp_path):
        from knowledge.generate_envoy_json import from_mcp

        with patch("knowledge.generate_envoy_json._call_llm") as mock_llm:
            mock_llm.return_value = {}
            result = from_mcp(
                tool_schemas=[],
                package_info={"name": "test", "version": "1.0", "doc": ""},
                envoy_id="EMPTY",
                output_dir=tmp_path,
            )

        data = json.loads(result.read_text())
        assert data["instruments"] == {}


class TestFromPackage:
    """Test from_package() generates valid envoy JSON from Python package introspection."""

    def test_generates_from_package(self, tmp_path):
        from knowledge.generate_envoy_json import from_package

        with patch("knowledge.generate_envoy_json._call_llm") as mock_llm:
            mock_llm.return_value = {
                "name": "Test Package",
                "description": "A test package",
                "keywords": ["test"],
                "instruments": {}
            }
            with patch("knowledge.generate_envoy_json._introspect_package") as mock_intro:
                mock_intro.return_value = {
                    "name": "test_pkg",
                    "version": "1.0.0",
                    "doc": "Test package docstring",
                    "functions": [
                        {"name": "do_thing", "signature": "(x: int) -> int", "doc": "Does a thing"}
                    ]
                }
                result = from_package(
                    package_name="test_pkg",
                    envoy_id="TEST_PKG",
                    output_dir=tmp_path,
                )

        assert result.exists()
        data = json.loads(result.read_text())
        assert data["id"] == "TEST_PKG"
        assert data["_generator_source"] == "package"

    def test_fallback_without_llm(self, tmp_path):
        from knowledge.generate_envoy_json import from_package

        with patch("knowledge.generate_envoy_json._call_llm") as mock_llm:
            mock_llm.return_value = {}
            with patch("knowledge.generate_envoy_json._introspect_package") as mock_intro:
                mock_intro.return_value = {
                    "name": "test_pkg",
                    "version": "2.0.0",
                    "doc": "A test",
                    "functions": [
                        {"name": "func_a", "signature": "()", "doc": "Function A"},
                        {"name": "func_b", "signature": "(x)", "doc": "Function B"},
                    ]
                }
                result = from_package(
                    package_name="test_pkg",
                    envoy_id="TEST_PKG",
                    output_dir=tmp_path,
                )

        data = json.loads(result.read_text())
        assert data["id"] == "TEST_PKG"
        assert "Functions" in data["instruments"]
        assert "func_a" in data["instruments"]["Functions"]["datasets"]
        assert "func_b" in data["instruments"]["Functions"]["datasets"]

    def test_skips_if_version_matches(self, tmp_path):
        from knowledge.generate_envoy_json import from_package

        existing = {"id": "PKG", "name": "PKG", "_generator_version": "1.0.0", "instruments": {}, "profile": {}, "keywords": []}
        (tmp_path / "pkg.json").write_text(json.dumps(existing))

        with patch("knowledge.generate_envoy_json._introspect_package") as mock_intro:
            mock_intro.return_value = {
                "name": "test_pkg",
                "version": "1.0.0",
                "doc": "",
                "functions": [],
            }
            result = from_package(
                package_name="test_pkg",
                envoy_id="PKG",
                output_dir=tmp_path,
            )

        assert result == tmp_path / "pkg.json"
        # File should not have been modified
        data = json.loads(result.read_text())
        assert data == existing


class TestIntrospectPackage:
    """Test _introspect_package() extracts metadata from real packages."""

    def test_introspects_json_module(self):
        from knowledge.generate_envoy_json import _introspect_package

        info = _introspect_package("json")
        assert info["name"] == "json"
        assert info["version"] is not None
        assert isinstance(info["functions"], list)
        # json module has public functions like dumps, loads
        func_names = [f["name"] for f in info["functions"]]
        assert "dumps" in func_names or "loads" in func_names

    def test_handles_missing_package(self):
        from knowledge.generate_envoy_json import _introspect_package

        info = _introspect_package("nonexistent_package_xyz_123")
        assert info["name"] == "nonexistent_package_xyz_123"
        assert info["version"] == "unknown"
        assert info["functions"] == []


class TestFallbackInstruments:
    """Test fallback instrument builders."""

    def test_mcp_fallback_groups_all_tools(self):
        from knowledge.generate_envoy_json import _fallback_instruments_from_mcp

        schemas = [
            {"name": "tool_a", "description": "Does A"},
            {"name": "tool_b", "description": "Does B"},
        ]
        result = _fallback_instruments_from_mcp(schemas)
        assert "Tools" in result
        assert "tool_a" in result["Tools"]["datasets"]
        assert "tool_b" in result["Tools"]["datasets"]
        assert result["Tools"]["datasets"]["tool_a"]["description"] == "Does A"

    def test_mcp_fallback_empty_schemas(self):
        from knowledge.generate_envoy_json import _fallback_instruments_from_mcp

        result = _fallback_instruments_from_mcp([])
        assert result == {}

    def test_package_fallback_groups_all_functions(self):
        from knowledge.generate_envoy_json import _fallback_instruments_from_package

        pkg_info = {
            "functions": [
                {"name": "fn1", "doc": "First function"},
                {"name": "fn2", "doc": "Second function"},
            ]
        }
        result = _fallback_instruments_from_package(pkg_info)
        assert "Functions" in result
        assert "fn1" in result["Functions"]["datasets"]
        assert "fn2" in result["Functions"]["datasets"]

    def test_package_fallback_empty_functions(self):
        from knowledge.generate_envoy_json import _fallback_instruments_from_package

        result = _fallback_instruments_from_package({"functions": []})
        assert result == {}


class TestBuildLLMPrompt:
    """Test _build_llm_prompt() produces reasonable prompts."""

    def test_includes_package_info(self):
        from knowledge.generate_envoy_json import _build_llm_prompt

        prompt = _build_llm_prompt({
            "package": {"name": "mypackage", "version": "1.0", "doc": "My package does things"},
            "tools": [],
        })
        assert "mypackage" in prompt
        assert "1.0" in prompt
        assert "My package does things" in prompt

    def test_includes_tool_info(self):
        from knowledge.generate_envoy_json import _build_llm_prompt

        prompt = _build_llm_prompt({
            "package": {},
            "tools": [
                {"name": "get_data", "description": "Gets data", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}},
            ],
        })
        assert "get_data" in prompt
        assert "Gets data" in prompt
        assert "id: string" in prompt

    def test_includes_function_info(self):
        from knowledge.generate_envoy_json import _build_llm_prompt

        prompt = _build_llm_prompt({
            "package": {},
            "functions": [
                {"name": "compute", "signature": "(x: float) -> float", "doc": "Computes things"},
            ],
        })
        assert "compute" in prompt
        assert "Computes things" in prompt
