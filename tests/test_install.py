"""Smoke tests for installation: verify imports, entry points, and package structure.

Skips mission JSON bootstrap (slow network download) — only tests that the
installed package is structurally sound.
"""

import importlib
import subprocess
import sys

import pytest


# Every top-level package that must be importable after `pip install -e .`
REQUIRED_PACKAGES = [
    "agent",
    "agent.base_agent",
    "agent.tools",
    "agent.memory",
    "agent.memory_agent",
    "agent.envoy_agent",
    "agent.viz_agent",
    "agent.data_ops_agent",
    "agent.data_io_agent",
    "agent.event_bus",
    "agent.session",
    "agent.llm",
    "agent.llm.gemini.adapter",
    "api",
    "api.routes",
    "api.session_manager",
    "api.streaming",
    "data_ops",
    "data_ops.store",
    "data_ops.custom_ops",
    "data_ops.dag",
    "knowledge",
    "knowledge.prompt_builder",
    "rendering",
    "rendering.plotly_renderer",
    "rendering.registry",
    "config",
]

# Third-party dependencies that must be installed
REQUIRED_DEPS = [
    "google.genai",
    "plotly",
    "kaleido",
    "numpy",
    "pandas",
    "scipy",
    "cdflib",
    "fastembed",
    "fastapi",
    "uvicorn",
    "dotenv",
    "mcp",
]


class TestPackageImports:
    """Verify all project packages import without error."""

    @pytest.mark.parametrize("module_name", REQUIRED_PACKAGES)
    def test_import_package(self, module_name):
        mod = importlib.import_module(module_name)
        assert mod is not None


class TestDependencyImports:
    """Verify all required third-party dependencies are installed."""

    @pytest.mark.parametrize("module_name", REQUIRED_DEPS)
    def test_import_dependency(self, module_name):
        mod = importlib.import_module(module_name)
        assert mod is not None


class TestEntryPoint:
    """Verify the CLI entry point is accessible."""

    def test_xhelio_cli_importable(self):
        """The xhelio_cli module must be importable with a main() function."""
        import xhelio_cli

        assert hasattr(xhelio_cli, "main")
        assert callable(xhelio_cli.main)

    def test_xhelio_help(self):
        """Running `xhelio --help` should exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "xhelio_cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "xhelio" in result.stdout.lower() or "XHelio" in result.stdout


class TestPackageMetadata:
    """Verify package metadata is correct."""

    def test_pyproject_exists(self):
        """pyproject.toml should exist and define the package."""
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found"
        content = pyproject.read_text()
        assert 'name = "xhelio"' in content
        assert "version" in content

    def test_python_version_constraint(self):
        """The running Python should satisfy the package's requires-python."""
        vi = sys.version_info
        assert vi >= (3, 11), f"Python 3.11+ required, got {vi.major}.{vi.minor}"


class TestKeyClasses:
    """Verify key classes and functions are importable from their modules."""

    def test_orchestrator_agent(self):
        from agent.core import OrchestratorAgent

        assert OrchestratorAgent is not None

    def test_sub_agent_base(self):
        from agent.base_agent import BaseAgent

        assert BaseAgent is not None

    def test_event_bus(self):
        from agent.event_bus import EventBus, get_event_bus

        assert EventBus is not None
        assert callable(get_event_bus)

    def test_data_store(self):
        from data_ops.store import DataStore

        assert DataStore is not None

    def test_plotly_renderer(self):
        from rendering.plotly_renderer import PlotlyRenderer

        assert PlotlyRenderer is not None

    def test_memory_store(self):
        from agent.memory import MemoryStore

        assert MemoryStore is not None

    def test_config_module(self):
        import config

        assert hasattr(config, "SMART_MODEL")
        assert hasattr(config, "LLM_PROVIDER")
