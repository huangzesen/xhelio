"""Tests for the manage_envoy tool handler."""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.tool_handlers.manage_envoy import handle_manage_envoy


@pytest.fixture
def mock_orch():
    orch = MagicMock()
    orch._event_bus = MagicMock()
    return orch


@pytest.fixture
def temp_envoys_dir(tmp_path):
    """Create a temporary envoys directory for testing."""
    envoys_dir = tmp_path / "envoys"
    envoys_dir.mkdir()
    (envoys_dir / "__init__.py").write_text("")
    return envoys_dir


@pytest.fixture
def temp_prompts_dir(tmp_path):
    """Create a temporary prompts directory for testing."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    return prompts_dir


class TestManageEnvoyCreate:
    def test_create_writes_kind_module(self, mock_orch, temp_prompts_dir, monkeypatch):
        """Test that create writes __init__.py and handlers.py to the real envoys dir.

        Uses the real knowledge/envoys/ directory so importlib can validate the module.
        Cleans up after itself.
        """
        from agent.tool_handlers.manage_envoy import _ENVOYS_DIR
        monkeypatch.setattr(
            "agent.tool_handlers.manage_envoy._PROMPTS_DIR", temp_prompts_dir
        )
        kind_dir = _ENVOYS_DIR / "_test_pfss"
        try:
            result = handle_manage_envoy(mock_orch, {
                "action": "create",
                "kind": "_test_pfss",
                "envoy_id": "_TEST_PFSS",
                "source": "test_package",
                "source_type": "package",
                "tools": [
                    {
                        "name": "compute_pfss",
                        "description": "Compute PFSS solution",
                        "parameters": {"type": "object", "properties": {}},
                        "handler_code": (
                            "def handle_compute_pfss(orch, tool_args):\n"
                            "    return {'status': 'success'}\n"
                        ),
                    }
                ],
            })
            assert result["status"] == "success"
            assert (kind_dir / "__init__.py").exists()
            assert (kind_dir / "handlers.py").exists()
        finally:
            # Cleanup
            import shutil, sys
            if kind_dir.exists():
                shutil.rmtree(kind_dir)
            sys.modules.pop("knowledge.envoys._test_pfss", None)
            from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
            ENVOY_KIND_REGISTRY.unregister_mission("_TEST_PFSS")

    def test_create_generates_prompt_template(self, mock_orch, temp_prompts_dir, monkeypatch):
        from agent.tool_handlers.manage_envoy import _ENVOYS_DIR
        monkeypatch.setattr(
            "agent.tool_handlers.manage_envoy._PROMPTS_DIR", temp_prompts_dir
        )
        kind_dir = _ENVOYS_DIR / "_test_pfss2"
        try:
            handle_manage_envoy(mock_orch, {
                "action": "create",
                "kind": "_test_pfss2",
                "envoy_id": "_TEST_PFSS2",
                "source": "sunkit_magex.pfss",
                "source_type": "package",
                "tools": [],
            })
            template = temp_prompts_dir / "envoy__test_pfss2" / "role.md"
            assert template.exists()
            content = template.read_text()
            assert "pfss" in content.lower() or "PFSS" in content
        finally:
            import shutil, sys
            if kind_dir.exists():
                shutil.rmtree(kind_dir)
            sys.modules.pop("knowledge.envoys._test_pfss2", None)
            from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
            ENVOY_KIND_REGISTRY.unregister_mission("_TEST_PFSS2")

    def test_create_rejects_existing_kind(self, mock_orch, temp_envoys_dir, temp_prompts_dir, monkeypatch):
        monkeypatch.setattr(
            "agent.tool_handlers.manage_envoy._ENVOYS_DIR", temp_envoys_dir
        )
        monkeypatch.setattr(
            "agent.tool_handlers.manage_envoy._PROMPTS_DIR", temp_prompts_dir
        )
        (temp_envoys_dir / "pfss").mkdir()
        result = handle_manage_envoy(mock_orch, {
            "action": "create",
            "kind": "pfss",
            "envoy_id": "PFSS",
            "source": "test",
            "source_type": "package",
            "tools": [],
        })
        assert result["status"] == "error"
        assert "already exists" in result["message"]


class TestManageEnvoyRemove:
    def test_remove_deletes_kind_dir(self, mock_orch, temp_envoys_dir, temp_prompts_dir, monkeypatch):
        monkeypatch.setattr(
            "agent.tool_handlers.manage_envoy._ENVOYS_DIR", temp_envoys_dir
        )
        monkeypatch.setattr(
            "agent.tool_handlers.manage_envoy._PROMPTS_DIR", temp_prompts_dir
        )
        # Create a kind first
        kind_dir = temp_envoys_dir / "pfss"
        kind_dir.mkdir()
        (kind_dir / "__init__.py").write_text("TOOLS = []\nHANDLERS = {}\nGLOBAL_TOOLS = []")
        (kind_dir / "handlers.py").write_text("")

        result = handle_manage_envoy(mock_orch, {
            "action": "remove",
            "kind": "pfss",
            "envoy_id": "PFSS",
        })
        assert result["status"] == "success"
        assert not kind_dir.exists()

    def test_remove_warns_on_prebuilt(self, mock_orch, temp_envoys_dir, monkeypatch):
        monkeypatch.setattr(
            "agent.tool_handlers.manage_envoy._ENVOYS_DIR", temp_envoys_dir
        )
        result = handle_manage_envoy(mock_orch, {
            "action": "remove",
            "kind": "cdaweb",
            "envoy_id": "PSP",
        })
        assert result["status"] == "error"
        assert "prebuilt" in result["message"].lower()

    def test_remove_nonexistent(self, mock_orch, temp_envoys_dir, monkeypatch):
        monkeypatch.setattr(
            "agent.tool_handlers.manage_envoy._ENVOYS_DIR", temp_envoys_dir
        )
        result = handle_manage_envoy(mock_orch, {
            "action": "remove",
            "kind": "nonexistent",
            "envoy_id": "TEST",
        })
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_unknown_action(self, mock_orch):
        result = handle_manage_envoy(mock_orch, {
            "action": "invalid",
            "kind": "test",
            "envoy_id": "TEST",
        })
        assert result["status"] == "error"
        assert "Unknown action" in result["message"]
