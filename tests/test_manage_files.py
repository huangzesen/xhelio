"""Tests for agent.tool_handlers.files — manage_files handler."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from data_ops.store import DataStore
from data_ops.asset_registry import AssetRegistry


@pytest.fixture
def session_dir(tmp_path):
    d = tmp_path / "session_test"
    d.mkdir()
    return d


@pytest.fixture
def data_store(session_dir):
    return DataStore(session_dir / "data")


@pytest.fixture
def registry(session_dir, data_store):
    return AssetRegistry(session_dir, data_store)


@pytest.fixture
def ctx(session_dir, data_store, registry):
    """Minimal ToolContext-like object."""
    mock = MagicMock()
    mock.session_dir = session_dir
    mock.store = data_store
    mock.event_bus = MagicMock()
    mock.asset_registry = registry
    mock.agent_state = {"orchestrator": MagicMock()}
    return mock


@pytest.fixture(autouse=True)
def _allow_tmp_dirs(tmp_path, monkeypatch):
    """Allow tmp_path in file validation for tests."""
    monkeypatch.setattr(
        "agent.tool_handlers.files._get_allowed_dirs",
        lambda session_dir=None: [tmp_path] + ([session_dir] if session_dir else []),
    )


class TestManageFilesRegister:
    def test_register_valid_file(self, ctx, tmp_path):
        from agent.tool_handlers.files import handle_manage_files

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")

        result = handle_manage_files(ctx, {
            "action": "register",
            "file_path": str(f),
        })
        assert result["status"] == "success"
        assert result["asset_id"].startswith("file_")
        assert result["name"] == "data.csv"
        assert result["size_bytes"] > 0

    def test_register_with_custom_name(self, ctx, tmp_path):
        from agent.tool_handlers.files import handle_manage_files

        f = tmp_path / "data.csv"
        f.write_text("x")

        result = handle_manage_files(ctx, {
            "action": "register",
            "file_path": str(f),
            "name": "My Dataset",
        })
        assert result["name"] == "My Dataset"

    def test_register_missing_file(self, ctx):
        from agent.tool_handlers.files import handle_manage_files

        result = handle_manage_files(ctx, {
            "action": "register",
            "file_path": "/nonexistent/file.csv",
        })
        assert result["status"] == "error"

    def test_register_no_path(self, ctx):
        from agent.tool_handlers.files import handle_manage_files

        result = handle_manage_files(ctx, {"action": "register"})
        assert result["status"] == "error"

    def test_register_copies_external_file_to_session(self, ctx, tmp_path, session_dir):
        from agent.tool_handlers.files import handle_manage_files

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")

        result = handle_manage_files(ctx, {
            "action": "register",
            "file_path": str(f),
        })
        assert result["status"] == "success"
        assert "session_path" in result
        assert Path(result["session_path"]).exists()
        assert Path(result["session_path"]).read_text() == "a,b\n1,2\n"
        # Original file still exists (copy, not move)
        assert f.exists()


class TestManageFilesList:
    def test_list_empty(self, ctx):
        from agent.tool_handlers.files import handle_manage_files

        result = handle_manage_files(ctx, {"action": "list"})
        assert result["status"] == "success"
        assert result["files"] == []

    def test_list_after_register(self, ctx, tmp_path):
        from agent.tool_handlers.files import handle_manage_files

        f = tmp_path / "data.csv"
        f.write_text("x")
        handle_manage_files(ctx, {"action": "register", "file_path": str(f)})

        result = handle_manage_files(ctx, {"action": "list"})
        assert len(result["files"]) == 1


class TestManageFilesInfo:
    def test_info_returns_metadata(self, ctx, tmp_path):
        from agent.tool_handlers.files import handle_manage_files

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        reg = handle_manage_files(ctx, {"action": "register", "file_path": str(f)})
        asset_id = reg["asset_id"]

        result = handle_manage_files(ctx, {"action": "info", "asset_id": asset_id})
        assert result["status"] == "success"
        assert result["name"] == "data.csv"
        assert "size_bytes" in result
        assert "mime_type" in result

    def test_info_missing_asset(self, ctx):
        from agent.tool_handlers.files import handle_manage_files

        result = handle_manage_files(ctx, {"action": "info", "asset_id": "nonexistent"})
        assert result["status"] == "error"


class TestManageFilesPrepareDeprecated:
    def test_prepare_returns_deprecation_error(self, ctx, tmp_path):
        from agent.tool_handlers.files import handle_manage_files

        f = tmp_path / "data.csv"
        f.write_text("x")
        reg = handle_manage_files(ctx, {"action": "register", "file_path": str(f)})

        result = handle_manage_files(ctx, {"action": "prepare", "asset_id": reg["asset_id"]})
        assert result["status"] == "error"
        assert "register" in result["message"].lower()


class TestManageFilesDelete:
    def test_delete_removes_asset(self, ctx, tmp_path):
        from agent.tool_handlers.files import handle_manage_files

        f = tmp_path / "del.csv"
        f.write_text("x")
        reg = handle_manage_files(ctx, {"action": "register", "file_path": str(f)})
        asset_id = reg["asset_id"]

        result = handle_manage_files(ctx, {"action": "delete", "asset_id": asset_id})
        assert result["status"] == "success"

        info = handle_manage_files(ctx, {"action": "info", "asset_id": asset_id})
        assert info["status"] == "error"
