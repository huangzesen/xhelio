"""Tests for persistent sandbox directory in run_code."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from data_ops.store import DataStore
from data_ops.asset_registry import AssetRegistry


@pytest.fixture
def session_dir(tmp_path):
    d = tmp_path / "session"
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
    mock = MagicMock()
    mock.session_dir = session_dir
    mock.store = data_store
    mock.asset_registry = registry
    mock.event_bus = MagicMock()
    mock.agent_state = {"orchestrator": MagicMock()}
    return mock


@pytest.fixture(autouse=True)
def _allow_tmp_dirs(tmp_path, monkeypatch):
    """Allow tmp_path in file validation for tests."""
    original = __import__("agent.tool_handlers.files", fromlist=["_get_allowed_dirs"])._get_allowed_dirs

    def patched(session_dir=None):
        dirs = [tmp_path]
        if session_dir is not None:
            dirs.append(session_dir)
        return dirs

    monkeypatch.setattr("agent.tool_handlers.files._get_allowed_dirs", patched)


class TestPersistentSandbox:
    def test_files_persist_between_calls(self, ctx, session_dir):
        from agent.tool_handlers.sandbox import handle_run_code

        # Call 1: write a file
        r1 = handle_run_code(ctx, {
            "code": "with open('test_file.txt', 'w') as f: f.write('hello')",
            "description": "write test file",
        })
        assert r1["status"] == "success"

        # Call 2: read the file back
        r2 = handle_run_code(ctx, {
            "code": "with open('test_file.txt') as f: print(f.read())",
            "description": "read test file",
        })
        assert r2["status"] == "success"
        assert "hello" in r2["output"]

    def test_sandbox_dir_is_session_scoped(self, ctx, session_dir):
        from agent.tool_handlers.sandbox import handle_run_code

        handle_run_code(ctx, {
            "code": "with open('marker.txt', 'w') as f: f.write('x')",
            "description": "write marker",
        })
        assert (session_dir / "sandbox" / "marker.txt").exists()

    def test_fallback_to_tempdir_when_no_session(self):
        from agent.tool_handlers.sandbox import handle_run_code

        ctx = MagicMock()
        ctx.session_dir = None
        ctx.store = MagicMock()
        ctx.store.get.return_value = None
        ctx.asset_registry = None

        r = handle_run_code(ctx, {
            "code": "print('works')",
            "description": "test fallback",
        })
        assert r["status"] == "success"
        assert "works" in r["output"]


class TestRegisterFromSandbox:
    def test_register_moves_file_to_files_dir(self, ctx, session_dir):
        from agent.tool_handlers.sandbox import handle_run_code
        from agent.tool_handlers.files import handle_manage_files

        # Download a file into sandbox
        handle_run_code(ctx, {
            "code": "with open('data.h5', 'wb') as f: f.write(b'fake hdf5 data')",
            "description": "create test file",
        })
        sandbox_file = session_dir / "sandbox" / "data.h5"
        assert sandbox_file.exists()

        # Register it
        result = handle_manage_files(ctx, {
            "action": "register",
            "file_path": str(sandbox_file),
            "name": "Test Data",
        })
        assert result["status"] == "success"
        assert result["asset_id"].startswith("file_")
        assert "session_path" in result
        assert result["original_filename"] == "data.h5"

        # File moved out of sandbox
        assert not sandbox_file.exists()
        # File exists in files/ dir
        assert Path(result["session_path"]).exists()
        assert (session_dir / "files").exists()

    def test_register_with_source_url(self, ctx, session_dir):
        from agent.tool_handlers.sandbox import handle_run_code
        from agent.tool_handlers.files import handle_manage_files

        handle_run_code(ctx, {
            "code": "with open('map.fits', 'wb') as f: f.write(b'fits')",
            "description": "create test file",
        })

        result = handle_manage_files(ctx, {
            "action": "register",
            "file_path": str(session_dir / "sandbox" / "map.fits"),
            "source_url": "https://example.com/map.fits",
        })
        assert result["status"] == "success"
        # Verify source_url is stored in metadata
        asset = ctx.asset_registry.get_asset(result["asset_id"])
        assert asset.metadata.get("source_url") == "https://example.com/map.fits"

    def test_register_outside_session_unchanged(self, ctx, tmp_path):
        """Files outside sandbox use existing behavior (no move)."""
        from agent.tool_handlers.files import handle_manage_files

        f = tmp_path / "external.csv"
        f.write_text("a,b\n1,2\n")

        result = handle_manage_files(ctx, {
            "action": "register",
            "file_path": str(f),
        })
        assert result["status"] == "success"
        # File NOT moved — still exists at original path
        assert f.exists()

    def test_register_rejects_disallowed_paths(self, ctx):
        """Paths outside allowed dirs are rejected."""
        from agent.tool_handlers.files import handle_manage_files

        result = handle_manage_files(ctx, {
            "action": "register",
            "file_path": "/etc/passwd",
        })
        assert result["status"] == "error"
        assert "outside" in result["message"].lower() or "not found" in result["message"].lower()


class TestFileAssetInputStaging:
    def test_stage_registered_file_as_input(self, ctx, session_dir):
        from agent.tool_handlers.sandbox import handle_run_code
        from agent.tool_handlers.files import handle_manage_files

        # Create and register a file
        handle_run_code(ctx, {
            "code": "with open('data.csv', 'w') as f: f.write('a,b\\n1,2\\n')",
            "description": "create csv",
        })
        reg = handle_manage_files(ctx, {
            "action": "register",
            "file_path": str(session_dir / "sandbox" / "data.csv"),
            "name": "Test CSV",
        })
        asset_id = reg["asset_id"]

        # Process it via inputs
        r = handle_run_code(ctx, {
            "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(f'rows={len(df)}')",
            "inputs": [asset_id],
            "description": "read registered file",
        })
        assert r["status"] == "success"
        assert "rows=1" in r["output"]

    def test_stage_nonexistent_file_asset_errors(self, ctx):
        from agent.tool_handlers.sandbox import handle_run_code

        r = handle_run_code(ctx, {
            "code": "print('hi')",
            "inputs": ["file_nonexistent"],
            "description": "test bad input",
        })
        assert r["status"] == "error"
        assert "not found" in r["message"]

    def test_stage_filename_collision_errors(self, ctx, session_dir):
        from agent.tool_handlers.sandbox import handle_run_code
        from agent.tool_handlers.files import handle_manage_files

        # Create two files with the same original name in different calls
        handle_run_code(ctx, {
            "code": "with open('data.csv', 'w') as f: f.write('v1')",
            "description": "create v1",
        })
        r1 = handle_manage_files(ctx, {
            "action": "register",
            "file_path": str(session_dir / "sandbox" / "data.csv"),
        })

        handle_run_code(ctx, {
            "code": "with open('data.csv', 'w') as f: f.write('v2')",
            "description": "create v2",
        })
        r2 = handle_manage_files(ctx, {
            "action": "register",
            "file_path": str(session_dir / "sandbox" / "data.csv"),
        })

        # Both have original_filename "data.csv" — staging both should error
        r = handle_run_code(ctx, {
            "code": "print('hi')",
            "inputs": [r1["asset_id"], r2["asset_id"]],
            "description": "test collision",
        })
        assert r["status"] == "error"
        assert "collision" in r["message"].lower()
