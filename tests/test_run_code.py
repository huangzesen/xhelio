"""
Tests for agent.tool_handlers.sandbox — handle_run_code.

Run with: python -m pytest tests/test_run_code.py
"""

from unittest.mock import MagicMock
from pathlib import Path

import pandas as pd
import pytest

from data_ops.store import DataStore, DataEntry


def _make_mock_orch(tmp_path):
    """Create a mock orchestrator with a real DataStore."""
    orch = MagicMock()
    store = DataStore(tmp_path / "data")
    orch._store = store
    orch._session_dir = tmp_path
    orch._event_bus = MagicMock()
    (tmp_path / "sandbox").mkdir()
    return orch, store


class TestRunCode:
    def test_standalone_print(self, tmp_path):
        from agent.tool_handlers.sandbox import handle_run_code

        orch, store = _make_mock_orch(tmp_path)
        result = handle_run_code(
            orch,
            {
                "code": "print('hello from sandbox')",
                "description": "test print",
            },
        )
        assert result["status"] == "success"
        assert "hello from sandbox" in result["output"]

    def test_store_result(self, tmp_path):
        from agent.tool_handlers.sandbox import handle_run_code

        orch, store = _make_mock_orch(tmp_path)
        result = handle_run_code(
            orch,
            {
                "code": "import pandas as pd\nresult = pd.DataFrame({'x': [1,2,3]})",
                "store_as": "test_df",
                "description": "test store",
            },
        )
        assert result["status"] == "success"
        assert result["stored"]["label"] == "test_df"
        assert store.get("test_df") is not None

    def test_inputs_staged(self, tmp_path):
        from agent.tool_handlers.sandbox import handle_run_code

        orch, store = _make_mock_orch(tmp_path)

        # Pre-populate store
        df = pd.DataFrame(
            {"Bx": [1.0, 2.0]},
            index=pd.date_range("2026-01-01", periods=2, freq="h"),
        )
        entry = DataEntry(label="ACE.Bgsm", data=df, units="nT")
        store.put(entry)

        result = handle_run_code(
            orch,
            {
                "inputs": ["ACE.Bgsm"],
                "code": (
                    "import pandas as pd\n"
                    "df = pd.read_parquet('ACE.Bgsm.parquet')\n"
                    "print(f'rows: {len(df)}')\n"
                    "result = df"
                ),
                "store_as": "ACE.Bgsm.copy",
                "description": "test inputs",
            },
        )
        assert result["status"] == "success"
        assert "rows: 2" in result["output"]

    def test_blocked_code(self, tmp_path):
        from agent.tool_handlers.sandbox import handle_run_code

        orch, store = _make_mock_orch(tmp_path)
        result = handle_run_code(
            orch,
            {
                "code": "import os\nos.system('echo pwned')",
                "description": "test blocked",
            },
        )
        assert result["status"] == "error"
        assert "Blocked" in result["message"]

    def test_missing_input(self, tmp_path):
        from agent.tool_handlers.sandbox import handle_run_code

        orch, store = _make_mock_orch(tmp_path)
        result = handle_run_code(
            orch,
            {
                "inputs": ["nonexistent"],
                "code": "result = 1",
                "description": "test missing",
            },
        )
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_no_store_as_no_stored_key(self, tmp_path):
        from agent.tool_handlers.sandbox import handle_run_code

        orch, store = _make_mock_orch(tmp_path)
        result = handle_run_code(
            orch,
            {
                "code": "result = 42",
                "description": "no store",
            },
        )
        assert result["status"] == "success"
        assert "stored" not in result

    def test_store_dict_result(self, tmp_path):
        from agent.tool_handlers.sandbox import handle_run_code

        orch, store = _make_mock_orch(tmp_path)
        result = handle_run_code(
            orch,
            {
                "code": "result = {'answer': 42, 'pi': 3.14}",
                "store_as": "my_dict",
                "description": "dict result",
            },
        )
        assert result["status"] == "success"
        assert result["stored"]["type"] == "dict"
        loaded = store.get("my_dict")
        assert loaded is not None
        assert loaded.data["answer"] == 42
