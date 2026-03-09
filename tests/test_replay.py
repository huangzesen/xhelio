"""Tests for the replay engine (scripts/replay.py)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from data_ops.store import DataEntry, DataStore
from scripts.replay import (
    ReplayResult,
    _replay_run_code,
    _replay_fetch,
    _replay_render,
    replay_pipeline,
    replay_session,
    replay_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts_dataframe(n: int = 10, columns: list[str] | None = None) -> pd.DataFrame:
    """Create a simple timeseries DataFrame for testing."""
    if columns is None:
        columns = ["value"]
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    data = {col: np.random.randn(n) for col in columns}
    return pd.DataFrame(data, index=idx)


def _make_fetch_result(n: int = 10, columns: list[str] | None = None) -> dict:
    """Simulated return value of fetch_data()."""
    return {
        "data": _make_ts_dataframe(n, columns),
        "units": "nT",
        "description": "Test magnetic field",
    }


def _fetch_record(
    op_id: str = "op_001",
    dataset_id: str = "AC_H2_MFI",
    parameter_id: str = "BGSEc",
    time_range: str = "2024-01-01 to 2024-01-02",
    outputs: list[str] | None = None,
) -> dict:
    return {
        "id": op_id,
        "tool": "fetch_data",
        "status": "success",
        "inputs": [],
        "outputs": outputs or ["AC_H2_MFI.BGSEc"],
        "args": {
            "dataset_id": dataset_id,
            "parameter_id": parameter_id,
            "time_range_resolved": time_range,
        },
    }


def _custom_op_record(
    op_id: str = "op_002",
    code: str = "result = df.abs()",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    description: str = "Absolute value",
    units: str = "nT",
) -> dict:
    return {
        "id": op_id,
        "tool": "run_code",
        "status": "success",
        "inputs": inputs or ["AC_H2_MFI.BGSEc"],
        "outputs": outputs or ["Bmag"],
        "args": {
            "code": code,
            "description": description,
            "units": units,
        },
    }


def _store_df_record(
    op_id: str = "op_003",
    code: str = "result = pd.DataFrame({'x': [1,2,3]})",
    outputs: list[str] | None = None,
) -> dict:
    return {
        "id": op_id,
        "tool": "run_code",
        "status": "success",
        "inputs": [],
        "outputs": outputs or ["my_data"],
        "args": {
            "code": code,
            "description": "Test dataframe",
            "units": "",
        },
    }


def _render_record(
    op_id: str = "op_004",
    inputs: list[str] | None = None,
    data_labels: list[str] | None = None,
) -> dict:
    labels = data_labels or ["AC_H2_MFI.BGSEc"]
    traces = [{"data_label": lbl, "type": "scatter"} for lbl in labels]
    return {
        "id": op_id,
        "tool": "render_plotly_json",
        "status": "success",
        "inputs": inputs or labels,
        "outputs": [],
        "args": {
            "figure_json": {
                "data": traces,
                "layout": {"title": "Test", "yaxis": {"title": "Value"}},
            },
        },
    }


# ---------------------------------------------------------------------------
# Tier 1: Unit tests for individual handlers
# ---------------------------------------------------------------------------


class TestReplayFetch:
    """Tests for _replay_fetch."""

    @patch("scripts.replay.fetch_data")
    def test_stores_entry_correctly(self, mock_fetch, tmp_path):
        mock_fetch.return_value = _make_fetch_result()
        store = DataStore(tmp_path / "data")
        rec = _fetch_record()

        _replay_fetch(rec, store)

        assert store.has("AC_H2_MFI.BGSEc")
        entry = store.get("AC_H2_MFI.BGSEc")
        assert entry.source == "cdf"
        assert entry.units == "nT"
        assert len(entry.data) == 10

    @patch("scripts.replay.fetch_data")
    def test_uses_time_range_resolved(self, mock_fetch, tmp_path):
        mock_fetch.return_value = _make_fetch_result()
        store = DataStore(tmp_path / "data")
        rec = _fetch_record(time_range="2024-06-01 to 2024-06-15")

        _replay_fetch(rec, store)

        mock_fetch.assert_called_once_with(
            "AC_H2_MFI", "BGSEc", "2024-06-01", "2024-06-15"
        )

    @patch("scripts.replay.fetch_data")
    def test_fallback_to_time_min_max(self, mock_fetch, tmp_path):
        """Falls back to time_min/time_max when time_range_resolved is absent."""
        mock_fetch.return_value = _make_fetch_result()
        store = DataStore(tmp_path / "data")
        rec = _fetch_record()
        del rec["args"]["time_range_resolved"]
        rec["args"]["time_min"] = "2024-03-01"
        rec["args"]["time_max"] = "2024-03-10"

        _replay_fetch(rec, store)

        mock_fetch.assert_called_once_with(
            "AC_H2_MFI", "BGSEc", "2024-03-01", "2024-03-10"
        )

    @patch("scripts.replay.fetch_data")
    def test_multiple_outputs(self, mock_fetch, tmp_path):
        """Each output label gets a DataEntry."""
        mock_fetch.return_value = _make_fetch_result()
        store = DataStore(tmp_path / "data")
        rec = _fetch_record(outputs=["label_a", "label_b"])

        _replay_fetch(rec, store)

        assert store.has("label_a")
        assert store.has("label_b")

    @patch("scripts.replay.fetch_data")
    def test_fetch_error_propagates(self, mock_fetch, tmp_path):
        mock_fetch.side_effect = ValueError("No data available")
        store = DataStore(tmp_path / "data")
        rec = _fetch_record()

        with pytest.raises(ValueError, match="No data available"):
            _replay_fetch(rec, store)


class TestReplayRunCode:
    """Tests for _replay_run_code (unified replay for run_code / legacy ops)."""

    def test_single_source(self, tmp_path):
        store = DataStore(tmp_path / "data")
        df = _make_ts_dataframe(5, ["value"])
        store.put(DataEntry(label="src.data", data=df, source="cdf"))

        rec = _custom_op_record(
            code="import pandas as pd\ndf = pd.read_parquet('src.data.parquet')\nresult = df.abs()",
            inputs=["src.data"],
            outputs=["abs_data"],
        )

        _replay_run_code(rec, store)

        assert store.has("abs_data")
        entry = store.get("abs_data")
        assert entry.source == "computed"
        assert len(entry.data) == 5

    def test_creates_dataframe(self, tmp_path):
        store = DataStore(tmp_path / "data")
        rec = _store_df_record(
            code="import pandas as pd\nresult = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})",
            outputs=["my_table"],
        )

        _replay_run_code(rec, store)

        assert store.has("my_table")
        entry = store.get("my_table")
        assert entry.source == "computed"
        assert len(entry.data) == 3
        assert list(entry.data.columns) == ["a", "b"]

    def test_invalid_code_skips(self, tmp_path):
        """Invalid code (blocked builtin) is silently skipped in replay."""
        store = DataStore(tmp_path / "data")
        rec = _store_df_record(
            code="eval('1+1')\nresult = 1",
            outputs=["out"],
        )

        _replay_run_code(rec, store)

        # Should not have stored anything — validation failed
        assert not store.has("out")


class TestReplayRender:
    """Tests for _replay_render."""

    def test_produces_figure(self, tmp_path):
        store = DataStore(tmp_path / "data")
        df = _make_ts_dataframe(10, ["value"])
        store.put(DataEntry(label="test.data", data=df, source="cdf"))

        rec = _render_record(data_labels=["test.data"])
        fig = _replay_render(rec, store)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_missing_label_raises(self, tmp_path):
        store = DataStore(tmp_path / "data")
        rec = _render_record(data_labels=["missing.label"])

        with pytest.raises(ValueError, match="not found in replay store"):
            _replay_render(rec, store)

    def test_missing_figure_json_raises(self, tmp_path):
        store = DataStore(tmp_path / "data")
        rec = {
            "id": "op_999",
            "tool": "render_plotly_json",
            "status": "success",
            "inputs": [],
            "outputs": [],
            "args": {},
        }

        with pytest.raises(ValueError, match="missing 'figure_json'"):
            _replay_render(rec, store)


# ---------------------------------------------------------------------------
# Tier 2: Integration tests
# ---------------------------------------------------------------------------


class TestReplayPipeline:
    """Integration tests for replay_pipeline."""

    @patch("scripts.replay.fetch_data")
    def test_full_pipeline_fetch_compute_render(self, mock_fetch):
        """fetch → run_code → render_plotly_json end-to-end."""
        mock_fetch.return_value = _make_fetch_result(20, ["value"])

        records = [
            _fetch_record(op_id="op_001", outputs=["src.field"]),
            _custom_op_record(
                op_id="op_002",
                code="import pandas as pd\ndf = pd.read_parquet('src.field.parquet')\nresult = df.abs()",
                inputs=["src.field"],
                outputs=["abs.field"],
            ),
            _render_record(
                op_id="op_003",
                inputs=["abs.field"],
                data_labels=["abs.field"],
            ),
        ]

        result = replay_pipeline(records)

        assert result.steps_completed == 3
        assert result.steps_total == 3
        assert len(result.errors) == 0
        assert result.store.has("src.field")
        assert result.store.has("abs.field")
        assert isinstance(result.figure, go.Figure)

    @patch("scripts.replay.fetch_data")
    def test_error_cascades_to_dependents(self, mock_fetch):
        """Fetch fails → dependent custom_op fails → independent branch succeeds."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Network error")
            return _make_fetch_result(5, ["value"])

        mock_fetch.side_effect = side_effect

        records = [
            # Branch A: fetch fails
            _fetch_record(op_id="op_001", outputs=["branch_a.data"],
                          dataset_id="FAIL_DS", parameter_id="X"),
            # Branch A dependent: should fail (missing source)
            _custom_op_record(
                op_id="op_002",
                code="import pandas as pd\ndf = pd.read_parquet('branch_a.data.parquet')\nresult = df.abs()",
                inputs=["branch_a.data"],
                outputs=["branch_a.computed"],
            ),
            # Branch B: independent fetch succeeds
            _fetch_record(op_id="op_003", outputs=["branch_b.data"],
                          dataset_id="AC_H2_MFI", parameter_id="BGSEc"),
        ]

        result = replay_pipeline(records)

        assert result.steps_completed == 1  # only branch B fetch
        assert len(result.errors) == 2  # branch A fetch + branch A compute (missing input file)
        assert result.store.has("branch_b.data")
        assert not result.store.has("branch_a.data")

    @patch("scripts.replay.fetch_data")
    def test_progress_callback(self, mock_fetch):
        mock_fetch.return_value = _make_fetch_result()

        calls = []
        def cb(step, total, tool):
            calls.append((step, total, tool))

        records = [
            _fetch_record(op_id="op_001"),
            _fetch_record(op_id="op_002", dataset_id="WIND_MFI",
                          parameter_id="BF1", outputs=["WIND_MFI.BF1"]),
        ]

        replay_pipeline(records, progress_callback=cb)

        assert len(calls) == 2
        assert calls[0] == (1, 2, "fetch_data")
        assert calls[1] == (2, 2, "fetch_data")

    def test_empty_records(self):
        result = replay_pipeline([])

        assert result.steps_completed == 0
        assert result.steps_total == 0
        assert len(result.errors) == 0
        assert result.figure is None
        assert len(result.store) == 0

    @patch("scripts.replay.fetch_data")
    def test_manage_plot_skipped(self, mock_fetch):
        """manage_plot records are skipped without error."""
        mock_fetch.return_value = _make_fetch_result()

        records = [
            _fetch_record(op_id="op_001"),
            {
                "id": "op_002",
                "tool": "manage_plot",
                "status": "success",
                "inputs": [],
                "outputs": [],
                "args": {"action": "export", "filepath": "/tmp/test.png"},
            },
        ]

        result = replay_pipeline(records)

        assert result.steps_completed == 2
        assert len(result.errors) == 0

    @patch("scripts.replay.fetch_data")
    def test_unknown_tool_skipped(self, mock_fetch):
        """Unknown tools are skipped without error."""
        mock_fetch.return_value = _make_fetch_result()

        records = [
            _fetch_record(op_id="op_001"),
            {
                "id": "op_002",
                "tool": "some_future_tool",
                "status": "success",
                "inputs": [],
                "outputs": [],
                "args": {},
            },
        ]

        result = replay_pipeline(records)

        assert result.steps_completed == 2
        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Tier 3: replay_session loading
# ---------------------------------------------------------------------------


class TestReplaySession:
    """Tests for replay_session."""

    @patch("scripts.replay.fetch_data")
    def test_loads_from_temp_dir(self, mock_fetch):
        mock_fetch.return_value = _make_fetch_result()

        records = [
            _fetch_record(op_id="op_001"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session_id = "test-session-001"
            session_path = session_dir / session_id
            session_path.mkdir()
            ops_file = session_path / "operations.json"
            with open(ops_file, "w") as f:
                json.dump(records, f)

            result = replay_session(
                session_id, session_dir=session_dir,
            )

        assert result.steps_completed >= 1
        assert result.store.has("AC_H2_MFI.BGSEc")

    def test_missing_operations_json_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session_id = "nonexistent"
            (session_dir / session_id).mkdir()

            with pytest.raises(FileNotFoundError, match="operations.json"):
                replay_session(session_id, session_dir=session_dir)


class TestReplayState:
    """Tests for replay_state."""

    @patch("scripts.replay.fetch_data")
    def test_replays_only_selected_state(self, mock_fetch):
        """replay_state replays only the ops for the selected render."""
        mock_fetch.return_value = _make_fetch_result(10, ["value"])

        # Two fetch+render pairs, each overwriting the same label
        records = [
            {
                "id": "op_001", "tool": "fetch_data", "status": "success",
                "inputs": [], "outputs": ["Bx"],
                "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc",
                         "time_range_resolved": "2024-01-01 to 2024-01-31"},
                "input_producers": {},
            },
            {
                "id": "op_002", "tool": "render_plotly_json", "status": "success",
                "inputs": ["Bx"], "outputs": [],
                "args": {"figure_json": {
                    "data": [{"data_label": "Bx", "type": "scatter"}],
                    "layout": {"title": "Jan"},
                }},
                "input_producers": {"Bx": "op_001"},
            },
            {
                "id": "op_003", "tool": "fetch_data", "status": "success",
                "inputs": [], "outputs": ["Bx"],
                "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc",
                         "time_range_resolved": "2024-02-01 to 2024-02-28"},
                "input_producers": {},
            },
            {
                "id": "op_004", "tool": "render_plotly_json", "status": "success",
                "inputs": ["Bx"], "outputs": [],
                "args": {"figure_json": {
                    "data": [{"data_label": "Bx", "type": "scatter"}],
                    "layout": {"title": "Feb"},
                }},
                "input_producers": {"Bx": "op_003"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            session_id = "state-test"
            session_path = session_dir / session_id
            session_path.mkdir()
            with open(session_path / "operations.json", "w") as f:
                json.dump(records, f)

            result = replay_state(
                session_id, "op_004",
                session_dir=session_dir, use_cache=False,
            )

        # Only the second fetch should have been called (op_003)
        assert mock_fetch.call_count == 1
        call_args = mock_fetch.call_args[0]
        assert call_args[2] == "2024-02-01"  # time_min
        assert call_args[3] == "2024-02-28"  # time_max

        assert result.steps_completed == 2  # fetch + render
        assert result.figure is not None
