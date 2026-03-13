"""Tests for agent/session_persistence.py and agent/tool_handlers/planning.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# find_latest_png (static helper, no orch needed)
# ---------------------------------------------------------------------------


def test_find_latest_png_empty_dir():
    from agent.session_persistence import find_latest_png

    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_latest_png(Path(tmpdir))
    assert result is None


def test_find_latest_png_returns_newest(tmp_path):
    from agent.session_persistence import find_latest_png

    out_dir = tmp_path / "plotly_outputs"
    out_dir.mkdir()
    old_file = out_dir / "old.png"
    new_file = out_dir / "new.png"
    old_file.write_bytes(b"PNG1")
    new_file.write_bytes(b"PNG2")
    # Force mtime ordering
    import time
    time.sleep(0.01)
    new_file.touch()

    result = find_latest_png(tmp_path)
    assert result == new_file


def test_find_latest_png_skips_empty_files(tmp_path):
    from agent.session_persistence import find_latest_png

    out_dir = tmp_path / "mpl_outputs"
    out_dir.mkdir()
    empty_file = out_dir / "empty.png"
    real_file = out_dir / "real.png"
    empty_file.write_bytes(b"")
    real_file.write_bytes(b"PNG")

    result = find_latest_png(tmp_path)
    assert result == real_file


def test_find_latest_png_ignores_non_png(tmp_path):
    from agent.session_persistence import find_latest_png

    out_dir = tmp_path / "plotly_outputs"
    out_dir.mkdir()
    (out_dir / "fig.html").write_bytes(b"<html>")

    result = find_latest_png(tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# get_plot_status
# ---------------------------------------------------------------------------


def _make_mock_renderer(figure=None, panel_count=0, trace_labels=None):
    renderer = MagicMock()
    renderer.get_figure.return_value = figure
    renderer._panel_count = panel_count
    renderer._trace_labels = trace_labels or []
    return renderer


def _make_orch_for_plot_status(figure=None, deferred=None, panel_count=0, trace_labels=None):
    """Create a mock that looks like a SessionContext to get_plot_status.

    The functions duck-type on ``hasattr(ctx, 'deferred_figure_state')`` to
    distinguish SessionContext from OrchestratorAgent.
    """
    ctx = SimpleNamespace()
    ctx.renderer = _make_mock_renderer(figure=figure, panel_count=panel_count, trace_labels=trace_labels)
    ctx.deferred_figure_state = deferred
    return ctx


def test_get_plot_status_none():
    from agent.session_persistence import get_plot_status

    orch = _make_orch_for_plot_status(figure=None, deferred=None)
    status = get_plot_status(orch)
    assert status == {"state": "none", "panel_count": 0, "traces": []}


def test_get_plot_status_active():
    from agent.session_persistence import get_plot_status

    mock_fig = MagicMock()
    orch = _make_orch_for_plot_status(figure=mock_fig, panel_count=3, trace_labels=["A", "B"])
    status = get_plot_status(orch)
    assert status["state"] == "active"
    assert status["panel_count"] == 3
    assert status["traces"] == ["A", "B"]


def test_get_plot_status_restorable():
    from agent.session_persistence import get_plot_status

    deferred = {"panel_count": 2, "trace_labels": ["X"], "last_fig_json": None}
    orch = _make_orch_for_plot_status(figure=None, deferred=deferred)
    status = get_plot_status(orch)
    assert status["state"] == "restorable"
    assert status["panel_count"] == 2
    assert status["traces"] == ["X"]


# ---------------------------------------------------------------------------
# get_data_status
# ---------------------------------------------------------------------------


def test_get_data_status_no_store():
    from agent.session_persistence import get_data_status

    ctx = SimpleNamespace(deferred_figure_state=None, store=None)
    result = get_data_status(ctx)
    assert result == {"total_entries": 0, "loaded": 0, "deferred": 0}


def test_get_data_status_with_store():
    from agent.session_persistence import get_data_status
    import threading

    store = SimpleNamespace()
    store._lock = threading.Lock()
    store._ids = {"a": 1, "b": 2, "c": 3}
    store._cache = {"a": "data_a"}

    ctx = SimpleNamespace(deferred_figure_state=None, store=store)
    result = get_data_status(ctx)
    assert result["total_entries"] == 3
    assert result["loaded"] == 1
    assert result["deferred"] == 2


# ---------------------------------------------------------------------------
# handle_plan
# ---------------------------------------------------------------------------


def _make_ctx_for_plan():
    """Create a minimal mock SessionContext for plan tests."""
    from agent.tool_caller import OrchestratorState
    orch = OrchestratorState()
    ctx = SimpleNamespace()
    ctx.agent_state = {"orchestrator": orch}
    ctx.session_dir = None  # not used directly in plan tests (save_plan is mocked)
    ctx.event_bus = MagicMock()
    ctx.deferred_figure_state = None  # duck-type marker for SessionContext
    return ctx


def _orch(ctx):
    """Get OrchestratorState from ctx."""
    return ctx.agent_state["orchestrator"]


def test_handle_plan_check_no_plan():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()
    with patch("agent.tool_handlers.planning._persist_plan"):
        result = handle_plan(ctx, {"action": "check"})
    assert result["status"] == "error"
    assert "No active plan" in result["message"]


def test_handle_plan_create():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()

    tasks = [
        {"description": "Fetch data", "instruction": "Call fetch tool"},
        {"description": "Plot data", "instruction": "Call render tool"},
    ]
    with patch("agent.tool_handlers.planning._persist_plan") as mock_save:
        result = handle_plan(ctx, {
            "action": "create",
            "summary": "Two step plan",
            "reasoning": "Need data then plot",
            "tasks": tasks,
        })

    assert result["status"] == "success"
    assert result["task_count"] == 2
    assert _orch(ctx).current_plan is not None
    assert len(_orch(ctx).current_plan["tasks"]) == 2
    assert _orch(ctx).current_plan["tasks"][0]["status"] == "pending"
    mock_save.assert_called_once()


def test_handle_plan_create_empty_tasks():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()

    result = handle_plan(ctx, {"action": "create", "tasks": []})
    assert result["status"] == "error"
    assert "tasks" in result["message"]


def test_handle_plan_check_with_plan():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()
    _orch(ctx).current_plan = {"summary": "test", "tasks": [{"description": "do it", "status": "pending"}]}

    result = handle_plan(ctx, {"action": "check"})
    assert result["status"] == "success"
    assert result["plan"] == _orch(ctx).current_plan


def test_handle_plan_update():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()
    _orch(ctx).current_plan = {
        "summary": "test",
        "tasks": [
            {"description": "step 0", "status": "pending"},
            {"description": "step 1", "status": "pending"},
        ],
    }

    with patch("agent.tool_handlers.planning._persist_plan") as mock_save:
        result = handle_plan(ctx, {"action": "update", "step": 0, "status": "done"})
    assert result["status"] == "success"
    assert result["step"] == 0
    assert result["new_status"] == "done"
    assert _orch(ctx).current_plan["tasks"][0]["status"] == "done"
    mock_save.assert_called_once()


def test_handle_plan_update_no_plan():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()

    result = handle_plan(ctx, {"action": "update", "step": 0, "status": "done"})
    assert result["status"] == "error"


def test_handle_plan_update_invalid_step():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()
    _orch(ctx).current_plan = {"summary": "test", "tasks": [{"description": "step 0", "status": "pending"}]}

    result = handle_plan(ctx, {"action": "update", "step": 99, "status": "done"})
    assert result["status"] == "error"
    assert "Invalid step" in result["message"]


def test_handle_plan_drop():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()
    _orch(ctx).current_plan = {"summary": "test", "tasks": []}

    with patch("agent.tool_handlers.planning._persist_plan") as mock_save:
        result = handle_plan(ctx, {"action": "drop"})
    assert result["status"] == "success"
    assert result["dropped"] is True
    assert _orch(ctx).current_plan is None
    mock_save.assert_called_once()


def test_handle_plan_drop_no_plan():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()

    with patch("agent.tool_handlers.planning._persist_plan"):
        result = handle_plan(ctx, {"action": "drop"})
    assert result["status"] == "success"
    assert result["dropped"] is False


def test_handle_plan_unknown_action():
    from agent.tool_handlers.planning import handle_plan

    ctx = _make_ctx_for_plan()

    result = handle_plan(ctx, {"action": "bogus"})
    assert result["status"] == "error"
    assert "bogus" in result["message"]


# ---------------------------------------------------------------------------
# plan_path / save_plan / load_plan
# ---------------------------------------------------------------------------


def _make_ctx_for_persistence(tmp_path=None):
    """Create a minimal SimpleNamespace ctx with agent_state for save/load tests."""
    from agent.tool_caller import OrchestratorState
    orch = OrchestratorState()
    ctx = SimpleNamespace(
        deferred_figure_state=None,
        session_dir=tmp_path,
        agent_state={"orchestrator": orch},
    )
    return ctx


def test_save_and_load_plan(tmp_path):
    from agent.session_persistence import save_plan, load_plan

    ctx = _make_ctx_for_persistence(tmp_path)
    _orch(ctx).current_plan = {"summary": "my plan", "tasks": [{"description": "a", "status": "pending"}]}

    save_plan(ctx)

    # Verify file written
    plan_file = tmp_path / "plan.json"
    assert plan_file.exists()
    data = json.loads(plan_file.read_text())
    assert data["summary"] == "my plan"

    # Load into fresh ctx
    ctx2 = _make_ctx_for_persistence(tmp_path)
    load_plan(ctx2)
    assert _orch(ctx2).current_plan is not None
    assert _orch(ctx2).current_plan["summary"] == "my plan"


def test_save_plan_none_deletes_file(tmp_path):
    from agent.session_persistence import save_plan, load_plan

    ctx = _make_ctx_for_persistence(tmp_path)
    _orch(ctx).current_plan = {"summary": "x", "tasks": []}
    save_plan(ctx)
    assert (tmp_path / "plan.json").exists()

    # Now save None to delete
    _orch(ctx).current_plan = None
    save_plan(ctx)
    assert not (tmp_path / "plan.json").exists()


def test_load_plan_missing_file(tmp_path):
    from agent.session_persistence import load_plan

    ctx = _make_ctx_for_persistence(tmp_path)
    _orch(ctx).current_plan = "should be cleared"
    load_plan(ctx)
    assert _orch(ctx).current_plan is None


def test_load_plan_corrupt_file(tmp_path):
    from agent.session_persistence import load_plan

    plan_file = tmp_path / "plan.json"
    plan_file.write_text("not valid json{{{")

    ctx = _make_ctx_for_persistence(tmp_path)
    _orch(ctx).current_plan = "old"
    load_plan(ctx)
    assert _orch(ctx).current_plan is None
