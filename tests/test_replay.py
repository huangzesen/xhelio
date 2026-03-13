"""Tests for the pipeline replay engine."""

import pytest
from data_ops.dag import PipelineDAG
from data_ops.replay import ReplayEngine, ReplayResult, StepResult
from data_ops.store import DataStore
from agent.tool_context import ReplayContext


def _build_simple_dag():
    """Build a DAG: op_000 (run_code) -> op_001 (render_plotly_json)."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="run_code", agent="orchestrator",
        args={"code": "x = 1", "inputs": [], "outputs": {"result": "x"}},
        inputs=[], outputs={"result": "result_id"}, status="success",
    )
    dag.add_node(
        "op_001", tool="render_plotly_json", agent="orchestrator",
        args={"figure_json": {"data": [{"data_label": "result"}]}},
        inputs=["result"], outputs={}, status="success",
    )
    return dag


def test_replay_result_shape():
    """ReplayResult should have frontend-compatible fields."""
    r = ReplayResult(
        steps_completed=1, steps_total=2, errors=[], figure=None,
    )
    assert r.steps_completed == 1
    assert r.steps_total == 2
    assert r.errors == []
    assert r.figure is None


def test_replay_skips_unknown_tools(tmp_path):
    """Replay should skip tools not in TOOL_REGISTRY (silent skip, not error)."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="nonexistent_tool", agent="test",
        args={}, inputs=[], outputs={}, status="success",
    )
    ctx = ReplayContext(store=DataStore(tmp_path / "data"))
    engine = ReplayEngine(dag, ctx)
    result = engine.replay("op_000")
    assert result.steps_completed == 0
    assert result.steps_total == 1
    assert len(result.errors) == 0  # skipped tools are not errors


def test_replay_skips_failed_original(tmp_path):
    """Replay should skip nodes whose original status was not 'success'."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="run_code", agent="test",
        args={}, inputs=[], outputs={}, status="error", error="original failed",
    )
    ctx = ReplayContext(store=DataStore(tmp_path / "data"))
    engine = ReplayEngine(dag, ctx)
    result = engine.replay("op_000")
    assert result.steps_completed == 0
    assert result.steps_total == 1
    assert len(result.errors) == 0


def test_replay_executes_subgraph_in_order(tmp_path):
    """Replay extracts subgraph and executes in topological order."""
    dag = _build_simple_dag()
    from agent.tool_handlers import TOOL_REGISTRY

    call_log = []
    def mock_run_code(ctx, tool_args):
        call_log.append("run_code")
        return {"status": "success"}

    def mock_render(ctx, tool_args):
        call_log.append("render_plotly_json")
        return {"status": "success", "figure": {"data": [], "layout": {}}}

    old_rc = TOOL_REGISTRY.get("run_code")
    old_rp = TOOL_REGISTRY.get("render_plotly_json")
    TOOL_REGISTRY["run_code"] = mock_run_code
    TOOL_REGISTRY["render_plotly_json"] = mock_render

    try:
        ctx = ReplayContext(store=DataStore(tmp_path / "data"))
        engine = ReplayEngine(dag, ctx)
        result = engine.replay("op_001")

        assert result.steps_completed == 2
        assert result.steps_total == 2
        assert call_log == ["run_code", "render_plotly_json"]
        assert result.figure is not None
    finally:
        if old_rc:
            TOOL_REGISTRY["run_code"] = old_rc
        else:
            TOOL_REGISTRY.pop("run_code", None)
        if old_rp:
            TOOL_REGISTRY["render_plotly_json"] = old_rp
        else:
            TOOL_REGISTRY.pop("render_plotly_json", None)
