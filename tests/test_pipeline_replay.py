"""Tests for pipeline replay action."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from data_ops.dag import PipelineDAG


def _make_dag_with_fetch_and_compute(session_dir: Path) -> PipelineDAG:
    """Create a small DAG: fetch -> compute -> render."""
    dag = PipelineDAG(session_dir=session_dir)
    dag.add_node(
        op_id="op_000", tool="fetch_data", agent="orchestrator",
        args={"dataset": "AC_H2_MFI", "start_time": "2025-01-01", "end_time": "2025-01-02"},
        inputs=[], outputs={"AC_H2_MFI.BGSEc": "magnetic field data"},
        status="success",
    )
    dag.add_node(
        op_id="op_001", tool="run_code", agent="data_ops",
        args={"code": "magnitude = np.sqrt(x**2)", "description": "compute magnitude"},
        inputs=["AC_H2_MFI.BGSEc"], outputs={"Bmag": "field magnitude"},
        status="success",
    )
    dag.add_node(
        op_id="op_002", tool="render_plotly_json", agent="viz_plotly",
        args={"description": "time series plot"},
        inputs=["Bmag"], outputs={},
        status="success",
    )
    dag.save()
    return dag


class TestPipelineReplay:
    def test_replay_loads_past_session_dag(self):
        """Replay action loads pipeline.json from a past session and replays it."""
        with tempfile.TemporaryDirectory() as past_dir, \
             tempfile.TemporaryDirectory() as current_dir:

            past_path = Path(past_dir)
            dag = _make_dag_with_fetch_and_compute(past_path)

            ctx = MagicMock()
            ctx.store = MagicMock()
            ctx.session_dir = Path(current_dir)
            ctx.dag = PipelineDAG(session_dir=Path(current_dir))

            from agent.tool_handlers.pipeline import handle_pipeline

            class FakeReplayEngine:
                def __init__(self, dag, ctx):
                    self.dag = dag
                def replay(self, target_op_id):
                    from data_ops.replay import ReplayResult
                    return ReplayResult(steps_completed=3, steps_total=3)

            with patch("agent.session.SessionManager") as MockSM, \
                 patch("data_ops.replay.ReplayEngine", FakeReplayEngine):
                MockSM.return_value.base_dir = past_path.parent
                result = handle_pipeline(ctx, {
                    "action": "replay",
                    "session_id": past_path.name,
                })

            assert result["status"] == "success"
            assert result["nodes_replayed"] == 3

    def test_replay_with_specific_op_id(self):
        """Replay with op_id replays only that node's subgraph."""
        with tempfile.TemporaryDirectory() as past_dir, \
             tempfile.TemporaryDirectory() as current_dir:

            past_path = Path(past_dir)
            dag = _make_dag_with_fetch_and_compute(past_path)

            ctx = MagicMock()
            ctx.store = MagicMock()
            ctx.session_dir = Path(current_dir)
            ctx.dag = PipelineDAG(session_dir=Path(current_dir))

            from agent.tool_handlers.pipeline import handle_pipeline

            replayed_targets = []

            class FakeReplayEngine:
                def __init__(self, dag, ctx):
                    self.dag = dag
                def replay(self, target_op_id):
                    replayed_targets.append(target_op_id)
                    from data_ops.replay import ReplayResult
                    return ReplayResult(steps_completed=2, steps_total=2)

            with patch("agent.session.SessionManager") as MockSM, \
                 patch("data_ops.replay.ReplayEngine", FakeReplayEngine):
                MockSM.return_value.base_dir = past_path.parent
                result = handle_pipeline(ctx, {
                    "action": "replay",
                    "session_id": past_path.name,
                    "op_id": "op_001",  # only compute node subgraph
                })

            # Should replay only the specific target, not all leaves
            assert replayed_targets == ["op_001"]
            assert result["nodes_replayed"] == 2

    def test_replay_missing_session_returns_error(self):
        """Replay with nonexistent session_id returns error."""
        ctx = MagicMock()

        from agent.tool_handlers.pipeline import handle_pipeline

        with patch("agent.session.SessionManager") as MockSM:
            MockSM.return_value.base_dir = Path("/tmp/nonexistent_sessions")
            result = handle_pipeline(ctx, {
                "action": "replay",
                "session_id": "no_such_session",
            })

        assert result["status"] == "error"
        assert "No pipeline" in result["message"]

    def test_replay_requires_session_id(self):
        """Replay without session_id returns error."""
        ctx = MagicMock()

        from agent.tool_handlers.pipeline import handle_pipeline
        result = handle_pipeline(ctx, {"action": "replay"})

        assert result["status"] == "error"
        assert "session_id" in result["message"].lower()

    def test_replay_patches_time_range(self):
        """Replay with time_range overrides fetch node args."""
        with tempfile.TemporaryDirectory() as past_dir, \
             tempfile.TemporaryDirectory() as current_dir:

            past_path = Path(past_dir)
            dag = _make_dag_with_fetch_and_compute(past_path)

            ctx = MagicMock()
            ctx.store = MagicMock()
            ctx.session_dir = Path(current_dir)
            ctx.dag = PipelineDAG(session_dir=Path(current_dir))

            from agent.tool_handlers.pipeline import handle_pipeline

            # Mock ReplayEngine to capture the DAG it receives
            captured_dags = []

            class FakeReplayEngine:
                def __init__(self, dag, ctx):
                    captured_dags.append(dag)
                    self.dag = dag
                def replay(self, target_op_id):
                    from data_ops.replay import ReplayResult
                    return ReplayResult(steps_completed=1, steps_total=1)

            with patch("agent.session.SessionManager") as MockSM, \
                 patch("data_ops.replay.ReplayEngine", FakeReplayEngine):
                MockSM.return_value.base_dir = past_path.parent
                result = handle_pipeline(ctx, {
                    "action": "replay",
                    "session_id": past_path.name,
                    "time_range": {"start": "2026-06-01", "end": "2026-06-02"},
                })

            # Verify fetch node args were patched
            assert len(captured_dags) == 1
            patched_dag = captured_dags[0]
            fetch_node = patched_dag.node("op_000")
            assert fetch_node["args"]["start_time"] == "2026-06-01"
            assert fetch_node["args"]["end_time"] == "2026-06-02"

    def test_replay_records_summary_node_in_current_dag(self):
        """After replay, a pipeline_replay summary node is added to current DAG."""
        with tempfile.TemporaryDirectory() as past_dir, \
             tempfile.TemporaryDirectory() as current_dir:

            past_path = Path(past_dir)
            dag = _make_dag_with_fetch_and_compute(past_path)

            current_dag = PipelineDAG(session_dir=Path(current_dir))
            ctx = MagicMock()
            ctx.store = MagicMock()
            ctx.session_dir = Path(current_dir)
            ctx.dag = current_dag

            from agent.tool_handlers.pipeline import handle_pipeline

            class FakeReplayEngine:
                def __init__(self, dag, ctx):
                    self.dag = dag
                def replay(self, target_op_id):
                    from data_ops.replay import ReplayResult
                    return ReplayResult(steps_completed=3, steps_total=3)

            with patch("agent.session.SessionManager") as MockSM, \
                 patch("data_ops.replay.ReplayEngine", FakeReplayEngine):
                MockSM.return_value.base_dir = past_path.parent
                result = handle_pipeline(ctx, {
                    "action": "replay",
                    "session_id": past_path.name,
                })

            # Current session DAG should have a summary node
            assert current_dag.node_count() == 1
            node = current_dag.node("op_000")
            assert node["tool"] == "pipeline_replay"

    def test_old_stub_actions_removed(self):
        """Old stub actions return 'Unknown action', not 'not yet reimplemented'."""
        ctx = MagicMock()

        from agent.tool_handlers.pipeline import handle_pipeline
        for action in ("save", "run", "search", "modify", "execute"):
            result = handle_pipeline(ctx, {"action": action})
            assert result["status"] == "error"
            assert "Unknown action" in result["message"]
