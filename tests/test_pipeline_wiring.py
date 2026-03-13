"""Tests for pipeline DAG wiring into production sessions."""

from unittest.mock import patch, MagicMock
from data_ops.dag import PipelineDAG
from agent.event_bus import EventBus, PipelineDAGListener, DATA_COMPUTED


class TestPipelineDAGListenerWiring:
    def test_listener_subscribed_via_event_bus(self):
        """PipelineDAGListener routes pipeline events to DAG when subscribed.

        Tests the wiring pattern used by Session.start() without instantiating
        the full session (avoids heavy dependencies like MemoryStore, EurekaStore).
        """
        dag = PipelineDAG()
        bus = EventBus()

        # This is the exact wiring that Session.start() should add:
        listener = PipelineDAGListener(lambda: dag)
        bus.subscribe(listener)

        bus.emit(
            DATA_COMPUTED,
            agent="test",
            msg="test",
            data={
                "tool": "run_code",
                "args": {"code": "x=1", "description": "test"},
                "inputs": [],
                "outputs": {"test_label": "test output"},
                "status": "success",
            },
        )
        assert dag.node_count() == 1
        node = dag.node("op_000")
        assert node["tool"] == "run_code"
        assert node["status"] == "success"

    def test_session_start_imports_pipeline_listener(self):
        """session_lifecycle.py imports and uses PipelineDAGListener."""
        import inspect
        from agent import session_lifecycle

        source = inspect.getsource(session_lifecycle.Session.start)
        assert "PipelineDAGListener" in source, \
            "Session.start() must subscribe PipelineDAGListener"


class TestRunCodeEmitsPipelineEvent:
    def test_successful_run_code_emits_data_computed(self):
        """handle_run_code emits DATA_COMPUTED on success."""
        from agent.event_bus import EventBus, DATA_COMPUTED

        bus = EventBus()
        events_captured = []
        bus.subscribe(lambda e: events_captured.append(e))

        ctx = MagicMock()
        ctx.event_bus = bus
        ctx.store = MagicMock()
        ctx.store.get.return_value = None  # no inputs needed
        ctx.session_dir = None  # use temp sandbox
        ctx.asset_registry = None

        caller = MagicMock()
        caller.agent_id = "test_agent"

        tool_args = {
            "code": "result = 42",
            "inputs": [],
            "outputs": {},
            "description": "test computation",
        }

        with patch("agent.tool_handlers.sandbox.validate_code_blocklist", return_value=[]), \
             patch("agent.tool_handlers.sandbox.execute_sandboxed", return_value=("", {})):
            from agent.tool_handlers.sandbox import handle_run_code
            result = handle_run_code(ctx, tool_args, caller)

        pipeline_events = [e for e in events_captured if e.type == DATA_COMPUTED]
        assert len(pipeline_events) == 1
        assert pipeline_events[0].data["tool"] == "run_code"
        assert pipeline_events[0].data["status"] == "success"
        assert pipeline_events[0].agent == "test_agent"

    def test_failed_run_code_does_not_emit(self):
        """handle_run_code does NOT emit DATA_COMPUTED when code is blocked."""
        from agent.event_bus import EventBus, DATA_COMPUTED

        bus = EventBus()
        events_captured = []
        bus.subscribe(lambda e: events_captured.append(e))

        ctx = MagicMock()
        ctx.event_bus = bus

        tool_args = {
            "code": "import os; os.system('rm -rf /')",
            "inputs": [],
            "outputs": {},
            "description": "evil",
        }

        with patch("agent.tool_handlers.sandbox.validate_code_blocklist",
                    return_value=["blocked: os.system"]):
            from agent.tool_handlers.sandbox import handle_run_code
            result = handle_run_code(ctx, tool_args, MagicMock())

        pipeline_events = [e for e in events_captured if e.type == DATA_COMPUTED]
        assert len(pipeline_events) == 0
        assert result["status"] == "error"


class TestPlotlyEmitsRenderEvent:
    def test_successful_render_emits_render_executed(self):
        """handle_render_plotly_json emits RENDER_EXECUTED on success."""
        from agent.event_bus import EventBus, RENDER_EXECUTED
        from data_ops.store import DataEntry
        import pandas as pd

        bus = EventBus()
        events_captured = []
        bus.subscribe(lambda e: events_captured.append(e))

        ctx = MagicMock()
        ctx.event_bus = bus
        ctx.session_dir = MagicMock()
        ctx.session_dir.__truediv__ = MagicMock(return_value=MagicMock())
        ctx.renderer.render_plotly_json.return_value = {"status": "success", "panels": 1}
        ctx.renderer.get_figure.return_value = None  # skip PNG caching

        # Mock store entry
        entry = MagicMock(spec=DataEntry)
        entry.data = pd.DataFrame({"x": [1, 2, 3]})

        with patch("agent.tool_handlers.visualization.resolve_entry",
                    return_value=(entry, None)):
            from agent.tool_handlers.visualization import handle_render_plotly_json
            caller = MagicMock()
            caller.agent_id = "VizAgent[Plotly]"

            result = handle_render_plotly_json(ctx, {
                "figure_json": {
                    "data": [{"type": "scatter", "data_label": "test.label"}],
                    "layout": {},
                }
            }, caller)

        render_events = [e for e in events_captured if e.type == RENDER_EXECUTED]
        assert len(render_events) == 1
        assert render_events[0].data["tool"] == "render_plotly_json"
        assert render_events[0].data["inputs"] == ["test.label"]
        assert render_events[0].data["outputs"] == {}  # sink

    def test_plotly_handler_safe_without_eureka_hooks(self):
        """Plotly handler does not crash when ctx has no eureka_hooks (ReplayContext)."""
        from data_ops.store import DataEntry
        import pandas as pd
        import tempfile
        from pathlib import Path

        # Build a minimal mock that mimics ReplayContext:
        # has event_bus=None, no eureka_hooks attr, but has renderer + session_dir
        ctx = MagicMock()
        ctx.event_bus = None  # ReplayContext returns None
        # Delete eureka_hooks so hasattr/getattr returns False/None
        del ctx.eureka_hooks

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx.session_dir = Path(tmpdir)
            ctx.renderer.render_plotly_json.return_value = {
                "status": "success", "panels": 1
            }
            ctx.renderer.get_figure.return_value = None

            entry = MagicMock(spec=DataEntry)
            entry.data = pd.DataFrame({"x": [1, 2, 3]})

            with patch("agent.tool_handlers.visualization.resolve_entry",
                        return_value=(entry, None)):
                from agent.tool_handlers.visualization import handle_render_plotly_json

                # Should not raise AttributeError on eureka_hooks
                result = handle_render_plotly_json(ctx, {
                    "figure_json": {
                        "data": [{"type": "scatter", "data_label": "test.label"}],
                        "layout": {},
                    }
                }, None)

            # Should complete without AttributeError
            assert result["status"] == "success"
