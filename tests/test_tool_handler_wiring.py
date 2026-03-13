"""Verify tool handlers use SessionContext public attributes correctly.

Each test constructs a minimal SessionContext with mocked services and
calls a handler — the assertion is that no AttributeError is raised
(i.e., handlers don't reference dead orchestrator private attributes).
"""
import pytest
from unittest.mock import MagicMock
from agent.session_context import SessionContext
from agent.tool_caller import OrchestratorState


def _make_ctx(**overrides):
    """Build a minimal SessionContext with mocked services."""
    store = MagicMock()
    store.list_entries.return_value = []
    dag = MagicMock()
    event_bus = MagicMock()
    service = MagicMock()
    renderer = MagicMock()
    delegation = MagicMock()
    work_tracker = MagicMock()

    orch_state = OrchestratorState(
        current_plan=None,
        event_feed=MagicMock(),
        ctx_tracker=MagicMock(),
    )

    asset_registry = MagicMock()
    asset_registry.list_assets_enriched.return_value = {
        "summary": {"data": 0, "files": 0, "figures": 0},
        "data": [], "files": [], "figures": [],
    }

    ctx = SessionContext(
        store=store, dag=dag, event_bus=event_bus,
        service=service, renderer=renderer,
        delegation=delegation, work_tracker=work_tracker,
        session_dir=None, session_id="test",
        model_name="test-model",
        agent_state={"orchestrator": orch_state},
        asset_registry=asset_registry,
    )
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _orch(ctx) -> OrchestratorState:
    """Get OrchestratorState from ctx.agent_state."""
    return ctx.agent_state.get("orchestrator", OrchestratorState())


class TestHandlerWiring:
    """Every handler must be callable with a SessionContext without AttributeError."""

    def test_plan_create(self):
        ctx = _make_ctx()
        from agent.tool_handlers.planning import handle_plan
        result = handle_plan(ctx, {
            "action": "create",
            "tasks": [{"description": "step 1"}],
        })
        assert result["status"] == "success"
        assert _orch(ctx).current_plan is not None

    def test_plan_check_no_plan(self):
        ctx = _make_ctx()
        from agent.tool_handlers.planning import handle_plan
        result = handle_plan(ctx, {"action": "check"})
        assert result["status"] == "error"

    def test_plan_update(self):
        ctx = _make_ctx()
        _orch(ctx).current_plan = {
            "summary": "",
            "reasoning": "",
            "tasks": [{"description": "step 1", "status": "pending"}],
        }
        from agent.tool_handlers.planning import handle_plan
        result = handle_plan(ctx, {"action": "update", "step": 0, "status": "done"})
        assert result["status"] == "success"

    def test_plan_drop(self):
        ctx = _make_ctx()
        _orch(ctx).current_plan = {"summary": "", "tasks": []}
        from agent.tool_handlers.planning import handle_plan
        result = handle_plan(ctx, {"action": "drop"})
        assert result["status"] == "success"
        assert _orch(ctx).current_plan is None

    def test_manage_workers_list(self):
        ctx = _make_ctx()
        ctx.work_tracker.list_active.return_value = []
        from agent.tool_handlers.session import handle_manage_workers
        result = handle_manage_workers(ctx, {"action": "list"})
        assert result["status"] == "success"
        assert result["count"] == 0

    def test_manage_workers_cancel(self):
        ctx = _make_ctx()
        ctx.work_tracker.cancel.return_value = True
        from agent.tool_handlers.session import handle_manage_workers
        result = handle_manage_workers(ctx, {"action": "cancel", "work_id": "w1"})
        assert result["status"] == "success"
        ctx.work_tracker.cancel.assert_called_once_with("w1")

    def test_manage_workers_cancel_all(self):
        ctx = _make_ctx()
        ctx.work_tracker.cancel_all.return_value = 3
        from agent.tool_handlers.session import handle_manage_workers
        result = handle_manage_workers(ctx, {"action": "cancel"})
        assert result["status"] == "success"
        assert result["cancelled_count"] == 3

    def test_assets_list(self):
        ctx = _make_ctx()
        from agent.tool_handlers.data_ops import handle_assets
        result = handle_assets(ctx, {"action": "list"})
        assert result["status"] == "success"

    def test_events_check(self):
        ctx = _make_ctx()
        _orch(ctx).event_feed.check.return_value = {"events": []}
        from agent.tool_handlers.session import handle_events
        result = handle_events(ctx, {"action": "check"})
        # Should not raise AttributeError — event_feed is used via agent_state
        assert result is not None

    def test_delegate_to_viz_no_crash(self):
        """Delegation handler should not AttributeError on ctx attributes."""
        ctx = _make_ctx()
        agent_mock = MagicMock()
        agent_mock.state = "sleeping"
        agent_mock.inbox.qsize.return_value = 0
        ctx.delegation.get_or_create_viz_agent.return_value = agent_mock
        ctx.delegation.delegate_to_sub_agent.return_value = {"status": "success"}
        from agent.tool_handlers.delegation import handle_delegate_to_viz
        result = handle_delegate_to_viz(ctx, {"request": "plot something", "backend": "plotly"})
        # Should reach delegation.delegate_to_sub_agent without AttributeError
        assert result is not None
