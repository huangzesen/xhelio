"""Tests for DelegationBus with new BaseAgent-based agents."""
from unittest.mock import MagicMock, patch
from agent.delegation import DelegationBus


def _make_mock_orchestrator():
    """Create a mock orchestrator with the attributes DelegationBus needs."""
    orch = MagicMock()
    orch.session_ctx = MagicMock()
    orch.session_ctx.service = MagicMock()
    orch.session_ctx.event_bus = MagicMock()
    orch.session_ctx.store = MagicMock()
    orch.session_ctx.store.list_entries.return_value = []
    orch.session_ctx.work_tracker = MagicMock()
    orch.session_ctx.agent_state = {
        "orchestrator": MagicMock(ctx_tracker=MagicMock()),
    }
    orch._cancel_event = MagicMock()
    return orch


def test_delegation_bus_creates_with_session_ctx():
    """DelegationBus can be constructed with a SessionContext."""
    orch = _make_mock_orchestrator()
    bus = DelegationBus(ctx=orch.session_ctx)
    assert bus is not None


def test_get_or_create_data_io_agent():
    """DelegationBus creates a DataIOAgent using new BaseAgent class."""
    orch = _make_mock_orchestrator()
    bus = DelegationBus(ctx=orch.session_ctx)
    with patch("agent.delegation.DataIOAgent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.agent_id = "data_io:abc123"
        MockAgent.return_value = mock_instance
        agent = bus.get_or_create_data_io_agent()
        # Verify session_ctx was passed
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args
        assert "session_ctx" in call_kwargs.kwargs or (
            len(call_kwargs.args) >= 2 and call_kwargs.args[1] is not None
        )


def test_get_or_create_viz_agent():
    """DelegationBus creates a VizAgent with backend parameter."""
    orch = _make_mock_orchestrator()
    bus = DelegationBus(ctx=orch.session_ctx)
    with patch("agent.delegation.VizAgent") as MockAgent:
        mock_instance = MagicMock()
        mock_instance.agent_id = "VizAgent[Plotly]"
        MockAgent.return_value = mock_instance
        agent = bus.get_or_create_viz_agent("plotly")
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args
        assert call_kwargs.kwargs.get("backend") == "plotly"
        assert "session_ctx" in call_kwargs.kwargs


def test_register_agent():
    """Can register an externally-created agent."""
    orch = _make_mock_orchestrator()
    bus = DelegationBus(ctx=orch.session_ctx)
    mock_agent = MagicMock()
    bus.register_agent("TestAgent", mock_agent)
    assert bus.has_agent("TestAgent")
