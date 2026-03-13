"""Verify BaseAgent dispatch passes ToolCaller as third argument."""
import threading
from unittest.mock import MagicMock

from agent.base_agent import BaseAgent, LoopGuard
from agent.session_context import SessionContext
from agent.tool_caller import ToolCaller


def _make_agent():
    ctx = MagicMock(spec=SessionContext)
    ctx.event_bus = MagicMock()
    ctx.cancel_event = threading.Event()
    service = MagicMock()
    service.provider = "mock"
    agent = BaseAgent(
        agent_id="test:001",
        service=service,
        tool_schemas=[],
        system_prompt="test",
        session_ctx=ctx,
        event_bus=ctx.event_bus,
        cancel_event=ctx.cancel_event,
    )
    return agent


def test_dispatch_passes_tool_caller():
    """Handler receives ToolCaller as third argument."""
    agent = _make_agent()
    received = {}

    def mock_handler(ctx, args, caller):
        received["ctx"] = ctx
        received["args"] = args
        received["caller"] = caller
        return {"status": "ok"}

    agent._local_tools["test_tool"] = mock_handler

    tc = MagicMock()
    tc.name = "test_tool"
    tc.id = "tc_123"
    tc.args = {"key": "value"}

    guard = LoopGuard(max_total=100, dup_free_passes=5, dup_hard_block=10)
    agent._execute_single_tool(tc, guard, [])

    assert "caller" in received
    caller = received["caller"]
    assert isinstance(caller, ToolCaller)
    assert caller.agent_id == "test:001"
    assert caller.agent_type == ""  # BaseAgent default
    assert caller.tool_call_id == "tc_123"
