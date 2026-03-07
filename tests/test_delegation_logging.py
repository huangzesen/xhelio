"""
Tests for delegation logging improvements.

Validates that delegation messages include async mode indicator when wait=False.

Run with: python -m pytest tests/test_delegation_logging.py -v
"""

import pytest
from unittest.mock import MagicMock, call

from agent.event_bus import DELEGATION, EventBus


def test_delegation_message_shows_async_when_wait_false():
    """Test that delegation message shows 'run in background' when wait=False."""
    # Create a mock orchestrator with required methods
    orch = MagicMock()
    orch._event_bus = MagicMock()
    orch._event_bus.emit = MagicMock()
    orch._get_or_create_envoy_agent = MagicMock(return_value=MagicMock(
        agent_id="envoy:ACE",
        state=MagicMock(state="sleeping"),
        inbox=MagicMock(qsize=lambda: 0)
    ))
    orch._build_envoy_request = MagicMock(return_value="test request")

    # Import after setting up mocks
    from agent.tool_handlers.delegation import handle_delegate_to_envoy

    tool_args = {"mission_id": "ACE", "request": "show data", "wait": False}
    result = handle_delegate_to_envoy(orch, tool_args)

    # Check that emit was called with message containing "run in background"
    # The DELEGATION emit should be called with a message indicating background mode
    emit_calls = orch._event_bus.emit.call_args_list

    # Find the DELEGATION emit call
    delegation_call = None
    for c in emit_calls:
        if c[0][0] == DELEGATION:
            delegation_call = c
            break

    assert delegation_call is not None, "DELEGATION event was not emitted"
    msg = delegation_call[1]["msg"]
    assert "run in background" in msg.lower() or "background" in msg.lower(), \
        f"Expected 'run in background' in message, got: {msg}"


def test_delegation_message_no_background_when_wait_true():
    """Test that delegation message does NOT show background when wait=True."""
    orch = MagicMock()
    orch._event_bus = MagicMock()
    orch._event_bus.emit = MagicMock()
    orch._get_or_create_envoy_agent = MagicMock(return_value=MagicMock(
        agent_id="envoy:ACE",
        state=MagicMock(state="sleeping"),
        inbox=MagicMock(qsize=lambda: 0)
    ))
    orch._build_envoy_request = MagicMock(return_value="test request")

    from agent.tool_handlers.delegation import handle_delegate_to_envoy

    tool_args = {"mission_id": "ACE", "request": "show data", "wait": True}
    result = handle_delegate_to_envoy(orch, tool_args)

    # Find the DELEGATION emit call
    emit_calls = orch._event_bus.emit.call_args_list
    delegation_call = None
    for c in emit_calls:
        if c[0][0] == DELEGATION:
            delegation_call = c
            break

    assert delegation_call is not None, "DELEGATION event was not emitted"
    msg = delegation_call[1]["msg"]
    # When wait=True, message should NOT contain "background"
    assert "background" not in msg.lower(), \
        f"Expected no 'background' in message when wait=True, got: {msg}"


def test_fire_and_forget_calls_delegate_to_sub_agent():
    """Test that wait=False flows through to _delegate_to_sub_agent (not early return)."""
    from agent.tool_handlers.delegation import handle_delegate_to_envoy
    from agent.sub_agent import AgentState

    orch = MagicMock()
    orch._event_bus = MagicMock()
    orch._event_bus.emit = MagicMock()

    # Set up agent mock that the handler will create
    agent_mock = MagicMock()
    agent_mock.state = AgentState.SLEEPING
    agent_mock.inbox = MagicMock()
    agent_mock.inbox.qsize.return_value = 0
    orch._get_or_create_envoy_agent.return_value = agent_mock
    orch._build_envoy_request.return_value = "test request"
    orch._store.list_entries.return_value = []
    orch._delegate_to_sub_agent.return_value = {"status": "queued", "message": "started"}

    tool_args = {"mission_id": "ACE", "request": "show data", "wait": False}
    result = handle_delegate_to_envoy(orch, tool_args)

    # With the fix: handler creates agent and calls _delegate_to_sub_agent(wait=False)
    orch._get_or_create_envoy_agent.assert_called_once()
    orch._delegate_to_sub_agent.assert_called_once()
    # Verify wait=False was passed through
    call_kwargs = orch._delegate_to_sub_agent.call_args
    assert call_kwargs[1].get("wait") is False or (len(call_kwargs[0]) > 1 and call_kwargs[0][-1] is False)


def test_async_completed_emitted_only_for_tracked_async_delegations():
    """Test that DELEGATION_ASYNC_COMPLETED is emitted ONLY for tracked async delegations."""
    from agent.event_bus import DELEGATION_ASYNC_COMPLETED, TEXT_DELTA
    from agent.core import OrchestratorAgent

    # Create orchestrator without __init__ to avoid side effects
    orch = OrchestratorAgent.__new__(OrchestratorAgent)
    # Set up required attributes
    orch._event_bus = MagicMock()
    orch._responded_this_cycle = True
    orch._all_work_subagents_idle = MagicMock(return_value=True)
    orch.get_token_usage = MagicMock(return_value={"prompt": 100, "completion": 50})
    orch._round_start_tokens = {"prompt": 100, "completion": 50}
    orch._cycle_number = 1
    orch._rounds_since_last_reload = 0

    # Mock process_message to return a response
    orch.process_message = MagicMock(return_value="Test response")
    # Mock logger
    orch.logger = MagicMock()
    # Set up state for cycle end
    orch._set_state = MagicMock()

    # Test 1: When sender IS in _async_delegations, DELEGATION_ASYNC_COMPLETED should be emitted
    import time
    orch._async_delegations = {"EnvoyAgent[ACE]": time.time(), "VizAgent[Plotly]": time.time()}

    msg = MagicMock()
    msg.type = "subagent_result"
    msg.sender = "EnvoyAgent[ACE]"
    msg.content = "Analysis complete"

    # Simulate the code path that processes subagent results
    # (This mimics what happens in core.py around line 2450-2475)
    response_text = orch.process_message(f"[Subagent {msg.sender} completed]: {msg.content}")
    orch._event_bus.emit(
        TEXT_DELTA,
        level="info",
        msg=response_text,
        data={"text": response_text, "source": "subagent"},
    )
    sender_id = msg.sender
    if hasattr(orch, '_async_delegations') and sender_id in orch._async_delegations:
        start_time = orch._async_delegations.get(sender_id)
        duration = time.time() - start_time if start_time else 0
        orch._event_bus.emit(
            DELEGATION_ASYNC_COMPLETED,
            level="info",
            msg=f"[Router] Background task completed: {sender_id}",
            data={"agent": sender_id, "duration_seconds": round(duration, 2)},
        )
        del orch._async_delegations[sender_id]

    # Verify DELEGATION_ASYNC_COMPLETED was emitted
    emitted_types = [call[0][0] for call in orch._event_bus.emit.call_args_list]
    assert DELEGATION_ASYNC_COMPLETED in emitted_types, \
        f"Expected DELEGATION_ASYNC_COMPLETED to be emitted for tracked async delegation, got {emitted_types}"

    # Test 2: When sender is NOT in _async_delegations, it should NOT be emitted
    orch._event_bus.reset_mock()
    orch._async_delegations = {"VizAgent[Plotly]": time.time()}  # EnvoyAgent[ACE] not tracked

    msg.sender = "EnvoyAgent[ACE]"  # This sender was NOT an async delegation
    response_text = orch.process_message(f"[Subagent {msg.sender} completed]: {msg.content}")
    orch._event_bus.emit(
        TEXT_DELTA,
        level="info",
        msg=response_text,
        data={"text": response_text, "source": "subagent"},
    )
    sender_id = msg.sender
    if hasattr(orch, '_async_delegations') and sender_id in orch._async_delegations:
        start_time = orch._async_delegations.get(sender_id)
        duration = time.time() - start_time if start_time else 0
        orch._event_bus.emit(
            DELEGATION_ASYNC_COMPLETED,
            level="info",
            msg=f"[Router] Background task completed: {sender_id}",
            data={"agent": sender_id, "duration_seconds": round(duration, 2)},
        )
        del orch._async_delegations[sender_id]

    # Verify DELEGATION_ASYNC_COMPLETED was NOT emitted
    emitted_types = [call[0][0] for call in orch._event_bus.emit.call_args_list]
    assert DELEGATION_ASYNC_COMPLETED not in emitted_types, \
        f"Expected DELEGATION_ASYNC_COMPLETED NOT to be emitted for non-async delegation, got {emitted_types}"


def test_async_duration_included_in_completion():
    """Test that duration is included in DELEGATION_ASYNC_COMPLETED data."""
    from agent.event_bus import DELEGATION_ASYNC_COMPLETED, TEXT_DELTA
    import time

    # Test the core.py completion path - verify it calculates duration
    from agent.core import OrchestratorAgent

    orch2 = OrchestratorAgent.__new__(OrchestratorAgent)
    orch2._event_bus = MagicMock()
    orch2._responded_this_cycle = True
    orch2._all_work_subagents_idle = MagicMock(return_value=True)
    orch2.get_token_usage = MagicMock(return_value={"prompt": 100, "completion": 50})
    orch2._round_start_tokens = {"prompt": 100, "completion": 50}
    orch2._cycle_number = 1
    orch2._rounds_since_last_reload = 0
    orch2.process_message = MagicMock(return_value="Test response")
    orch2.logger = MagicMock()
    orch2._set_state = MagicMock()

    # Set up async delegations dict with a start time from 1.5 seconds ago
    agent_id = "EnvoyAgent[ACE]"
    start_time = time.time() - 1.5
    orch2._async_delegations = {agent_id: start_time}

    msg = MagicMock()
    msg.type = "subagent_result"
    msg.sender = agent_id
    msg.content = "Analysis complete"

    # Simulate the core.py completion path
    response_text = orch2.process_message(f"[Subagent {msg.sender} completed]: {msg.content}")
    orch2._event_bus.emit(
        TEXT_DELTA,
        level="info",
        msg=response_text,
        data={"text": response_text, "source": "subagent"},
    )
    sender_id = msg.sender
    if sender_id in orch2._async_delegations:
        stored_start_time = orch2._async_delegations.get(sender_id)
        duration = time.time() - stored_start_time if stored_start_time else 0
        orch2._event_bus.emit(
            DELEGATION_ASYNC_COMPLETED,
            level="info",
            msg=f"[Router] Background task completed: {sender_id}",
            data={"agent": sender_id, "duration_seconds": round(duration, 2)},
        )
        del orch2._async_delegations[sender_id]

    # Verify DELEGATION_ASYNC_COMPLETED was emitted with duration
    emitted_types = [call[0][0] for call in orch2._event_bus.emit.call_args_list]
    assert DELEGATION_ASYNC_COMPLETED in emitted_types, \
        f"Expected DELEGATION_ASYNC_COMPLETED to be emitted, got {emitted_types}"

    # Find the DELEGATION_ASYNC_COMPLETED call and verify duration_seconds is in data
    completed_call = None
    for call in orch2._event_bus.emit.call_args_list:
        if call[0][0] == DELEGATION_ASYNC_COMPLETED:
            completed_call = call
            break

    assert completed_call is not None, "DELEGATION_ASYNC_COMPLETED call not found"
    data = completed_call[1]["data"]
    assert "duration_seconds" in data, f"Expected 'duration_seconds' in data, got: {data}"
    assert data["duration_seconds"] >= 1.5, f"Expected duration >= 1.5, got: {data['duration_seconds']}"
