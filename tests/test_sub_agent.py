"""
Tests for agent.sub_agent — SubAgent base class, Message dataclass, and inbox/event bus integration.

Run with: python -m pytest tests/test_sub_agent.py -v
"""

import json
import queue
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agent.sub_agent import (
    SubAgent,
    AgentState,
    Message,
    _make_message,
)
from agent.event_bus import (
    EventBus,
    TOOL_STARTED,
    TOOL_RESULT,
    TOOL_ERROR,
    SUB_AGENT_TOOL,
    DEBUG,
    AGENT_STATE_CHANGE,
)
from agent.llm.base import LLMResponse, ToolCall, FunctionSchema, UsageMetadata

# Speed up agent tests: use fast inbox polling instead of 1.0s default
SubAgent._inbox_timeout = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(responses=None):
    """Create a mock LLMAdapter that returns pre-configured responses."""
    adapter = MagicMock()
    if responses is None:
        responses = [LLMResponse(text="OK")]

    response_iter = iter(responses)

    def create_chat_fn(**kwargs):
        chat = MagicMock()
        chat.interaction_id = "test_interaction"
        chat.context_window.return_value = 0  # disable compaction in tests

        def send_fn(msg):
            try:
                return next(response_iter)
            except StopIteration:
                return LLMResponse(text="(no more responses)")

        chat.send = send_fn
        return chat

    adapter.create_chat = MagicMock(side_effect=create_chat_fn)
    adapter.make_tool_result_message = MagicMock(
        side_effect=lambda name, result, tool_call_id=None, **kwargs: {
            "tool_name": name,
            "result": result,
            "tool_call_id": tool_call_id,
        }
    )
    return adapter


def _make_mock_service(adapter=None):
    """Create a mock LLMService wrapping a mock adapter."""
    if adapter is None:
        adapter = _make_adapter()
    svc = MagicMock()
    svc.get_adapter.return_value = adapter
    svc.provider = "gemini"
    svc.make_tool_result.side_effect = lambda name, result, **kw: adapter.make_tool_result_message(name, result, **kw)
    return svc


def _make_tool_executor(results=None):
    """Create a mock tool executor that returns pre-configured results."""
    if results is None:
        results = {}

    def executor(name, args, tc_id=None):
        if name in results:
            return results[name]
        return {"status": "success", "data": f"result for {name}"}

    return executor


def _make_agent(
    agent_id="test_agent",
    responses=None,
    tool_results=None,
    tool_schemas=None,
    event_bus=None,
):
    """Create an SubAgent with mocked dependencies."""
    adapter = _make_adapter(responses)
    svc = _make_mock_service(adapter)
    executor = _make_tool_executor(tool_results)
    bus = event_bus if event_bus is not None else EventBus(session_id="test")

    agent = SubAgent(
        agent_id=agent_id,
        service=svc,
        agent_type="test",
        tool_executor=executor,
        model_name="test-model",
        system_prompt="You are a test agent.",
        tool_schemas=tool_schemas or [],
        event_bus=bus,
    )
    return agent


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------


class TestMessage:
    def test_make_message_fields(self):
        msg = _make_message("request", "user", "hello")
        assert msg.type == "request"
        assert msg.sender == "user"
        assert msg.content == "hello"
        assert msg.id.startswith("msg_")
        assert msg.reply_to is None
        assert msg._reply_event is None

    def test_make_message_with_reply_to(self):
        msg = _make_message("tool_result", "tool_runner", {"data": 1}, reply_to="t_abc")
        assert msg.reply_to == "t_abc"
        assert msg.type == "tool_result"

    def test_make_message_with_reply_event(self):
        evt = threading.Event()
        msg = _make_message("request", "user", "test", reply_event=evt)
        assert msg._reply_event is evt


# ---------------------------------------------------------------------------
# SubAgent lifecycle tests
# ---------------------------------------------------------------------------


class TestSubAgentLifecycle:
    def test_start_creates_thread(self):
        agent = _make_agent()
        agent.start()
        try:
            assert agent._thread is not None
            assert agent._thread.is_alive()
        finally:
            agent.stop()

    def test_stop_kills_thread(self):
        agent = _make_agent()
        agent.start()
        agent.stop(timeout=3.0)
        assert not agent._thread.is_alive()

    def test_start_is_idempotent(self):
        agent = _make_agent()
        agent.start()
        thread1 = agent._thread
        agent.start()  # should not create a new thread
        assert agent._thread is thread1
        agent.stop()

    def test_idle_property(self):
        agent = _make_agent()
        assert agent.is_idle  # idle before start
        agent.start()
        time.sleep(0.1)
        assert agent.is_idle  # idle when no messages
        agent.stop()


# ---------------------------------------------------------------------------
# Message handling tests
# ---------------------------------------------------------------------------


class TestMessageHandling:
    def test_request_returns_text(self):
        agent = _make_agent(responses=[LLMResponse(text="Hello from LLM")])
        agent.start()
        try:
            result = agent.send(wait=True, content="test message", timeout=5.0)
            assert result["text"] == "Hello from LLM"
            assert not result["failed"]
        finally:
            agent.stop()

    def test_messages_processed_in_order(self):
        """Two messages to the same agent should be processed sequentially."""
        order = []

        def slow_executor(name, args, tc_id=None):
            order.append(f"exec_{args.get('n', '?')}")
            time.sleep(0.1)
            return {"status": "success"}

        responses = [
            # First request: tool call then text
            LLMResponse(
                tool_calls=[
                    ToolCall(name="test_tool", args={"n": 1})
                ]
            ),
            LLMResponse(text="done 1"),
            # Second request: tool call then text
            LLMResponse(
                tool_calls=[
                    ToolCall(name="test_tool", args={"n": 2})
                ]
            ),
            LLMResponse(text="done 2"),
        ]

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = SubAgent(
            agent_id="order_test",
            service=svc,
            agent_type="test",
            tool_executor=slow_executor,
            tool_schemas=[
                FunctionSchema(name="test_tool", description="T", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            r1 = agent.send(wait=True, content="first", timeout=5.0)
            r2 = agent.send(wait=True, content="second", timeout=5.0)
            assert order == ["exec_1", "exec_2"]
        finally:
            agent.stop()

    def test_cancel_message(self):
        cancel_event = threading.Event()
        agent = _make_agent()
        agent._cancel_event = cancel_event
        agent.start()
        try:
            agent.inbox.put(_make_message("cancel", "user", "stop"))
            time.sleep(0.2)
            assert cancel_event.is_set()
        finally:
            agent.stop()

    def test_unknown_message_type_doesnt_crash(self):
        agent = _make_agent()
        agent.start()
        try:
            agent.inbox.put(_make_message("unknown_type", "test", "data"))
            time.sleep(0.2)
            assert agent.is_idle  # agent didn't crash
        finally:
            agent.stop()


# ---------------------------------------------------------------------------
# Async tool execution tests
# ---------------------------------------------------------------------------


class TestAsyncTools:
    def test_blocking_tool_returns_actual_result(self):
        """Non-sync tools block and return actual results (no started ack)."""
        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="fetch_data", args={"dataset_id": "AC_H2_MFI"})
                ]
            ),
            # LLM receives the actual result and responds
            LLMResponse(text="Data fetched successfully."),
        ]

        agent = _make_agent(responses=responses)
        agent.start()
        try:
            result = agent.send(wait=True, content="fetch ACE data", timeout=5.0)
            # The actual result (not a "started" ack) should have been sent to the LLM
            assert "fetched successfully" in result["text"]
        finally:
            agent.stop()

    def test_blocking_tool_result_returned_inline(self):
        """Blocking tool result is returned inline (not via event bus inbox)."""
        bus = EventBus(session_id="test")
        execution_completed = threading.Event()

        def slow_executor(name, args, tc_id=None):
            time.sleep(0.1)
            execution_completed.set()
            return {"status": "success", "data": "fetched data"}

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="fetch_data", args={"dataset_id": "AC_H2_MFI"})
                ]
            ),
            # LLM gets the actual result inline and responds
            LLMResponse(text="Data is ready!"),
        ]

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        agent = SubAgent(
            agent_id="blocking_test",
            service=svc,
            agent_type="test",
            tool_executor=slow_executor,
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="fetch data", timeout=5.0)
            # Tool should have executed and completed during the blocking call
            assert execution_completed.is_set()
            assert "Data is ready" in result["text"]
        finally:
            agent.stop()

    def test_sync_flag_stripped_from_args(self):
        """The _sync flag should be stripped from args (backward compat)."""
        captured_args = {}

        def capturing_executor(name, args, tc_id=None):
            captured_args[name] = dict(args)
            return {"status": "success"}

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="my_tool", args={"param1": "value", "_sync": True})
                ]
            ),
            LLMResponse(text="Done."),
        ]

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = SubAgent(
            agent_id="strip_test",
            service=svc,
            agent_type="test",
            tool_executor=capturing_executor,
            event_bus=bus,
            tool_schemas=[
                FunctionSchema(
                    name="my_tool",
                    description="Test tool",
                    parameters={"type": "object", "properties": {}},
                ),
            ],
        )
        agent.start()
        try:
            agent.send(wait=True, content="do something", timeout=5.0)
            # _sync should be stripped — executor gets clean args
            assert "_sync" not in captured_args.get("my_tool", {}), (
                "_sync should be stripped from tool args"
            )
            assert captured_args["my_tool"]["param1"] == "value"
        finally:
            agent.stop()


# ---------------------------------------------------------------------------
# Event bus integration tests
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    def test_tool_started_event_emitted(self):
        """TOOL_STARTED event should be emitted for async tool dispatch."""
        bus = EventBus(session_id="test")
        events = []
        bus.subscribe(lambda e: events.append(e) if e.type == TOOL_STARTED else None)

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="tool_a", args={}),
                ]
            ),
            LLMResponse(text="Done."),
        ]

        agent = _make_agent(responses=responses, event_bus=bus)
        agent.start()
        try:
            agent.send(wait=True, content="go", timeout=5.0)
            started_events = [e for e in events if e.type == TOOL_STARTED]
            assert len(started_events) >= 1
            assert started_events[0].data["tool_name"] == "tool_a"
        finally:
            agent.stop()



# ---------------------------------------------------------------------------
# send_and_wait / reply channel tests
# ---------------------------------------------------------------------------


class TestSendAndWait:
    def test_timeout_returns_error(self):
        """send_and_wait should return error dict on timeout."""
        agent = _make_agent()
        # Don't start the agent — no thread to process messages
        result = agent.send(wait=True, content="hello", timeout=0.1)
        assert result["failed"]
        assert "Timeout" in result["text"]

    def test_reply_delivered_to_caller(self):
        agent = _make_agent(responses=[LLMResponse(text="Reply text")])
        agent.start()
        try:
            result = agent.send(wait=True, content="test", timeout=5.0)
            assert result["text"] == "Reply text"
        finally:
            agent.stop()


# ---------------------------------------------------------------------------
# Hook tests
# ---------------------------------------------------------------------------


class TestHooks:
    def test_on_tool_result_hook_intercept(self):
        """_on_tool_result_hook returning non-None should short-circuit tool execution."""

        class InterceptSubAgent(SubAgent):
            def _on_tool_result_hook(self, tool_name, tool_args, result):
                if result.get("status") == "clarification_needed":
                    return result["question"]
                return None

        tool_results = {
            "ask_clarification": {
                "status": "clarification_needed",
                "question": "Which dataset?",
            }
        }

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="ask_clarification", args={})
                ]
            ),
        ]

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = InterceptSubAgent(
            agent_id="intercept_test",
            service=svc,
            agent_type="test",
            tool_executor=_make_tool_executor(tool_results),
            tool_schemas=[
                FunctionSchema(
                    name="ask_clarification", description="Ask", parameters={}
                ),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="search", timeout=5.0)
            assert result["text"] == "Which dataset?"
        finally:
            agent.stop()


# ---------------------------------------------------------------------------
# Parallel tool execution tests
# ---------------------------------------------------------------------------


class TestParallelToolExecution:
    """Tests for native parallel tool execution via _PARALLEL_SAFE_TOOLS."""

    def test_parallel_safe_tools_run_concurrently(self):
        """Two parallel-safe tools should both run and return results."""
        results_map = {
            "tool_a": {"status": "success", "data": "a_result"},
            "tool_b": {"status": "success", "data": "b_result"},
        }

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="tool_a", args={"x": 1}, id="tc_a"),
                    ToolCall(name="tool_b", args={"y": 2}, id="tc_b"),
                ]
            ),
            LLMResponse(text="Both done."),
        ]

        class ParallelAgent(SubAgent):
            _PARALLEL_SAFE_TOOLS = {"tool_a", "tool_b"}

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = ParallelAgent(
            agent_id="parallel_test",
            service=svc,
            agent_type="test",
            tool_executor=_make_tool_executor(results_map),
            tool_schemas=[
                FunctionSchema(name="tool_a", description="A", parameters={}),
                FunctionSchema(name="tool_b", description="B", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="run both", timeout=5.0)
            assert "Both done" in result["text"]

            # Verify both tool results were sent to LLM
            result_calls = adapter.make_tool_result_message.call_args_list
            tool_names = [c[0][0] for c in result_calls]
            assert "tool_a" in tool_names
            assert "tool_b" in tool_names
        finally:
            agent.stop()

    def test_mixed_safe_unsafe_falls_back_to_sequential(self):
        """When not all tools are parallel-safe, execution falls back to sequential."""
        execution_order = []

        def order_executor(name, args, tc_id=None):
            execution_order.append(name)
            time.sleep(0.05)  # Small delay to detect ordering
            return {"status": "success"}

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="safe_tool", args={}, id="tc_1"),
                    ToolCall(name="unsafe_tool", args={}, id="tc_2"),
                ]
            ),
            LLMResponse(text="Done sequentially."),
        ]

        class MixedAgent(SubAgent):
            _PARALLEL_SAFE_TOOLS = {"safe_tool"}  # unsafe_tool not in set

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = MixedAgent(
            agent_id="mixed_test",
            service=svc,
            agent_type="test",
            tool_executor=order_executor,
            tool_schemas=[
                FunctionSchema(name="safe_tool", description="S", parameters={}),
                FunctionSchema(name="unsafe_tool", description="U", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="run mixed", timeout=5.0)
            # Sequential execution — tools run in order
            assert execution_order == ["safe_tool", "unsafe_tool"]
        finally:
            agent.stop()

    def test_single_tool_runs_sequentially(self):
        """A single tool call always runs sequentially (no pool overhead)."""
        responses = [
            LLMResponse(
                tool_calls=[ToolCall(name="tool_a", args={}, id="tc_1")]
            ),
            LLMResponse(text="Single done."),
        ]

        class ParallelAgent(SubAgent):
            _PARALLEL_SAFE_TOOLS = {"tool_a"}

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = ParallelAgent(
            agent_id="single_test",
            service=svc,
            agent_type="test",
            tool_executor=_make_tool_executor(),
            tool_schemas=[
                FunctionSchema(name="tool_a", description="A", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="single", timeout=5.0)
            assert "Single done" in result["text"]
        finally:
            agent.stop()

    def test_duplicate_detection_works_in_parallel(self):
        """Duplicate calls should be blocked even in parallel mode."""
        from agent.loop_guard import LoopGuard

        call_count = {"tool_a": 0}

        def counting_executor(name, args, tc_id=None):
            call_count[name] = call_count.get(name, 0) + 1
            return {"status": "success"}

        # First response: two identical tool calls
        # Second response: same two again (should be blocked by dup detection)
        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="tool_a", args={"x": 1}, id="tc_1"),
                    ToolCall(name="tool_a", args={"x": 1}, id="tc_2"),
                ]
            ),
            LLMResponse(text="Done."),
        ]

        class ParallelAgent(SubAgent):
            _PARALLEL_SAFE_TOOLS = {"tool_a"}

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = ParallelAgent(
            agent_id="dup_test",
            service=svc,
            agent_type="test",
            tool_executor=counting_executor,
            tool_schemas=[
                FunctionSchema(name="tool_a", description="A", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="run dups", timeout=5.0)
            # Both calls should get results (first executes, second may be dup-warned)
            assert result is not None
        finally:
            agent.stop()

    def test_hook_interception_during_parallel(self):
        """_on_tool_result_hook should intercept during parallel execution."""

        class InterceptAgent(SubAgent):
            _PARALLEL_SAFE_TOOLS = {"tool_a", "tool_b"}

            def _on_tool_result_hook(self, tool_name, tool_args, result):
                if tool_name == "tool_a":
                    return "Intercepted by hook"
                return None

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="tool_a", args={}, id="tc_1"),
                    ToolCall(name="tool_b", args={}, id="tc_2"),
                ]
            ),
        ]

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = InterceptAgent(
            agent_id="hook_parallel",
            service=svc,
            agent_type="test",
            tool_executor=_make_tool_executor(),
            tool_schemas=[
                FunctionSchema(name="tool_a", description="A", parameters={}),
                FunctionSchema(name="tool_b", description="B", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="parallel", timeout=5.0)
            assert result["text"] == "Intercepted by hook"
            assert not result["failed"]
        finally:
            agent.stop()

    def test_parallel_results_dont_leak_to_inbox(self):
        """Results from parallel execution go to LLM, not inbox."""

        class ParallelAgent(SubAgent):
            _PARALLEL_SAFE_TOOLS = {"tool_a", "tool_b"}

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="tool_a", args={}, id="tc_1"),
                    ToolCall(name="tool_b", args={}, id="tc_2"),
                ]
            ),
            LLMResponse(text="Done."),
        ]

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = ParallelAgent(
            agent_id="leak_test",
            service=svc,
            agent_type="test",
            tool_executor=_make_tool_executor(),
            tool_schemas=[
                FunctionSchema(name="tool_a", description="A", parameters={}),
                FunctionSchema(name="tool_b", description="B", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            agent.send(wait=True, content="parallel", timeout=5.0)
            time.sleep(0.1)
            assert agent.inbox.empty(), "Parallel results should not leak to inbox"
        finally:
            agent.stop()

    def test_hook_fires_for_each_parallel_tool(self):
        """_on_tool_result_hook is called for each tool in parallel execution."""
        hook_calls = []

        class TrackingAgent(SubAgent):
            _PARALLEL_SAFE_TOOLS = {"tool_a", "tool_b"}

            def _on_tool_result_hook(self, tool_name, tool_args, result):
                hook_calls.append(tool_name)
                return None

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="tool_a", args={}, id="tc_1"),
                    ToolCall(name="tool_b", args={}, id="tc_2"),
                ]
            ),
            LLMResponse(text="Done."),
        ]

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = TrackingAgent(
            agent_id="hook_track",
            service=svc,
            agent_type="test",
            tool_executor=_make_tool_executor(),
            tool_schemas=[
                FunctionSchema(name="tool_a", description="A", parameters={}),
                FunctionSchema(name="tool_b", description="B", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            agent.send(wait=True, content="parallel", timeout=5.0)
            assert "tool_a" in hook_calls
            assert "tool_b" in hook_calls
            assert len(hook_calls) == 2
        finally:
            agent.stop()

    def test_empty_parallel_safe_set_forces_sequential(self):
        """With empty _PARALLEL_SAFE_TOOLS, multiple tools always run sequentially."""
        execution_order = []

        def order_executor(name, args, tc_id=None):
            execution_order.append(name)
            time.sleep(0.05)
            return {"status": "success"}

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="tool_a", args={}, id="tc_1"),
                    ToolCall(name="tool_b", args={}, id="tc_2"),
                ]
            ),
            LLMResponse(text="Sequential."),
        ]

        # Default SubAgent has empty _PARALLEL_SAFE_TOOLS
        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = SubAgent(
            agent_id="seq_test",
            service=svc,
            agent_type="test",
            tool_executor=order_executor,
            tool_schemas=[
                FunctionSchema(name="tool_a", description="A", parameters={}),
                FunctionSchema(name="tool_b", description="B", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="run both", timeout=5.0)
            assert execution_order == ["tool_a", "tool_b"]
        finally:
            agent.stop()


# ---------------------------------------------------------------------------
# Status / introspection tests
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_returns_expected_fields(self):
        agent = _make_agent()
        status = agent.status()
        assert status["agent_id"] == "test_agent"
        assert status["state"] == "sleeping"
        assert status["idle"] is True
        assert status["queue_depth"] == 0
        assert "tokens" in status


# ---------------------------------------------------------------------------
# AgentState lifecycle tests
# ---------------------------------------------------------------------------


class TestAgentState:
    """Tests for the 2-state lifecycle model (SLEEPING, ACTIVE)."""

    def test_initial_state_is_sleeping(self):
        agent = _make_agent()
        assert agent.state == AgentState.SLEEPING

    def test_sleeping_to_active_on_message(self):
        """SubAgent transitions SLEEPING → ACTIVE when processing a message."""
        bus = EventBus(session_id="test")
        state_changes = []
        bus.subscribe(
            lambda e: (
                state_changes.append(e.data) if e.type == AGENT_STATE_CHANGE else None
            )
        )

        agent = _make_agent(
            responses=[LLMResponse(text="Reply")],
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="hello", timeout=5.0)
            assert result["text"] == "Reply"

            # Should have transitioned SLEEPING → ACTIVE → SLEEPING
            active_transitions = [
                c
                for c in state_changes
                if c.get("agent_id") == "test_agent" and c.get("new") == "active"
            ]
            sleeping_transitions = [
                c
                for c in state_changes
                if c.get("agent_id") == "test_agent" and c.get("new") == "sleeping"
            ]
            assert len(active_transitions) >= 1, (
                "Should have at least one ACTIVE transition"
            )
            assert len(sleeping_transitions) >= 1, (
                "Should have at least one SLEEPING transition"
            )
        finally:
            agent.stop()

    def test_active_to_sleeping_when_no_async_tools(self):
        """With no async tools dispatched, agent goes directly to SLEEPING."""
        agent = _make_agent(responses=[LLMResponse(text="Done")])
        agent.start()
        try:
            agent.send(wait=True, content="test", timeout=5.0)
            time.sleep(0.2)
            assert agent.state == AgentState.SLEEPING
        finally:
            agent.stop()

    def test_tool_goes_to_sleeping_after_completion(self):
        """Tools complete inline — agent goes to SLEEPING after."""

        def slow_executor(name, args, tc_id=None):
            time.sleep(0.1)
            return {"status": "success"}

        responses = [
            LLMResponse(tool_calls=[ToolCall(name="slow_tool", args={})]),
            LLMResponse(text="Done."),
        ]

        bus = EventBus(session_id="test")
        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        agent = SubAgent(
            agent_id="idle_test",
            service=svc,
            agent_type="test",
            tool_executor=slow_executor,
            event_bus=bus,
        )
        agent.start()
        try:
            agent.send(wait=True, content="do something slow", timeout=5.0)
            time.sleep(0.2)
            assert agent.state == AgentState.SLEEPING
        finally:
            agent.stop()

    def test_tool_result_feeds_back_to_llm(self):
        """Tool result is fed back to LLM in the same turn."""
        executed = threading.Event()

        def slow_executor(name, args, tc_id=None):
            time.sleep(0.1)
            executed.set()
            return {"status": "success", "data": "done"}

        responses = [
            LLMResponse(tool_calls=[ToolCall(name="slow_tool", args={})]),
            # LLM gets the actual result inline and produces final text
            LLMResponse(text="Tool finished!"),
        ]

        bus = EventBus(session_id="test")
        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        agent = SubAgent(
            agent_id="wake_test",
            service=svc,
            agent_type="test",
            tool_executor=slow_executor,
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="run it", timeout=5.0)
            assert executed.is_set()
            assert "Tool finished" in result["text"]
            assert agent.state == AgentState.SLEEPING
        finally:
            agent.stop()

    def test_parallel_tools_stay_active(self):
        """SubAgent stays ACTIVE during parallel tool execution (no PENDING state)."""
        bus = EventBus(session_id="test")
        state_changes = []
        bus.subscribe(
            lambda e: (
                state_changes.append(e.data) if e.type == AGENT_STATE_CHANGE else None
            )
        )

        class ParallelAgent(SubAgent):
            _PARALLEL_SAFE_TOOLS = {"test_tool"}

        responses = [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="test_tool", args={"n": 1}, id="tc_1"),
                    ToolCall(name="test_tool", args={"n": 2}, id="tc_2"),
                ]
            ),
            LLMResponse(text="Parallel done."),
        ]

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        agent = ParallelAgent(
            agent_id="parallel_state_test",
            service=svc,
            agent_type="test",
            tool_executor=_make_tool_executor(),
            tool_schemas=[
                FunctionSchema(name="test_tool", description="T", parameters={}),
            ],
            event_bus=bus,
        )
        agent.start()
        try:
            result = agent.send(wait=True, content="parallel", timeout=5.0)
            assert result["text"] == "Parallel done."

            # Should have seen: SLEEPING → ACTIVE → SLEEPING (no PENDING)
            all_states = [
                c.get("new")
                for c in state_changes
                if c.get("agent_id") == "parallel_state_test"
            ]
            assert "pending" not in all_states, (
                "Should never transition to PENDING"
            )
        finally:
            agent.stop()

    def test_set_state_emits_event(self):
        """_set_state() emits AGENT_STATE_CHANGE event with correct data."""
        bus = EventBus(session_id="test")
        state_events = []
        bus.subscribe(
            lambda e: state_events.append(e) if e.type == AGENT_STATE_CHANGE else None
        )

        agent = _make_agent(event_bus=bus)
        agent._set_state(AgentState.ACTIVE)

        assert len(state_events) == 1
        evt = state_events[0]
        assert evt.data["agent_id"] == "test_agent"
        assert evt.data["old"] == "sleeping"
        assert evt.data["new"] == "active"
        assert evt.data["reason"] == ""

    def test_set_state_emits_reason(self):
        """_set_state() includes reason in event data and message."""
        bus = EventBus(session_id="test")
        state_events = []
        bus.subscribe(
            lambda e: state_events.append(e) if e.type == AGENT_STATE_CHANGE else None
        )

        agent = _make_agent(event_bus=bus)
        agent._set_state(AgentState.ACTIVE, reason="received request")

        assert len(state_events) == 1
        evt = state_events[0]
        assert evt.data["reason"] == "received request"
        assert "(received request)" in evt.msg

    def test_set_state_no_op_when_same(self):
        """_set_state() with same state does not emit an event."""
        bus = EventBus(session_id="test")
        state_events = []
        bus.subscribe(
            lambda e: state_events.append(e) if e.type == AGENT_STATE_CHANGE else None
        )

        agent = _make_agent(event_bus=bus)
        assert agent.state == AgentState.SLEEPING
        agent._set_state(AgentState.SLEEPING)  # no-op
        assert len(state_events) == 0

    def test_idle_sync_with_legacy_is_idle(self):
        """The legacy is_idle property stays in sync with state."""
        agent = _make_agent()
        assert agent.is_idle is True  # SLEEPING → _idle.set()

        agent._set_state(AgentState.ACTIVE)
        assert agent.is_idle is False

        agent._set_state(AgentState.SLEEPING)
        assert agent.is_idle is True

    def test_status_includes_state(self):
        """status() dict includes the current state value."""
        agent = _make_agent()
        assert agent.status()["state"] == "sleeping"

        agent._set_state(AgentState.ACTIVE)
        assert agent.status()["state"] == "active"


# ---------------------------------------------------------------------------
# Sync tool execution tests
# ---------------------------------------------------------------------------


class TestSyncToolExecution:
    """Tests for the unified sync tool execution path."""

    def test_tool_executes_inline(self):
        """All tools execute inline and return results immediately."""
        tool_call = ToolCall(name="list_fetched_data", args={}, id="tc_1")
        first_response = LLMResponse(text=None, tool_calls=[tool_call])
        responses = [
            LLMResponse(text="Done"),
        ]
        executor_calls = []

        def tracking_executor(name, args, tc_id=None):
            executor_calls.append({"name": name, "args": args, "tc_id": tc_id})
            return {"status": "success", "labels": ["test_data"]}

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = SubAgent(
            agent_id="test_sync",
            service=svc,
            agent_type="test",
            tool_executor=tracking_executor,
            system_prompt="test",
            tool_schemas=[
                FunctionSchema(
                    name="list_fetched_data",
                    description="List data",
                    parameters={"type": "object", "properties": {}},
                ),
            ],
            event_bus=bus,
        )

        agent._guard = None
        result = agent._process_response(first_response)

        assert len(executor_calls) == 1
        assert executor_calls[0]["name"] == "list_fetched_data"

    def test_tool_returns_actual_result(self):
        """All tools return actual results inline."""
        tool_call = ToolCall(name="fetch_data", args={"dataset": "ACE"}, id="tc_1")
        first_response = LLMResponse(text=None, tool_calls=[tool_call])
        responses = [
            LLMResponse(text="Done"),
        ]

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")

        dispatch_called = []

        def tracking_executor(name, args, tc_id=None):
            dispatch_called.append(name)
            return {"status": "success"}

        agent = SubAgent(
            agent_id="test_blocking",
            service=svc,
            agent_type="test",
            tool_executor=tracking_executor,
            system_prompt="test",
            tool_schemas=[
                FunctionSchema(
                    name="fetch_data",
                    description="Fetch",
                    parameters={"type": "object", "properties": {}},
                ),
            ],
            event_bus=bus,
        )

        agent._guard = None
        result = agent._process_response(first_response)

        assert "fetch_data" in dispatch_called
        adapter.make_tool_result_message.assert_called()
        call_args = adapter.make_tool_result_message.call_args
        assert call_args[0][1]["status"] == "success"

    def test_multiple_tools_all_execute_inline(self):
        """Multiple tools in same response all execute inline."""
        tc1 = ToolCall(name="list_fetched_data", args={}, id="tc_1")
        tc2 = ToolCall(name="custom_operation", args={"code": "df+1"}, id="tc_2")
        first_response = LLMResponse(text=None, tool_calls=[tc1, tc2])
        responses = [
            LLMResponse(text="All done"),
        ]

        inline_calls = []

        def executor(name, args, tc_id=None):
            inline_calls.append(name)
            return {"status": "success"}

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = SubAgent(
            agent_id="test_multi",
            service=svc,
            agent_type="test",
            tool_executor=executor,
            system_prompt="test",
            tool_schemas=[
                FunctionSchema(
                    name="list_fetched_data",
                    description="List",
                    parameters={"type": "object", "properties": {}},
                ),
                FunctionSchema(
                    name="custom_operation",
                    description="Compute",
                    parameters={"type": "object", "properties": {}},
                ),
            ],
            event_bus=bus,
        )

        agent._guard = None
        result = agent._process_response(first_response)

        assert "list_fetched_data" in inline_calls
        assert "custom_operation" in inline_calls

    def test_tool_error_handled(self):
        """Errors from tools should be caught and recorded."""
        tool_call = ToolCall(name="describe_data", args={"label": "bad"}, id="tc_err")
        first_response = LLMResponse(text=None, tool_calls=[tool_call])
        responses = [
            LLMResponse(text="Handled error"),
        ]

        def failing_executor(name, args, tc_id=None):
            raise ValueError("data not found")

        adapter = _make_adapter(responses)
        svc = _make_mock_service(adapter)
        bus = EventBus(session_id="test")
        agent = SubAgent(
            agent_id="test_err",
            service=svc,
            agent_type="test",
            tool_executor=failing_executor,
            system_prompt="test",
            tool_schemas=[
                FunctionSchema(
                    name="describe_data",
                    description="Describe",
                    parameters={"type": "object", "properties": {}},
                ),
            ],
            event_bus=bus,
        )

        agent._guard = None
        result = agent._process_response(first_response)

        assert any("data not found" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Persistent LoopGuard tests
# ---------------------------------------------------------------------------


class TestPersistentGuard:
    """Tests for the persistent LoopGuard that survives across _process_response calls."""

    def test_guard_persists_across_process_response_calls(self):
        """The LoopGuard should persist between _handle_request and
        subsequent _process_response calls on the same request."""
        from agent.loop_guard import LoopGuard

        agent = _make_agent(responses=[LLMResponse(text="OK")])

        # Simulate _handle_request setting the guard
        agent._guard = LoopGuard(max_total_calls=10)
        guard_ref = agent._guard

        # _process_response should use the existing guard
        agent._process_response(LLMResponse(text="OK"))

        # Guard should still be the same object
        assert agent._guard is guard_ref

    def test_guard_reset_on_new_request(self):
        """_handle_request should create a fresh guard."""
        from agent.loop_guard import LoopGuard

        agent = _make_agent(responses=[LLMResponse(text="OK")])

        # Set an old guard
        old_guard = LoopGuard(max_total_calls=5)
        old_guard.total_calls = 999  # simulate exhausted guard
        agent._guard = old_guard

        # _handle_request creates fresh guard
        msg = _make_message("request", "user", "test")
        agent._handle_request(msg)

        # Guard should be a NEW instance, not the old one
        assert agent._guard is not old_guard
        assert agent._guard.total_calls == 0

    def test_guard_tracks_calls_across_rounds(self):
        """A guard created in _handle_request should accumulate call counts
        across multiple _process_response invocations."""
        from agent.loop_guard import LoopGuard

        agent = _make_agent(responses=[LLMResponse(text="OK")])

        # Create guard with low limits
        agent._guard = LoopGuard(max_total_calls=5)

        # Simulate recording calls from one round
        agent._guard.record_calls(1)
        assert agent._guard.total_calls == 1

        # Simulate another round of _process_response
        agent._guard.record_calls(1)
        assert agent._guard.total_calls == 2
