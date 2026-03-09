"""
Tests for agent.event_bus — EventBus, SessionEvent, and listeners.

Run with: python -m pytest tests/test_event_bus.py -v
"""

import json
import logging
import threading
from unittest.mock import MagicMock, call

import pytest

from agent.event_bus import (
    EventBus,
    SessionEvent,
    DebugLogListener,
    SSEEventListener,
    OperationsLogListener,
    DisplayLogBuilder,
    TokenLogListener,
    _resolve_tags,
    INFRASTRUCTURE_TAGS,
    ALL_CTX_TAGS,
    UNIVERSAL_CTX_EVENTS,
    ROUTING_ONLY_EVENTS,
    get_event_bus,
    set_event_bus,
    # Event types
    USER_MESSAGE,
    AGENT_RESPONSE,
    TOOL_CALL,
    TOOL_RESULT,
    TOOL_CALL_LOG,
    TOOL_RESULT_LOG,
    TOOL_ERROR_LOG,
    DATA_FETCHED,
    DATA_COMPUTED,
    DATA_CREATED,
    RENDER_EXECUTED,
    MPL_RENDER_EXECUTED,
    JSX_RENDER_EXECUTED,
    PLOT_ACTION,
    DELEGATION,
    DELEGATION_DONE,
    SUB_AGENT_TOOL,
    SUB_AGENT_ERROR,
    PLAN_CREATED,
    PLAN_TASK,
    PROGRESS,
    THINKING,
    FETCH_ERROR,
    RENDER_ERROR,
    CUSTOM_OP_FAILURE,
    TOOL_ERROR,
    CDF_FILE_QUERY,
    CDF_DOWNLOAD,
    CDF_DOWNLOAD_WARN,
    DEBUG,
    TOKEN_USAGE,
    CONTEXT_COMPACTION,
    INSIGHT_RESULT,
)
from agent.agent_registry import (
    AGENT_CALL_REGISTRY,
    AGENT_INFORMED_REGISTRY,
    InformedRegistry,
)


# ---- EventBus core ----

class TestEventBus:
    def test_emit_creates_event(self):
        bus = EventBus(session_id="test")
        event = bus.emit(TOOL_CALL, agent="orchestrator", msg="test call",
                         data={"tool_name": "test_tool", "tool_args": {}})
        assert event.type == TOOL_CALL
        assert event.agent == "orchestrator"
        assert "test_tool" in event.summary  # formatter generates summary from data
        assert event.msg == event.summary   # backward compat property
        assert event.level == "debug"
        assert event.tags == frozenset({"display", "memory", "console"})
        assert len(bus) == 1

    def test_emit_with_explicit_summary(self):
        """When summary= is provided, it's used directly (no formatter)."""
        bus = EventBus()
        event = bus.emit(DEBUG, agent="test", summary="My summary", details="My details")
        assert event.summary == "My summary"
        assert event.details == "My details"
        assert event.msg == "My summary"  # backward compat

    def test_emit_with_msg_backward_compat(self):
        """When msg= is provided but not summary=, formatter generates summary from msg."""
        bus = EventBus()
        event = bus.emit(DEBUG, msg="hello world")
        # The debug formatter uses _msg from data, which is set to msg value
        assert "hello world" in event.summary
        assert event.msg == event.summary  # backward compat

    def test_emit_resolves_tags_from_registry(self):
        bus = EventBus()
        event = bus.emit(
            DELEGATION, agent="orch", level="info",
            msg="Delegating to PSP",
            data={"target": "PSP"},
        )
        assert event.tags == frozenset({"display", "memory", "console", "ctx:planner", "ctx:orchestrator"})
        assert event.data["target"] == "PSP"
        assert event.level == "info"

    def test_emit_calls_listeners(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda e: received.append(e))
        bus.emit(DEBUG, msg="hello")
        assert len(received) == 1
        assert "hello" in received[0].summary

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        listener = lambda e: received.append(e)
        bus.subscribe(listener)
        bus.emit(DEBUG, msg="one")
        bus.unsubscribe(listener)
        bus.emit(DEBUG, msg="two")
        assert len(received) == 1

    def test_unsubscribe_nonexistent_is_noop(self):
        bus = EventBus()
        bus.unsubscribe(lambda e: None)  # Should not raise

    def test_listener_exception_doesnt_break_emit(self):
        bus = EventBus()
        good_received = []

        def bad_listener(e):
            raise RuntimeError("boom")

        bus.subscribe(bad_listener)
        bus.subscribe(lambda e: good_received.append(e))
        bus.emit(DEBUG, msg="test")
        assert len(good_received) == 1

    def test_get_events_all(self):
        bus = EventBus()
        bus.emit(TOOL_CALL, msg="a")
        bus.emit(TOOL_RESULT, msg="b")
        bus.emit(DELEGATION, msg="c")
        events = bus.get_events()
        assert len(events) == 3

    def test_get_events_filter_by_types(self):
        bus = EventBus()
        bus.emit(TOOL_CALL, msg="a")
        bus.emit(TOOL_RESULT, msg="b")
        bus.emit(DELEGATION, msg="c")
        events = bus.get_events(types={TOOL_CALL, DELEGATION})
        assert len(events) == 2
        assert {e.type for e in events} == {TOOL_CALL, DELEGATION}

    def test_get_events_filter_by_tags(self):
        bus = EventBus()
        # DELEGATION has {"display", "memory", "console"} — includes "memory"
        bus.emit(DELEGATION, msg="a")
        # CDF_FILE_QUERY has empty tags — no "memory"
        bus.emit(CDF_FILE_QUERY, msg="b")
        # DEBUG has {"console"} — no "memory"
        bus.emit(DEBUG, msg="c")
        events = bus.get_events(tags={"memory"})
        assert len(events) == 1
        assert events[0].type == DELEGATION

    def test_get_events_since_index(self):
        bus = EventBus()
        bus.emit(DEBUG, msg="old")
        bus.emit(DEBUG, msg="new")
        events = bus.get_events(since_index=1)
        assert len(events) == 1
        assert "new" in events[0].summary

    def test_clear(self):
        bus = EventBus()
        bus.emit(DEBUG, msg="a")
        bus.emit(DEBUG, msg="b")
        assert len(bus) == 2
        bus.clear()
        assert len(bus) == 0
        assert bus.get_events() == []

    def test_thread_safety(self):
        bus = EventBus()
        errors = []

        def emitter(prefix):
            try:
                for i in range(100):
                    bus.emit(DEBUG, msg=f"{prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emitter, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(bus) == 400

    def test_event_ts_is_iso8601(self):
        bus = EventBus()
        event = bus.emit(DEBUG, msg="ts test")
        # Should be parseable ISO 8601
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(event.ts)
        assert dt.tzinfo is not None  # UTC timezone

    def test_tags_deterministic(self):
        """Every registered event type gets its tags from _resolve_tags()."""
        bus = EventBus()
        for event_type in INFRASTRUCTURE_TAGS:
            expected_tags = _resolve_tags(event_type)
            event = bus.emit(event_type, msg=f"test {event_type}")
            assert event.tags == expected_tags, (
                f"Event type {event_type!r}: expected tags {expected_tags}, got {event.tags}"
            )

    def test_unknown_type_gets_empty_tags(self):
        """An unregistered event type gets an empty frozenset."""
        bus = EventBus()
        event = bus.emit("totally_unknown_type", msg="mystery")
        assert event.tags == frozenset()

    def test_ctx_orchestrator_tags(self):
        """ctx:orchestrator tag is present on the 9 expected event types."""
        orchestrator_types = {
            SUB_AGENT_TOOL, SUB_AGENT_ERROR,
            DATA_FETCHED, DATA_COMPUTED,
            RENDER_EXECUTED, CUSTOM_OP_FAILURE,
            FETCH_ERROR, RENDER_ERROR, PLOT_ACTION,
        }
        for event_type in orchestrator_types:
            tags = _resolve_tags(event_type)
            assert "ctx:orchestrator" in tags, (
                f"{event_type!r} should have ctx:orchestrator tag"
            )

    def test_ctx_planner_and_orchestrator_synced(self):
        """Every event with ctx:orchestrator must also have ctx:planner and vice versa."""
        for event_type in INFRASTRUCTURE_TAGS:
            tags = _resolve_tags(event_type)
            has_orch = "ctx:orchestrator" in tags
            has_plan = "ctx:planner" in tags
            assert has_orch == has_plan, (
                f"{event_type}: ctx:orchestrator={has_orch} but ctx:planner={has_plan}"
            )

    def test_sub_agent_error_has_memory_tag(self):
        """SUB_AGENT_ERROR should have 'memory' in its infrastructure tags."""
        tags = INFRASTRUCTURE_TAGS[SUB_AGENT_ERROR]
        assert "memory" in tags, "SUB_AGENT_ERROR must have 'memory' tag for memory extraction"
        assert "console" in tags, "SUB_AGENT_ERROR must keep 'console' tag"

    def test_context_compaction_event_type(self):
        """CONTEXT_COMPACTION is registered with display, console, and all ctx:* tags."""
        assert CONTEXT_COMPACTION in INFRASTRUCTURE_TAGS
        expected = frozenset({
            "display", "console",
            "ctx:envoy", "ctx:viz_plotly", "ctx:viz_mpl", "ctx:viz_jsx",
            "ctx:dataops", "ctx:planner", "ctx:orchestrator",
        })
        assert _resolve_tags(CONTEXT_COMPACTION) == expected

    def test_context_compaction_emit(self):
        """CONTEXT_COMPACTION events are emitted with correct tags and summary."""
        bus = EventBus()
        event = bus.emit(
            CONTEXT_COMPACTION,
            agent="orchestrator",
            level="info",
            data={
                "before_tokens": 150000,
                "after_tokens": 50000,
                "context_window": 200000,
            },
        )
        assert event.type == CONTEXT_COMPACTION
        assert event.agent == "orchestrator"
        assert "display" in event.tags
        assert "console" in event.tags
        assert "ctx:envoy" in event.tags
        assert "ctx:orchestrator" in event.tags
        # Formatter generates summary from data
        assert "150000" in event.summary
        assert "50000" in event.summary

    def test_event_has_summary_details(self):
        """Events always have summary and details fields."""
        bus = EventBus()
        event = bus.emit(DATA_FETCHED, msg="test", data={
            "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            "outputs": ["AC_H2_MFI.BGSEc"],
        })
        assert event.summary  # non-empty
        assert event.details  # non-empty
        assert "AC_H2_MFI.BGSEc" in event.summary

    def test_span_id_and_children_default(self):
        """Default span_id and children are empty."""
        bus = EventBus()
        event = bus.emit(DEBUG, msg="test")
        assert event.span_id == ""
        assert event.children == ()


# ---- ContextVar singleton ----

class TestContextVar:
    def test_set_and_get(self):
        bus = EventBus(session_id="ctx-test")
        set_event_bus(bus)
        assert get_event_bus() is bus

    def test_fallback_bus(self):
        """get_event_bus() returns a fallback when no context bus is set."""
        # Clear the context var first, then copy
        from agent.event_bus import _bus_var
        token = _bus_var.set(None)
        try:
            bus = get_event_bus()
            assert bus is not None
            assert bus.session_id == "<fallback>"
        finally:
            _bus_var.reset(token)


# ---- DebugLogListener ----

class TestDebugLogListener:
    def test_logs_at_correct_level(self):
        mock_logger = MagicMock(spec=logging.Logger)
        listener = DebugLogListener(mock_logger)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(DELEGATION, level="info", msg="[Router] Delegating",
                 data={"target": "[Router] Delegating"})
        # Listener logs event.summary (generated by formatter)
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.INFO
        assert call_args[1]["extra"]["log_tag"] == "delegation"

    def test_error_level(self):
        mock_logger = MagicMock(spec=logging.Logger)
        listener = DebugLogListener(mock_logger)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(FETCH_ERROR, level="error", msg="Failed to fetch")
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.ERROR
        assert call_args[1]["extra"]["log_tag"] == "error"

    def test_debug_level_no_tag(self):
        mock_logger = MagicMock(spec=logging.Logger)
        listener = DebugLogListener(mock_logger)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(DEBUG, level="debug", msg="Some debug info")
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.DEBUG
        assert "Some debug info" in call_args[0][1]
        assert call_args[1]["extra"]["log_tag"] == ""

    def test_tag_mapping(self):
        """Verify event types map to correct legacy tags."""
        assert DebugLogListener._TYPE_TO_TAG.get(DELEGATION, "") == "delegation"
        assert DebugLogListener._TYPE_TO_TAG.get(THINKING, "") == "thinking"
        assert DebugLogListener._TYPE_TO_TAG.get(DATA_FETCHED, "") == "data_fetched"
        assert DebugLogListener._TYPE_TO_TAG.get(DEBUG, "") == ""

    def test_new_type_tag_mapping(self):
        """Verify new event types map to correct legacy tags."""
        assert DebugLogListener._TYPE_TO_TAG.get(TOOL_CALL_LOG, "") == ""
        assert DebugLogListener._TYPE_TO_TAG.get(TOOL_RESULT_LOG, "") == ""
        assert DebugLogListener._TYPE_TO_TAG.get(TOOL_ERROR_LOG, "") == "error"


# ---- SSEEventListener ----

class TestSSEEventListener:
    def test_forwards_display_tagged_events(self):
        callback = MagicMock()
        listener = SSEEventListener(callback)

        bus = EventBus()
        bus.subscribe(listener)

        # DELEGATION default tags include "display" and "console"
        bus.emit(DELEGATION, level="info", msg="Routing to PSP",
                 data={"target": "PSP"})
        callback.assert_called_once()
        payload = callback.call_args[0][0]
        assert payload["type"] == "log_line"
        assert "PSP" in payload["text"]

    def test_skips_non_display_non_console_events(self):
        callback = MagicMock()
        listener = SSEEventListener(callback)

        bus = EventBus()
        bus.subscribe(listener)

        # CDF_FILE_QUERY has no tags — should be skipped
        bus.emit(CDF_FILE_QUERY, level="debug", msg="internal stuff")
        callback.assert_not_called()

    def test_forwards_warnings_regardless_of_tags(self):
        callback = MagicMock()
        listener = SSEEventListener(callback)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(DEBUG, level="warning", msg="Something unusual")
        callback.assert_called_once()

    def test_forwards_errors_regardless_of_tags(self):
        callback = MagicMock()
        listener = SSEEventListener(callback)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(TOOL_ERROR, level="error", msg="Tool failed")
        callback.assert_called_once()


# ---- OperationsLogListener ----

class TestOperationsLogListener:
    def test_records_data_fetched(self):
        ops_log = MagicMock()
        listener = OperationsLogListener(lambda: ops_log)

        bus = EventBus()
        bus.subscribe(listener)

        # DATA_FETCHED default tags include "pipeline"
        bus.emit(DATA_FETCHED, msg="Fetched data", data={
            "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            "outputs": ["AC_H2_MFI.BGSEc"],
            "status": "success",
        })
        ops_log.record.assert_called_once_with(
            tool="fetch_data",
            args={"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            outputs=["AC_H2_MFI.BGSEc"],
            status="success",
            error=None,
        )

    def test_records_data_computed(self):
        ops_log = MagicMock()
        listener = OperationsLogListener(lambda: ops_log)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(DATA_COMPUTED, msg="Computed", data={
            "args": {"description": "magnitude", "code": "np.sqrt(x)"},
            "outputs": ["Bmag"],
            "inputs": ["AC_H2_MFI.BGSEc"],
            "status": "success",
        })
        ops_log.record.assert_called_once_with(
            tool="run_code",
            args={"description": "magnitude", "code": "np.sqrt(x)"},
            outputs=["Bmag"],
            inputs=["AC_H2_MFI.BGSEc"],
            status="success",
            error=None,
        )

    def test_render_executed_not_recorded_by_listener(self):
        """RENDER_EXECUTED recording is now done inline in core.py, not by OperationsLogListener."""
        ops_log = MagicMock()
        listener = OperationsLogListener(lambda: ops_log)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(RENDER_EXECUTED, msg="Rendered", data={
            "args": {"figure_json": "..."},
            "outputs": [],
            "inputs": ["Bmag"],
            "status": "success",
        })
        ops_log.record.assert_not_called()

    def test_records_plot_action(self):
        ops_log = MagicMock()
        listener = OperationsLogListener(lambda: ops_log)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(PLOT_ACTION, msg="Export", data={
            "args": {"action": "export_png"},
            "outputs": [],
            "status": "success",
        })
        ops_log.record.assert_called_once()
        assert ops_log.record.call_args.kwargs["tool"] == "manage_plot"

    def test_records_data_created(self):
        ops_log = MagicMock()
        listener = OperationsLogListener(lambda: ops_log)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(DATA_CREATED, msg="Created", data={
            "args": {"description": "manual df"},
            "outputs": ["manual_df"],
            "status": "success",
        })
        ops_log.record.assert_called_once()
        assert ops_log.record.call_args.kwargs["tool"] == "run_code"

    def test_records_mpl_render_executed(self):
        ops_log = MagicMock()
        listener = OperationsLogListener(lambda: ops_log)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(MPL_RENDER_EXECUTED, msg="MPL rendered", data={
            "args": {"script": "import matplotlib..."},
            "outputs": [],
            "inputs": ["Bmag"],
            "status": "success",
        })
        ops_log.record.assert_called_once_with(
            tool="generate_mpl_script",
            args={"script": "import matplotlib..."},
            outputs=[],
            inputs=["Bmag"],
            status="success",
            error=None,
        )

    def test_records_jsx_render_executed(self):
        ops_log = MagicMock()
        listener = OperationsLogListener(lambda: ops_log)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(JSX_RENDER_EXECUTED, msg="JSX rendered", data={
            "args": {"component": "<Plot .../>"},
            "outputs": [],
            "inputs": ["solar_wind"],
            "status": "success",
        })
        ops_log.record.assert_called_once_with(
            tool="generate_jsx_component",
            args={"component": "<Plot .../>"},
            outputs=[],
            inputs=["solar_wind"],
            status="success",
            error=None,
        )

    def test_ignores_non_pipeline_events(self):
        ops_log = MagicMock()
        listener = OperationsLogListener(lambda: ops_log)

        bus = EventBus()
        bus.subscribe(listener)

        # DEBUG has {"console"} — no "pipeline" tag
        bus.emit(DEBUG, msg="Routing")
        ops_log.record.assert_not_called()


# ---- DisplayLogBuilder ----

class TestDisplayLogBuilder:
    def test_user_message(self):
        builder = DisplayLogBuilder()
        bus = EventBus()
        bus.subscribe(builder)

        bus.emit(USER_MESSAGE, msg="[User] Show ACE data",
                 data={"text": "Show ACE data"})
        assert len(builder.entries) == 1
        assert builder.entries[0]["role"] == "user"
        assert builder.entries[0]["content"] == "Show ACE data"

    def test_agent_response(self):
        builder = DisplayLogBuilder()
        bus = EventBus()
        bus.subscribe(builder)

        bus.emit(AGENT_RESPONSE, msg="[Agent] Here's the data",
                 data={"text": "Here's the data"})
        assert len(builder.entries) == 1
        assert builder.entries[0]["role"] == "agent"

    def test_milestone_event(self):
        builder = DisplayLogBuilder()
        bus = EventBus()
        bus.subscribe(builder)

        # DELEGATION default tags include "display"
        bus.emit(DELEGATION, level="info",
                 msg="[Router] Delegating to PSP specialist")
        assert len(builder.entries) == 1
        assert builder.entries[0]["role"] == "milestone"

    def test_non_display_event_ignored(self):
        builder = DisplayLogBuilder()
        bus = EventBus()
        bus.subscribe(builder)

        # CDF_FILE_QUERY has no tags (no "display")
        bus.emit(CDF_FILE_QUERY, msg="internal debug")
        assert len(builder.entries) == 0

    def test_full_conversation_flow(self):
        builder = DisplayLogBuilder()
        bus = EventBus()
        bus.subscribe(builder)

        bus.emit(USER_MESSAGE, data={"text": "Show me ACE data"})
        bus.emit(DELEGATION, level="info", msg="[Router] Routing to ACE")
        bus.emit(DATA_FETCHED, level="info", msg="[DataOps] Stored 'ACE.Bmag'")
        bus.emit(AGENT_RESPONSE, data={"text": "Here's the data"})

        assert len(builder.entries) == 4
        assert builder.entries[0]["role"] == "user"
        assert builder.entries[1]["role"] == "milestone"
        assert builder.entries[2]["role"] == "milestone"
        assert builder.entries[3]["role"] == "agent"


# ---- TokenLogListener ----

class TestTokenLogListener:
    def test_writes_jsonl(self, tmp_path):
        path = tmp_path / "token.jsonl"
        listener = TokenLogListener(path)

        bus = EventBus()
        bus.subscribe(listener)

        bus.emit(TOKEN_USAGE, msg="[Tokens] test", data={
            "agent_name": "OrchestratorAgent",
            "tool_context": "initial_message",
            "input_tokens": 100,
            "output_tokens": 50,
            "thinking_tokens": 10,
            "cached_tokens": 0,
            "cumulative_input": 100,
            "cumulative_output": 50,
            "cumulative_thinking": 10,
            "cumulative_cached": 0,
            "api_calls": 1,
        })
        listener.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["agent_name"] == "OrchestratorAgent"
        assert record["input_tokens"] == 100
        assert record["output_tokens"] == 50
        assert record["api_calls"] == 1

    def test_ignores_non_token_events(self, tmp_path):
        path = tmp_path / "token.jsonl"
        listener = TokenLogListener(path)

        bus = EventBus()
        bus.subscribe(listener)

        # DEBUG has {"console"} — no "token" tag
        bus.emit(DEBUG, msg="Some debug info")
        # TOOL_CALL has {"display", "memory", "console"} — no "token" tag
        bus.emit(TOOL_CALL, msg="test call")
        listener.close()

        content = path.read_text().strip()
        assert content == ""

    def test_token_usage_default_tags(self):
        assert _resolve_tags(TOKEN_USAGE) == frozenset({"token", "console"})

    def test_sse_forwards_token_usage(self):
        callback = MagicMock()
        listener = SSEEventListener(callback)

        bus = EventBus()
        bus.subscribe(listener)

        data = {
            "agent_name": "OrchestratorAgent",
            "input_tokens": 200,
            "output_tokens": 100,
            "api_calls": 2,
        }
        bus.emit(TOKEN_USAGE, msg="[Tokens] test", data=data)

        # SSEEventListener should send both a token_usage payload and a log_line
        calls = callback.call_args_list
        payloads = [c[0][0] for c in calls]
        types = [p["type"] for p in payloads]
        assert "token_usage" in types
        assert "log_line" in types

        token_payload = next(p for p in payloads if p["type"] == "token_usage")
        assert token_payload["data"]["agent_name"] == "OrchestratorAgent"


# ---- Agent Tool Registry ----

class TestAgentToolRegistry:
    """Smoke tests for AGENT_CALL_REGISTRY and AGENT_INFORMED_REGISTRY."""

    def test_call_registry_has_expected_keys(self):
        expected_keys = {"ctx:orchestrator", "ctx:envoy", "ctx:viz_plotly", "ctx:viz_mpl", "ctx:viz_jsx", "ctx:dataops", "ctx:planner", "ctx:data_io", "ctx:eureka"}
        assert set(AGENT_CALL_REGISTRY.keys()) == expected_keys

    def test_informed_registry_has_expected_keys(self):
        expected_keys = {"ctx:orchestrator", "ctx:envoy", "ctx:viz_plotly", "ctx:viz_mpl", "ctx:viz_jsx", "ctx:dataops", "ctx:planner", "ctx:data_io", "ctx:eureka"}
        assert set(AGENT_INFORMED_REGISTRY.keys()) == expected_keys

    def test_call_registry_values_are_nonempty_frozensets(self):
        for ctx, tools in AGENT_CALL_REGISTRY.items():
            assert isinstance(tools, frozenset), f"{ctx} value is not frozenset"
            assert len(tools) > 0, f"{ctx} has empty tool set"

    def test_informed_registry_values_are_nonempty_frozensets(self):
        for ctx, tools in AGENT_INFORMED_REGISTRY.items():
            assert isinstance(tools, frozenset), f"{ctx} value is not frozenset"
            assert len(tools) > 0, f"{ctx} has empty tool set"

    def test_informed_is_superset_of_call(self):
        """AGENT_INFORMED_REGISTRY should always be a superset of AGENT_CALL_REGISTRY."""
        for ctx in AGENT_CALL_REGISTRY:
            call_tools = AGENT_CALL_REGISTRY[ctx]
            informed_tools = AGENT_INFORMED_REGISTRY.get(ctx)
            assert call_tools <= informed_tools, (
                f"{ctx}: call tools not subset of informed tools. "
                f"Missing: {call_tools - informed_tools}"
            )

    def test_specific_tool_in_expected_agents(self):
        """Verify key tools appear in the expected agents' call sets."""
        # fetch_data should be in mission (call)
        assert "fetch_data" in AGENT_CALL_REGISTRY["ctx:envoy"]
        # render_plotly_json should be in viz (call)
        assert "render_plotly_json" in AGENT_CALL_REGISTRY["ctx:viz_plotly"]
        # run_code should be in dataops (call)
        assert "run_code" in AGENT_CALL_REGISTRY["ctx:dataops"]
        # list_fetched_data should be in all agents (call)
        for ctx in AGENT_CALL_REGISTRY:
            assert "list_fetched_data" in AGENT_CALL_REGISTRY[ctx], (
                f"{ctx} should have list_fetched_data"
            )

    def test_data_inspection_tools_in_viz_agents(self):
        """Data inspection tools should be callable by viz agents."""
        for ctx in ["ctx:viz_plotly", "ctx:viz_mpl", "ctx:viz_jsx"]:
            assert "describe_data" in AGENT_CALL_REGISTRY[ctx], (
                f"{ctx} should have describe_data"
            )
            assert "preview_data" in AGENT_CALL_REGISTRY[ctx], (
                f"{ctx} should have preview_data"
            )

    def test_informed_tools_for_viz(self):
        """VizAgent should be informed about fetch_data and run_code."""
        viz_informed = AGENT_INFORMED_REGISTRY.get("ctx:viz_plotly")
        assert "fetch_data" in viz_informed
        assert "run_code" in viz_informed

    def test_informed_tools_for_dataops(self):
        """DataOpsAgent should be informed about fetch_data."""
        dataops_informed = AGENT_INFORMED_REGISTRY.get("ctx:dataops")
        assert "fetch_data" in dataops_informed


# ---- Data-driven tag derivation ----

class TestDataDrivenTags:
    """Verify that _resolve_tags() correctly computes tags from the layered system."""

    def test_universal_events_have_all_ctx_tags(self):
        """Events in UNIVERSAL_CTX_EVENTS should have all 5 ctx:* tags."""
        for event_type in UNIVERSAL_CTX_EVENTS:
            tags = _resolve_tags(event_type)
            for ctx in ALL_CTX_TAGS:
                assert ctx in tags, (
                    f"Universal event {event_type!r} missing {ctx}"
                )

    def test_routing_events_have_only_planner_orchestrator(self):
        """Events in ROUTING_ONLY_EVENTS should only have ctx:planner and ctx:orchestrator."""
        for event_type in ROUTING_ONLY_EVENTS:
            tags = _resolve_tags(event_type)
            ctx_tags = {t for t in tags if t.startswith("ctx:")}
            assert ctx_tags == {"ctx:planner", "ctx:orchestrator"}, (
                f"Routing event {event_type!r} has wrong ctx tags: {ctx_tags}"
            )

    def test_data_fetched_visible_to_all_agents(self):
        """DATA_FETCHED should now be visible to all 5 agent types."""
        tags = _resolve_tags(DATA_FETCHED)
        for ctx in ALL_CTX_TAGS:
            assert ctx in tags, f"DATA_FETCHED missing {ctx}"

    def test_data_computed_visible_to_all_agents(self):
        """DATA_COMPUTED should now be visible to all 5 agent types."""
        tags = _resolve_tags(DATA_COMPUTED)
        for ctx in ALL_CTX_TAGS:
            assert ctx in tags, f"DATA_COMPUTED missing {ctx}"

    def test_render_executed_visible_to_viz_planner_orch(self):
        """RENDER_EXECUTED should be visible to viz, planner, orchestrator (not mission)."""
        tags = _resolve_tags(RENDER_EXECUTED)
        assert "ctx:viz_plotly" in tags
        assert "ctx:planner" in tags
        assert "ctx:orchestrator" in tags
        # Mission agents don't need render events — they discover and fetch, not plot
        assert "ctx:envoy" not in tags

    def test_infrastructure_tags_preserved(self):
        """Infrastructure tags should be preserved in _resolve_tags()."""
        for event_type, infra in INFRASTRUCTURE_TAGS.items():
            resolved = _resolve_tags(event_type)
            # All infrastructure tags should be present
            assert infra <= resolved, (
                f"{event_type}: infrastructure tags {infra - resolved} missing from _resolve_tags()"
            )

    def test_resolve_tags_covers_all_infrastructure_events(self):
        """Every event in INFRASTRUCTURE_TAGS should produce non-empty tags."""
        for event_type in INFRASTRUCTURE_TAGS:
            tags = _resolve_tags(event_type)
            infra = INFRASTRUCTURE_TAGS[event_type]
            assert infra <= tags, (
                f"{event_type!r}: infrastructure tags not included in _resolve_tags()"
            )

    def test_dynamic_tags_after_registry_mutation(self):
        """After adding a tool to an agent's informed set, _resolve_tags() reflects it."""
        tags_before = _resolve_tags(RENDER_EXECUTED)

        added = AGENT_INFORMED_REGISTRY.add(
            "ctx:dataops", "render_plotly_json", "test dynamic tags"
        )
        try:
            if added:
                tags_after = _resolve_tags(RENDER_EXECUTED)
                assert "ctx:dataops" in tags_after, (
                    "After adding render_plotly_json to dataops, "
                    "RENDER_EXECUTED should include ctx:dataops"
                )
        finally:
            if added:
                AGENT_INFORMED_REGISTRY.drop(
                    "ctx:dataops", "render_plotly_json", "test cleanup"
                )


# ---- Span tests ----

class TestSpan:
    def test_span_collects_children(self):
        """Sub-events emitted inside a span are collected as children."""
        bus = EventBus()
        all_events = []
        bus.subscribe(lambda e: all_events.append(e))

        with bus.span(DATA_FETCHED, agent="orchestrator", data={
            "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            "outputs": ["AC_H2_MFI.BGSEc"],
        }) as span_id:
            assert span_id  # non-empty
            bus.emit(CDF_FILE_QUERY, agent="cdf", msg="Found 3 files",
                     data={"file_count": 3, "dataset_id": "AC_H2_MFI"})
            bus.emit(CDF_DOWNLOAD, agent="cdf", msg="Downloaded file.cdf",
                     data={"filename": "file.cdf"})

        # The span-closing parent event should have children
        parent_events = [e for e in all_events if e.type == DATA_FETCHED and e.children]
        assert len(parent_events) == 1
        parent = parent_events[0]
        assert len(parent.children) == 2
        assert parent.children[0]["type"] == CDF_FILE_QUERY
        assert parent.children[1]["type"] == CDF_DOWNLOAD

    def test_span_generates_summary(self):
        """Span closing generates summary from formatter."""
        bus = EventBus()
        with bus.span(DATA_FETCHED, agent="orchestrator", data={
            "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            "outputs": ["AC_H2_MFI.BGSEc"],
        }):
            pass

        events = bus.get_events(types={DATA_FETCHED})
        assert len(events) == 1
        assert "AC_H2_MFI.BGSEc" in events[0].summary

    def test_span_sub_events_stored(self):
        """Sub-events inside a span are still stored in the event list."""
        bus = EventBus()
        with bus.span(DATA_FETCHED, agent="orchestrator", data={
            "args": {}, "outputs": ["test"],
        }):
            bus.emit(CDF_FILE_QUERY, agent="cdf", msg="query")
            bus.emit(CDF_DOWNLOAD, agent="cdf", msg="download")

        # Sub-events + parent = 3 events total
        all_events = bus.get_events()
        types = [e.type for e in all_events]
        assert CDF_FILE_QUERY in types
        assert CDF_DOWNLOAD in types
        assert DATA_FETCHED in types

    def test_nested_spans_not_broken(self):
        """Nested spans don't break (though uncommon)."""
        bus = EventBus()
        with bus.span(DELEGATION, agent="orch", data={"target": "outer"}):
            bus.emit(DEBUG, msg="outer sub-event")
            with bus.span(DATA_FETCHED, agent="mission", data={
                "args": {}, "outputs": ["inner"],
            }):
                bus.emit(CDF_FILE_QUERY, agent="cdf", msg="inner sub")

        events = bus.get_events()
        delegation_events = [e for e in events if e.type == DELEGATION and e.children]
        assert len(delegation_events) == 1


# ---- Formatter tests ----

class TestFormatters:
    def test_data_fetched_formatter(self):
        from agent.event_formatters import format_event
        summary, details = format_event("data_fetched", "orchestrator", {
            "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            "outputs": ["AC_H2_MFI.BGSEc"],
            "n_pts": 1000,
            "n_cols": 3,
        })
        assert "AC_H2_MFI.BGSEc" in summary
        assert "1000" in summary
        assert "AC_H2_MFI" in details

    def test_data_fetched_already_loaded(self):
        from agent.event_formatters import format_event
        summary, details = format_event("data_fetched", "orchestrator", {
            "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc",
                     "already_loaded": True},
            "outputs": ["AC_H2_MFI.BGSEc"],
        })
        assert "already loaded" in summary

    def test_data_fetched_error(self):
        from agent.event_formatters import format_event
        summary, details = format_event("data_fetched", "orchestrator", {
            "args": {"dataset_id": "AC_H2_MFI"},
            "status": "error",
            "error": "Dataset not found",
        })
        assert "FAILED" in summary
        assert "Dataset not found" in summary

    def test_data_computed_formatter(self):
        from agent.event_formatters import format_event
        summary, details = format_event("data_computed", "orchestrator", {
            "args": {"code": "np.sqrt(x**2 + y**2)"},
            "outputs": ["Bmag"],
            "inputs": ["AC_H2_MFI.BGSEc"],
            "n_pts": 500,
        })
        assert "Bmag" in summary
        assert "500" in summary

    def test_render_executed_formatter(self):
        from agent.event_formatters import format_event
        summary, details = format_event("render_executed", "orchestrator", {
            "inputs": ["Bmag", "Bx", "By"],
            "n_panels": 2,
            "status": "success",
        })
        assert "2 panel" in summary
        assert "Bmag" in summary

    def test_delegation_formatter(self):
        from agent.event_formatters import format_event
        summary, details = format_event("delegation", "orchestrator", {
            "target": "PSP",
        })
        assert "PSP" in summary

    def test_sub_agent_tool_formatter(self):
        from agent.event_formatters import format_event
        summary, details = format_event("sub_agent_tool", "PSP_agent", {
            "tool_name": "fetch_data",
            "tool_result": {"status": "success"},
        })
        assert "PSP_agent" in summary
        assert "fetch_data" in summary

    def test_default_formatter_with_msg(self):
        from agent.event_formatters import format_event
        summary, details = format_event("unknown_type", "test", {
            "_msg": "Hello world",
        })
        assert "Hello world" in summary

    def test_default_formatter_empty(self):
        from agent.event_formatters import format_event
        summary, details = format_event("unknown_type", "test", {})
        assert summary == ""

    def test_user_message_formatter(self):
        from agent.event_formatters import format_event
        summary, details = format_event("user_message", "user", {
            "text": "Show me ACE magnetic field data",
        })
        assert "User:" in summary
        assert "ACE" in summary

    def test_token_usage_formatter(self):
        from agent.event_formatters import format_event
        summary, details = format_event("token_usage", "orchestrator", {
            "agent_name": "OrchestratorAgent",
            "input_tokens": 100,
            "output_tokens": 50,
            "cumulative_input": 200,
            "cumulative_output": 100,
        })
        assert "OrchestratorAgent" in summary
        assert "100" in summary

    def test_memory_extraction_done_formatter(self):
        from agent.event_formatters import format_event
        summary, details = format_event("memory_extraction_done", "MemoryAgent", {
            "actions": {"add": 2, "drop": 1, "edit": 0},
        })
        assert "+2" in summary
        assert "-1" in summary


# ---- EventFeedBuffer peek tests ----

class TestEventFeedPeek:
    """Tests for EventFeedBuffer.peek() and peek_details() methods."""

    def _make_feed(self, bus, ctx_tag="ctx:orchestrator"):
        from agent.event_feed import EventFeedBuffer
        return EventFeedBuffer(bus, ctx_tag)

    def test_peek_returns_untagged_events(self):
        """Events with no ctx tags (e.g. DEBUG, THINKING) are visible via peek."""
        bus = EventBus()
        # DEBUG has {"console"} — no ctx:orchestrator
        bus.emit(DEBUG, agent="viz_plotly", msg="internal debug")
        # THINKING has {"display", "console"} — no ctx:orchestrator
        bus.emit(THINKING, agent="viz_plotly", msg="thinking about data")

        feed = self._make_feed(bus)
        result = feed.peek()
        assert result["status"] == "success"
        assert result["count"] == 2
        types = {e["type"] for e in result["events"]}
        assert DEBUG in types
        assert THINKING in types

    def test_peek_filters_by_agent(self):
        """Agent name filter returns only matching events."""
        bus = EventBus()
        bus.emit(DEBUG, agent="viz_plotly", msg="viz debug")
        bus.emit(DEBUG, agent="dataops", msg="dataops debug")
        bus.emit(DEBUG, agent="viz_mpl", msg="mpl debug")

        feed = self._make_feed(bus)
        result = feed.peek(agent_filter="viz_plotly")
        assert result["count"] == 1
        assert result["events"][0]["agent"] == "viz_plotly"

    def test_peek_excludes_orchestrator_visible(self):
        """Events with ctx:orchestrator tag are excluded from peek."""
        bus = EventBus()
        # DELEGATION has ctx:orchestrator — should be excluded
        bus.emit(DELEGATION, agent="orchestrator", msg="delegating",
                 data={"target": "PSP"})
        # DEBUG has no ctx:orchestrator — should be included
        bus.emit(DEBUG, agent="viz_plotly", msg="internal")

        feed = self._make_feed(bus, ctx_tag="ctx:orchestrator")
        result = feed.peek()
        assert result["count"] == 1
        assert result["events"][0]["type"] == DEBUG

    def test_peek_does_not_advance_cursor(self):
        """Peek is stateless — calling it does NOT affect check() results."""
        bus = EventBus()
        # Emit an event visible to orchestrator
        bus.emit(DELEGATION, agent="orchestrator", msg="delegating",
                 data={"target": "PSP"})
        # Emit an event NOT visible to orchestrator
        bus.emit(DEBUG, agent="viz_plotly", msg="internal")

        feed = self._make_feed(bus, ctx_tag="ctx:orchestrator")

        # Peek first
        peek_result = feed.peek()
        assert peek_result["count"] == 1  # only the DEBUG event

        # Check should still return the DELEGATION event (cursor not advanced)
        check_result = feed.check()
        assert check_result["count"] == 1
        assert check_result["events"][0]["type"] == DELEGATION

        # Second check should return nothing (cursor advanced by check)
        check_result2 = feed.check()
        assert check_result2["count"] == 0

    def test_peek_since_seconds(self):
        """Time window filtering works."""
        bus = EventBus()
        bus.emit(DEBUG, agent="viz_plotly", msg="recent event")

        feed = self._make_feed(bus)

        # Very large window — should include the event
        result = feed.peek(since_seconds=3600)
        assert result["count"] == 1

        # Very small window (0 seconds) — event was just created, might still match
        # Use a time in the past to reliably exclude
        result_tight = feed.peek(since_seconds=0.0)
        # The event was created within the last second, so 0.0 excludes it
        assert result_tight["count"] == 0

    def test_peek_agent_prefix_match(self):
        """Prefix match: 'envoy' matches 'envoy:PSP', 'envoy:ACE'."""
        bus = EventBus()
        bus.emit(DEBUG, agent="envoy:PSP", msg="PSP event")
        bus.emit(DEBUG, agent="envoy:ACE", msg="ACE event")
        bus.emit(DEBUG, agent="viz_plotly", msg="viz event")
        bus.emit(DEBUG, agent="envoy", msg="bare envoy event")

        feed = self._make_feed(bus)
        result = feed.peek(agent_filter="envoy")
        assert result["count"] == 3
        agents = {e["agent"] for e in result["events"]}
        assert agents == {"envoy:PSP", "envoy:ACE", "envoy"}

    def test_peek_details_returns_full_data(self):
        """ID-based detail lookup works for untagged events."""
        bus = EventBus()
        ev = bus.emit(DEBUG, agent="viz_plotly",
                      summary="internal debug", details="Full debug details here")

        feed = self._make_feed(bus)
        result = feed.peek_details([ev.id])
        assert result["status"] == "success"
        assert result["count"] == 1
        assert result["events"][0]["id"] == ev.id
        assert result["events"][0]["details"] == "Full debug details here"

    def test_peek_includes_tags_in_summary(self):
        """Peek summaries include the tags field."""
        bus = EventBus()
        bus.emit(DEBUG, agent="viz_plotly", msg="test")

        feed = self._make_feed(bus)
        result = feed.peek()
        assert result["count"] == 1
        assert "tags" in result["events"][0]
        assert isinstance(result["events"][0]["tags"], list)

    def test_peek_with_event_types_filter(self):
        """Event type filter works for peek."""
        bus = EventBus()
        bus.emit(DEBUG, agent="viz_plotly", msg="debug event")
        bus.emit(THINKING, agent="viz_plotly", msg="thinking event")

        feed = self._make_feed(bus)
        result = feed.peek(event_types=[DEBUG])
        assert result["count"] == 1
        assert result["events"][0]["type"] == DEBUG

    def test_peek_max_events_truncation(self):
        """Peek respects max_events and reports has_earlier."""
        bus = EventBus()
        for i in range(10):
            bus.emit(DEBUG, agent="viz_plotly", msg=f"event {i}")

        feed = self._make_feed(bus)
        result = feed.peek(max_events=3)
        assert result["count"] == 3
        assert result["has_earlier"] is True
        # Should return the LAST 3 events (most recent)
        assert "event 7" in result["events"][0]["summary"]
