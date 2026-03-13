"""Tests for MemoryHooks."""

import threading
from unittest.mock import MagicMock, patch
from agent.memory_hooks import MemoryHooks


def test_initial_state():
    hooks = MemoryHooks(ctx=MagicMock())
    assert hooks._agent is None
    assert hooks._turn_counter == 0
    assert hooks._last_op_index == 0


def test_enumerate_pipeline_candidates_from_dag():
    """enumerate_pipeline_candidates returns candidates from a populated DAG."""
    from data_ops.dag import PipelineDAG

    dag = PipelineDAG()
    dag.add_node(
        op_id="op_000", tool="fetch_data", agent="orchestrator",
        args={"dataset": "AC_H2_MFI"}, inputs=[],
        outputs={"AC_H2_MFI.BGSEc": "mag field"}, status="success",
    )
    dag.add_node(
        op_id="op_001", tool="render_plotly_json", agent="viz",
        args={"description": "plot"}, inputs=["AC_H2_MFI.BGSEc"],
        outputs={}, status="success",
    )

    ctx = MagicMock()
    ctx.dag = dag
    ctx.session_id = "test123"
    ctx.memory_store = MagicMock()
    ctx.memory_store.total_tokens.return_value = 0

    hooks = MemoryHooks(ctx)
    candidates = hooks.enumerate_pipeline_candidates()

    assert len(candidates) == 1
    c = candidates[0]
    assert c["session_id"] == "test123"
    assert c["node_count"] == 2
    assert len(c["sources"]) == 1
    assert c["sources"][0]["tool"] == "fetch_data"
    assert c["sink"]["tool"] == "render_plotly_json"


def test_enumerate_pipeline_candidates_empty_dag():
    """enumerate_pipeline_candidates returns [] when DAG is empty."""
    ctx = MagicMock()
    ctx.dag = MagicMock()
    ctx.dag.node_count.return_value = 0

    hooks = MemoryHooks(ctx)
    assert hooks.enumerate_pipeline_candidates() == []


def test_on_memory_mutated_ignores_wrong_event():
    """on_memory_mutated ignores events that aren't MEMORY_EXTRACTION_DONE."""
    from agent.event_bus import MEMORY_EXTRACTION_START

    ctx = MagicMock()
    hooks = MemoryHooks(ctx=ctx)
    event = MagicMock()
    event.type = MEMORY_EXTRACTION_START
    hooks.on_memory_mutated(event)
    # Should not touch _memory_store
    ctx._memory_store.format_for_injection.assert_not_called()


def test_on_memory_mutated_updates_system_prompt():
    """on_memory_mutated refreshes the system prompt when memory changes."""
    from agent.event_bus import MEMORY_EXTRACTION_DONE

    ctx = MagicMock()
    ctx._memory_store.format_for_injection.return_value = "## Memory\nsome memory"
    ctx.chat = MagicMock()

    hooks = MemoryHooks(ctx=ctx)
    event = MagicMock()
    event.type = MEMORY_EXTRACTION_DONE

    with patch("agent.memory_hooks.get_system_prompt", return_value="base prompt"):
        hooks.on_memory_mutated(event)

    assert "some memory" in ctx._system_prompt
    assert "base prompt" in ctx._system_prompt
    ctx.chat.update_system_prompt.assert_called_once()


def test_on_memory_mutated_no_memory():
    """on_memory_mutated uses base prompt when no memory available."""
    from agent.event_bus import MEMORY_EXTRACTION_DONE

    ctx = MagicMock()
    ctx._memory_store.format_for_injection.return_value = ""
    ctx.chat = None

    hooks = MemoryHooks(ctx=ctx)
    event = MagicMock()
    event.type = MEMORY_EXTRACTION_DONE

    with patch("agent.memory_hooks.get_system_prompt", return_value="base prompt"):
        hooks.on_memory_mutated(event)

    assert ctx._system_prompt == "base prompt"


def test_reset():
    """reset() clears agent and counters."""
    ctx = MagicMock()
    hooks = MemoryHooks(ctx=ctx)
    hooks._agent = MagicMock()
    hooks._turn_counter = 5
    hooks._last_op_index = 10

    hooks.reset()

    assert hooks._agent is None
    assert hooks._turn_counter == 0
    assert hooks._last_op_index == 0


def test_trigger_hot_reload_no_chat():
    """trigger_hot_reload is a no-op when chat is None."""
    ctx = MagicMock()
    ctx.chat = None
    hooks = MemoryHooks(ctx=ctx)
    # Should not raise
    hooks.trigger_hot_reload()


def test_persist_operations_log_no_session():
    """persist_operations_log is a no-op when no session ID."""
    ctx = MagicMock()
    ctx._session_id = ""
    hooks = MemoryHooks(ctx=ctx)
    # Should not raise or call save_to_file
    hooks.persist_operations_log()
    ctx._ops_log.save_to_file.assert_not_called()


def test_maybe_extract_no_new_events():
    """maybe_extract returns early when no new console events."""
    ctx = MagicMock()
    ctx._event_bus.get_events.return_value = []
    hooks = MemoryHooks(ctx=ctx)
    hooks.maybe_extract()
    # Lock should not have been acquired
    assert not hooks._lock.locked()


def test_run_for_pipelines_no_candidates():
    """run_for_pipelines returns empty when no candidates."""
    ctx = MagicMock()
    ctx._event_bus.get_events.return_value = []
    ctx._memory_store.total_tokens.return_value = 0
    ctx._delegation = MagicMock()
    ctx._delegation.lock = threading.Lock()
    ctx._delegation.agents = {}
    ctx._ops_log = MagicMock()
    ctx._ops_log.get_records.return_value = []
    ctx._session_id = "test-session"

    hooks = MemoryHooks(ctx=ctx)

    with patch("agent.memory_hooks.config") as mock_config:
        mock_config.get_data_dir.return_value = MagicMock()
        mock_config.get_data_dir.return_value.exists.return_value = False
        result = hooks.run_for_pipelines()

    assert result == []
