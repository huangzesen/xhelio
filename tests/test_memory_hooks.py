"""Tests for MemoryHooks."""

from unittest.mock import MagicMock, patch
from agent.memory_hooks import MemoryHooks, candidates_from_log


def test_initial_state():
    hooks = MemoryHooks(ctx=MagicMock())
    assert hooks._agent is None
    assert hooks._turn_counter == 0
    assert hooks._last_op_index == 0


def test_candidates_from_log_none():
    """candidates_from_log handles None input gracefully."""
    result = candidates_from_log(None)
    assert result == []


def test_candidates_from_log_empty():
    """candidates_from_log handles an empty ops log."""
    mock_log = MagicMock()
    mock_log.get_records.return_value = []
    result = candidates_from_log(mock_log)
    assert result == []


def test_candidates_from_log_no_render_ops():
    """candidates_from_log skips non-render operations."""
    mock_log = MagicMock()
    mock_log.get_records.return_value = [
        {"tool": "fetch_data", "status": "success", "outputs": ["label1"], "id": "1"},
    ]
    result = candidates_from_log(mock_log)
    assert result == []


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
    ctx._sub_agents = {}
    ctx._sub_agents_lock = MagicMock()
    ctx._sub_agents_lock.__enter__ = MagicMock(return_value=None)
    ctx._sub_agents_lock.__exit__ = MagicMock(return_value=False)
    ctx._ops_log = MagicMock()
    ctx._ops_log.get_records.return_value = []
    ctx._session_id = "test-session"

    hooks = MemoryHooks(ctx=ctx)

    with patch("agent.memory_hooks.config") as mock_config:
        mock_config.get_data_dir.return_value = MagicMock()
        mock_config.get_data_dir.return_value.exists.return_value = False
        result = hooks.run_for_pipelines()

    assert result == []
