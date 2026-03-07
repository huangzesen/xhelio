"""
Tests for agent.memory_agent — MemoryAgent (SubAgent subclass) tool-loop memory extraction.

Run with: python -m pytest tests/test_memory_agent.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.memory import Memory, MemoryStore
from agent.memory_agent import (
    MemoryAgent,
    MemoryContext,
    _validate_scopes,
    _VALID_SCOPE_RE,
    _VALID_TYPES,
)
from agent.event_bus import SessionEvent


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for state/memory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def memory_store(tmp_dir):
    """Provide a MemoryStore backed by a temp file."""
    return MemoryStore(path=tmp_dir / "memory.json")


@pytest.fixture
def agent(memory_store):
    """Provide a MemoryAgent (SubAgent) with mock adapter, started and stopped."""
    adapter = MagicMock()
    # Mock create_chat to return a mock ChatSession
    mock_chat = MagicMock()
    mock_chat.context_window.return_value = 0  # skip compaction
    adapter.create_chat.return_value = mock_chat
    a = MemoryAgent(
        adapter=adapter,
        model_name="test-model",
        memory_store=memory_store,
        verbose=False,
        session_id="test-session",
    )
    a.start()
    yield a
    a.stop()


def _make_event(type="user_message", agent="orchestrator", summary="test event", details=None, level="info"):
    """Build a minimal SessionEvent for testing."""
    return SessionEvent(
        id="evt_0001",
        type=type,
        ts="2026-01-01T00:00:00.000Z",
        agent=agent,
        level=level,
        summary=summary,
        details=details or {},
    )


def _make_context(**overrides):
    """Build a MemoryContext with sensible defaults, overridable."""
    defaults = dict(
        console_events=[
            _make_event(type="user_message", summary="[User] Show me ACE data"),
            _make_event(type="agent_response", summary="[Agent] Fetching ACE magnetic field data"),
            _make_event(type="tool_result", summary="fetch_data(AC_H2_MFI/BGSEc) -> error ERROR: missing parameter"),
        ],
        active_scopes=["generic", "envoy:ACE"],
        total_memory_tokens=500,
    )
    defaults.update(overrides)
    return MemoryContext(**defaults)


def _make_mock_usage(input_tokens=100, output_tokens=50, thinking_tokens=0, cached_tokens=0):
    """Build a mock usage object for LLMResponse."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.thinking_tokens = thinking_tokens
    usage.cached_tokens = cached_tokens
    return usage


def _make_mock_response(text="", tool_calls=None, input_tokens=100, output_tokens=50):
    """Build a mock LLMResponse with usage."""
    resp = MagicMock()
    resp.text = text
    resp.tool_calls = tool_calls or []
    resp.usage = _make_mock_usage(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


# ---- run() via SubAgent ----

class TestRun:
    def test_empty_context_returns_empty(self, agent):
        """Should return empty for no console events (short-circuits before send())."""
        context = _make_context(console_events=[])
        result = agent.run(context)
        assert result == []

    def test_no_tool_calls_returns_empty_with_summary(self, agent):
        """When LLM returns text but no tool calls, result is [] and summary is emitted."""
        mock_response = _make_mock_response(text="Session involved ACE data fetching.")

        with patch.object(agent, "_llm_send", return_value=mock_response):
            context = _make_context()
            result = agent.run(context)

        assert result == []

    def test_tool_calls_execute_add(self, agent, memory_store):
        """Should execute add_memory tool calls and return executed actions."""
        tc = MagicMock()
        tc.name = "add_memory"
        tc.args = {"type": "pitfall", "scopes": ["envoy:ACE"], "content": "ACE data needs NaN check"}
        tc.id = "tc_001"
        first_response = _make_mock_response(tool_calls=[tc])
        second_response = _make_mock_response(text="Added a pitfall about ACE NaN values.")

        with patch.object(agent, "_llm_send", side_effect=[first_response, second_response]):
            context = _make_context()
            result = agent.run(context)

        assert len(result) == 1
        assert result[0]["action"] == "add"
        assert result[0]["type"] == "pitfall"
        assert len(memory_store.get_all()) == 1
        assert memory_store.get_all()[0].content == "ACE data needs NaN check"

    def test_tool_calls_execute_edit(self, agent, memory_store):
        """Should execute edit_memory tool calls."""
        # Pre-populate a memory
        memory_store.add(Memory(
            id="abc123", type="preference", scopes=["generic"],
            content="Old preference text",
        ))

        tc = MagicMock()
        tc.name = "edit_memory"
        tc.args = {"memory_id": "abc123", "content": "Updated preference text"}
        tc.id = "tc_002"
        first_response = _make_mock_response(tool_calls=[tc])
        second_response = _make_mock_response(text="Edited preference.")

        with patch.object(agent, "_llm_send", side_effect=[first_response, second_response]):
            context = _make_context()
            result = agent.run(context)

        assert len(result) == 1
        active = memory_store.get_all()
        assert len(active) == 1
        assert active[0].content == "Updated preference text"
        assert active[0].supersedes == "abc123"
        assert active[0].version == 2

    def test_tool_calls_execute_drop(self, agent, memory_store):
        """Should execute drop_memory tool calls."""
        memory_store.add(Memory(
            id="drop123", type="pitfall", scopes=["generic"],
            content="Outdated pitfall",
        ))

        tc = MagicMock()
        tc.name = "drop_memory"
        tc.args = {"memory_id": "drop123"}
        tc.id = "tc_003"
        first_response = _make_mock_response(tool_calls=[tc])
        second_response = _make_mock_response(text="Dropped outdated pitfall.")

        with patch.object(agent, "_llm_send", side_effect=[first_response, second_response]):
            context = _make_context()
            result = agent.run(context)

        assert len(result) == 1
        assert len(memory_store.get_all()) == 0  # archived

    def test_multiple_tool_rounds(self, agent, memory_store):
        """Should handle multiple rounds of tool calls."""
        # Round 1: add a pitfall
        tc1 = MagicMock()
        tc1.name = "add_memory"
        tc1.args = {"type": "pitfall", "scopes": ["generic"], "content": "Pitfall 1"}
        tc1.id = "tc_r1"
        resp1 = _make_mock_response(tool_calls=[tc1])

        # Round 2: add a preference
        tc2 = MagicMock()
        tc2.name = "add_memory"
        tc2.args = {"type": "preference", "scopes": ["generic"], "content": "Preference 1"}
        tc2.id = "tc_r2"
        resp2 = _make_mock_response(tool_calls=[tc2])

        # Round 3: done
        resp3 = _make_mock_response(text="Added two memories.")

        with patch.object(agent, "_llm_send", side_effect=[resp1, resp2, resp3]):
            context = _make_context()
            result = agent.run(context)

        assert len(result) == 2
        assert len(memory_store.get_all()) == 2

    def test_llm_failure_returns_empty(self, agent):
        """Should return empty on LLM failure — SubAgent catches and returns error."""
        with patch.object(agent, "_llm_send", side_effect=Exception("API error")):
            context = _make_context()
            result = agent.run(context)
        assert result == []


# ---- Scope validation ----

class TestScopeValidation:
    def test_valid_scopes(self):
        assert _validate_scopes(["generic"]) == ["generic"]
        assert _validate_scopes(["visualization"]) == ["visualization"]
        assert _validate_scopes(["data_ops"]) == ["data_ops"]
        assert _validate_scopes(["envoy:PSP"]) == ["envoy:PSP"]
        assert _validate_scopes(["envoy:ACE", "visualization"]) == ["envoy:ACE", "visualization"]

    def test_string_input_normalized(self):
        assert _validate_scopes("generic") == ["generic"]
        assert _validate_scopes("envoy:PSP") == ["envoy:PSP"]

    def test_invalid_scopes_default_to_generic(self):
        assert _validate_scopes(["invalid_scope"]) == ["generic"]
        assert _validate_scopes(["", "also_bad"]) == ["generic"]

    def test_non_list_non_string_defaults(self):
        assert _validate_scopes(42) == ["generic"]
        assert _validate_scopes(None) == ["generic"]

    def test_mixed_valid_invalid(self):
        result = _validate_scopes(["envoy:PSP", "bad_scope", "visualization"])
        assert result == ["envoy:PSP", "visualization"]

    def test_valid_types_constant(self):
        assert _VALID_TYPES == {"preference", "summary", "pitfall", "reflection"}

    def test_scope_regex(self):
        assert _VALID_SCOPE_RE.match("generic")
        assert _VALID_SCOPE_RE.match("visualization")
        assert _VALID_SCOPE_RE.match("data_ops")
        assert _VALID_SCOPE_RE.match("envoy:PSP")
        assert not _VALID_SCOPE_RE.match("invalid")
        assert not _VALID_SCOPE_RE.match("envoy:")


# ---- MemoryStore.execute_actions() ----

class TestMemoryStoreExecuteActions:
    def test_add_via_store(self, memory_store):
        actions = [
            {"action": "add", "type": "pitfall", "scopes": ["data_ops"], "content": "Check NaN before merge"},
        ]
        count = memory_store.execute_actions(actions, session_id="sess1")
        assert count == 1
        assert len(memory_store.get_all()) == 1
        assert memory_store.get_all()[0].source_session == "sess1"

    def test_edit_via_store(self, memory_store):
        memory_store.add(Memory(id="s1", type="preference", scopes=["generic"], content="Old"))
        actions = [
            {"action": "edit", "id": "s1", "content": "New content"},
        ]
        count = memory_store.execute_actions(actions)
        assert count == 1
        active = memory_store.get_all()
        assert len(active) == 1
        assert active[0].content == "New content"

    def test_drop_via_store(self, memory_store):
        memory_store.add(Memory(id="s2", type="pitfall", scopes=["generic"], content="Old pitfall"))
        actions = [
            {"action": "drop", "id": "s2"},
        ]
        count = memory_store.execute_actions(actions)
        assert count == 1
        assert len(memory_store.get_all()) == 0

    def test_invalid_actions_skipped(self, memory_store):
        actions = [
            {"action": "add", "type": "bad_type", "content": "test"},
            {"action": "edit", "id": "nonexist", "content": "x"},
            {"action": "drop", "id": "nonexist"},
            "not a dict",
        ]
        count = memory_store.execute_actions(actions)
        assert count == 0

    def test_mixed_actions_via_store(self, memory_store):
        memory_store.add(Memory(id="x1", type="preference", scopes=["generic"], content="Keep"))
        memory_store.add(Memory(id="x2", type="pitfall", scopes=["generic"], content="Remove"))
        actions = [
            {"action": "add", "type": "summary", "scopes": ["generic"], "content": "Session summary"},
            {"action": "edit", "id": "x1", "content": "Updated keep"},
            {"action": "drop", "id": "x2"},
        ]
        count = memory_store.execute_actions(actions)
        assert count == 3
        active = memory_store.get_all()
        assert len(active) == 2
        contents = {m.content for m in active}
        assert "Session summary" in contents
        assert "Updated keep" in contents

    def test_review_via_store_rejected(self, memory_store):
        """Review action is no longer supported via execute_actions (removed)."""
        memory_store.add(Memory(id="rv1", type="pitfall", scopes=["generic"], content="Test"))
        actions = [
            {"action": "review", "id": "rv1", "stars": 4, "comment": "Useful pitfall"},
        ]
        count = memory_store.execute_actions(actions, session_id="sess1")
        assert count == 0
        m = memory_store.get_all()[0]
        assert m.review_of == ""  # No review via execute_actions
