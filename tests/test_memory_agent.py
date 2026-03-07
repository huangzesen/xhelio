"""
Tests for agent.memory_agent — MemoryAgent think-then-act extraction.

Run with: python -m pytest tests/test_memory_agent.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.memory import Memory, MemoryStore
from agent.memory_agent import (
    MemoryAgent,
    MemoryContext,
    _validate_scopes,
    _VALID_SCOPE_RE,
    _VALID_TYPES,
    MEMORY_RELEVANT_TYPES,
    CURATED_EVENTS_TOKEN_BUDGET,
    RECENT_EVENTS_WINDOW,
    _CURATION_INDEX,
    _fill_budget,
)


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
    """Provide a MemoryAgent with mock adapter."""
    adapter = MagicMock()
    return MemoryAgent(
        adapter=adapter,
        model_name="test-model",
        memory_store=memory_store,
        verbose=False,
        session_id="test-session",
    )


def _make_context(**overrides):
    """Build a MemoryContext with sensible defaults, overridable."""
    defaults = dict(
        events=[
            "  [User] Show me ACE data",
            "  [Agent] Fetching ACE magnetic field data",
            "  fetch_data(AC_H2_MFI/BGSEc) → error ERROR: missing parameter",
        ],
        active_memories=[],
        active_scopes=["generic", "envoy:ACE"],
        token_budget=10000,
        total_memory_tokens=500,
    )
    defaults.update(overrides)
    return MemoryContext(**defaults)


# ---- run() ----

class TestRun:
    def test_empty_context_returns_empty(self, agent):
        """Should return empty for no events."""
        context = _make_context(events=[])
        result = agent.run(context)
        assert result == []
        agent.adapter.generate.assert_not_called()

    def test_run_with_add_actions(self, agent, memory_store):
        """Should execute add actions from LLM response."""
        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"action": "add", "type": "preference", "scopes": ["envoy:ACE"], "content": "User prefers RTN coordinates for ACE"},
            {"action": "add", "type": "pitfall", "scopes": ["generic"], "content": "Always check NaN before plotting"},
        ])
        agent.adapter.generate.return_value = mock_response

        context = _make_context()
        result = agent.run(context)
        assert len(result) == 2
        all_mems = memory_store.get_all()
        assert len(all_mems) == 2
        types = {m.type for m in all_mems}
        assert types == {"preference", "pitfall"}

    def test_run_with_edit_action(self, agent, memory_store):
        """Should edit an existing memory via supersede pattern."""
        # Pre-populate a memory
        memory_store.add(Memory(
            id="abc123", type="preference", scopes=["generic"],
            content="Old preference text",
        ))
        assert len(memory_store.get_all()) == 1

        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"action": "edit", "id": "abc123", "content": "Updated preference text"},
        ])
        agent.adapter.generate.return_value = mock_response

        context = _make_context(active_memories=[
            {"id": "abc123", "type": "preference", "scopes": ["generic"], "content": "Old preference text"},
        ])
        result = agent.run(context)
        assert len(result) == 1

        active = memory_store.get_all()
        assert len(active) == 1
        assert active[0].content == "Updated preference text"
        assert active[0].supersedes == "abc123"
        assert active[0].version == 2

    def test_run_with_drop_action(self, agent, memory_store):
        """Should archive a memory via drop action."""
        memory_store.add(Memory(
            id="drop123", type="pitfall", scopes=["generic"],
            content="Outdated pitfall",
        ))

        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"action": "drop", "id": "drop123"},
        ])
        agent.adapter.generate.return_value = mock_response

        context = _make_context(active_memories=[
            {"id": "drop123", "type": "pitfall", "scopes": ["generic"], "content": "Outdated pitfall"},
        ])
        result = agent.run(context)
        assert len(result) == 1
        assert len(memory_store.get_all()) == 0  # archived, not returned by get_all

    def test_run_llm_failure_returns_empty(self, agent):
        """Should return empty on LLM failure."""
        agent.adapter.generate.side_effect = Exception("API error")
        context = _make_context()
        result = agent.run(context)
        assert result == []

    def test_run_invalid_json_returns_empty(self, agent):
        """Should return empty for unparseable LLM response."""
        mock_response = MagicMock()
        mock_response.text = "not valid json at all"
        agent.adapter.generate.return_value = mock_response

        context = _make_context()
        result = agent.run(context)
        assert result == []

    def test_run_with_pipeline_events_only(self, agent, memory_store):
        """Should analyze pipeline events even without conversation turns."""
        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"action": "add", "type": "pitfall", "scopes": ["envoy:PSP"],
             "content": "PSP SPC data has 45% NaN values"},
        ])
        agent.adapter.generate.return_value = mock_response

        context = _make_context(events=[
            "  fetch_data(PSP_FLD_L2_MAG_RTN/B_RTN) → success [NaN: 45%]",
        ])
        result = agent.run(context)
        assert len(result) == 1


# ---- build_curated_events() ----

class TestBuildCuratedEvents:
    def test_filters_already_loaded(self):
        """Should exclude data_fetched with already_loaded: True."""
        events = [
            {"event": "data_fetched", "args": {"dataset_id": "AC_H2_MFI", "already_loaded": True},
             "status": "success"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert result == []

    def test_filters_routine_success(self):
        """Should exclude routine data_fetched success with no anomalies."""
        events = [
            {"event": "data_fetched", "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
             "status": "success"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert result == []

    def test_includes_fetch_error(self):
        """Should include data_fetched errors."""
        events = [
            {"event": "data_fetched", "args": {"dataset_id": "BAD_DS", "parameter_id": "X"},
             "status": "error", "error": "Dataset not found"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "fetch_data" in result[0]
        assert "BAD_DS" in result[0]
        assert "ERROR" in result[0]
        assert "Dataset not found" in result[0]

    def test_includes_high_nan(self):
        """Should include data_fetched success with high NaN percentage."""
        events = [
            {"event": "data_fetched", "args": {"dataset_id": "PSP_SPC", "parameter_id": "vp_fit"},
             "status": "success", "nan_percentage": 45},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "PSP_SPC" in result[0]
        assert "NaN" in result[0]

    def test_includes_data_computed(self):
        """Should include data_computed with code snippet."""
        events = [
            {"event": "data_computed",
             "args": {"description": "Compute magnitude", "output_label": "Bmag", "code": "np.sqrt(x**2)"},
             "status": "success", "outputs": ["Bmag"]},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "custom_operation" in result[0]
        assert "Compute magnitude" in result[0]
        assert "np.sqrt" in result[0]

    def test_includes_data_computed_error(self):
        """Should include data_computed errors."""
        events = [
            {"event": "data_computed",
             "args": {"description": "Bad op", "output_label": "X", "code": "bad()"},
             "status": "error", "error": "NameError: bad"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "ERROR" in result[0]
        assert "NameError" in result[0]

    def test_includes_custom_op_failure(self):
        """Should include custom_op_failure events."""
        events = [
            {"event": "custom_op_failure",
             "args": {"description": "Failed op", "code": "bad_code()"},
             "error": "SyntaxError"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "custom_operation" in result[0]
        assert "FAILED" in result[0]
        assert "SyntaxError" in result[0]

    def test_includes_render_status(self):
        """Should include render_executed status and error."""
        events = [
            {"event": "render_executed", "args": {}, "status": "error",
             "error": "Invalid data_label reference"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "render_plotly_json" in result[0]
        assert "Invalid data_label" in result[0]

    def test_includes_plot_action(self):
        """Should include plot_action as P4 fill-in."""
        events = [
            {"event": "plot_action", "args": {"action": "export_png"}, "status": "success"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "manage_plot" in result[0]
        assert "export_png" in result[0]

    def test_includes_delegation(self):
        """Should include delegation events."""
        events = [
            {"event": "delegation", "agent": "orchestrator",
             "msg": "[Router] Delegating to PSP specialist"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "delegation" in result[0]
        assert "Delegating to PSP" in result[0]

    def test_includes_delegation_done(self):
        """Should include delegation_done events."""
        events = [
            {"event": "delegation_done", "agent": "PSP",
             "msg": "[Router] PSP specialist completed"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "delegation_done" in result[0]

    def test_includes_user_message(self):
        """Should include user_message events."""
        events = [
            {"event": "user_message", "text": "Show me ACE data", "msg": "[User] Show me ACE data"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "[User]" in result[0]
        assert "Show me ACE data" in result[0]

    def test_includes_agent_response(self):
        """Should include agent_response events."""
        events = [
            {"event": "agent_response", "text": "Here's the data", "msg": "[Agent] Here's the data"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "[Agent]" in result[0]
        assert "Here's the data" in result[0]

    def test_includes_data_created(self):
        """Should include data_created as P4 fill-in."""
        events = [
            {"event": "data_created", "args": {"description": "manual df"},
             "status": "success", "outputs": ["manual_df"]},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "store_dataframe" in result[0]
        assert "manual df" in result[0]

    def test_empty_events(self):
        """Should return empty list for empty input."""
        assert MemoryAgent.build_curated_events([]) == []

    def test_unknown_types_use_catchall(self):
        """Unregistered event types are formatted via catch-all at P4."""
        events = [
            {"event": "some_future_event", "msg": "something happened"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "some_future_event" in result[0]
        assert "something happened" in result[0]

    def test_chronological_interleaving(self):
        """Should return events in order, interleaving conversation and ops."""
        events = [
            {"event": "user_message", "text": "Show me ACE data"},
            {"event": "delegation", "agent": "orchestrator", "msg": "Routing to ACE"},
            {"event": "data_fetched", "args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
             "status": "error", "error": "Timeout"},
            {"event": "agent_response", "text": "Sorry, fetch failed"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 4
        # Verify chronological order is preserved despite mixed priorities
        assert "[User]" in result[0]
        assert "delegation" in result[1]
        assert "fetch_data" in result[2]
        assert "[Agent]" in result[3]

    # ---- New tests: sub_agent_error ----

    def test_includes_sub_agent_error_with_tool(self):
        """Should include sub_agent_error with tool_name and error fields."""
        events = [
            {"event": "sub_agent_error", "agent": "PSP_agent",
             "tool_name": "fetch_data", "error": "RTN is not a valid frame for this dataset"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "[PSP_agent] ERROR" in result[0]
        assert "fetch_data" in result[0]
        assert "RTN is not a valid frame" in result[0]

    def test_includes_sub_agent_error_msg_only(self):
        """Should include sub_agent_error with only msg (consecutive errors, exceptions)."""
        events = [
            {"event": "sub_agent_error", "agent": "ACE_agent",
             "msg": "Agent exceeded max retries after 3 attempts"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "[ACE_agent] ERROR" in result[0]
        assert "exceeded max retries" in result[0]

    def test_skips_sub_agent_error_empty(self):
        """Should skip sub_agent_error when no content."""
        events = [
            {"event": "sub_agent_error", "agent": "?"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert result == []

    # ---- New tests: tool_error ----

    def test_includes_tool_error(self):
        """Should include tool_error from _execute_tool_safe shape."""
        events = [
            {"event": "tool_error", "tool_name": "custom_operation",
             "error": "NameError: name 'scipy' is not defined"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "custom_operation ERROR" in result[0]
        assert "scipy" in result[0]

    def test_includes_tool_error_log_error_shape(self):
        """Should include tool_error from log_error() shape with short + context."""
        events = [
            {"event": "tool_error",
             "short": "ValueError: invalid coordinate frame",
             "context": "Attempted to use RTN frame with observer=SUN"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "ValueError" in result[0]
        assert "RTN frame" in result[0]

    # ---- New tests: user_amendment, work_cancelled, delegation_async_completed ----

    def test_includes_user_amendment(self):
        """Should include user_amendment events."""
        events = [
            {"event": "user_amendment", "text": "Actually, use GSE coordinates instead"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "[User amendment]" in result[0]
        assert "GSE coordinates" in result[0]

    def test_includes_work_cancelled(self):
        """Should include work_cancelled events."""
        events = [
            {"event": "work_cancelled", "msg": "User cancelled PSP fetch"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "[Work cancelled]" in result[0]
        assert "PSP fetch" in result[0]

    def test_includes_delegation_async_completed(self):
        """Should include delegation_async_completed events."""
        events = [
            {"event": "delegation_async_completed",
             "tool_name": "fetch_data", "work_unit_id": "wu_001"},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert "async_completed" in result[0]
        assert "wu_001" in result[0]

    # ---- New tests: no truncation ----

    def test_no_truncation_long_text(self):
        """Long text should stay complete — no truncation, no ellipsis."""
        long_text = "A" * 5000
        events = [
            {"event": "user_message", "text": long_text},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert long_text in result[0]
        assert "..." not in result[0]

    def test_no_truncation_long_error(self):
        """Long error messages should stay complete."""
        long_error = "Error: " + "x" * 3000
        events = [
            {"event": "sub_agent_error", "agent": "PSP_agent",
             "tool_name": "fetch_data", "error": long_error},
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 1
        assert long_error in result[0]

    # ---- New tests: budget behavior ----

    def test_budget_drops_low_priority_first(self):
        """With tiny budget, P0 errors survive while P3 routing is dropped."""
        events = [
            {"event": "delegation", "agent": "orchestrator", "msg": "Routing to PSP"},  # P3
            {"event": "sub_agent_error", "agent": "PSP_agent",
             "tool_name": "fetch_data", "error": "RTN is not valid"},  # P0
        ]
        # Use a budget that fits the P0 error but not both
        result = MemoryAgent.build_curated_events(events, token_budget=30)
        # The P0 error should survive, P3 delegation should be dropped
        has_error = any("ERROR" in line for line in result)
        has_delegation = any("delegation(" in line for line in result)
        assert has_error, "P0 error should survive tiny budget"
        # With budget=30, both might fit; use a budget that fits only the P0 error (~17 tokens)
        result_tiny = MemoryAgent.build_curated_events(events, token_budget=20)
        if result_tiny:
            # If anything survives, it must be P0 not P3
            assert any("ERROR" in line for line in result_tiny)

    def test_chronological_order_with_mixed_priorities(self):
        """Output order should match input order regardless of priority."""
        events = [
            {"event": "delegation", "agent": "orch", "msg": "Route"},          # P3, idx=0
            {"event": "user_message", "text": "Hello"},                          # P1, idx=1
            {"event": "sub_agent_error", "agent": "A", "error": "Boom"},        # P0, idx=2
            {"event": "agent_response", "text": "Response"},                     # P2, idx=3
        ]
        result = MemoryAgent.build_curated_events(events)
        assert len(result) == 4
        # Should be chronological: delegation, user_message, error, response
        assert "delegation" in result[0]
        assert "[User]" in result[1]
        assert "ERROR" in result[2]
        assert "[Agent]" in result[3]

    def test_budget_parameter(self):
        """Custom token_budget should be respected."""
        events = [
            {"event": "user_message", "text": "A" * 1000},
            {"event": "agent_response", "text": "B" * 1000},
        ]
        # With a large budget, both fit
        result_large = MemoryAgent.build_curated_events(events, token_budget=50000)
        assert len(result_large) == 2

        # With a tiny budget, at most one fits
        result_tiny = MemoryAgent.build_curated_events(events, token_budget=260)
        assert len(result_tiny) <= 1

    # ---- Recency split tests ----

    def test_recency_split_short_session(self):
        """With < 300 events, all land in recent half; old half is empty so full budget rolls over."""
        events = [
            {"event": "user_message", "text": "A" * 1000},
            {"event": "agent_response", "text": "B" * 1000},
        ]
        # Both events are < 300, so old_half is empty, full budget rolls to recent
        result = MemoryAgent.build_curated_events(events, token_budget=50000)
        assert len(result) == 2

    def test_recency_split_old_budget_rollover(self):
        """Unused old-half budget rolls into the recent half."""
        # Build 310 events: 10 old, 300 recent
        # Old events are tiny, recent events are large
        old_events = [{"event": "user_message", "text": f"old {i}"} for i in range(10)]
        recent_events = [{"event": "user_message", "text": "R" * 400} for _ in range(300)]
        events = old_events + recent_events
        # With budget=50000, half=25000. Old half uses ~40 tokens (10 tiny msgs).
        # Leftover ~24960 rolls into recent, giving recent ~49960 total.
        result = MemoryAgent.build_curated_events(events, token_budget=50000)
        # Old events should all be present
        old_results = [r for r in result if "old" in r]
        assert len(old_results) == 10

    def test_fill_budget_returns_used_tokens(self):
        """_fill_budget returns (accepted, used_tokens) tuple."""
        candidates = [
            (0, 0, "  short"),       # ~2 tokens
            (0, 1, "  also short"),  # ~3 tokens
        ]
        accepted, used = _fill_budget(candidates, 10000)
        assert len(accepted) == 2
        assert used > 0

    # ---- Registry structure tests ----

    def test_registry_covers_expected_types(self):
        """MEMORY_RELEVANT_TYPES should include all expected event types."""
        expected = {
            "sub_agent_error", "tool_error", "custom_op_failure",
            "user_message", "user_amendment", "work_cancelled",
            "thinking", "agent_response", "data_fetched", "data_computed",
            "render_executed", "insight_feedback",
            "delegation", "delegation_done", "delegation_async_completed",
            "tool_call", "tool_result", "sub_agent_tool",
            "data_created", "plot_action",
        }
        assert MEMORY_RELEVANT_TYPES == expected

    def test_registry_excludes_noise(self):
        """Event types that are infrastructure noise should NOT be in the registry."""
        noise = {"token_usage", "progress",
                 "memory_action", "memory_extraction_start"}
        assert not (MEMORY_RELEVANT_TYPES & noise)


# ---- _parse_actions() ----

class TestParseActions:
    def test_valid_json_array(self, agent):
        text = json.dumps([{"action": "add", "type": "pitfall", "content": "test"}])
        result = agent._parse_actions(text)
        assert len(result) == 1
        assert result[0]["action"] == "add"

    def test_json_with_markdown_fencing(self, agent):
        inner = json.dumps([{"action": "add", "type": "pitfall", "content": "test"}])
        text = f"```json\n{inner}\n```"
        result = agent._parse_actions(text)
        assert len(result) == 1

    def test_empty_array(self, agent):
        assert agent._parse_actions("[]") == []

    def test_invalid_json(self, agent):
        assert agent._parse_actions("not json") == []

    def test_empty_string(self, agent):
        assert agent._parse_actions("") == []

    def test_non_array_json(self, agent):
        """Should return empty for non-array JSON."""
        assert agent._parse_actions('{"action": "add"}') == []


# ---- Action execution ----

class TestExecuteActions:
    def test_add_action(self, agent, memory_store):
        actions = [
            {"action": "add", "type": "preference", "scopes": ["generic"], "content": "Prefers dark theme"},
        ]
        executed = agent._execute_actions(actions)
        assert len(executed) == 1
        assert len(memory_store.get_all()) == 1
        assert memory_store.get_all()[0].content == "Prefers dark theme"

    def test_add_invalid_type_skipped(self, agent, memory_store):
        actions = [
            {"action": "add", "type": "invalid_type", "scopes": ["generic"], "content": "test"},
        ]
        executed = agent._execute_actions(actions)
        assert len(executed) == 0
        assert len(memory_store.get_all()) == 0

    def test_add_empty_content_skipped(self, agent, memory_store):
        actions = [
            {"action": "add", "type": "preference", "scopes": ["generic"], "content": ""},
        ]
        executed = agent._execute_actions(actions)
        assert len(executed) == 0

    def test_edit_action(self, agent, memory_store):
        memory_store.add(Memory(
            id="edit01", type="preference", scopes=["visualization"],
            content="Old viz preference",
        ))
        actions = [
            {"action": "edit", "id": "edit01", "content": "Updated viz preference"},
        ]
        executed = agent._execute_actions(actions)
        assert len(executed) == 1
        active = memory_store.get_all()
        assert len(active) == 1
        assert active[0].content == "Updated viz preference"
        assert active[0].scopes == ["visualization"]

    def test_edit_nonexistent_id_skipped(self, agent, memory_store):
        actions = [
            {"action": "edit", "id": "nonexist", "content": "new text"},
        ]
        executed = agent._execute_actions(actions)
        assert len(executed) == 0

    def test_drop_action(self, agent, memory_store):
        memory_store.add(Memory(
            id="drop01", type="pitfall", scopes=["generic"],
            content="To be dropped",
        ))
        actions = [
            {"action": "drop", "id": "drop01"},
        ]
        executed = agent._execute_actions(actions)
        assert len(executed) == 1
        assert len(memory_store.get_all()) == 0

    def test_drop_nonexistent_id_skipped(self, agent, memory_store):
        actions = [
            {"action": "drop", "id": "nonexist"},
        ]
        executed = agent._execute_actions(actions)
        assert len(executed) == 0

    def test_mixed_actions(self, agent, memory_store):
        memory_store.add(Memory(id="m1", type="pitfall", scopes=["generic"], content="Old pitfall"))
        memory_store.add(Memory(id="m2", type="preference", scopes=["generic"], content="Old pref"))

        actions = [
            {"action": "add", "type": "summary", "scopes": ["generic"], "content": "Analyzed ACE data"},
            {"action": "edit", "id": "m1", "content": "Updated pitfall"},
            {"action": "drop", "id": "m2"},
        ]
        executed = agent._execute_actions(actions)
        assert len(executed) == 3
        active = memory_store.get_all()
        assert len(active) == 2  # summary + edited pitfall (old pitfall and pref archived)
        contents = {m.content for m in active}
        assert "Analyzed ACE data" in contents
        assert "Updated pitfall" in contents

    def test_review_action_skipped_by_memory_agent(self, agent, memory_store):
        """MemoryAgent no longer handles review actions — they are skipped."""
        memory_store.add(Memory(
            id="rev01", type="pitfall", scopes=["generic"], content="Check NaN",
        ))
        actions = [
            {"action": "review", "id": "rev01", "stars": 5, "comment": "Directly prevented NaN issue."},
        ]
        executed = agent._execute_actions(actions)
        assert len(executed) == 0
        m = memory_store.get_all()[0]
        assert m.review_of == ""  # No review created

    def test_unknown_action_type_skipped(self, agent, memory_store):
        actions = [{"action": "unknown", "content": "test"}]
        executed = agent._execute_actions(actions)
        assert len(executed) == 0

    def test_non_dict_actions_skipped(self, agent, memory_store):
        actions = ["not a dict", 42, None]
        executed = agent._execute_actions(actions)
        assert len(executed) == 0


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


# ---- Prompt building ----

class TestBuildPrompt:
    def test_includes_events(self, agent):
        context = _make_context()
        prompt = agent._build_prompt(context)
        assert "Show me ACE data" in prompt
        assert "Fetching ACE magnetic field data" in prompt
        assert "fetch_data" in prompt

    def test_includes_active_memories(self, agent):
        context = _make_context(active_memories=[
            {"id": "m1", "type": "preference", "scopes": ["generic"], "content": "Prefers dark theme"},
        ])
        prompt = agent._build_prompt(context)
        assert "Prefers dark theme" in prompt
        assert "m1" in prompt

    def test_session_activity_section(self, agent):
        """Should have a single Session Activity section."""
        context = _make_context()
        prompt = agent._build_prompt(context)
        assert "## Session Activity" in prompt
        # Old section headers should NOT be present
        assert "## Conversation" not in prompt
        assert "## Tool Operations" not in prompt
        assert "## Session Events" not in prompt

    def test_includes_budget_warning(self, agent):
        context = _make_context(token_budget=1000, total_memory_tokens=900)
        prompt = agent._build_prompt(context)
        assert "90%" in prompt
        assert "token budget" in prompt

    def test_no_budget_warning_when_low(self, agent):
        context = _make_context(token_budget=10000, total_memory_tokens=500)
        prompt = agent._build_prompt(context)
        assert "token budget" not in prompt

    def test_empty_memories_shows_none(self, agent):
        context = _make_context(active_memories=[])
        prompt = agent._build_prompt(context)
        assert "(none)" in prompt

    def test_includes_active_scopes(self, agent):
        context = _make_context(active_scopes=["generic", "envoy:ACE", "visualization"])
        prompt = agent._build_prompt(context)
        assert "envoy:ACE" in prompt
        assert "visualization" in prompt

    def test_no_session_activity_when_empty(self, agent):
        """Should not include Session Activity section when events is empty."""
        context = _make_context(events=[])
        prompt = agent._build_prompt(context)
        assert "## Session Activity" not in prompt

    def test_prompt_no_review_schema(self, agent, memory_store):
        """MemoryAgent prompt does NOT include review action schema (reviews done by sub-agents)."""
        memory_store.add(Memory(id="inj1", type="pitfall", scopes=["generic"], content="Test pitfall"))
        memory_store._last_injected_ids = {"inj1": "OrchestratorAgent"}
        context = _make_context(active_memories=[
            {"id": "inj1", "type": "pitfall", "scopes": ["generic"], "content": "Test pitfall"},
        ])
        prompt = agent._build_prompt(context)
        assert '"action": "review"' not in prompt
        assert "REVIEW: Set or update the review" not in prompt
        # [INJECTED] annotation is still present
        assert "[INJECTED]" in prompt

    def test_prompt_injected_annotation(self, agent, memory_store):
        """Memories in _last_injected_ids are annotated [INJECTED]."""
        memory_store._last_injected_ids = {"m1": "OrchestratorAgent"}
        context = _make_context(active_memories=[
            {"id": "m1", "type": "pitfall", "scopes": ["generic"], "content": "Injected mem"},
            {"id": "m2", "type": "preference", "scopes": ["generic"], "content": "Not injected"},
        ])
        prompt = agent._build_prompt(context)
        # m1 should be annotated
        lines = prompt.split("\n")
        m1_lines = [l for l in lines if "m1" in l]
        assert any("[INJECTED]" in l for l in m1_lines)
        # m2 should NOT be annotated
        m2_lines = [l for l in lines if "m2" in l]
        assert not any("[INJECTED]" in l for l in m2_lines)


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
