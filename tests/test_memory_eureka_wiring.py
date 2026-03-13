"""Tests for the memory & eureka wiring — agents, hooks, and session plumbing."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Lightweight fakes for SessionContext dependencies
# ---------------------------------------------------------------------------


class FakeEventBus:
    """Minimal event bus for tests."""
    def __init__(self):
        self.events = []
    def emit(self, *args, **kwargs):
        self.events.append((args, kwargs))
    def get_events(self, **kwargs):
        return []
    def event_count(self):
        return 0


class FakeStore:
    """Minimal data store."""
    def list_entries(self):
        return []


class FakeRenderer:
    """Minimal renderer."""
    def get_figure(self):
        return None
    def get_current_state(self):
        return {}


class FakeService:
    """Minimal LLM service stub."""
    def create_session(self, **kwargs):
        return MagicMock()
    def get_adapter(self, provider=None):
        return MagicMock()


# ---------------------------------------------------------------------------
# MemoryAgent: manage_memory tool
# ---------------------------------------------------------------------------


class TestManageMemoryHandler:
    """Test MemoryAgent._handle_manage_memory dispatch."""

    def _make_agent(self, memory_store=None):
        from agent.memory_agent import MemoryAgent
        agent = MemoryAgent(
            service=FakeService(),
            memory_store=memory_store,
        )
        return agent

    def test_list_returns_memories(self):
        """list action returns formatted memories from store."""
        mock_store = MagicMock()
        mock_mem = MagicMock()
        mock_mem.id = "m1"
        mock_mem.type = "preference"
        mock_mem.content = "User likes scatter plots"
        mock_mem.scopes = ["visualization"]
        mock_mem.tags = ["plot"]
        mock_mem.created = "2026-01-01"
        mock_store.get_all.return_value = [mock_mem]

        agent = self._make_agent(memory_store=mock_store)
        result = agent._handle_manage_memory(None, {"action": "list"}, None)

        assert result["status"] == "ok"
        assert len(result["memories"]) == 1
        assert result["memories"][0]["id"] == "m1"

    def test_add_delegates_to_execute_actions(self):
        """add action delegates to MemoryStore.execute_actions."""
        mock_store = MagicMock()
        mock_store.execute_actions.return_value = 1

        agent = self._make_agent(memory_store=mock_store)
        result = agent._handle_manage_memory(None, {
            "action": "add",
            "content": "Test memory",
            "type": "preference",
            "scopes": ["generic"],
        }, None)

        assert result["status"] == "ok"
        assert result["actions_applied"] == 1
        mock_store.execute_actions.assert_called_once()
        action = mock_store.execute_actions.call_args[0][0][0]
        assert action["action"] == "add"
        assert action["content"] == "Test memory"

    def test_edit_delegates_to_execute_actions(self):
        """edit action passes id and content."""
        mock_store = MagicMock()
        mock_store.execute_actions.return_value = 1

        agent = self._make_agent(memory_store=mock_store)
        result = agent._handle_manage_memory(None, {
            "action": "edit",
            "id": "m1",
            "content": "Updated content",
        }, None)

        assert result["status"] == "ok"
        action = mock_store.execute_actions.call_args[0][0][0]
        assert action["action"] == "edit"
        assert action["id"] == "m1"

    def test_drop_delegates_to_execute_actions(self):
        """drop action passes id."""
        mock_store = MagicMock()
        mock_store.execute_actions.return_value = 1

        agent = self._make_agent(memory_store=mock_store)
        result = agent._handle_manage_memory(None, {
            "action": "drop",
            "id": "m1",
        }, None)

        assert result["status"] == "ok"
        action = mock_store.execute_actions.call_args[0][0][0]
        assert action["action"] == "drop"

    def test_no_store_returns_error(self):
        """All actions fail gracefully when store is None."""
        agent = self._make_agent(memory_store=None)

        for action in ("list", "add", "edit", "drop"):
            result = agent._handle_manage_memory(None, {"action": action}, None)
            assert result["status"] == "error"

    def test_unknown_action_returns_error(self):
        """Unknown action returns error."""
        agent = self._make_agent(memory_store=MagicMock())
        result = agent._handle_manage_memory(None, {"action": "explode"}, None)
        assert result["status"] == "error"

    def test_tool_schema_registered(self):
        """manage_memory schema is in _tool_schemas."""
        agent = self._make_agent()
        names = [s.name for s in agent._tool_schemas]
        assert "manage_memory" in names
        assert "add_memory" not in names  # old stub removed

    def test_local_tool_registered(self):
        """manage_memory handler is in _local_tools."""
        agent = self._make_agent()
        assert "manage_memory" in agent._local_tools
        assert "add_memory" not in agent._local_tools


# ---------------------------------------------------------------------------
# EurekaAgent: submit_finding + submit_suggestion tools
# ---------------------------------------------------------------------------


class TestEurekaAgentTools:
    """Test EurekaAgent tool handlers."""

    def _make_agent(self, eureka_store=None, session_id="test123"):
        from agent.eureka_agent import EurekaAgent
        agent = EurekaAgent(
            service=FakeService(),
            eureka_store=eureka_store,
            session_id=session_id,
        )
        return agent

    def test_submit_finding_creates_entry(self):
        """submit_finding creates an EurekaEntry and adds to store."""
        mock_store = MagicMock()
        agent = self._make_agent(eureka_store=mock_store, session_id="sess1")

        result = agent._handle_submit_finding(None, {
            "title": "CME detection",
            "observation": "Sudden increase in Bz",
            "hypothesis": "Coronal mass ejection passage",
            "evidence": ["Bz dropped to -20nT at 12:00"],
            "confidence": 0.8,
            "tags": ["anomaly", "cme"],
        }, None)

        assert result["status"] == "ok"
        assert "eureka_id" in result
        mock_store.add.assert_called_once()
        entry = mock_store.add.call_args[0][0]
        assert entry.title == "CME detection"
        assert entry.session_id == "sess1"
        assert entry.status == "active"

    def test_submit_suggestion_creates_entry(self):
        """submit_suggestion creates an EurekaSuggestion and adds to store."""
        mock_store = MagicMock()
        agent = self._make_agent(eureka_store=mock_store)

        result = agent._handle_submit_suggestion(None, {
            "action": "fetch_data",
            "description": "Fetch ACE solar wind data",
            "rationale": "Compare upstream conditions",
            "parameters": {"mission": "ACE", "dataset": "swepam"},
            "priority": "high",
            "linked_eureka_id": "e1",
        }, None)

        assert result["status"] == "ok"
        assert "suggestion_id" in result
        mock_store.add_suggestion.assert_called_once()
        sug = mock_store.add_suggestion.call_args[0][0]
        assert sug.action == "fetch_data"
        assert sug.status == "proposed"

    def test_no_store_returns_error(self):
        """Tools fail gracefully when store is None."""
        agent = self._make_agent(eureka_store=None)

        result = agent._handle_submit_finding(None, {
            "title": "test", "observation": "test", "hypothesis": "test",
            "evidence": [], "confidence": 0.5, "tags": [],
        }, None)
        assert result["status"] == "error"

        result = agent._handle_submit_suggestion(None, {
            "action": "compute", "description": "test", "rationale": "test",
            "parameters": {}, "priority": "low", "linked_eureka_id": "",
        }, None)
        assert result["status"] == "error"

    def test_tool_schemas_registered(self):
        """Both tool schemas are in _tool_schemas."""
        agent = self._make_agent()
        names = [s.name for s in agent._tool_schemas]
        assert "submit_finding" in names
        assert "submit_suggestion" in names
        assert "submit_eureka" not in names  # old stub removed

    def test_local_tools_registered(self):
        """Both handlers are in _local_tools."""
        agent = self._make_agent()
        assert "submit_finding" in agent._local_tools
        assert "submit_suggestion" in agent._local_tools
        assert "submit_eureka" not in agent._local_tools


# ---------------------------------------------------------------------------
# Hook context formatting
# ---------------------------------------------------------------------------


class TestContextFormatting:
    """Test format_context_message and format_eureka_message."""

    def test_format_context_message(self):
        """format_context_message produces valid string with all sections."""
        from agent.memory_hooks import format_context_message, MemoryContext

        ctx = MemoryContext(
            console_events=[],
            active_scopes=["generic", "visualization"],
            total_memory_tokens=500,
        )
        msg = format_context_message(ctx)
        assert "generic, visualization" in msg
        assert "500" in msg
        assert "No session events" in msg

    def test_format_context_message_with_events(self):
        """format_context_message includes event summaries."""
        from agent.memory_hooks import format_context_message, MemoryContext

        mock_event = MagicMock()
        mock_event.agent = "orchestrator"
        mock_event.summary = "User asked for a plot"
        mock_event.msg = "User asked for a plot"

        ctx = MemoryContext(
            console_events=[mock_event],
            active_scopes=["generic"],
            total_memory_tokens=0,
        )
        msg = format_context_message(ctx)
        assert "User asked for a plot" in msg
        assert "1 total" in msg

    def test_format_eureka_message(self):
        """format_eureka_message produces valid string."""
        from agent.eureka_hooks import format_eureka_message

        context = {
            "session_id": "abc123",
            "data_store_keys": ["PSP.mag_rtn", "PSP.spc_fit"],
            "has_figure": True,
            "recent_messages": ["Plot the magnetic field"],
        }
        msg = format_eureka_message(context)
        assert "abc123" in msg
        assert "PSP.mag_rtn" in msg
        assert "Active figure: Yes" in msg
        assert "Plot the magnetic field" in msg

    def test_format_eureka_message_empty(self):
        """format_eureka_message handles empty context."""
        from agent.eureka_hooks import format_eureka_message

        context = {
            "session_id": "x",
            "data_store_keys": [],
            "has_figure": False,
            "recent_messages": [],
        }
        msg = format_eureka_message(context)
        assert "No data in memory" in msg
        assert "Active figure: No" in msg


# ---------------------------------------------------------------------------
# SessionContext fields
# ---------------------------------------------------------------------------


class TestSessionContextFields:
    """Test that eureka_store and eureka_hooks fields exist on SessionContext."""

    def test_eureka_fields_exist(self):
        """SessionContext has eureka_store and eureka_hooks fields."""
        from agent.session_context import SessionContext
        import dataclasses

        field_names = [f.name for f in dataclasses.fields(SessionContext)]
        assert "eureka_store" in field_names
        assert "eureka_hooks" in field_names

    def test_eureka_fields_default_none(self):
        """eureka_store and eureka_hooks default to None."""
        from agent.session_context import SessionContext
        import dataclasses

        fields_dict = {f.name: f for f in dataclasses.fields(SessionContext)}
        # These should have default None (or default_factory returning None)
        assert fields_dict["eureka_store"].default is None
        assert fields_dict["eureka_hooks"].default is None


# ---------------------------------------------------------------------------
# Ensure agent passes stores
# ---------------------------------------------------------------------------


class TestEnsureAgentPlumbing:
    """Test that ensure_agent passes stores to agent constructors."""

    def test_memory_hooks_ensure_agent_passes_store(self):
        """MemoryHooks.ensure_agent passes memory_store to MemoryAgent."""
        from agent.memory_hooks import MemoryHooks

        mock_store = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.memory_store = mock_store
        mock_ctx.service = FakeService()
        mock_ctx.event_bus = FakeEventBus()
        mock_ctx.session_id = "test123"
        mock_ctx.delegation = None

        hooks = MemoryHooks(ctx=mock_ctx)
        agent = hooks.ensure_agent()

        assert agent._memory_store is mock_store

    def test_eureka_hooks_ensure_agent_passes_store(self):
        """EurekaHooks.ensure_eureka_agent passes eureka_store and session_id."""
        from agent.eureka_hooks import EurekaHooks

        mock_store = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.eureka_store = mock_store
        mock_ctx.service = FakeService()
        mock_ctx.event_bus = FakeEventBus()
        mock_ctx.session_id = "test456"
        mock_ctx.delegation = None

        hooks = EurekaHooks(ctx=mock_ctx)
        agent = hooks.ensure_eureka_agent()

        assert agent._eureka_store is mock_store
        assert agent._session_id == "test456"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class TestPromptBuilder:
    """Test prompt builder functions for memory and eureka agents."""

    def test_build_memory_system_prompt_no_store(self):
        """build_memory_system_prompt works without a store."""
        from knowledge.prompt_builder import build_memory_system_prompt

        prompt = build_memory_system_prompt(memory_store=None)
        assert "manage_memory" in prompt
        assert "No memories stored" in prompt

    def test_build_memory_system_prompt_with_store(self):
        """build_memory_system_prompt injects current memories."""
        from knowledge.prompt_builder import build_memory_system_prompt

        mock_store = MagicMock()
        mock_mem = MagicMock()
        mock_mem.id = "m1"
        mock_mem.type = "preference"
        mock_mem.content = "User prefers dark theme plots"
        mock_mem.scopes = ["visualization"]
        mock_store.get_all.return_value = [mock_mem]

        prompt = build_memory_system_prompt(memory_store=mock_store)
        assert "m1" in prompt
        assert "dark theme" in prompt

    def test_build_eureka_system_prompt(self):
        """build_eureka_system_prompt returns valid prompt."""
        from knowledge.prompt_builder import build_eureka_system_prompt

        prompt = build_eureka_system_prompt()
        assert "submit_finding" in prompt
        assert "submit_suggestion" in prompt
