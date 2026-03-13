# tests/test_session_context.py
"""Tests for SessionContext dataclass."""
import threading
from unittest.mock import MagicMock
from agent.session_context import SessionContext


def test_session_context_construction():
    """SessionContext can be constructed with all required fields."""
    ctx = SessionContext(
        store=MagicMock(),
        dag=MagicMock(),
        event_bus=MagicMock(),
        service=MagicMock(),
        renderer=MagicMock(),
        memory_store=MagicMock(),
        memory_hooks=MagicMock(),
        session_manager=MagicMock(),
        session_dir=MagicMock(),
        delegation=MagicMock(),
        work_tracker=MagicMock(),
    )
    assert ctx.store is not None
    assert ctx.web_mode is False
    assert ctx.mcp_client is None
    assert ctx.request_permission is None
    assert ctx.agent_state == {}


def test_session_context_optional_fields():
    """Optional fields can be set at construction."""
    perm_fn = MagicMock()
    ctx = SessionContext(
        store=MagicMock(),
        dag=MagicMock(),
        event_bus=MagicMock(),
        service=MagicMock(),
        renderer=MagicMock(),
        memory_store=MagicMock(),
        memory_hooks=MagicMock(),
        session_manager=MagicMock(),
        session_dir=MagicMock(),
        delegation=MagicMock(),
        work_tracker=MagicMock(),
        web_mode=True,
        request_permission=perm_fn,
    )
    assert ctx.web_mode is True
    assert ctx.request_permission is perm_fn


def test_replay_context_has_all_protocol_properties():
    """ReplayContext must expose all ToolContext protocol properties."""
    import tempfile
    from pathlib import Path
    from agent.tool_context import ReplayContext
    from data_ops.store import DataStore
    ctx = ReplayContext(store=DataStore(data_dir=Path(tempfile.mkdtemp())))
    # These must not AttributeError — they can return None
    assert ctx.request_permission is None
    assert ctx.work_tracker is None
    assert ctx.delegation is None
    assert ctx.service is None
    assert ctx.model_name == ""
