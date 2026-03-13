# tests/test_session_lifecycle.py
"""Tests for Session lifecycle class."""
from unittest.mock import MagicMock

from agent.session_lifecycle import Session


def test_session_start_returns_context():
    """Session.start() creates a fully populated SessionContext."""
    session = Session(service=MagicMock())
    ctx = session.start()
    assert ctx.store is not None
    assert ctx.dag is not None
    assert ctx.event_bus is not None
    assert ctx.service is not None
    assert ctx.renderer is not None
    assert ctx.work_tracker is not None
    assert ctx.session_dir is not None


def test_session_end_does_not_raise():
    """Session.end() cleans up without raising."""
    session = Session(service=MagicMock())
    session.start()
    session.end()  # should not raise


def test_session_context_is_reusable():
    """SessionContext from start() can be passed to agents."""
    session = Session(service=MagicMock())
    ctx = session.start()
    # Verify it's the right type
    from agent.session_context import SessionContext
    assert isinstance(ctx, SessionContext)


def test_session_creates_unique_session_id():
    """Each start() creates a unique session ID."""
    s1 = Session(service=MagicMock())
    s2 = Session(service=MagicMock())
    ctx1 = s1.start()
    ctx2 = s2.start()
    assert ctx1.session_id != ctx2.session_id
