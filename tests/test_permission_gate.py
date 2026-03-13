"""Tests for PermissionGate — the blocking user-approval callback."""

import threading
from unittest.mock import MagicMock

import pytest

from agent.permission_gate import PermissionGate


class TestPermissionGate:
    def _make_gate(self, event_bus=None):
        return PermissionGate(event_bus=event_bus)

    def test_resolve_approved(self):
        gate = self._make_gate(event_bus=MagicMock())
        def _approve():
            import time; time.sleep(0.05)
            gate.resolve("req-1", approved=True, reason="ok")
        t = threading.Thread(target=_approve)
        t.start()
        result = gate(request_id="req-1", action="install_package",
                      description="Install foo", command="pip install foo")
        t.join()
        assert result["approved"] is True
        assert result["reason"] == "ok"

    def test_resolve_denied(self):
        gate = self._make_gate(event_bus=MagicMock())
        def _deny():
            import time; time.sleep(0.05)
            gate.resolve("req-2", approved=False, reason="not now")
        t = threading.Thread(target=_deny)
        t.start()
        result = gate(request_id="req-2", action="install_package",
                      description="Install foo", command="pip install foo")
        t.join()
        assert result["approved"] is False
        assert result["reason"] == "not now"

    def test_emits_permission_request_event(self):
        bus = MagicMock()
        gate = self._make_gate(event_bus=bus)
        def _approve():
            import time; time.sleep(0.05)
            gate.resolve("req-3", approved=True, reason="ok")
        t = threading.Thread(target=_approve)
        t.start()
        gate(request_id="req-3", action="test", description="d", command="c")
        t.join()
        bus.emit.assert_called_once()
        call_args = bus.emit.call_args
        assert call_args[1]["request_id"] == "req-3"
        assert call_args[1]["action"] == "test"

    def test_timeout(self):
        gate = self._make_gate(event_bus=MagicMock())
        result = gate(request_id="req-4", action="test",
                      description="d", command="c", timeout=0.1)
        assert result["approved"] is False
        assert "timed out" in result["reason"].lower()

    def test_resolve_unknown_request_id_is_noop(self):
        gate = self._make_gate()
        # Should not raise
        gate.resolve("nonexistent", approved=True, reason="ok")

    def test_concurrent_requests(self):
        gate = self._make_gate(event_bus=MagicMock())
        results = {}
        def _request(rid):
            results[rid] = gate(request_id=rid, action="test",
                                description="d", command="c")
        def _resolve_all():
            import time; time.sleep(0.1)
            gate.resolve("a", approved=True, reason="yes")
            gate.resolve("b", approved=False, reason="no")
        t1 = threading.Thread(target=_request, args=("a",))
        t2 = threading.Thread(target=_request, args=("b",))
        t3 = threading.Thread(target=_resolve_all)
        t1.start(); t2.start(); t3.start()
        t1.join(); t2.join(); t3.join()
        assert results["a"]["approved"] is True
        assert results["b"]["approved"] is False

    def test_cli_mode_auto_approves(self):
        """No event bus → auto-approve without blocking."""
        gate = self._make_gate(event_bus=None)
        result = gate(request_id="cli-1", action="install",
                      description="Install foo", command="pip install foo")
        assert result["approved"] is True
        assert "auto-approved" in result["reason"]


class TestPermissionGateWiring:
    def test_session_lifecycle_sets_request_permission(self):
        """Session.start() must wire up request_permission as a PermissionGate."""
        from agent.session_lifecycle import Session

        service = MagicMock()
        session = Session(service=service)
        ctx = session.start()
        try:
            assert ctx.request_permission is not None
            assert callable(ctx.request_permission)
            assert isinstance(ctx.request_permission, PermissionGate)
        finally:
            session.end()
