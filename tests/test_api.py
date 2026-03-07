"""Tests for the FastAPI backend API endpoints.

Uses FastAPI TestClient with a mock agent — no API key needed.
"""

import json
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers: build a fake agent that doesn't need an LLM backend
# ---------------------------------------------------------------------------

def _make_mock_agent():
    """Create a mock OrchestratorAgent with the minimal interface we need.

    Supports the persistent event loop architecture:
    - push_input() queues messages
    - run_loop() processes queued messages via process_message()
    - shutdown_loop() signals the loop to stop
    """
    agent = MagicMock()
    agent.model_name = "test-model"
    agent._current_plan = None
    agent._sse_callback = None
    agent._renderer = MagicMock()
    agent._renderer.get_figure.return_value = None

    # Disk-backed DataStore mock — must behave like a real empty store
    mock_store = MagicMock()
    mock_store.list_entries.return_value = []
    mock_store.__len__ = lambda self: 0
    mock_store.memory_usage_bytes.return_value = 0
    agent._store = mock_store

    # Real queue for the persistent event loop
    _msg_queue = queue.Queue()
    _shutdown = threading.Event()

    def _process_message(msg):
        # Emit events via the SSE callback if one is set
        cb = agent._sse_callback
        if cb:
            cb({"type": "tool_call", "tool_name": "search_datasets", "tool_args": {"query": "ACE"}})
            cb({"type": "tool_result", "tool_name": "search_datasets", "status": "success"})
        return f"Echo: {msg}"

    agent.process_message.side_effect = _process_message

    def _push_input(msg):
        _msg_queue.put(msg)
    agent.push_input.side_effect = _push_input

    def _run_loop():
        """Minimal persistent loop for tests."""
        while not _shutdown.is_set():
            try:
                msg = _msg_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            cb = agent._sse_callback
            response = _process_message(msg)
            if cb:
                cb({"type": "text_delta", "text": response})
                cb({"type": "done", "token_usage": agent.get_token_usage.return_value})
    agent.run_loop.side_effect = _run_loop

    def _shutdown_loop():
        _shutdown.set()
    agent.shutdown_loop.side_effect = _shutdown_loop

    def _subscribe_sse(cb):
        agent._sse_callback = cb
        return MagicMock()  # Return a mock listener
    agent.subscribe_sse.side_effect = _subscribe_sse

    def _unsubscribe_sse():
        agent._sse_callback = None
    agent.unsubscribe_sse.side_effect = _unsubscribe_sse

    agent.get_token_usage.return_value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "thinking_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 150,
        "api_calls": 1,
    }
    agent.request_cancel.return_value = None
    agent._cleanup_caches.return_value = None
    agent.generate_follow_ups.return_value = ["What about PSP?", "Show velocity"]
    # start_session() is called by APISessionManager.create_session() to
    # enable auto-save; it must return a unique string session ID per agent.
    agent.start_session.return_value = f"20260219_120000_{id(agent):08x}"
    return agent


@pytest.fixture()
def client():
    """Create a TestClient with a mocked session manager."""
    # Patch create_agent at its definition site — the local import in
    # session_manager.create_session resolves to agent.core.create_agent
    with patch("agent.core.create_agent", side_effect=lambda **kw: _make_mock_agent()):
        from api.app import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestServerStatus:
    def test_status(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["active_sessions"] == 0


class TestSessionCRUD:
    def test_create_session(self, client):
        r = client.post("/api/sessions")
        assert r.status_code == 201
        data = r.json()
        assert "session_id" in data
        assert data["model"] == "test-model"

    def test_list_sessions(self, client):
        client.post("/api/sessions")
        client.post("/api/sessions")
        r = client.get("/api/sessions")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_get_session(self, client):
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]
        r = client.get(f"/api/sessions/{sid}")
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == sid
        assert "token_usage" in data
        assert data["data_entries"] == 0

    def test_get_session_404(self, client):
        r = client.get("/api/sessions/nonexistent")
        assert r.status_code == 404

    def test_delete_session(self, client):
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]
        r = client.delete(f"/api/sessions/{sid}")
        assert r.status_code == 204
        # Should be gone
        r2 = client.get(f"/api/sessions/{sid}")
        assert r2.status_code == 404

    def test_delete_session_404(self, client):
        r = client.delete("/api/sessions/nonexistent")
        assert r.status_code == 404


class TestChat:
    def test_chat_queues_message(self, client):
        """POST /chat pushes message and returns {status: queued}."""
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]

        r = client.post(
            f"/api/sessions/{sid}/chat",
            json={"message": "Hello agent"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "queued"

    def test_events_endpoint_available(self, client):
        """GET /events returns SSE stream after loop is started."""
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]

        # Before the loop starts, GET /events should return 400
        r = client.get(f"/api/sessions/{sid}/events")
        assert r.status_code == 400

        # Start the loop by sending a chat message
        r = client.post(f"/api/sessions/{sid}/chat", json={"message": "Hello agent"})
        assert r.json()["status"] == "queued"

        # Give the loop a moment to start and process
        time.sleep(0.5)

        # Now the sse_bridge should exist — verify by checking the session
        # has a running loop (we can't easily consume the SSE stream in
        # a sync test client, so we just verify the endpoint is active)
        from api.routes import session_manager
        state = session_manager.get_session(sid)
        assert state is not None
        assert state.sse_bridge is not None
        assert state._loop_thread is not None
        assert state._loop_thread.is_alive()

        # Verify push_input was called and the loop processed it
        assert state.agent.push_input.called
        assert state.agent.run_loop.called

    def test_chat_empty_message(self, client):
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]
        r = client.post(f"/api/sessions/{sid}/chat", json={"message": ""})
        assert r.status_code == 422  # Pydantic validation

    def test_chat_session_not_found(self, client):
        r = client.post("/api/sessions/nope/chat", json={"message": "hi"})
        assert r.status_code == 404


class TestCancel:
    def test_cancel(self, client):
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]
        r = client.post(f"/api/sessions/{sid}/cancel")
        assert r.status_code == 202
        assert r.json()["status"] in ("cancelled", "cancel_requested")


class TestData:
    def test_data_empty(self, client):
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]
        r = client.get(f"/api/sessions/{sid}/data")
        assert r.status_code == 200
        assert r.json() == []


class TestFigure:
    def test_figure_none(self, client):
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]
        r = client.get(f"/api/sessions/{sid}/figure")
        assert r.status_code == 200
        assert r.json()["figure"] is None


class TestFollowUps:
    def test_follow_ups(self, client):
        create_r = client.post("/api/sessions")
        sid = create_r.json()["session_id"]
        r = client.post(f"/api/sessions/{sid}/follow-ups")
        assert r.status_code == 200
        data = r.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)


class TestSessionLimit:
    def test_max_sessions(self, client):
        # Create max_sessions (default 10)
        for _ in range(10):
            r = client.post("/api/sessions")
            assert r.status_code == 201

        # 11th should fail
        r = client.post("/api/sessions")
        assert r.status_code == 429
