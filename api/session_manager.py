"""Multi-session agent lifecycle manager for the FastAPI backend."""

import asyncio
import contextvars
import threading
from datetime import datetime, timezone
from typing import Optional

from data_ops.store import DataStore, get_store, set_store
from data_ops.operations_log import OperationsLog, get_operations_log, set_operations_log


class SessionState:
    """State for a single API session."""

    def __init__(self, session_id: str, agent: "OrchestratorAgent"):
        self.session_id = session_id
        self.agent = agent
        self.store = DataStore()
        self.ops_log = OperationsLog()
        self.created_at = datetime.now(timezone.utc)
        self.last_active = self.created_at
        self.busy = False
        self._busy_lock = threading.Lock()

    def touch(self) -> None:
        self.last_active = datetime.now(timezone.utc)

    @property
    def model(self) -> str:
        return self.agent.model_name


class APISessionManager:
    """Manages multiple concurrent agent sessions.

    Each session owns its own OrchestratorAgent, DataStore, and OperationsLog.
    Tool execution runs inside a copied context so that ContextVar-based
    get_store() / get_operations_log() resolve to the session's instances.
    """

    def __init__(self, max_sessions: int = 10, idle_timeout_seconds: float = 3600):
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.idle_timeout_seconds = idle_timeout_seconds
        self._cleanup_task: Optional[asyncio.Task] = None

    # ---- Lifecycle ----

    def create_session(self) -> SessionState:
        """Create a new agent session. Raises RuntimeError if limit reached."""
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"Maximum sessions ({self.max_sessions}) reached. "
                    f"Delete an existing session first."
                )
        # Import here to avoid circular import at module load time
        from agent.core import create_agent
        agent = create_agent(verbose=False)
        # Start a disk session so auto-save works after each process_message()
        session_id = agent.start_session()
        state = SessionState(session_id, agent)
        with self._lock:
            self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return False
        # Flush any in-memory memories/discoveries to disk
        try:
            state.agent._memory_store.save()
        except Exception:
            pass
        try:
            state.agent._discovery_store.save()
        except Exception:
            pass
        # Best-effort cleanup
        try:
            state.agent._cleanup_caches()
        except Exception:
            pass
        return True

    def list_sessions(self) -> list[SessionState]:
        with self._lock:
            return list(self._sessions.values())

    # ---- Scoped execution ----

    def run_in_session_context(self, session: SessionState, fn, *args, **kwargs):
        """Run *fn* in a context where get_store() and get_operations_log()
        resolve to this session's instances.

        Uses contextvars.copy_context().run() so that ContextVar changes
        inside *fn* don't leak to the calling context.
        """
        ctx = contextvars.copy_context()

        def _scoped():
            set_store(session.store)
            set_operations_log(session.ops_log)
            return fn(*args, **kwargs)

        return ctx.run(_scoped)

    # ---- Idle cleanup ----

    async def start_cleanup_loop(self) -> None:
        """Start a background task that evicts idle sessions."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_loop(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            now = datetime.now(timezone.utc)
            to_remove = []
            with self._lock:
                for sid, state in self._sessions.items():
                    idle = (now - state.last_active).total_seconds()
                    if idle > self.idle_timeout_seconds and not state.busy:
                        to_remove.append(sid)
            for sid in to_remove:
                self.delete_session(sid)

    # ---- Resume saved session ----

    def resume_session(self, session_id: str) -> tuple[SessionState, dict, list[dict] | None, list[dict] | None]:
        """Resume a saved session from disk into a new API session.

        Creates a fresh agent, calls agent.load_session() to restore chat
        history + DataStore + operations log, then copies the restored state
        into the SessionState so that session-scoped context vars work.

        Returns (state, metadata, display_log, event_log).

        Raises RuntimeError if session limit is reached, FileNotFoundError
        if the saved session doesn't exist.
        """
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"Maximum sessions ({self.max_sessions}) reached. "
                    f"Delete an existing session first."
                )

        from agent.core import create_agent
        agent = create_agent(verbose=False, defer_chat=True)

        # load_session restores chat history, DataStore, ops log, figure state
        # and sets _auto_save=True so subsequent saves go to the same directory
        metadata, display_log, event_log = agent.load_session(session_id)

        # Use the original disk session ID so auto-saves write back to the same directory
        state = SessionState(session_id, agent)

        # Copy the restored DataStore and ops log from the agent's context
        # into the SessionState so run_in_session_context works correctly
        from data_ops.store import get_store
        from data_ops.operations_log import get_operations_log
        state.store = get_store()
        state.ops_log = get_operations_log()

        with self._lock:
            self._sessions[session_id] = state

        return state, metadata, display_log, event_log

    # ---- Shutdown ----

    def shutdown(self) -> None:
        """Delete all sessions (called on server shutdown)."""
        with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            self.delete_session(sid)
