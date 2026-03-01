"""Multi-session agent lifecycle manager for the FastAPI backend."""

import asyncio
import threading
from datetime import datetime, timezone
from typing import Optional

from data_ops.store import set_store
from data_ops.operations_log import OperationsLog, get_operations_log, set_operations_log
from agent.sub_agent import AgentState
from agent.event_bus import set_event_bus


class SessionState:
    """State for a single API session."""

    def __init__(self, session_id: str, agent: "OrchestratorAgent"):
        self.session_id = session_id
        self.agent = agent
        self.ops_log = OperationsLog()
        self.created_at = datetime.now(timezone.utc)
        self.last_active = self.created_at
        self._busy_lock = threading.Lock()
        # Persistent event loop (turnless mode)
        self.sse_bridge: "SessionSSEBridge | None" = None
        self._loop_thread: threading.Thread | None = None

    @property
    def store(self):
        """The session's DataStore, owned by the agent."""
        return self.agent._store

    @property
    def busy(self) -> bool:
        """Derive busy from orchestrator state.

        ACTIVE and PENDING mean the orchestrator is processing and cannot
        accept new work safely. IDLE and SLEEPING both accept input.
        """
        state = self.agent.state
        return state in (AgentState.ACTIVE, AgentState.PENDING)

    def touch(self) -> None:
        self.last_active = datetime.now(timezone.utc)

    @property
    def model(self) -> str:
        return self.agent.model_name


class APISessionManager:
    """Manages multiple concurrent agent sessions.

    Each session owns its own OrchestratorAgent, DataStore, and OperationsLog.
    The DataStore is owned by the agent (disk-backed, shared across threads).
    The OperationsLog is set via module-level global for the session context.
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

    def start_loop(self, state: SessionState, loop: asyncio.AbstractEventLoop) -> None:
        """Start the persistent event loop for a turnless session.

        Creates a SessionSSEBridge and launches run_loop() in a daemon thread.
        Safe to call multiple times — no-ops if loop is already running.
        """
        if state._loop_thread is not None and state._loop_thread.is_alive():
            return

        from api.streaming import SessionSSEBridge

        state.sse_bridge = SessionSSEBridge(loop)
        state.agent.subscribe_sse(state.sse_bridge.callback)

        def _run():
            self.run_in_session_context(state, state.agent.run_loop)

        t = threading.Thread(
            target=_run, daemon=True,
            name=f"loop-{state.session_id[:8]}",
        )
        state._loop_thread = t
        t.start()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return False
        # Shut down persistent event loop if running
        try:
            state.agent.shutdown_loop()
        except Exception:
            pass
        if state._loop_thread is not None:
            state._loop_thread.join(timeout=5.0)
        if state.sse_bridge is not None:
            state.agent.unsubscribe_sse()
            state.sse_bridge.close()
            state.sse_bridge = None
        # Flush any in-memory memories/discoveries to disk
        try:
            state.agent._memory_store.save()
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

        The DataStore is owned by the agent (disk-backed, thread-safe).
        set_store() is called so that any code still using get_store()
        (e.g. sub-agents, data_ops) sees the right store.
        """
        # Set module-level globals for this thread
        set_store(session.store)
        set_operations_log(session.ops_log)
        set_event_bus(session.agent._event_bus)
        return fn(*args, **kwargs)

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
        history + DataStore + operations log, then wraps in a SessionState.

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

        # Copy the restored ops log from the agent's context
        from data_ops.operations_log import get_operations_log
        state.ops_log = get_operations_log()

        with self._lock:
            self._sessions[session_id] = state

        return state, metadata, display_log, event_log

    # ---- Config hot-reload ----

    def reload_config_for_sessions(self) -> dict:
        """Hot-reload all active sessions after config changes.

        Each agent compares its own cached state (adapter type, model name)
        against the current config values. If nothing diverges for an agent,
        it's a no-op. Busy sessions are skipped.

        Returns:
            dict with keys: reloaded, skipped, failed, unchanged (counts).
        """
        with self._lock:
            sessions = list(self._sessions.items())

        reloaded = 0
        skipped = 0
        failed = 0
        unchanged = 0

        for sid, state in sessions:
            if state.busy:
                skipped += 1
                continue
            try:
                result = state.agent.hot_reload_config()
                if result["status"] == "unchanged":
                    unchanged += 1
                else:
                    reloaded += 1
            except Exception:
                failed += 1

        return {
            "reloaded": reloaded,
            "skipped": skipped,
            "failed": failed,
            "unchanged": unchanged,
        }

    # ---- Shutdown ----

    def shutdown(self) -> None:
        """Delete all sessions (called on server shutdown)."""
        with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            self.delete_session(sid)
