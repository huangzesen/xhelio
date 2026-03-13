"""Multi-session agent lifecycle manager for the FastAPI backend."""

import asyncio
import threading
from datetime import datetime, timezone
from typing import Optional

from data_ops.store import set_store
from agent.base_agent import AgentState
from agent.event_bus import set_event_bus
from agent.session_persistence import (
    start_session as _start_session,
    save_session as _save_session,
    load_session as _load_session,
)


class SessionState:
    """State for a single API session."""

    def __init__(self, session_id: str, agent: "OrchestratorAgent"):
        self.session_id = session_id
        self.agent = agent
        self.ops_log = None  # Pipeline tracking now uses PipelineDAG
        self.created_at = datetime.now(timezone.utc)
        self.last_active = self.created_at
        self._busy_lock = threading.Lock()
        # Persistent event loop (turnless mode)
        self.sse_bridge: "SessionSSEBridge | None" = None
        self._loop_thread: threading.Thread | None = None

    @property
    def store(self):
        """The session's DataStore, via SessionContext."""
        return self.agent.session_ctx.store

    @property
    def busy(self) -> bool:
        """Derive busy from orchestrator state.

        ACTIVE means the orchestrator is processing and cannot accept new
        work safely. SLEEPING accepts input.
        """
        return self.agent.state == AgentState.ACTIVE

    def touch(self) -> None:
        self.last_active = datetime.now(timezone.utc)

    @property
    def model(self) -> str:
        return self.agent.model_name


class APISessionManager:
    """Manages multiple concurrent agent sessions.

    Each session owns its own OrchestratorAgent and DataStore.
    The DataStore is owned by SessionContext (disk-backed, shared across threads).
    Pipeline operations are tracked via PipelineDAG on the SessionContext.
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
        session_id = _start_session(agent)
        state = SessionState(session_id, agent)
        with self._lock:
            self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._sessions.get(session_id)

    def start_loop(self, state: SessionState, loop: asyncio.AbstractEventLoop) -> None:
        """Start the persistent event loop for a turnless session.

        Creates a SessionSSEBridge and launches _run_loop() in a daemon thread.
        Safe to call multiple times — no-ops if loop is already running.
        """
        if state._loop_thread is not None and state._loop_thread.is_alive():
            return

        from api.streaming import SessionSSEBridge

        state.sse_bridge = SessionSSEBridge(loop)
        state.agent.subscribe_sse(state.sse_bridge.callback)

        def _run():
            self.run_in_session_context(state, state.agent._run_loop)

        t = threading.Thread(
            target=_run,
            daemon=True,
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
            state.agent.stop()
        except Exception:
            pass
        if state._loop_thread is not None:
            state._loop_thread.join(timeout=5.0)
        if state.sse_bridge is not None:
            state.agent.unsubscribe_sse()
            state.sse_bridge.close()
            state.sse_bridge = None
        # Flush any in-memory memories to disk
        try:
            ms = state.agent.session_ctx.memory_store
            if ms is not None:
                ms.save()
        except Exception:
            pass
        return True

    def list_sessions(self) -> list[SessionState]:
        with self._lock:
            return list(self._sessions.values())

    # ---- Scoped execution ----

    def run_in_session_context(self, session: SessionState, fn, *args, **kwargs):
        """Run *fn* in a context where get_store() resolves to this session's instance.

        The DataStore is owned by SessionContext (disk-backed, thread-safe).
        set_store() is called so that any code still using get_store()
        (e.g. sub-agents, data_ops) sees the right store.
        """
        # Set module-level globals for this thread
        set_store(session.store)
        set_event_bus(session.agent.session_ctx.event_bus)
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
                # Re-check busy state — session may have become active
                with self._lock:
                    state = self._sessions.get(sid)
                    if state is None or state.busy:
                        continue
                self.delete_session(sid)

    # ---- Resume saved session ----

    def resume_session(
        self, session_id: str
    ) -> tuple[SessionState, dict, list[dict] | None, list[dict] | None]:
        """Resume a saved session from disk into a new API session.

        Creates a fresh agent, calls load_session() to restore chat
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
        metadata, display_log, event_log = _load_session(agent, session_id)

        # Use the original disk session ID so auto-saves write back to the same directory
        state = SessionState(session_id, agent)

        with self._lock:
            self._sessions[session_id] = state

        return state, metadata, display_log, event_log

    # ---- Fork session (independent history) ----

    def fork_session(
        self, session_id: str
    ) -> tuple[SessionState, dict, list[dict] | None, list[dict] | None]:
        """Fork a live session into a new independent session.

        Unlike resume_session, this:
        1. Saves the source session to disk first
        2. Copies the session directory to a new ID
        3. Resumes the copy with skip_interaction_resume=True (fresh chat,
           no shared server-side history)
        4. Updates metadata with fork lineage

        Returns (state, metadata, display_log, event_log).
        """
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"Maximum sessions ({self.max_sessions}) reached. "
                    f"Delete an existing session first."
                )
            source = self._sessions.get(session_id)
        if source is None:
            raise ValueError(f"Session {session_id} not found in active sessions")

        # 1. Force-save the source session to disk
        self.run_in_session_context(
            source, lambda: _save_session(source.agent)
        )

        # 2. Copy session directory with a new proper ID
        from agent.session import SessionManager as SM
        import shutil
        import json
        from datetime import datetime

        sm = SM()
        new_id = sm.generate_session_id()
        src_dir = sm.base_dir / session_id
        dst_dir = sm.base_dir / new_id
        shutil.copytree(str(src_dir), str(dst_dir))

        # 3. Update metadata with fork lineage + new ID
        meta_path = dst_dir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["id"] = new_id
        meta["forked_from"] = session_id
        meta["forked_at"] = datetime.now().isoformat()
        # Set a clear title indicating this is a fork
        original_name = meta.get("name") or "Untitled"
        meta["name"] = f"Fork of {original_name}"
        # Clear interaction_id so load_session uses client history instead
        meta.pop("interaction_id", None)
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(meta_path)

        # 4. Resume with fresh chat (skip_interaction_resume=True)
        from agent.core import create_agent

        agent = create_agent(verbose=False, defer_chat=True)
        metadata, display_log, event_log = _load_session(
            agent, new_id, skip_interaction_resume=True
        )

        state = SessionState(new_id, agent)

        with self._lock:
            self._sessions[new_id] = state

        return state, metadata, display_log, event_log

    # ---- Config hot-reload ----

    def reload_config_for_sessions(self) -> dict:
        """Hot-reload all active sessions after config changes.

        Not yet implemented on the BaseAgent architecture — reports all
        sessions as unchanged.

        Returns:
            dict with keys: reloaded, skipped, failed, unchanged (counts).
        """
        with self._lock:
            sessions = list(self._sessions.items())

        # hot_reload_config not yet implemented on BaseAgent architecture.
        unchanged = len(sessions)

        return {
            "reloaded": 0,
            "skipped": 0,
            "failed": 0,
            "unchanged": unchanged,
        }

    # ---- Shutdown ----

    def shutdown(self) -> None:
        """Save and tear down all sessions (called on server shutdown)."""
        with self._lock:
            states = list(self._sessions.values())
            session_ids = list(self._sessions.keys())
        # Save each session to disk before tearing down
        for state in states:
            try:
                self.run_in_session_context(state, _save_session, state.agent)
            except Exception:
                pass  # Best-effort save on shutdown
        for sid in session_ids:
            self.delete_session(sid)
