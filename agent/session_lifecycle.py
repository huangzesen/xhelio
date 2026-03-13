"""Session lifecycle — builds SessionContext and manages session start/end.

This replaces the resource creation logic scattered across
OrchestratorAgent.__init__ and the session management code in core.py.
A Session object creates a fully populated SessionContext that can be
passed to any agent constructor.

Usage:
    session = Session(service=service)
    ctx = session.start()
    orchestrator = OrchestratorAgent(session_ctx=ctx, service=service)
    orchestrator.start()
    # ... run session ...
    session.end()
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .llm import LLMService

from .session_context import SessionContext
from .work_tracker import WorkTracker


class Session:
    """Lifecycle manager that creates and owns a SessionContext.

    Attributes:
        ctx: The SessionContext created by start(). None before start().
    """

    def __init__(
        self,
        service: LLMService,
        *,
        session_dir: Path | None = None,
        web_mode: bool = False,
    ):
        self._service = service
        self._session_dir = session_dir
        self._web_mode = web_mode
        self.ctx: SessionContext | None = None

    def start(self) -> SessionContext:
        """Create all session-scoped resources and return SessionContext."""
        from data_ops.store import DataStore
        from data_ops.dag import PipelineDAG
        from rendering.plotly_renderer import PlotlyRenderer
        from .event_bus import get_event_bus
        from .session import SessionManager

        session_id = uuid4().hex[:12]

        # Resolve session directory
        session_dir = self._session_dir
        if session_dir is None:
            import tempfile
            session_dir = Path(tempfile.mkdtemp(prefix=f"xhelio-{session_id}-"))

        session_dir.mkdir(parents=True, exist_ok=True)

        # Create core resources
        store = DataStore(data_dir=session_dir)
        dag = PipelineDAG(session_dir=session_dir)
        event_bus = get_event_bus()
        renderer = PlotlyRenderer()
        work_tracker = WorkTracker()

        # Subscribe PipelineDAGListener for pipeline tracking
        from .event_bus import PipelineDAGListener
        self._pipeline_listener = PipelineDAGListener(lambda: dag)
        event_bus.subscribe(self._pipeline_listener)

        # Memory store — create if available, otherwise None
        memory_store = None
        try:
            from .memory import MemoryStore
            memory_store = MemoryStore()
        except ImportError:
            pass

        self.ctx = SessionContext(
            store=store,
            dag=dag,
            event_bus=event_bus,
            service=self._service,
            renderer=renderer,
            memory_store=memory_store,
            memory_hooks=None,  # set below (needs self.ctx)
            session_manager=SessionManager(),
            session_dir=session_dir,
            delegation=None,  # set below (needs self.ctx)
            work_tracker=work_tracker,
            web_mode=self._web_mode,
            session_id=session_id,
            cancel_event=threading.Event(),
        )

        # Wire PermissionGate — uses event_bus for SSE notifications
        from .permission_gate import PermissionGate
        self.ctx.request_permission = PermissionGate(event_bus=event_bus)

        # Wire DelegationBus — needs self.ctx to exist first
        from .delegation import DelegationBus
        self.ctx.delegation = DelegationBus(ctx=self.ctx)

        # Wire MemoryHooks — callbacks are connected later in core.py
        # after the orchestrator is created (they need agent internals).
        if memory_store is not None:
            from .memory_hooks import MemoryHooks
            self.ctx.memory_hooks = MemoryHooks(ctx=self.ctx)

        # Wire EurekaStore + EurekaHooks — inbox_injector callback is
        # connected in core.py after the orchestrator is created.
        try:
            from .eureka_store import EurekaStore
            from .eureka_hooks import EurekaHooks
            self.ctx.eureka_store = EurekaStore()
            self.ctx.eureka_hooks = EurekaHooks(ctx=self.ctx)
        except ImportError:
            pass

        # Ensure OrchestratorState exists (other code may still read it)
        if "orchestrator" not in self.ctx.agent_state:
            from .tool_caller import OrchestratorState
            self.ctx.agent_state["orchestrator"] = OrchestratorState()

        # Wire EventFeedBuffer so the events tool works
        from .event_feed import EventFeedBuffer
        orch_state = self.ctx.agent_state["orchestrator"]
        if orch_state.event_feed is None:
            orch_state.event_feed = EventFeedBuffer(
                self.ctx.event_bus, "ctx:orchestrator"
            )

        return self.ctx

    def end(self) -> None:
        """Clean up session resources."""
        if self.ctx is None:
            return

        # Unsubscribe pipeline listener to prevent stale references
        if hasattr(self, '_pipeline_listener') and self.ctx.event_bus is not None:
            self.ctx.event_bus.unsubscribe(self._pipeline_listener)

        # Cancel active work, then clear completed entries
        if self.ctx.work_tracker:
            self.ctx.work_tracker.cancel_all()
            self.ctx.work_tracker.clear()

        self.ctx = None
