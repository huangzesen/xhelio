"""Session-scoped resources shared by all agents and tool handlers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agent.delegation import DelegationBus
    from agent.eureka_hooks import EurekaHooks
    from agent.eureka_store import EurekaStore
    from agent.event_bus import EventBus
    from agent.llm.service import LLMService
    from agent.memory import MemoryStore
    from agent.memory_hooks import MemoryHooks
    from agent.work_tracker import WorkTracker
    from data_ops.store import DataStore
    from rendering.plotly_renderer import PlotlyRenderer


@dataclass
class SessionContext:
    """All session-scoped resources in one place.

    Created once per session before any agents are constructed.
    Passed to all agents and tool handlers. Extendable by adding fields.

    Orchestrator-specific state (plan, inline) lives in
    ``agent_state["orchestrator"]`` (an ``OrchestratorState`` dataclass).
    Handlers access it via ``ctx.agent_state["orchestrator"]``.
    """

    # Core
    store: DataStore
    dag: Any  # PipelineDAG
    event_bus: EventBus
    service: LLMService

    # Rendering
    renderer: PlotlyRenderer

    # Memory
    memory_store: MemoryStore | None = None
    memory_hooks: MemoryHooks | None = None

    # Eureka
    eureka_store: EurekaStore | None = None
    eureka_hooks: EurekaHooks | None = None

    # Session lifecycle
    session_manager: Any = None  # SessionManager
    session_dir: Path | None = None

    # Asset registry
    asset_registry: Any = None  # AssetRegistry

    # Delegation & work tracking
    delegation: DelegationBus | None = None
    work_tracker: WorkTracker | None = None

    # Environment
    web_mode: bool = False
    mcp_client: Any | None = None  # MCPClient

    # Permission gate
    request_permission: Callable | None = None

    # Per-agent-type state sections — keyed by agent type string
    # e.g. ctx.agent_state["orchestrator"] = OrchestratorState()
    agent_state: dict[str, Any] = field(default_factory=dict)

    # Session identity (shared — API layer and persistence use these)
    session_id: str = ""
    model_name: str = ""

    # Deferred figure state (API layer accesses without agent reference)
    deferred_figure_state: Any = None

    # Cancel signal
    cancel_event: Any = None  # threading.Event
