"""ToolContext protocol — minimal interface for tool handler dispatch.

SessionContext satisfies this protocol at runtime. ReplayContext provides
a lightweight stand-in for headless pipeline replay without a full session.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from data_ops.store import DataStore
    from agent.event_bus import EventBus
    from data_ops.dag import PipelineDAG
    from rendering.plotly_renderer import PlotlyRenderer


@runtime_checkable
class ToolContext(Protocol):
    """Minimal interface that tool handlers receive.

    At runtime, this is a SessionContext instance. For headless replay,
    a ReplayContext can be used instead.

    Properties may return None in contexts where the resource is unavailable
    (e.g., event_bus is None during replay, mcp_client is None without MCP).

    Per-agent-type state is accessed via ``agent_state``, keyed by agent
    type string (e.g., ``ctx.agent_state["orchestrator"]``).
    """

    @property
    def store(self) -> "DataStore": ...

    @property
    def event_bus(self) -> "EventBus | None": ...

    @property
    def dag(self) -> "PipelineDAG | None": ...

    @property
    def session_dir(self) -> Path | None: ...

    @property
    def asset_registry(self): ...  # AssetRegistry | None

    @property
    def renderer(self) -> "PlotlyRenderer | None": ...

    @property
    def mcp_client(self): ...  # MCPClientManager | None

    @property
    def web_mode(self) -> bool: ...

    @property
    def agent_state(self) -> dict[str, Any]: ...

    @property
    def request_permission(self): ...  # Callable | None

    @property
    def work_tracker(self): ...  # WorkTracker | None

    @property
    def delegation(self): ...  # DelegationBus | None

    @property
    def service(self): ...  # LLMService | None

    @property
    def model_name(self) -> str: ...


class ReplayContext:
    """Minimal ToolContext for replay — no LLM, no delegation."""

    def __init__(
        self,
        store: "DataStore",
        session_dir: Path | None = None,
        mcp_client=None,
    ):
        self._store = store
        self._session_dir = session_dir
        self._mcp_client = mcp_client
        self._renderer = None  # lazy init
        self._agent_state: dict[str, Any] = {}

    @property
    def store(self):
        return self._store

    @property
    def event_bus(self):
        return None

    @property
    def dag(self):
        return None

    @property
    def session_dir(self):
        return self._session_dir

    @property
    def asset_registry(self):
        return None

    @property
    def renderer(self):
        if self._renderer is None:
            from rendering.plotly_renderer import PlotlyRenderer
            self._renderer = PlotlyRenderer()
        return self._renderer

    @property
    def mcp_client(self):
        return self._mcp_client

    @property
    def web_mode(self):
        return True

    @property
    def agent_state(self):
        return self._agent_state

    @property
    def request_permission(self):
        return None

    @property
    def work_tracker(self):
        return None

    @property
    def delegation(self):
        return None

    @property
    def service(self):
        return None

    @property
    def model_name(self):
        return ""
