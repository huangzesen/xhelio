"""Delegation subsystem — manages sub-agent lifecycle and dispatch.

The ``DelegationBus`` owns the sub-agent registry (``_agents`` dict),
ephemeral agent counters, retired usage tracking, and all factory +
dispatch methods.  It receives a ``SessionContext`` as ``ctx``.
"""

from __future__ import annotations

import json
import queue
import time
import threading
import uuid
from typing import Callable, TYPE_CHECKING

import config
from .base_agent import BaseAgent, AgentState
from .envoy_agent import EnvoyAgent
from .viz_agent import VizAgent
from .viz_backends import VIZ_BACKENDS
from .data_ops_agent import DataOpsAgent
from .data_io_agent import DataIOAgent
from .truncation import get_item_limit
from .event_bus import DEBUG
from .logging import get_logger

if TYPE_CHECKING:
    from .session_context import SessionContext


# ---------------------------------------------------------------------------
# Agent ID constants — single source of truth for agent registry keys
# ---------------------------------------------------------------------------

AGENT_ID_DATAOPS = "DataOpsAgent"
AGENT_ID_DATA_IO = "DataIOAgent"
AGENT_ID_EUREKA = "EurekaAgent"
AGENT_ID_MEMORY = "MemoryAgent"


class DelegationBus:
    """Manages sub-agent creation, delegation dispatch, and cleanup.

    Args:
        ctx: The SessionContext instance (shared resource pool for all agents).
    """

    def __init__(self, ctx: "SessionContext"):
        self._ctx = ctx
        self._agents: dict[str, BaseAgent] = {}
        self._lock = threading.Lock()
        self._dataops_seq: int = 0
        self._mission_seq: int = 0
        self._retired_usage: list[dict] = []
        self._async_delegations: dict[str, float] = {}

    # ------------------------------------------------------------------
    # State accessors (used by orchestrator for token tracking, etc.)
    # ------------------------------------------------------------------

    @property
    def agents(self) -> dict[str, BaseAgent]:
        """Direct access to the agents dict (caller must hold lock if mutating)."""
        return self._agents

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    @property
    def retired_usage(self) -> list[dict]:
        return self._retired_usage

    # ------------------------------------------------------------------
    # Agent idle check
    # ------------------------------------------------------------------

    def all_work_subagents_idle(self) -> bool:
        """Check if all work subagents (excluding Eureka/Memory) are idle."""
        with self._lock:
            for agent_id, agent in self._agents.items():
                if agent_id in (AGENT_ID_EUREKA, AGENT_ID_MEMORY):
                    continue  # Post-cycle agents, not part of work cycle
                if not agent.is_idle:
                    return False
        return True

    # ------------------------------------------------------------------
    # Active envoy IDs
    # ------------------------------------------------------------------

    def get_active_envoy_ids(self) -> set[str]:
        """Return set of active mission IDs from current agents. Thread-safe."""
        with self._lock:
            return {
                k.removeprefix("EnvoyAgent[").rstrip("]")
                for k in self._agents
                if k.startswith("EnvoyAgent[")
            }

    # ------------------------------------------------------------------
    # Reset / cleanup
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Stop and remove all sub-agents."""
        from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY

        with self._lock:
            for agent_id, agent in list(self._agents.items()):
                try:
                    agent.stop(timeout=2.0)
                except Exception as e:
                    get_logger().warning(
                        "Agent %s stop failed during reset: %s", agent_id, e
                    )
            self._agents.clear()
        ENVOY_KIND_REGISTRY.clear_active()
        sctx = self._ctx
        orch_state = sctx.agent_state.get("orchestrator") if sctx else None
        if orch_state and orch_state.ctx_tracker:
            orch_state.ctx_tracker.reset_all()
        if sctx and sctx.event_bus:
            sctx.event_bus.emit(
                DEBUG,
                level="debug",
                msg="[Config] Sub-agents invalidated after config reload",
            )

    def reset_full(self) -> None:
        """Full reset including sequence counters (used by new-session)."""
        with self._lock:
            for agent_id, agent in list(self._agents.items()):
                try:
                    agent.stop(timeout=2.0)
                except Exception as e:
                    get_logger().warning(
                        "Agent %s stop failed during reset_full: %s", agent_id, e
                    )
            self._agents.clear()
        from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY

        ENVOY_KIND_REGISTRY.clear_active()
        self._dataops_seq = 0
        self._mission_seq = 0
        self._retired_usage.clear()
        self._async_delegations.clear()

    def stop_all(self) -> None:
        """Stop and clear all agents (used during shutdown)."""
        with self._lock:
            for agent in self._agents.values():
                agent.stop(timeout=2.0)
            self._agents.clear()

    def cleanup_ephemeral(self, agent_id: str) -> None:
        """Shut down and remove an ephemeral agent from the registry.

        Preserves the agent's token usage in ``_retired_usage`` so it
        is still included in ``get_token_usage()`` and
        ``get_token_usage_breakdown()`` after the agent is removed.
        """
        with self._lock:
            agent = self._agents.pop(agent_id, None)
        if agent:
            # Preserve token usage before stopping the agent
            usage = agent.get_token_usage()
            if usage.get("api_calls", 0) > 0:
                self._retired_usage.append(
                    {
                        "agent": agent_id,
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "thinking_tokens": usage.get("thinking_tokens", 0),
                        "cached_tokens": usage.get("cached_tokens", 0),
                        "api_calls": usage["api_calls"],
                        "ctx_system_tokens": usage.get("ctx_system_tokens", 0),
                        "ctx_tools_tokens": usage.get("ctx_tools_tokens", 0),
                        "ctx_history_tokens": usage.get("ctx_history_tokens", 0),
                        "ctx_total_tokens": usage.get("ctx_total_tokens", 0),
                    }
                )
            agent.stop()
            sctx = self._ctx
            orch_state = sctx.agent_state.get("orchestrator") if sctx else None
            if orch_state and orch_state.ctx_tracker:
                orch_state.ctx_tracker.reset(agent_id)
            if sctx and sctx.event_bus:
                sctx.event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Cleaned up ephemeral agent {agent_id}",
                )

    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Register an externally-created agent (e.g. MemoryAgent, EurekaAgent)."""
        with self._lock:
            self._agents[agent_id] = agent

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def _get_session_ctx(self):
        """Return the SessionContext (self._ctx IS the SessionContext now)."""
        return self._ctx

    def get_or_create_envoy_agent(self, mission_id: str) -> EnvoyAgent:
        """Get the persistent envoy agent, creating it on first use."""
        agent_id = f"EnvoyAgent[{mission_id}]"
        sctx = self._get_session_ctx()
        with self._lock:
            if agent_id not in self._agents:
                from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
                kind_info = ENVOY_KIND_REGISTRY.get_kind(mission_id)
                kind_name = kind_info.get("kind", mission_id) if kind_info else mission_id

                agent = EnvoyAgent(
                    kind=kind_name,
                    instance_id=mission_id,
                    service=sctx.service,
                    session_ctx=sctx,
                    tool_schemas=[],
                    system_prompt=f"You are the {mission_id} envoy specialist.",
                    event_bus=sctx.event_bus,
                    cancel_event=self._ctx.cancel_event,
                )
                agent.start()
                self._agents[agent_id] = agent
                sctx.event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created {mission_id} envoy agent",
                )
                if ENVOY_KIND_REGISTRY.mark_active(mission_id):
                    from .agent_registry import AGENT_INFORMED_REGISTRY, CTX_ORCHESTRATOR

                    for tool_name in ENVOY_KIND_REGISTRY.get_tool_names(mission_id):
                        AGENT_INFORMED_REGISTRY._registry.setdefault(
                            CTX_ORCHESTRATOR, set()
                        ).add(tool_name)
            return self._agents[agent_id]

    def create_ephemeral_envoy_agent(self, mission_id: str) -> EnvoyAgent:
        """Create an ephemeral overflow envoy agent for parallel delegation."""
        sctx = self._get_session_ctx()
        with self._lock:
            seq = self._mission_seq
            self._mission_seq = seq + 1
            ephemeral_id = f"EnvoyAgent[{mission_id}]#{seq}"

            from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
            kind_info = ENVOY_KIND_REGISTRY.get_kind(mission_id)
            kind_name = kind_info.get("kind", mission_id) if kind_info else mission_id

            agent = EnvoyAgent(
                kind=kind_name,
                instance_id=mission_id,
                service=sctx.service,
                session_ctx=sctx,
                tool_schemas=[],
                system_prompt=f"You are the {mission_id} envoy specialist.",
                event_bus=sctx.event_bus,
                cancel_event=self._ctx.cancel_event,
            )
            agent.start()
            self._agents[ephemeral_id] = agent
        sctx.event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[Router] Created ephemeral envoy agent {ephemeral_id}",
        )
        return agent

    def get_or_create_viz_agent(self, backend: str) -> VizAgent:
        """Get the cached viz agent for this backend or create a new one. Thread-safe."""
        cfg = VIZ_BACKENDS[backend]
        agent_id = cfg["agent_id"]
        sctx = self._get_session_ctx()
        with self._lock:
            if agent_id not in self._agents:
                agent = VizAgent(
                    backend=backend,
                    service=sctx.service,
                    session_ctx=sctx,
                    event_bus=sctx.event_bus,
                    cancel_event=self._ctx.cancel_event,
                )
                agent.start()
                self._agents[agent_id] = agent
                sctx.event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created {backend} visualization agent",
                )
            return self._agents[agent_id]

    def get_available_dataops_agent(self) -> DataOpsAgent:
        """Get an idle DataOps agent or create a new ephemeral one.

        Priority: (1) idle primary agent, (2) create primary if it doesn't
        exist, (3) create an ephemeral overflow instance.  Ephemeral agents
        are cleaned up after their delegation completes.
        """
        primary_id = AGENT_ID_DATAOPS
        sctx = self._get_session_ctx()
        with self._lock:
            if primary_id in self._agents:
                agent = self._agents[primary_id]
                if agent.state == AgentState.SLEEPING and agent.inbox.qsize() == 0:
                    return agent
            else:
                # Create the primary (persistent) agent
                agent = DataOpsAgent(
                    service=sctx.service,
                    session_ctx=sctx,
                    event_bus=sctx.event_bus,
                    cancel_event=self._ctx.cancel_event,
                )
                agent.start()
                self._agents[primary_id] = agent
                sctx.event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created DataOps agent",
                )
                return agent

            # Primary is busy — create an ephemeral overflow instance
            seq = self._dataops_seq
            self._dataops_seq = seq + 1
            ephemeral_id = f"{AGENT_ID_DATAOPS}#{seq}"
            agent = DataOpsAgent(
                service=sctx.service,
                session_ctx=sctx,
                event_bus=sctx.event_bus,
                cancel_event=self._ctx.cancel_event,
            )
            agent.start()
            self._agents[ephemeral_id] = agent
            sctx.event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Router] Created ephemeral DataOps agent {ephemeral_id}",
            )
            return agent

    def get_or_create_data_io_agent(self) -> DataIOAgent:
        """Get the cached data I/O agent or create a new one. Thread-safe."""
        agent_id = AGENT_ID_DATA_IO
        sctx = self._get_session_ctx()
        with self._lock:
            if agent_id not in self._agents:
                agent = DataIOAgent(
                    service=sctx.service,
                    session_ctx=sctx,
                    event_bus=sctx.event_bus,
                    cancel_event=self._ctx.cancel_event,
                )
                agent.start()
                self._agents[agent_id] = agent
                sctx.event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created DataIO agent",
                )
            return self._agents[agent_id]

    # ------------------------------------------------------------------
    # Store-delta formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_store_delta(
        ctx_tracker, agent_id: str, entries: list[dict], fmt: str = "json"
    ) -> str:
        """Compute store delta for *agent_id* and return formatted injection text.

        Args:
            ctx_tracker: The ``ContextTracker`` instance.
            agent_id: Sub-agent identifier used for delta tracking.
            entries: Current store entries (from ``store.list_entries()``).
            fmt: ``"json"`` wraps new-entry metadata in a ```json code block;
                 ``"text"`` uses a plain-text bullet list.

        Returns:
            A string to append to the delegation request (empty if no delta).
        """
        new_entries, removed_labels, store_hash = ctx_tracker.get_store_delta(
            agent_id, entries
        )
        if not new_entries and not removed_labels:
            return ""

        # Format the "added" portion according to *fmt*
        if fmt == "json":
            added_block = (
                "```json\n" + json.dumps(new_entries, indent=2, default=str) + "\n```"
                if new_entries
                else ""
            )
        else:  # "text"
            added_block = (
                "\n".join(
                    f"  - {e['label']} ({e['num_points']} pts, "
                    f"{e['time_min']} to {e['time_max']})"
                    for e in new_entries
                )
                if new_entries
                else ""
            )

        # Compose the full delta text
        if new_entries and not removed_labels:
            text = "\n\nNew data added to memory:\n" + added_block
        elif removed_labels and not new_entries:
            text = "\n\nData removed from memory: " + ", ".join(removed_labels)
        else:
            text = "\n\nData store updated:\n" + added_block
            if removed_labels:
                text += "\nRemoved: " + ", ".join(removed_labels)

        # Record the new baseline so subsequent calls see no delta
        ctx_tracker.record(agent_id, store_entries=entries, store_hash=store_hash)
        return text

    # ------------------------------------------------------------------
    # Request builders
    # ------------------------------------------------------------------

    def build_envoy_request(self, mission_id: str, request: str, agent=None) -> str:
        """Build a full request string for an envoy delegation."""
        sctx = self._get_session_ctx()
        agent_id = agent.agent_id if agent else f"EnvoyAgent[{mission_id}]"
        store = sctx.store
        entries = store.list_entries()
        if entries:
            _orch_state = sctx.agent_state.get("orchestrator") if sctx else None
            _ctx_tracker = _orch_state.ctx_tracker if _orch_state else None
            delta_text = self.format_store_delta(
                _ctx_tracker, agent_id, entries, fmt="text"
            )
            if delta_text:
                request += delta_text
                request += "\nDo NOT re-fetch data that is already in memory with a matching label and time range."
        if not (agent and agent._interaction_id):
            request += "\n\n[Tip: Call events(action='check') to see what happened earlier in this session.]"
        return request

    def build_dataops_request(self, request: str, context: str, agent=None) -> str:
        """Build a full request string for a DataOps delegation."""
        sctx = self._get_session_ctx()
        _agent_id = agent.agent_id if agent else AGENT_ID_DATAOPS
        full_request = f"{request}\n\nContext: {context}" if context else request
        store = sctx.store
        entries = store.list_entries()
        if entries:
            _orch_state = sctx.agent_state.get("orchestrator") if sctx else None
            _ctx_tracker = _orch_state.ctx_tracker if _orch_state else None
            delta_text = self.format_store_delta(
                _ctx_tracker, _agent_id, entries, fmt="json"
            )
            if delta_text:
                full_request += delta_text
        if not (agent and agent._interaction_id):
            full_request += "\n\n[Tip: Call events(action='check') to see what happened earlier in this session.]"
        return full_request

    def build_data_io_request(self, request: str, context: str) -> str:
        """Build a full request string for a DataIO delegation."""
        full_request = f"{request}\n\nContext: {context}" if context else request
        return full_request

    # ------------------------------------------------------------------
    # Delegation result wrapping
    # ------------------------------------------------------------------

    @staticmethod
    def wrap_delegation_result(sub_result, store_snapshot=None) -> dict:
        """Convert an agent send result into a tool result dict.

        Success is determined by actual output, not by error heuristics.
        If the agent produced meaningful output (text or files), it's a success
        even if there were transient errors during retries. The LLM sees both
        the result and any errors in the text.

        Args:
            sub_result: Dict from agent's _handle_request ({text, failed, errors}).
            store_snapshot: Optional list of store entry summaries to include,
                so the orchestrator LLM sees concrete data state after delegation.
        """
        if isinstance(sub_result, dict):
            text = sub_result.get("text", "")
            failed = sub_result.get("failed", False)
            errors = sub_result.get("errors", [])
            output_files = sub_result.get("output_files", [])
        else:
            # Legacy: plain string (shouldn't happen, but be safe)
            text = str(sub_result)
            failed = False
            errors = []
            output_files = []

        # Check for actual success: has meaningful output (text or output_files)
        # Even if there were errors during retries, if we got output, it's a success
        has_output = bool(text.strip()) or bool(output_files)
        has_critical_errors = failed and errors

        if has_critical_errors and not has_output:
            # Failed with no output - true failure
            error_summary = "; ".join(errors[-get_item_limit("items.error_summary"):])
            result = {
                "status": "error",
                "message": f"Sub-agent failed. Errors: {error_summary}",
                "result": text,
            }
        else:
            # Has output (possibly with some errors during retries) - success
            # The LLM will see both the result and any errors in the text
            result = {"status": "success", "result": text}

        if output_files:
            result["output_files"] = output_files

        if store_snapshot is not None:
            result["data_in_memory"] = [
                {
                    "label": e["label"],
                    "columns": e.get("columns", []),
                    "shape": e.get("shape", ""),
                    "units": e.get("units", ""),
                    "num_points": e.get("num_points", 0),
                }
                for e in store_snapshot
            ]
        return result

    # ------------------------------------------------------------------
    # Core delegation dispatch
    # ------------------------------------------------------------------

    def delegate_to_sub_agent(
        self,
        agent: BaseAgent,
        request,
        timeout: float = 300.0,
        wait: bool = True,
        store_snapshot=None,
        tool_call_id: str | None = None,
        agent_type: str = "",
        agent_name: str = "",
        task_summary: str = "",
        post_process: Callable | None = None,
        post_complete: Callable | None = None,
    ) -> dict:
        """Dispatch delegation synchronously — blocks until the sub-agent finishes.

        Called from ``execute_tools_batch``'s ThreadPoolExecutor worker threads,
        so multiple delegations in the same LLM turn still run in parallel.
        Results are returned directly with proper tool_call_id pairing (no
        stale IDs, no pending_async).

        Args:
            agent: The target Agent instance.
            request: String or dict payload for the agent.
            timeout: Max seconds to wait.
            store_snapshot: Optional list of store entries to include in result.
            tool_call_id: LLM tool_call_id for result mapping.
            agent_type: Agent type for work tracking.
            agent_name: Agent name for work tracking.
            task_summary: Human-readable summary for work tracking.
            post_process: Optional callable(result) -> result to run after
                delegation completes.
            post_complete: Optional callable(result) to run after processing.
                For non-critical work (e.g. PNG export).
        """
        sctx = self._get_session_ctx()
        wt = sctx.work_tracker if sctx else None
        summary = task_summary or (
            request[:200] if isinstance(request, str) else str(request)[:200]
        )
        work_id = f"wu_{uuid.uuid4().hex[:8]}"

        if wt is not None:
            cancel_event = self._ctx.cancel_event or threading.Event()
            wt.register(
                work_id=work_id,
                agent_id=agent_name or agent.agent_id,
                description=summary,
                cancel_event=cancel_event,
            )

        # Handle fire-and-forget delegation — start agent but don't wait
        if not wait:
            agent.send(request, sender="orchestrator", timeout=timeout, wait=False)

            # Track as async delegation for completion notification
            self._async_delegations[agent_name or agent.agent_id] = time.time()

            if wt is not None:
                wt.mark_completed(work_id)
            return {
                "status": "queued",
                "message": f"Delegation to {agent_name or agent.agent_id} started (fire-and-forget)",
            }

        try:
            result = agent.send(
                request, sender="orchestrator", timeout=timeout, wait=wait
            )

            wrapped = self.wrap_delegation_result(
                result, store_snapshot=store_snapshot
            )
            try:
                if post_process is not None:
                    wrapped = post_process(wrapped)
            finally:
                if wt is not None:
                    wt.mark_completed(work_id)

            # Fire-and-forget callback after marking complete
            if post_complete is not None:
                try:
                    post_complete(wrapped)
                except Exception as pc_err:
                    get_logger().debug(f"post_complete callback failed: {pc_err}")

            return wrapped
        except Exception as e:
            error_msg = str(e)
            if wt is not None:
                wt.mark_failed(work_id, error_msg)
            return {
                "status": "error",
                "message": f"Delegation failed: {error_msg}",
            }

    # ------------------------------------------------------------------
    # Token usage aggregation helpers
    # ------------------------------------------------------------------

    def get_all_agent_usages(self) -> tuple[list[tuple[str, dict]], list[dict]]:
        """Return (active_agents, retired_usage) for token tracking.

        Returns:
            Tuple of:
            - list of (agent_id, usage_dict) for active agents
            - list of retired usage dicts
        """
        active = []
        with self._lock:
            for agent_id, agent in self._agents.items():
                active.append((agent_id, agent.get_token_usage()))
        return active, list(self._retired_usage)

    def has_agent(self, agent_id: str) -> bool:
        """Check if an agent exists in the registry."""
        with self._lock:
            return agent_id in self._agents

    def is_stale(self) -> bool:
        """Check if any sub-agents exist (used for hot-reload staleness check)."""
        with self._lock:
            return len(self._agents) > 0
