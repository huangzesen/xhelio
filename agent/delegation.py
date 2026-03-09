"""Delegation subsystem — manages sub-agent lifecycle and dispatch.

Extracted from OrchestratorAgent to reduce the size of core.py.
The ``DelegationBus`` owns the sub-agent registry (``_agents`` dict),
ephemeral agent counters, retired usage tracking, and all factory +
dispatch methods.  It receives the orchestrator (or any object that
satisfies the implicit DelegationContext interface) as ``ctx``.
"""

from __future__ import annotations

import json
import queue
import time
import threading
from typing import Callable, TYPE_CHECKING

import config
from .sub_agent import SubAgent, AgentState
from .envoy_agent import EnvoyAgent
from .viz_plotly_agent import VizPlotlyAgent
from .viz_mpl_agent import VizMplAgent
from .viz_jsx_agent import VizJsxAgent
from .data_ops_agent import DataOpsAgent
from .data_io_agent import DataIOAgent
from .insight_agent import InsightAgent
from .truncation import get_item_limit
from .event_bus import DEBUG
from .logging import get_logger

if TYPE_CHECKING:
    from .core import OrchestratorAgent


class DelegationBus:
    """Manages sub-agent creation, delegation dispatch, and cleanup.

    Args:
        ctx: The OrchestratorAgent instance (used to access service,
             event_bus, store, renderer, etc.).
    """

    def __init__(self, ctx: "OrchestratorAgent"):
        self._ctx = ctx
        self._agents: dict[str, SubAgent] = {}
        self._lock = threading.Lock()
        self._dataops_seq: int = 0
        self._mission_seq: int = 0
        self._retired_usage: list[dict] = []

    # ------------------------------------------------------------------
    # State accessors (used by orchestrator for token tracking, etc.)
    # ------------------------------------------------------------------

    @property
    def agents(self) -> dict[str, SubAgent]:
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
                if agent_id in ("EurekaAgent", "MemoryAgent"):
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
            for agent in self._agents.values():
                agent.stop(timeout=2.0)
            self._agents.clear()
        ENVOY_KIND_REGISTRY.clear_active()
        self._ctx._ctx_tracker.reset_all()
        self._ctx._event_bus.emit(
            DEBUG,
            level="debug",
            msg="[Config] Sub-agents invalidated after config reload",
        )

    def reset_full(self) -> None:
        """Full reset including sequence counters (used by new-session)."""
        with self._lock:
            for agent in self._agents.values():
                agent.stop(timeout=2.0)
            self._agents.clear()
        from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY

        ENVOY_KIND_REGISTRY.clear_active()
        self._dataops_seq = 0
        self._mission_seq = 0
        self._retired_usage.clear()

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
            self._ctx._ctx_tracker.reset(agent_id)
            self._ctx._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Router] Cleaned up ephemeral agent {agent_id}",
            )

    def register_agent(self, agent_id: str, agent: SubAgent) -> None:
        """Register an externally-created agent (e.g. MemoryAgent, EurekaAgent)."""
        with self._lock:
            self._agents[agent_id] = agent

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def get_or_create_envoy_agent(self, mission_id: str) -> EnvoyAgent:
        """Get the persistent envoy agent, creating it on first use."""
        agent_id = f"EnvoyAgent[{mission_id}]"
        ctx = self._ctx
        with self._lock:
            if agent_id not in self._agents:
                agent = EnvoyAgent(
                    mission_id=mission_id,
                    service=ctx.service,
                    tool_executor=lambda name, args, tc_id=None, _mid=mission_id: (
                        ctx._execute_tool_for_agent(
                            name, args, tc_id, agent_type=f"envoy:{_mid}",
                        )
                    ),
                    event_bus=ctx._event_bus,
                    memory_store=ctx._memory_store,
                    memory_scope=f"envoy:{mission_id}",
                    cancel_event=ctx._cancel_event,
                )
                agent._orchestrator_inbox = ctx._inbox
                agent.start()
                self._agents[agent_id] = agent
                ctx._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created {mission_id} envoy agent",
                )
                from agent.envoy_kinds.registry import ENVOY_KIND_REGISTRY
                if ENVOY_KIND_REGISTRY.mark_active(mission_id):
                    from .agent_registry import AGENT_INFORMED_REGISTRY

                    for tool_name in ENVOY_KIND_REGISTRY.get_tool_names(mission_id):
                        AGENT_INFORMED_REGISTRY._registry.setdefault(
                            "ctx:orchestrator", set()
                        ).add(tool_name)
            return self._agents[agent_id]

    def create_ephemeral_envoy_agent(self, mission_id: str) -> EnvoyAgent:
        """Create an ephemeral overflow envoy agent for parallel delegation."""
        ctx = self._ctx
        with self._lock:
            seq = self._mission_seq
            self._mission_seq = seq + 1
            ephemeral_id = f"EnvoyAgent[{mission_id}]#{seq}"

            agent = EnvoyAgent(
                mission_id=mission_id,
                service=ctx.service,
                tool_executor=lambda name, args, tc_id=None, _mid=mission_id: (
                    ctx._execute_tool_for_agent(
                        name, args, tc_id, agent_type=f"envoy:{_mid}",
                    )
                ),
                agent_id=ephemeral_id,
                event_bus=ctx._event_bus,
                memory_store=ctx._memory_store,
                memory_scope=f"envoy:{mission_id}",
                cancel_event=ctx._cancel_event,
            )
            agent._orchestrator_inbox = ctx._inbox
            agent.start()
            self._agents[ephemeral_id] = agent
        ctx._event_bus.emit(
            DEBUG,
            level="debug",
            msg=f"[Router] Created ephemeral envoy agent {ephemeral_id}",
        )
        return agent

    def get_or_create_viz_plotly_agent(self) -> VizPlotlyAgent:
        """Get the cached Plotly viz agent or create a new one. Thread-safe."""
        agent_id = "VizAgent[Plotly]"
        ctx = self._ctx
        with self._lock:
            if agent_id not in self._agents:
                agent = VizPlotlyAgent(
                    service=ctx.service,
                    tool_executor=lambda name, args, tc_id=None: (
                        ctx._execute_tool_for_agent(
                            name, args, tc_id, agent_type="viz_plotly"
                        )
                    ),
                    gui_mode=ctx.gui_mode,
                    event_bus=ctx._event_bus,
                    memory_store=ctx._memory_store,
                    memory_scope="visualization",
                    cancel_event=ctx._cancel_event,
                )
                agent._orchestrator_inbox = ctx._inbox
                agent.start()
                self._agents[agent_id] = agent
                ctx._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created Plotly Visualization agent",
                )
            return self._agents[agent_id]

    def get_or_create_viz_mpl_agent(self) -> VizMplAgent:
        """Get the cached MPL viz agent or create a new one. Thread-safe."""
        agent_id = "VizAgent[Mpl]"
        ctx = self._ctx
        with self._lock:
            if agent_id not in self._agents:
                session_dir = ctx._session_manager.base_dir / ctx._session_id
                agent = VizMplAgent(
                    service=ctx.service,
                    tool_executor=lambda name, args, tc_id=None: (
                        ctx._execute_tool_for_agent(
                            name, args, tc_id, agent_type="viz_mpl"
                        )
                    ),
                    gui_mode=ctx.gui_mode,
                    event_bus=ctx._event_bus,
                    memory_store=ctx._memory_store,
                    memory_scope="visualization",
                    session_dir=session_dir,
                    cancel_event=ctx._cancel_event,
                )
                agent._orchestrator_inbox = ctx._inbox
                agent.start()
                self._agents[agent_id] = agent
                ctx._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created MPL Visualization agent",
                )
            return self._agents[agent_id]

    def get_or_create_viz_jsx_agent(self) -> VizJsxAgent:
        """Get the cached JSX viz agent or create a new one. Thread-safe."""
        agent_id = "VizAgent[JSX]"
        ctx = self._ctx
        with self._lock:
            if agent_id not in self._agents:
                session_dir = ctx._session_manager.base_dir / ctx._session_id
                agent = VizJsxAgent(
                    service=ctx.service,
                    tool_executor=lambda name, args, tc_id=None: (
                        ctx._execute_tool_for_agent(
                            name, args, tc_id, agent_type="viz_jsx"
                        )
                    ),
                    gui_mode=ctx.gui_mode,
                    event_bus=ctx._event_bus,
                    memory_store=ctx._memory_store,
                    memory_scope="visualization",
                    session_dir=session_dir,
                    cancel_event=ctx._cancel_event,
                )
                agent._orchestrator_inbox = ctx._inbox
                agent.start()
                self._agents[agent_id] = agent
                ctx._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created JSX Visualization agent",
                )
            return self._agents[agent_id]

    def get_available_dataops_agent(self) -> DataOpsAgent:
        """Get an idle DataOps agent or create a new ephemeral one.

        Priority: (1) idle primary agent, (2) create primary if it doesn't
        exist, (3) create an ephemeral overflow instance.  Ephemeral agents
        are cleaned up after their delegation completes.
        """
        primary_id = "DataOpsAgent"
        ctx = self._ctx
        with self._lock:
            if primary_id in self._agents:
                agent = self._agents[primary_id]
                if agent.state == AgentState.SLEEPING and agent.inbox.qsize() == 0:
                    return agent
            else:
                # Create the primary (persistent) agent
                agent = DataOpsAgent(
                    service=ctx.service,
                    tool_executor=lambda name, args, tc_id=None: (
                        ctx._execute_tool_for_agent(
                            name, args, tc_id, agent_type="dataops"
                        )
                    ),
                    event_bus=ctx._event_bus,
                    memory_store=ctx._memory_store,
                    memory_scope="data_ops",
                    active_missions_fn=self.get_active_envoy_ids,
                    cancel_event=ctx._cancel_event,
                )
                agent._orchestrator_inbox = ctx._inbox
                agent.start()
                self._agents[primary_id] = agent
                ctx._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created DataOps agent",
                )
                return agent

            # Primary is busy — create an ephemeral overflow instance
            seq = self._dataops_seq
            self._dataops_seq = seq + 1
            ephemeral_id = f"DataOpsAgent#{seq}"
            agent = DataOpsAgent(
                service=ctx.service,
                tool_executor=lambda name, args, tc_id=None: (
                    ctx._execute_tool_for_agent(
                        name, args, tc_id, agent_type="dataops"
                    )
                ),
                agent_id=ephemeral_id,
                event_bus=ctx._event_bus,
                memory_store=ctx._memory_store,
                memory_scope="data_ops",
                active_missions_fn=self.get_active_envoy_ids,
                cancel_event=ctx._cancel_event,
            )
            agent._orchestrator_inbox = ctx._inbox
            agent.start()
            self._agents[ephemeral_id] = agent
            ctx._event_bus.emit(
                DEBUG,
                level="debug",
                msg=f"[Router] Created ephemeral DataOps agent {ephemeral_id}",
            )
            return agent

    def get_or_create_data_io_agent(self) -> DataIOAgent:
        """Get the cached data I/O agent or create a new one. Thread-safe."""
        agent_id = "DataIOAgent"
        ctx = self._ctx
        with self._lock:
            if agent_id not in self._agents:
                agent = DataIOAgent(
                    service=ctx.service,
                    tool_executor=lambda name, args, tc_id=None: (
                        ctx._execute_tool_for_agent(
                            name, args, tc_id, agent_type="data_io"
                        )
                    ),
                    event_bus=ctx._event_bus,
                    cancel_event=ctx._cancel_event,
                )
                agent._orchestrator_inbox = ctx._inbox
                agent.start()
                self._agents[agent_id] = agent
                ctx._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created DataIO agent",
                )
            return self._agents[agent_id]

    def get_or_create_insight_agent(self) -> InsightAgent:
        """Get the cached insight agent or create a new one. Thread-safe."""
        agent_id = "InsightAgent"
        ctx = self._ctx
        with self._lock:
            if agent_id not in self._agents:
                agent = InsightAgent(
                    service=ctx.service,
                    tool_executor=lambda name, args, tc_id=None: (
                        ctx._execute_tool_for_agent(
                            name, args, tc_id, agent_type="viz_plotly"
                        )
                    ),
                    event_bus=ctx._event_bus,
                    cancel_event=ctx._cancel_event,
                )
                agent._orchestrator_inbox = ctx._inbox
                agent.start()
                self._agents[agent_id] = agent
                ctx._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg="[Router] Created Insight agent",
                )
            return self._agents[agent_id]

    def get_or_create_planner_agent(self):
        """Get the cached planner agent or create a new one. Thread-safe."""
        agent_id = "PlannerAgent"
        ctx = self._ctx
        with self._lock:
            if agent_id not in self._agents:
                from .planner import PlannerAgent
                agent = PlannerAgent(
                    service=ctx.service,
                    tool_executor=lambda name, args, tc_id=None: (
                        ctx._execute_tool_for_agent(
                            name, args, tc_id, agent_type="planner"
                        )
                    ),
                    event_bus=ctx._event_bus,
                    cancel_event=ctx._cancel_event,
                    memory_store=ctx._memory_store,
                    memory_scope="planner",
                    session_id=ctx._session_id,
                )
                agent._orchestrator_inbox = ctx._inbox
                agent.start()
                self._agents[agent_id] = agent
                ctx._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[Router] Created PlannerAgent ({config.PLANNER_MODEL})",
                )
            return self._agents[agent_id]

    # ------------------------------------------------------------------
    # Request builders
    # ------------------------------------------------------------------

    def build_envoy_request(self, mission_id: str, request: str, agent=None) -> str:
        """Build a full request string for an envoy delegation."""
        ctx = self._ctx
        agent_id = agent.agent_id if agent else f"EnvoyAgent[{mission_id}]"
        store = ctx._store
        entries = store.list_entries()
        if entries:
            new_entries, removed_labels, store_hash = ctx._ctx_tracker.get_store_delta(
                agent_id, entries
            )
            if new_entries or removed_labels:
                labels = [
                    f"  - {e['label']} ({e['num_points']} pts, {e['time_min']} to {e['time_max']})"
                    for e in new_entries
                ]
                if new_entries and not removed_labels:
                    request += (
                        "\n\nNew data added to memory:\n"
                        + "\n".join(labels)
                        + "\nDo NOT re-fetch data that is already in memory with a matching label and time range."
                    )
                elif removed_labels and not new_entries:
                    request += (
                        "\n\nData removed from memory: "
                        + ", ".join(removed_labels)
                        + "\nDo NOT re-fetch data that is already in memory with a matching label and time range."
                    )
                else:
                    request += "\n\nData store updated:\n" + "\n".join(labels)
                    if removed_labels:
                        request += "\nRemoved: " + ", ".join(removed_labels)
                    request += "\nDo NOT re-fetch data that is already in memory with a matching label and time range."
                ctx._ctx_tracker.record(
                    agent_id, store_entries=entries, store_hash=store_hash
                )
            # else: store unchanged — skip injection (agent already has it)
        if not (agent and agent._interaction_id):
            request += "\n\n[Tip: Call events(action='check') to see what happened earlier in this session.]"
        return request

    def build_dataops_request(self, request: str, context: str, agent=None) -> str:
        """Build a full request string for a DataOps delegation."""
        ctx = self._ctx
        _agent_id = agent.agent_id if agent else "DataOpsAgent"
        full_request = f"{request}\n\nContext: {context}" if context else request
        store = ctx._store
        entries = store.list_entries()
        if entries:
            new_entries, removed_labels, store_hash = ctx._ctx_tracker.get_store_delta(
                _agent_id, entries
            )
            if new_entries or removed_labels:
                store_text = json.dumps(new_entries, indent=2, default=str)
                if new_entries and not removed_labels:
                    full_request += (
                        "\n\nNew data added to memory:\n```json\n"
                        + store_text
                        + "\n```"
                    )
                elif removed_labels and not new_entries:
                    full_request += "\n\nData removed from memory: " + ", ".join(
                        removed_labels
                    )
                else:
                    full_request += (
                        "\n\nData store updated:\n```json\n" + store_text + "\n```"
                    )
                    if removed_labels:
                        full_request += "\nRemoved: " + ", ".join(removed_labels)
                ctx._ctx_tracker.record(
                    _agent_id, store_entries=entries, store_hash=store_hash
                )
            # else: store unchanged — skip injection
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
        agent: SubAgent,
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
            tool_call_id: LLM tool_call_id for result mapping (unused now but
                kept for API compatibility with callers).
            agent_type: Agent type for ControlCenter tracking.
            agent_name: Agent name for ControlCenter tracking.
            task_summary: Human-readable summary for ControlCenter.
            post_process: Optional callable(result) -> result to run after
                delegation completes.
            post_complete: Optional callable(result) to run after processing.
                For non-critical work (e.g. PNG export).
        """
        ctx = self._ctx
        cc = ctx._control_center
        summary = task_summary or (
            request[:200] if isinstance(request, str) else str(request)[:200]
        )
        # Capture the full request for observability
        request_str = request if isinstance(request, str) else str(request)
        unit = cc.register(
            kind="delegation",
            agent_type=agent_type,
            agent_name=agent_name or agent.agent_id,
            task_summary=summary,
            request=request_str,
            tool_call_id=tool_call_id,
        )

        # Capture operation log index before the delegation starts so we can
        # collect only the operations produced by this delegation.
        ops_log = ctx._ops_log
        ops_start_index = len(ops_log.get_records())

        # Handle fire-and-forget delegation — start agent but don't wait
        if not wait:
            agent.send(request, sender="orchestrator", timeout=timeout, wait=False)

            # Track as async delegation for completion notification
            ctx._async_delegations[agent_name or agent.agent_id] = time.time()

            cc.mark_completed(
                unit.id,
                {
                    "status": "queued",
                    "message": f"Delegation to {agent_name or agent.agent_id} started (fire-and-forget)",
                },
            )
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
            if post_process is not None:
                wrapped = post_process(wrapped)

            # Build operation log from records added during this delegation
            all_records = ops_log.get_records()
            operation_log = all_records[ops_start_index:]

            cc.mark_completed(unit.id, wrapped, operation_log=operation_log)

            # Fire-and-forget callback after marking complete
            if post_complete is not None:
                try:
                    post_complete(wrapped)
                except Exception as pc_err:
                    get_logger().debug(f"post_complete callback failed: {pc_err}")

            return wrapped
        except Exception as e:
            error_msg = str(e)
            cc.mark_failed(unit.id, error_msg)
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
