"""Memory hooks subsystem — manages memory extraction, hot reload, and pipeline curation.

The ``MemoryHooks`` class owns the MemoryAgent lifecycle, periodic
extraction daemon, pipeline candidate enumeration, and memory hot
reload logic.  It receives a ``SessionContext`` as ``ctx``.
"""

from __future__ import annotations

import json
import threading
from typing import Callable, TYPE_CHECKING

import config
from .event_bus import (
    DEBUG,
    MEMORY_EXTRACTION_START,
    MEMORY_EXTRACTION_DONE,
    MEMORY_EXTRACTION_ERROR,
    set_event_bus,
)
from .logging import get_logger
from dataclasses import dataclass, field
from .memory_agent import MemoryAgent
from .prompts import get_system_prompt

if TYPE_CHECKING:
    from .session_context import SessionContext
    from .event_bus import SessionEvent

logger = get_logger()


@dataclass
class MemoryContext:
    """Context passed to the MemoryAgent for extraction."""
    console_events: list = field(default_factory=list)
    active_scopes: list[str] = field(default_factory=list)
    total_memory_tokens: int = 0
    pipeline_candidates: list[dict] = field(default_factory=list)


def format_context_message(context: MemoryContext) -> str:
    """Serialize a MemoryContext into a string for the MemoryAgent LLM.

    The message gives the agent all session context it needs to decide
    whether to add/edit/drop memories.
    """
    parts = []
    parts.append(f"Active scopes: {', '.join(context.active_scopes)}")
    parts.append(f"Total memory tokens: {context.total_memory_tokens}")

    if context.console_events:
        parts.append(f"\n## Session Events ({len(context.console_events)} total)")
        for i, ev in enumerate(context.console_events):
            agent_tag = f"[{ev.agent}] " if ev.agent else ""
            summary = ev.summary or ev.msg or "(no summary)"
            parts.append(f"  {i+1}. {agent_tag}{summary}")
    else:
        parts.append("\nNo session events to review.")

    if context.pipeline_candidates:
        parts.append(f"\n## Pipeline Candidates ({len(context.pipeline_candidates)})")
        for pc in context.pipeline_candidates:
            parts.append(f"  - {json.dumps(pc, default=str)}")

    parts.append(
        "\nReview the session events above. If you find reusable knowledge "
        "(user preferences, workflow patterns, lessons learned, session summaries), "
        "use the manage_memory tool to add/edit/drop memories. "
        "If the session is genuinely trivial with nothing novel, you may skip."
    )

    return "\n".join(parts)


class MemoryHooks:
    """Manages memory extraction, hot reload, and pipeline curation.

    Args:
        ctx: The SessionContext instance (shared resource pool).
        prompt_injector: Optional callback ``(new_system_prompt) -> None``
            to update the orchestrator's system prompt when memory changes.
        session_restarter: Optional callback ``() -> None``
            to restart the orchestrator's chat session for hot reload.
    """

    def __init__(
        self,
        ctx: "SessionContext",
        *,
        prompt_injector: "Callable[[str], None] | None" = None,
        session_restarter: "Callable[[], None] | None" = None,
    ):
        self._ctx = ctx
        self._prompt_injector = prompt_injector
        self._session_restarter = session_restarter
        self._agent: MemoryAgent | None = None
        self._lock = threading.Lock()
        self._turn_counter: int = 0
        self._last_op_index: int = 0

    # ------------------------------------------------------------------
    # Event listener
    # ------------------------------------------------------------------

    def on_memory_mutated(self, event: "SessionEvent") -> None:
        """Event bus listener: refresh core memory when long-term memory changes."""
        if event.type != MEMORY_EXTRACTION_DONE:
            return
        memory_section = self._ctx.memory_store.format_for_injection(
            scope="generic", include_review_instruction=False
        )
        base_prompt = get_system_prompt()
        if memory_section:
            new_prompt = f"{base_prompt}\n\n{memory_section}"
        else:
            new_prompt = base_prompt
        if self._prompt_injector is not None:
            self._prompt_injector(new_prompt)

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    def trigger_hot_reload(self) -> None:
        """Hot reload: restart chat session with fresh LTM injection.

        Called every N rounds to refresh the LLM context with latest memories.
        Uses the ``session_restarter`` callback provided by the orchestrator.
        """
        if self._session_restarter is None:
            logger.debug("[Memory] No session_restarter callback — skipping hot reload")
            return

        try:
            self._session_restarter()
            self._ctx.event_bus.emit(
                DEBUG,
                level="info",
                msg="[Memory] Hot reload complete via session_restarter callback",
            )
        except Exception as e:
            logger.error(f"[Memory] Hot reload failed: {e}")
            self._ctx.event_bus.emit(
                DEBUG,
                level="error",
                msg=f"[Memory] Hot reload failed: {e}",
            )

    # ------------------------------------------------------------------
    # Memory context building
    # ------------------------------------------------------------------

    def build_context(self) -> MemoryContext:
        """Build a MemoryContext from the current session state.

        Shared by maybe_extract() (periodic) and
        run_for_pipelines() (on-demand).
        """
        ctx = self._ctx
        # Detect active scopes from actors
        active_scopes = ["generic"]
        delegation = ctx.delegation
        if delegation is not None:
            from .delegation import AGENT_ID_DATAOPS
            with delegation.lock:
                agents = delegation.agents
                if (
                    "VizAgent[Plotly]" in agents
                    or "VizAgent[Mpl]" in agents
                ):
                    active_scopes.append("visualization")
                if AGENT_ID_DATAOPS in agents:
                    active_scopes.append("data_ops")
                for key in agents:
                    if key.startswith("EnvoyAgent["):
                        mission_id = key.removeprefix("EnvoyAgent[").rstrip("]")
                        active_scopes.append(f"envoy:{mission_id}")

        # Collect all console-tagged events (same log the user sees)
        console_events = ctx.event_bus.get_events(tags={"console"})

        # MemoryAgent reads ALL memories directly from the store in its
        # system prompt — no need to build active_memories here.
        return MemoryContext(
            console_events=console_events,
            active_scopes=active_scopes,
            total_memory_tokens=ctx.memory_store.total_tokens(),
        )

    # ------------------------------------------------------------------
    # Pipeline candidate enumeration
    # ------------------------------------------------------------------

    def enumerate_pipeline_candidates(self) -> list[dict]:
        """Identify pipeline candidates from the session's PipelineDAG.

        Each leaf (terminal output — usually a render) produces one
        candidate representing the full pipeline that produced it.
        """
        dag = self._ctx.dag
        if dag is None or dag.node_count() == 0:
            return []

        candidates = []
        for leaf_id in dag.leaves():
            sub = dag.subgraph(leaf_id)
            nodes = sub.topological_order()
            candidate = {
                "session_id": getattr(self._ctx, "session_id", ""),
                "root_node": leaf_id,
                "node_count": len(nodes),
                "sources": [sub.node(n) for n in sub.roots()],
                "transforms": [
                    sub.node(n) for n in nodes
                    if sub.node_kind(n) == "transform"
                ],
                "sink": sub.node(leaf_id),
                "labels": [
                    label for n in nodes
                    for label in sub.node(n).get("outputs", {}).keys()
                ],
            }
            candidates.append(candidate)
        return candidates

    # ------------------------------------------------------------------
    # MemoryAgent lifecycle
    # ------------------------------------------------------------------

    def ensure_agent(self, session_id: str = "", bus=None) -> MemoryAgent:
        """Lazily create or return the existing MemoryAgent."""
        ctx = self._ctx
        if bus is None:
            bus = ctx.event_bus
        if session_id == "":
            session_id = ctx.session_id or ""
        if self._agent is None:
            self._agent = MemoryAgent(
                service=ctx.service,
                session_ctx=ctx,
                event_bus=bus,
                memory_store=ctx.memory_store,
            )
            self._agent.start()
            delegation = ctx.delegation
            if delegation is not None:
                from .delegation import AGENT_ID_MEMORY
                delegation.register_agent(AGENT_ID_MEMORY, self._agent)
        return self._agent

    # ------------------------------------------------------------------
    # Pipeline curation (synchronous)
    # ------------------------------------------------------------------

    def run_for_pipelines(self) -> list[dict]:
        """Force a Memory Agent run focused on pipeline curation.

        Builds context with pipeline candidates from the current session,
        runs the Memory Agent synchronously, and returns pipeline actions.
        """
        context = self.build_context()
        context.pipeline_candidates = self.enumerate_pipeline_candidates()

        if not context.pipeline_candidates:
            return []  # Nothing to curate

        agent = self.ensure_agent()

        self._ctx.event_bus.emit(
            MEMORY_EXTRACTION_START,
            agent="Memory",
            level="info",
            msg="[Memory] Pipeline curation started",
            data={"pipeline_candidates": len(context.pipeline_candidates)},
        )

        try:
            msg_content = format_context_message(context)
            result = agent.send(msg_content, sender="orchestrator", timeout=120.0)
            # Persist ops log so pipeline_status changes are saved to disk
            self.persist_operations_log()
            return []  # Pipeline actions tracked by store, not return value
        except Exception as e:
            self._ctx.event_bus.emit(
                MEMORY_EXTRACTION_ERROR,
                agent="Memory",
                level="warning",
                msg=f"[Memory] Pipeline curation failed: {e}",
            )
            return []

    # ------------------------------------------------------------------
    # Operations log persistence
    # ------------------------------------------------------------------

    def persist_operations_log(self) -> None:
        """Save the pipeline DAG to the session directory on disk."""
        ctx = self._ctx
        if not ctx.session_id:
            return
        try:
            ctx.dag.save()
        except Exception:
            pass  # Best-effort persistence

    # ------------------------------------------------------------------
    # Async memory extraction (daemon thread)
    # ------------------------------------------------------------------

    def maybe_extract(self) -> None:
        """Trigger async memory extraction with full session context.

        Runs on a daemon thread. Lock prevents concurrent extractions.
        The MemoryAgent sees memory-tagged events from the EventBus,
        curated into concise summaries. Also includes pipeline candidates
        for the LLM to curate.
        """
        ctx = self._ctx
        # Check if there are new console events since last extraction
        console_events = ctx.event_bus.get_events(
            tags={"console"}, since_index=self._last_op_index
        )
        if not console_events:
            return  # No new events since last extraction

        if not self._lock.acquire(blocking=False):
            return  # Another extraction already running

        try:
            context = self.build_context()
            context.pipeline_candidates = self.enumerate_pipeline_candidates()

            self._last_op_index = ctx.event_bus.event_count()

            session_id = ctx.session_id or ""
            bus = ctx.event_bus  # capture before thread (ContextVar won't propagate)

            def _run():
                set_event_bus(bus)  # propagate session bus to daemon thread
                try:
                    bus.emit(
                        MEMORY_EXTRACTION_START,
                        agent="Memory",
                        level="info",
                        msg="[Memory] Extraction started",
                        data={
                            "console_events": len(context.console_events),
                            "active_scopes": context.active_scopes,
                        },
                    )

                    # Dump memory feed for debugging
                    if session_id:
                        try:
                            from datetime import datetime as _dt, timezone as _tz

                            feed_dir = config.get_data_dir() / "sessions" / session_id
                            feed_dir.mkdir(parents=True, exist_ok=True)
                            feed_payload = {
                                "timestamp": _dt.now(_tz.utc).isoformat(),
                                "active_scopes": context.active_scopes,
                                "console_events_count": len(context.console_events),
                                "console_events": [
                                    {
                                        "index": i,
                                        "type": ev.type,
                                        "agent": ev.agent,
                                        "summary": ev.summary,
                                    }
                                    for i, ev in enumerate(context.console_events)
                                ],
                                "total_memory_tokens": context.total_memory_tokens,
                                "pipeline_candidates_count": len(
                                    context.pipeline_candidates
                                ),
                            }
                            (feed_dir / "memory_feed.json").write_text(
                                json.dumps(feed_payload, indent=2, default=str)
                            )
                        except Exception:
                            pass  # Debug dump — never break extraction

                    agent = self.ensure_agent(session_id=session_id, bus=bus)
                    agent.reset_action_counts()
                    msg_content = format_context_message(context)
                    result = agent.send(
                        msg_content, sender="orchestrator", timeout=120.0
                    )

                    # Persist ops log so pipeline_status changes are saved to disk
                    self.persist_operations_log()

                    if result and result.get("failed"):
                        bus.emit(
                            MEMORY_EXTRACTION_ERROR,
                            agent="Memory",
                            level="warning",
                            msg=f"[Memory] Extraction failed: {result.get('text', 'unknown')}",
                        )
                    else:
                        counts = agent.get_action_counts()
                        bus.emit(
                            MEMORY_EXTRACTION_DONE,
                            agent="Memory",
                            level="info",
                            msg="[Memory] Extraction complete",
                            data={"actions": counts},
                        )
                except Exception as e:
                    bus.emit(
                        MEMORY_EXTRACTION_ERROR,
                        agent="Memory",
                        level="warning",
                        msg=f"[Memory] Extraction failed: {e}",
                    )
                finally:
                    self._lock.release()

            t = threading.Thread(target=_run, daemon=True)
            t.start()
        except Exception:
            self._lock.release()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset memory hooks state for a new session.

        Does NOT clear the memory store — only resets the agent and counters.
        """
        self._agent = None
        self._turn_counter = 0
        self._last_op_index = 0
