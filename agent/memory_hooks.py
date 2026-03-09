"""Memory hooks subsystem — manages memory extraction, hot reload, and pipeline curation.

Extracted from OrchestratorAgent to reduce the size of core.py.
The ``MemoryHooks`` class owns the MemoryAgent lifecycle, periodic
extraction daemon, pipeline candidate enumeration, and memory hot
reload logic.  It receives the orchestrator as ``ctx``.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

import config
from .event_bus import (
    DEBUG,
    MEMORY_EXTRACTION_START,
    MEMORY_EXTRACTION_DONE,
    MEMORY_EXTRACTION_ERROR,
    set_event_bus,
)
from .logging import get_logger
from .memory_agent import MemoryAgent, MemoryContext
from .prompts import get_system_prompt

if TYPE_CHECKING:
    from .core import OrchestratorAgent
    from .event_bus import SessionEvent

logger = get_logger()


def candidates_from_log(ops_log) -> list[dict]:
    """Extract fresh pipeline candidates from a single OperationsLog.

    Returns rich per-step detail for each candidate so the LLM can make
    informed registration decisions.  Only render ops with
    ``pipeline_status`` == ``"fresh"`` (or absent) are included.
    """
    if ops_log is None:
        return []

    from data_ops.pipeline import is_vanilla
    from knowledge.mission_prefixes import (
        match_dataset_to_mission,
        get_canonical_id,
    )

    records = ops_log.get_records()

    render_ops = [
        r
        for r in records
        if r["tool"] == "render_plotly_json"
        and r["status"] == "success"
        and r.get("pipeline_status", "fresh") == "fresh"
    ]

    if not render_ops:
        return []

    candidates = []
    all_labels = {
        l for r in records if r["status"] == "success" for l in r.get("outputs", [])
    }

    for render in render_ops:
        render_id = render["id"]
        sub_dag = ops_log.get_state_pipeline(render_id, all_labels)

        # Build step-like dicts for is_vanilla check
        step_dicts = [
            {"tool": op["tool"], "params": op.get("args", {})} for op in sub_dag
        ]

        # Auto-extract scopes from dataset IDs
        missions_set: set[str] = set()
        steps = []
        for op in sub_dag:
            tool = op["tool"]
            args = op.get("args", {})
            output_label = op.get("outputs", [""])[0] if op.get("outputs") else ""
            step: dict = {"tool": tool}

            if tool == "fetch_data":
                ds = args.get("dataset_id", "")
                param = args.get("parameter_id", "")
                step["dataset_id"] = ds
                step["parameter_id"] = param
                if output_label:
                    step["output_label"] = output_label
                if ds:
                    stem, _ = match_dataset_to_mission(ds)
                    if stem:
                        missions_set.add(get_canonical_id(stem))

            elif tool in ("run_code",):
                if args.get("code"):
                    step["code"] = args["code"]
                if args.get("description"):
                    step["description"] = args["description"]
                if args.get("units"):
                    step["units"] = args["units"]
                if output_label:
                    step["output_label"] = output_label

            elif tool == "render_plotly_json":
                step["inputs"] = list(op.get("inputs", []))
                # figure_json intentionally omitted — too large

            elif tool == "manage_plot":
                step["action"] = args.get("action", "")
                for k in ("plot_id", "title", "subplot"):
                    if args.get(k):
                        step[k] = args[k]

            else:
                if output_label:
                    step["output_label"] = output_label

            steps.append(step)

        scopes = sorted(f"mission:{m}" for m in missions_set)

        candidates.append(
            {
                "render_op_id": render_id,
                "step_count": len(sub_dag),
                "is_vanilla": is_vanilla(step_dicts),
                "scopes": scopes,
                "steps": steps,
            }
        )

    return candidates


class MemoryHooks:
    """Manages memory extraction, hot reload, and pipeline curation.

    Args:
        ctx: The OrchestratorAgent instance (used to access service,
             event_bus, store, memory_store, etc.).
    """

    def __init__(self, ctx: "OrchestratorAgent"):
        self._ctx = ctx
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
        memory_section = self._ctx._memory_store.format_for_injection(
            scope="generic", include_review_instruction=False
        )
        base_prompt = get_system_prompt()
        if memory_section:
            self._ctx._system_prompt = f"{base_prompt}\n\n{memory_section}"
        else:
            self._ctx._system_prompt = base_prompt
        if self._ctx.chat is not None:
            self._ctx.chat.update_system_prompt(self._ctx._system_prompt)

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    def trigger_hot_reload(self) -> None:
        """Hot reload: restart chat session with fresh LTM injection.

        Called every N rounds to refresh the LLM context with latest memories.
        """
        ctx = self._ctx
        if not ctx.chat:
            logger.debug("[Memory] No chat session to reload")
            return

        try:
            # Get canonical interface (works across all providers)
            interface = ctx.chat.interface
            logger.info(
                f"[Memory] Hot reload: preserving {len(interface.entries)} interface entries"
            )

            # Get fresh memory section
            memory_section = ctx._memory_store.format_for_injection(
                scope="generic", include_review_instruction=False
            )

            # Build new system prompt with fresh memory
            base_prompt = get_system_prompt()
            if memory_section:
                new_system_prompt = f"{base_prompt}\n\n{memory_section}"
            else:
                new_system_prompt = base_prompt

            # Update system prompt in interface before creating new session
            interface.add_system(new_system_prompt)

            # Create new chat session with canonical interface
            ctx.chat = ctx.service.create_session(
                system_prompt=new_system_prompt,
                tools=ctx._tool_schemas,
                model=ctx.model_name,
                thinking="high",
                tracked=False,
                interface=interface,
            )

            ctx._event_bus.emit(
                DEBUG,
                level="info",
                msg=f"[Memory] Hot reload complete: {len(interface.entries)} entries, memory injected",
            )
        except Exception as e:
            logger.error(f"[Memory] Hot reload failed: {e}")
            ctx._event_bus.emit(
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
        with ctx._sub_agents_lock:
            if (
                "VizAgent[Plotly]" in ctx._sub_agents
                or "VizAgent[Mpl]" in ctx._sub_agents
            ):
                active_scopes.append("visualization")
            if "DataOpsAgent" in ctx._sub_agents:
                active_scopes.append("data_ops")
            for key in ctx._sub_agents:
                if key.startswith("EnvoyAgent["):
                    mission_id = key.removeprefix("EnvoyAgent[").rstrip("]")
                    active_scopes.append(f"envoy:{mission_id}")

        # Collect all console-tagged events (same log the user sees)
        console_events = ctx._event_bus.get_events(tags={"console"})

        # MemoryAgent reads ALL memories directly from the store in its
        # system prompt — no need to build active_memories here.
        return MemoryContext(
            console_events=console_events,
            active_scopes=active_scopes,
            total_memory_tokens=ctx._memory_store.total_tokens(),
        )

    # ------------------------------------------------------------------
    # Pipeline candidate enumeration
    # ------------------------------------------------------------------

    def enumerate_pipeline_candidates(self) -> list[dict]:
        """Identify ALL fresh (unprocessed) pipelines across all sessions.

        Scans the current in-memory log and all past sessions'
        ``operations.json`` files for render ops with ``pipeline_status``
        == ``"fresh"`` (or absent).  The MemoryAgent sees every fresh
        pipeline and decides each one — register or discard.
        """
        from data_ops.operations_log import OperationsLog

        ctx = self._ctx
        candidates = []

        # -- Current session --
        ops_log = ctx._ops_log
        candidates.extend(candidates_from_log(ops_log))

        # -- Past sessions --
        sessions_dir = config.get_data_dir() / "sessions"
        current_sid = ctx._session_id or ""
        if sessions_dir.exists():
            for sdir in sorted(
                sessions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                if not sdir.is_dir():
                    continue
                if sdir.name == current_sid:
                    continue  # already scanned above
                ops_file = sdir / "operations.json"
                if not ops_file.exists():
                    continue
                try:
                    past_log = OperationsLog(session_id=sdir.name)
                    past_log.load_from_file(ops_file)
                    past_candidates = candidates_from_log(past_log)
                    if past_candidates:
                        candidates.extend(past_candidates)
                except Exception:
                    continue  # skip corrupt files

        return candidates

    # ------------------------------------------------------------------
    # MemoryAgent lifecycle
    # ------------------------------------------------------------------

    def ensure_agent(self, session_id: str = "", bus=None) -> MemoryAgent:
        """Lazily create or return the existing MemoryAgent."""
        ctx = self._ctx
        if bus is None:
            bus = ctx._event_bus
        if session_id == "":
            session_id = ctx._session_id or ""
        if self._agent is None:
            self._agent = MemoryAgent(
                service=ctx.service,
                memory_store=ctx._memory_store,
                pipeline_store=ctx._pipeline_store,
                verbose=ctx.verbose,
                session_id=session_id,
                event_bus=bus,
            )
            self._agent.start()
            with ctx._sub_agents_lock:
                ctx._sub_agents["MemoryAgent"] = self._agent
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

        self._ctx._event_bus.emit(
            MEMORY_EXTRACTION_START,
            agent="Memory",
            level="info",
            msg="[Memory] Pipeline curation started",
            data={"pipeline_candidates": len(context.pipeline_candidates)},
        )

        try:
            executed = agent.run(context)
            # Persist ops log so pipeline_status changes are saved to disk
            self.persist_operations_log()
            pipeline_actions = [
                a
                for a in (executed or [])
                if a.get("action") in ("register_pipeline", "discard_pipeline")
            ]
            return pipeline_actions
        except Exception as e:
            self._ctx._event_bus.emit(
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
        """Save the current operations log to the session directory on disk."""
        ctx = self._ctx
        if not ctx._session_id:
            return
        try:
            session_dir = config.get_data_dir() / "sessions" / ctx._session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            ctx._ops_log.save_to_file(session_dir / "operations.json")
        except Exception:
            pass  # Best-effort persistence

    # ------------------------------------------------------------------
    # Async memory extraction (daemon thread)
    # ------------------------------------------------------------------

    def maybe_extract(self) -> None:
        """Trigger async memory extraction with full session context.

        Runs on a daemon thread using SMART_MODEL.
        Lock prevents concurrent extractions. The MemoryAgent sees
        memory-tagged events from the EventBus, curated into concise summaries.
        Also includes pipeline candidates for the LLM to curate.
        """
        ctx = self._ctx
        # Check if there are new console events since last extraction
        console_events = ctx._event_bus.get_events(
            tags={"console"}, since_index=self._last_op_index
        )
        if not console_events:
            return  # No new events since last extraction

        if not self._lock.acquire(blocking=False):
            return  # Another extraction already running

        try:
            context = self.build_context()
            context.pipeline_candidates = self.enumerate_pipeline_candidates()

            self._last_op_index = len(ctx._event_bus._events)

            session_id = ctx._session_id or ""
            bus = ctx._event_bus  # capture before thread (ContextVar won't propagate)

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
                    executed = agent.run(context)

                    # Persist ops log so pipeline_status changes are saved to disk
                    self.persist_operations_log()

                    # Tally actions by type
                    counts = {}
                    for action in executed or []:
                        atype = action.get("action", "unknown")
                        counts[atype] = counts.get(atype, 0) + 1

                    bus.emit(
                        MEMORY_EXTRACTION_DONE,
                        agent="Memory",
                        level="info",
                        msg=f"[Memory] Extraction complete: {counts}"
                        if counts
                        else "[Memory] Extraction complete: no changes",
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
