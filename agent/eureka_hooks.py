"""Eureka/Insight hooks — extracted from OrchestratorAgent.

The ``EurekaHooks`` class owns eureka discovery state (agent, lock,
counters, pending suggestions) and insight review state (iteration
counter, latest render PNG).  It receives the orchestrator as ``ctx``
for access to shared services (store, renderer, event bus, etc.).
"""

from __future__ import annotations

import json as _json
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import config
from .logging import get_logger
from .event_bus import (
    DEBUG,
    INSIGHT_FEEDBACK,
    USER_MESSAGE,
)

if TYPE_CHECKING:
    from .core import OrchestratorAgent


class EurekaHooks:
    """Manages eureka discovery and insight auto-review lifecycle.

    Args:
        ctx: The OrchestratorAgent instance (used to access service,
             event_bus, store, renderer, etc.).
        eureka_mode: Whether eureka mode is initially enabled.
    """

    def __init__(self, ctx: "OrchestratorAgent", eureka_mode: bool = False):
        self._ctx = ctx
        self.logger = get_logger()

        # Eureka discovery state
        self._agent = None
        self._lock = threading.Lock()
        self._mode: bool = eureka_mode
        self._round_counter: int = 0
        self._turn_counter: int = 0
        self._pending_suggestion = None

        # Insight review state
        self._insight_review_iter: int = 0
        self._latest_render_png: bytes | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def eureka_mode(self) -> bool:
        return self._mode

    @eureka_mode.setter
    def eureka_mode(self, value: bool) -> None:
        self._mode = value

    @property
    def eureka_round_counter(self) -> int:
        return self._round_counter

    @eureka_round_counter.setter
    def eureka_round_counter(self, value: int) -> None:
        self._round_counter = value

    @property
    def eureka_turn_counter(self) -> int:
        return self._turn_counter

    @eureka_turn_counter.setter
    def eureka_turn_counter(self, value: int) -> None:
        self._turn_counter = value

    @property
    def eureka_pending_suggestion(self):
        return self._pending_suggestion

    @eureka_pending_suggestion.setter
    def eureka_pending_suggestion(self, value) -> None:
        self._pending_suggestion = value

    @property
    def eureka_agent(self):
        return self._agent

    @eureka_agent.setter
    def eureka_agent(self, value) -> None:
        self._agent = value

    @property
    def insight_review_iter(self) -> int:
        return self._insight_review_iter

    @insight_review_iter.setter
    def insight_review_iter(self, value: int) -> None:
        self._insight_review_iter = value

    @property
    def latest_render_png(self) -> bytes | None:
        return self._latest_render_png

    @latest_render_png.setter
    def latest_render_png(self, value: bytes | None) -> None:
        self._latest_render_png = value

    # ------------------------------------------------------------------
    # Insight review
    # ------------------------------------------------------------------

    def build_insight_context(self) -> str:
        """Build data context string for the InsightAgent.

        Gathers renderer state (trace labels, panel count) and store entries
        (label, num_points, time range, units, columns) for the context.
        """
        lines = []

        # Renderer state
        state = self._ctx._renderer.get_current_state()
        if state.get("traces"):
            lines.append(f"Traces on plot: {state['traces']}")
        if state.get("num_panels"):
            lines.append(f"Number of panels: {state['num_panels']}")
        if not state.get("traces") and self._latest_render_png is not None:
            lines.append("Visualization: matplotlib (static PNG)")

        # Store entries
        entries = self._ctx._store.list_entries()
        if entries:
            lines.append("\nData in memory:")
            for e in entries:
                parts = [f"  - {e['label']}"]
                if e.get("num_points"):
                    parts.append(f"{e['num_points']} pts")
                if e.get("units"):
                    parts.append(f"units={e['units']}")
                if e.get("time_min") and e.get("time_max"):
                    parts.append(f"range={e['time_min']} to {e['time_max']}")
                if e.get("columns"):
                    cols = e["columns"]
                    if len(cols) <= 5:
                        parts.append(f"columns={cols}")
                    else:
                        parts.append(
                            f"columns=[{cols[0]}, ..., {cols[-1]}] ({len(cols)} cols)"
                        )
                lines.append(", ".join(parts))

        return "\n".join(lines) if lines else "No data context available."

    def sync_insight_review(self) -> dict | None:
        """Synchronous InsightAgent figure review after a successful render.

        Exports the current figure to PNG, gathers context, and dispatches
        a synchronous review via agent.send(). Blocks until the
        review completes, then returns the result.

        Returns:
            The review result dict, or None if the review was skipped
            (disabled, iteration cap, no figure, etc.).
        """
        if not config.INSIGHT_FEEDBACK:
            return None

        if self._insight_review_iter >= config.INSIGHT_FEEDBACK_MAX_ITERS:
            return None

        image_bytes = self._ctx.get_latest_figure_png()
        if image_bytes is None:
            return None

        agent = self._ctx._get_or_create_insight_agent()
        data_context = self.build_insight_context()

        user_msgs = self._ctx._event_bus.get_events(types={USER_MESSAGE})
        user_request = (
            user_msgs[-1].data.get("text", user_msgs[-1].msg) if user_msgs else ""
        )

        self._insight_review_iter += 1

        review_instruction = (
            f'Review this figure for correctness and quality against the user\'s original request: "{user_request}"\n\n'
            "Check: (1) Does it show the requested datasets/parameters and time range? "
            "(2) Are axis labels, units, title, and legend correct? "
            "(3) Are traces distinguishable and scales appropriate? "
            "(4) Do value ranges look physically reasonable?\n\n"
            "Start your response with VERDICT: PASS or VERDICT: NEEDS_IMPROVEMENT, "
            "then explain. If NEEDS_IMPROVEMENT, list specific actionable suggestions as bullet points (max 5)."
        )
        result = agent.send(
            {
                "action": "review",
                "image_bytes": image_bytes,
                "data_context": data_context,
                "user_request": review_instruction,
            },
            sender="orchestrator",
            timeout=180,
        )

        passed = result.get("passed", True)
        review_text = result.get("text", "")
        verdict = "PASS" if passed else "NEEDS_IMPROVEMENT"
        self._ctx._event_bus.emit(
            INSIGHT_FEEDBACK,
            agent="InsightAgent",
            msg=f"[Figure Auto-Review — {verdict}]\n{review_text}",
            data={"verdict": verdict, "text": review_text},
        )
        return result

    # ------------------------------------------------------------------
    # Eureka discovery
    # ------------------------------------------------------------------

    def ensure_eureka_agent(self):
        """Lazily create or return the existing EurekaAgent (SubAgent)."""
        if self._agent is None:
            from .eureka_agent import EurekaAgent

            self._agent = EurekaAgent(
                service=self._ctx.service,
                tool_executor=lambda name, args, tc_id=None: (
                    self._ctx._execute_tool_for_agent(name, args, tc_id, agent_type="eureka")
                ),
                event_bus=self._ctx._event_bus,
                memory_store=self._ctx._memory_store,
                memory_scope="eureka",
                orchestrator_ref=self._ctx,
            )
            self._agent.start()
            with self._ctx._sub_agents_lock:
                self._ctx._sub_agents["EurekaAgent"] = self._agent
        return self._agent

    def build_eureka_context(self) -> dict:
        """Build context dict for Eureka discovery."""
        user_msgs = self._ctx._event_bus.get_events(types={USER_MESSAGE})
        return {
            "session_id": self._ctx._session_id or "unknown",
            "data_store_keys": [e["label"] for e in self._ctx._store.list_entries()]
            if self._ctx._store
            else [],
            "has_figure": self._ctx._renderer.get_figure() is not None,
            "recent_messages": [m.msg for m in user_msgs[-5:]],
        }

    def format_eureka_suggestion_as_user_msg(self, suggestion) -> str:
        """Convert a EurekaSuggestion into a natural-language user message."""
        parts = [f"[Eureka Mode] {suggestion.description}"]
        if suggestion.rationale:
            parts.append(f"Rationale: {suggestion.rationale}")
        if suggestion.parameters:
            parts.append(f"Parameters: {_json.dumps(suggestion.parameters)}")
        return "\n".join(parts)

    def maybe_extract_eurekas(self) -> None:
        """Trigger async Eureka extraction on a daemon thread.

        Uses EurekaAgent.send() via the SubAgent inbox pattern.
        Lock prevents concurrent extractions. After the agent returns,
        if Eureka Mode is ON, queues the top suggestion for execution.
        """
        if not self._lock.acquire(blocking=False):
            return

        try:
            agent = self.ensure_eureka_agent()
            context = self.build_eureka_context()
        except Exception as e:
            self.logger.warning(f"Eureka setup failed: {e}")
            self._lock.release()
            return

        from .event_bus import (
            EUREKA_EXTRACTION_START,
            EUREKA_EXTRACTION_DONE,
            EUREKA_EXTRACTION_ERROR,
            set_event_bus,
        )

        bus = self._ctx._event_bus

        def _run():
            set_event_bus(bus)
            try:
                bus.emit(EUREKA_EXTRACTION_START, agent="Eureka", level="info")
                # Build the context message for the SubAgent
                msg_content = agent.build_context_message(context)
                result = agent.send(
                    msg_content, sender="orchestrator", timeout=120.0
                )

                if result.get("failed"):
                    bus.emit(
                        EUREKA_EXTRACTION_ERROR,
                        agent="Eureka",
                        level="warning",
                        data={"error": result.get("text", "unknown error")},
                    )
                else:
                    # Count findings from the store for this session
                    session_id = context.get("session_id", "unknown")
                    findings = agent.eureka_store.list(session_id=session_id)
                    suggestions = agent.eureka_store.list_suggestions(
                        session_id=session_id, status="proposed"
                    )
                    bus.emit(
                        EUREKA_EXTRACTION_DONE,
                        agent="Eureka",
                        level="info",
                        data={
                            "n_findings": len(findings),
                            "n_suggestions": len(suggestions),
                        },
                    )

                    # If Eureka Mode is ON, inject the top suggestion as
                    # a synthetic user message into the orchestrator inbox
                    if self._mode and suggestions:
                        eureka_max_rounds = config.get("eureka_max_rounds", 5)
                        if self._round_counter >= eureka_max_rounds:
                            self._mode = False
                            self._round_counter = 0
                            bus.emit(
                                DEBUG,
                                agent="Eureka",
                                level="info",
                                msg=f"[Eureka Mode] Paused: reached max rounds ({eureka_max_rounds})",
                            )
                        else:
                            suggestion = suggestions[0]
                            self._pending_suggestion = suggestion
                            synthetic_msg = self.format_eureka_suggestion_as_user_msg(
                                suggestion
                            )
                            self._round_counter += 1
                            # Mark suggestion as approved
                            from .eureka_store import EurekaStore

                            EurekaStore().update_suggestion_status(
                                suggestion.id, "executed"
                            )
                            # Inject into orchestrator inbox if running in turnless mode
                            ctx = self._ctx
                            if hasattr(ctx, "_inbox") and ctx._inbox is not None:
                                from .sub_agent import _make_message

                                ctx._put_message(
                                    _make_message(
                                        "user_input", "eureka_mode", synthetic_msg
                                    ),
                                    priority=0,
                                )
                            else:
                                # Fallback: store for process_message to pick up
                                self._pending_suggestion = suggestion

            except Exception as e:
                self.logger.warning(f"Eureka extraction failed: {e}")
                bus.emit(
                    EUREKA_EXTRACTION_ERROR,
                    agent="Eureka",
                    level="warning",
                    data={"error": str(e)},
                )
            finally:
                self._lock.release()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all eureka and insight state for a new session."""
        self._agent = None
        self._turn_counter = 0
        self._mode = config.get("eureka_mode", False)
        self._round_counter = 0
        self._pending_suggestion = None
        self._insight_review_iter = 0
        self._latest_render_png = None

    def reset_per_message(self) -> None:
        """Reset per-message state (called at the start of each user message)."""
        self._insight_review_iter = 0
        self._latest_render_png = None

    def reset_eureka_on_user_message(self, user_message: str) -> None:
        """Reset eureka round counter when a real (non-eureka) user message arrives."""
        if self._mode and not user_message.startswith("[Eureka Mode]"):
            self._round_counter = 0
            self._pending_suggestion = None
