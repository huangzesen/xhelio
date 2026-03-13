"""Eureka/Insight hooks — manages eureka discovery and insight auto-review.

The ``EurekaHooks`` class owns eureka discovery state (agent, lock,
counters, pending suggestions) and insight review state (iteration
counter, latest render PNG).  It receives a ``SessionContext`` for
access to shared services (store, renderer, event bus, etc.).
"""

from __future__ import annotations

import json as _json
import threading
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import config
from .logging import get_logger
from .event_bus import (
    DEBUG,
    INSIGHT_FEEDBACK,
    USER_MESSAGE,
)

if TYPE_CHECKING:
    from .session_context import SessionContext


def format_eureka_message(context: dict) -> str:
    """Serialize an eureka context dict into a string for the EurekaAgent LLM.

    The message gives the agent all session context it needs to decide
    whether to submit findings or suggestions.
    """
    parts = []
    parts.append(f"Session: {context.get('session_id', 'unknown')}")

    data_keys = context.get("data_store_keys", [])
    if data_keys:
        parts.append(f"\nData in memory: {', '.join(data_keys)}")
    else:
        parts.append("\nNo data in memory.")

    has_figure = context.get("has_figure", False)
    parts.append(f"Active figure: {'Yes' if has_figure else 'No'}")

    recent = context.get("recent_messages", [])
    if recent:
        parts.append(f"\nRecent user messages ({len(recent)}):")
        for msg in recent:
            parts.append(f"  - {msg}")

    parts.append(
        "\nAnalyze the session state above. If you find scientifically "
        "interesting patterns, anomalies, or correlations, use submit_finding. "
        "Then suggest concrete follow-up actions with submit_suggestion. "
        "If nothing noteworthy is found, simply respond without calling tools."
    )

    return "\n".join(parts)


class EurekaHooks:
    """Manages eureka discovery and insight auto-review lifecycle.

    Args:
        ctx: The SessionContext instance (shared session resources).
        eureka_mode: Whether eureka mode is initially enabled.
        inbox_injector: Callback to inject a synthetic user message into
            the orchestrator's inbox (for eureka mode auto-execution).
    """

    def __init__(
        self,
        ctx: "SessionContext",
        *,
        eureka_mode: bool = False,
        inbox_injector: Callable[[str], None] | None = None,
    ):
        self._ctx = ctx
        self._inbox_injector = inbox_injector
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
        sctx = self._ctx
        lines = []

        # Renderer state
        state = sctx.renderer.get_current_state()
        if state.get("traces"):
            lines.append(f"Traces on plot: {state['traces']}")
        if state.get("num_panels"):
            lines.append(f"Number of panels: {state['num_panels']}")
        if not state.get("traces") and self._latest_render_png is not None:
            lines.append("Visualization: matplotlib (static PNG)")

        # Store entries
        entries = sctx.store.list_entries()
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
        """Synchronous vision-based figure review after a successful render.

        Uses LLMService.generate_vision() directly — no InsightAgent needed.
        Exports the current figure to PNG, sends it with a review prompt,
        and returns the result.

        Returns:
            The review result dict, or None if the review was skipped
            (disabled, iteration cap, no figure, no vision provider, etc.).
        """
        if not config.INSIGHT_FEEDBACK:
            return None

        if self._insight_review_iter >= config.INSIGHT_FEEDBACK_MAX_ITERS:
            return None

        from agent.session_persistence import get_latest_figure_png
        sctx = self._ctx
        image_bytes = get_latest_figure_png(sctx)
        if image_bytes is None:
            return None

        user_msgs = sctx.event_bus.get_events(types={USER_MESSAGE})
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

        response = sctx.service.generate_vision(review_instruction, image_bytes)
        if not response.text:
            return {"failed": True, "text": "Vision review unavailable — no vision provider configured."}

        review_text = response.text
        passed = "VERDICT: PASS" in review_text.upper()[:50]

        verdict = "PASS" if passed else "NEEDS_IMPROVEMENT"
        sctx.event_bus.emit(
            INSIGHT_FEEDBACK,
            agent="vision_review",
            msg=f"[Figure Auto-Review — {verdict}]\n{review_text}",
            data={"verdict": verdict, "text": review_text},
        )
        return {"passed": passed, "text": review_text, "suggestions": []}

    # ------------------------------------------------------------------
    # Eureka discovery
    # ------------------------------------------------------------------

    def ensure_eureka_agent(self):
        """Lazily create or return the existing EurekaAgent."""
        if self._agent is None:
            from .eureka_agent import EurekaAgent

            self._agent = EurekaAgent(
                service=self._ctx.service,
                session_ctx=self._ctx,
                event_bus=self._ctx.event_bus,
                eureka_store=self._ctx.eureka_store,
                session_id=self._ctx.session_id or "",
            )
            self._agent.start()
            delegation = self._ctx.delegation
            if delegation is not None:
                from .delegation import AGENT_ID_EUREKA
                delegation.register_agent(AGENT_ID_EUREKA, self._agent)
        return self._agent

    def build_eureka_context(self) -> dict:
        """Build context dict for Eureka discovery."""
        sctx = self._ctx
        user_msgs = sctx.event_bus.get_events(types={USER_MESSAGE})
        return {
            "session_id": sctx.session_id or "unknown",
            "data_store_keys": [e["label"] for e in sctx.store.list_entries()]
            if sctx.store
            else [],
            "has_figure": sctx.renderer.get_figure() is not None,
            "recent_messages": [m.msg for m in user_msgs[-5:]],
        }

    def format_eureka_suggestion_as_user_msg(self, suggestion) -> str:
        """Convert a EurekaSuggestion into a natural-language user message."""
        parts = [f"[eureka] {suggestion.description}"]
        if suggestion.rationale:
            parts.append(f"Rationale: {suggestion.rationale}")
        if suggestion.parameters:
            parts.append(f"Parameters: {_json.dumps(suggestion.parameters)}")
        return "\n".join(parts)

    def maybe_extract_eurekas(self) -> None:
        """Trigger async Eureka extraction on a daemon thread.

        Uses EurekaAgent.send() via the BaseAgent inbox pattern.
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

        bus = self._ctx.event_bus
        eureka_store = self._ctx.eureka_store

        def _run():
            set_event_bus(bus)
            try:
                bus.emit(EUREKA_EXTRACTION_START, agent="Eureka", level="info")
                # Build the context message for the EurekaAgent
                msg_content = format_eureka_message(context)
                result = agent.send(
                    msg_content, sender="orchestrator", timeout=120.0
                )

                if result and result.get("failed"):
                    bus.emit(
                        EUREKA_EXTRACTION_ERROR,
                        agent="Eureka",
                        level="warning",
                        data={"error": result.get("text", "unknown error")},
                    )
                else:
                    # Count findings from the store for this session
                    session_id = context.get("session_id", "unknown")
                    if eureka_store is not None:
                        findings = eureka_store.list(session_id=session_id)
                        suggestions = eureka_store.list_suggestions(
                            session_id=session_id, status="proposed"
                        )
                    else:
                        findings = []
                        suggestions = []
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
                            # Mark suggestion as executed via the shared store
                            if eureka_store is not None:
                                eureka_store.update_suggestion_status(
                                    suggestion.id, "executed"
                                )
                            # Inject into orchestrator inbox if callback available
                            if self._inbox_injector is not None:
                                self._inbox_injector(synthetic_msg)
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
        if self._mode and not user_message.startswith("[eureka]"):
            self._round_counter = 0
            self._pending_suggestion = None
