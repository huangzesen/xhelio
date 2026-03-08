"""Inline completion generation — follow-ups, session titles, autocomplete.

Extracted from OrchestratorAgent to reduce god-object complexity.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Optional

from agent.event_bus import DEBUG, USER_MESSAGE, AGENT_RESPONSE
from agent.truncation import get_limit, get_item_limit
import config

if TYPE_CHECKING:
    from agent.llm.service import LLMService
    from agent.token_tracker import TokenTracker
    from agent.event_bus import EventBus


def _extract_turns_from_history(history: list, max_text: int | None = None) -> list[str]:
    """Import and delegate to the module-level _extract_turns in core."""
    from agent.core import _extract_turns
    return _extract_turns(history, max_text=max_text)


_CIRCUIT_BREAKER_THRESHOLD = 5   # consecutive failures before disabling
_CIRCUIT_BREAKER_COOLDOWN = 60   # seconds to disable after threshold hit


class InlineCompletions:
    """Handles follow-up suggestions, session titles, and inline completions.

    Uses the cheapest model tier (INLINE_MODEL) for low latency.
    Includes a circuit breaker that disables inline completions after
    repeated failures.
    """

    def __init__(
        self,
        service: LLMService,
        inline_tracker: TokenTracker,
        event_bus: EventBus,
    ) -> None:
        self._service = service
        self._tracker = inline_tracker
        self._event_bus = event_bus
        self._fail_count: int = 0
        self._disabled_until: float = 0.0
        self._last_tool_context: str = ""

    def _track(self, response, context: str) -> None:
        self._last_tool_context = context
        self._tracker.track(response, last_tool_context=context)

    def generate_follow_ups(
        self,
        chat_history: list,
        store_labels: list[str],
        has_plot: bool,
        max_suggestions: int = 3,
    ) -> list[str]:
        """Generate contextual follow-up suggestions.

        Args:
            chat_history: The chat session history (from chat.get_history())
            store_labels: Labels of datasets in the DataStore
            has_plot: Whether a plot is currently displayed
            max_suggestions: Max number of suggestions to return

        Returns:
            List of suggestion strings, or [] on failure.
        """
        turns = _extract_turns_from_history(
            chat_history[-get_item_limit("items.follow_up_turns"):]
        )
        if not turns:
            return []

        conversation_text = "\n".join(turns)
        data_context = (
            f"Data in memory: {', '.join(store_labels)}"
            if store_labels
            else "No data in memory yet."
        )
        plot_context = (
            "A plot is currently displayed." if has_plot else "No plot is displayed."
        )

        prompt = f"""Based on this conversation, suggest {max_suggestions} short follow-up questions the user might ask next.

{conversation_text}

{data_context}
{plot_context}

Respond with a JSON array of strings only (no markdown fencing). Each suggestion should be:
- A natural, conversational question (max 12 words)
- Actionable — something the agent can actually do
- Different from what was already asked
- Related to the current context (data, plots, missions)

Example: ["Compare this with solar wind speed", "Zoom in to January 10-15", "Export the plot as PDF"]"""

        try:
            response = self._service.generate(
                prompt=prompt,
                model=config.INLINE_MODEL,
                temperature=0.7,
                tracked=False,
            )
            self._track(response, "follow_up_suggestions")

            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3].strip()

            suggestions = json.loads(text)
            if isinstance(suggestions, list):
                return [s for s in suggestions if isinstance(s, str)][:max_suggestions]
        except Exception as e:
            self._event_bus.emit(
                DEBUG, level="debug", msg=f"[FollowUp] Generation failed: {e}"
            )

        return []

    def generate_session_title(
        self,
        chat_history: list,
        event_bus_events: list | None = None,
    ) -> Optional[str]:
        """Generate a short title from the first exchange.

        Args:
            chat_history: The chat session history
            event_bus_events: Fallback events for Interactions API
        """
        turns = _extract_turns_from_history(chat_history[:4], max_text=500)

        # Fallback: EventBus events
        if not turns and event_bus_events:
            for ev in event_bus_events[:4]:
                text = (ev.data or {}).get("text", ev.msg)
                if text:
                    label = "User" if ev.type == USER_MESSAGE else "Agent"
                    turns.append(f"{label}: {text[:500]}")

        if not turns:
            return None

        conversation_text = "\n".join(turns)
        prompt = (
            "Generate a concise title (3-7 words) for this conversation. "
            "Summarize the user's main intent. Use plain English.\n\n"
            f"{conversation_text}\n\n"
            "Respond with ONLY the title text, no quotes, no punctuation at the end."
        )
        try:
            response = self._service.generate(
                prompt=prompt,
                model=config.INLINE_MODEL,
                temperature=0.3,
                tracked=False,
            )
            self._track(response, "session_title")
            text = (response.text or "").strip().strip("\"'")
            if text and len(text) <= 100:
                return text
        except Exception as e:
            self._event_bus.emit(
                DEBUG, level="debug", msg=f"[SessionTitle] Generation failed: {e}"
            )
        return None

    def generate_inline_completions(
        self,
        partial: str,
        chat_history: list,
        store_labels: list[str],
        memory_section: str,
        max_completions: int = 3,
    ) -> list[str]:
        """Complete the user's partial input.

        Circuit breaker: after 5 consecutive failures, disables for 60s.

        Args:
            partial: The user's partial input text
            chat_history: Chat session history
            store_labels: Labels of datasets in DataStore
            memory_section: Formatted memory for injection
            max_completions: Max completions to return
        """
        if time.time() < self._disabled_until:
            return []

        from knowledge.prompt_builder import build_inline_completion_prompt

        turns = _extract_turns_from_history(
            (chat_history or [])[-get_item_limit("items.inline_turns"):],
            max_text=get_limit("context.turn_text.inline"),
        )

        prompt = build_inline_completion_prompt(
            partial,
            conversation_context="\n".join(turns),
            memory_section=memory_section,
            data_labels=store_labels,
            max_completions=max_completions,
        )

        try:
            response = self._service.generate(
                prompt=prompt,
                model=config.INLINE_MODEL,
                temperature=0.5,
                max_output_tokens=get_limit("output.inline_tokens"),
                tracked=False,
            )
            self._track(response, "inline_completion")

            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3].strip()

            completions = json.loads(text)
            if isinstance(completions, list):
                valid = [
                    c for c in completions
                    if isinstance(c, str)
                    and c.startswith(partial)
                    and len(c) > len(partial)
                    and len(c) <= 120
                ]
                if valid:
                    self._fail_count = 0
                    return valid[:max_completions]
            return []
        except Exception as e:
            self._fail_count += 1
            if self._fail_count >= _CIRCUIT_BREAKER_THRESHOLD:
                self._disabled_until = time.time() + _CIRCUIT_BREAKER_COOLDOWN
                self._event_bus.emit(
                    DEBUG,
                    level="warning",
                    msg=f"[InlineComplete] {self._fail_count} consecutive failures, "
                    f"disabling for {_CIRCUIT_BREAKER_COOLDOWN}s",
                )
                self._fail_count = 0
            else:
                self._event_bus.emit(
                    DEBUG,
                    level="debug",
                    msg=f"[InlineComplete] Generation failed "
                    f"({self._fail_count}/{_CIRCUIT_BREAKER_THRESHOLD}): {e}",
                )

        return []

    def reset(self) -> None:
        """Reset circuit breaker state."""
        self._fail_count = 0
        self._disabled_until = 0.0
