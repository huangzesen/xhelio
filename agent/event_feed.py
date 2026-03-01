"""agent/event_feed.py — Cursor-tracked event feed for agents.

Provides a pull-based model where agents call check_events / get_event_details
on demand, with built-in cursor-based deduplication. Each agent gets its own
EventFeedBuffer with an independent cursor, so repeated calls never return
duplicate events.

When a response exceeds the token quota, the buffer returns a warning with
``quota_exceeded=True``. The agent can then re-call with ``compact=True``
to get an LLM-compacted summary via a compaction callback.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .event_bus import EventBus

from .token_counter import count_tokens as _estimate_tokens


class EventFeedBuffer:
    """Cursor-tracked event feed for an agent. Returns only unseen events.

    Args:
        event_bus: The session EventBus instance.
        ctx_tag: Context tag for filtering (e.g. "ctx:viz_plotly").
        token_quota: Max tokens before triggering quota_exceeded warning.
        compact_fn: Optional callback ``(raw_text, agent_type, budget) -> str``
                     that runs LLM compaction. Provided by OrchestratorAgent.
        agent_type: Agent type string for the compaction callback
                    (e.g. "viz", "mission", "orchestrator").
    """

    def __init__(
        self,
        event_bus: EventBus,
        ctx_tag: str,
        *,
        token_quota: int = 0,
        compact_fn: Optional[Callable[[str, str, int], str]] = None,
        agent_type: str = "",
    ):
        self._bus = event_bus
        self._ctx_tag = ctx_tag
        self._cursor: int = 0
        self._token_quota = token_quota
        self._compact_fn = compact_fn
        self._agent_type = agent_type

    def check(
        self,
        *,
        max_events: int = 50,
        event_types: list[str] | None = None,
        compact: bool = False,
    ) -> dict:
        """Return summaries of new events since last check.

        Args:
            max_events: Cap on returned events (default 50, max 200).
            event_types: Optional list of event type filters. If None, all
                         context-tagged events returned.
            compact: If True and the result exceeds the token quota, run
                     LLM compaction instead of returning a warning.
        """
        type_set = set(event_types) if event_types else None
        events = self._bus.get_events(
            tags={self._ctx_tag},
            types=type_set,
            since_index=self._cursor,
        )
        # Advance cursor to total event count
        self._cursor = len(self._bus)

        truncated = len(events) > max_events
        events = events[-max_events:]

        summaries = []
        for ev in events:
            entry = {
                "id": ev.id,
                "type": ev.type,
                "ts": ev.ts,
                "agent": ev.agent,
                "summary": ev.summary,
            }
            # Include key data fields for actionable context
            if ev.data.get("tool_name"):
                entry["tool_name"] = ev.data["tool_name"]
            if ev.data.get("status"):
                entry["status"] = ev.data["status"]
            if ev.data.get("label"):
                entry["label"] = ev.data["label"]
            summaries.append(entry)

        result = {
            "status": "success",
            "count": len(summaries),
            "events": summaries,
            "has_earlier": truncated,
        }

        # Token quota check
        if self._token_quota > 0 and summaries:
            text_size = _estimate_tokens(json.dumps(summaries, default=str))
            if text_size > self._token_quota:
                if compact and self._compact_fn:
                    raw = self._format_events_for_compaction(summaries)
                    compacted = self._compact_fn(
                        raw, self._agent_type, self._token_quota,
                    )
                    return {
                        "status": "success",
                        "count": len(summaries),
                        "compacted": True,
                        "summary": compacted,
                        "has_earlier": truncated,
                    }
                else:
                    result["quota_exceeded"] = True
                    result["estimated_tokens"] = text_size
                    result["token_quota"] = self._token_quota
                    result["hint"] = (
                        "Response exceeds token quota. Re-call with compact=true "
                        "to get an LLM-compacted summary, or use event_types filter "
                        "to narrow down."
                    )

        return result

    def get_details(
        self,
        event_ids: list[str],
        *,
        compact: bool = False,
    ) -> dict:
        """Return full details for specific events by ID.

        Args:
            event_ids: List of event IDs to retrieve.
            compact: If True and the result exceeds the token quota, run
                     LLM compaction instead of returning a warning.
        """
        found = self._bus.get_events_by_ids(set(event_ids))
        details = []
        for ev in found:
            details.append({
                "id": ev.id,
                "type": ev.type,
                "ts": ev.ts,
                "agent": ev.agent,
                "summary": ev.summary,
                "details": ev.details,
                "data": ev.data,
            })

        result = {"status": "success", "count": len(details), "events": details}

        # Token quota check
        if self._token_quota > 0 and details:
            text_size = _estimate_tokens(json.dumps(details, default=str))
            if text_size > self._token_quota:
                if compact and self._compact_fn:
                    raw = self._format_details_for_compaction(details)
                    compacted = self._compact_fn(
                        raw, self._agent_type, self._token_quota,
                    )
                    return {
                        "status": "success",
                        "count": len(details),
                        "compacted": True,
                        "summary": compacted,
                    }
                else:
                    result["quota_exceeded"] = True
                    result["estimated_tokens"] = text_size
                    result["token_quota"] = self._token_quota
                    result["hint"] = (
                        "Response exceeds token quota. Re-call with compact=true "
                        "to get an LLM-compacted summary."
                    )

        return result

    @staticmethod
    def _format_events_for_compaction(summaries: list[dict]) -> str:
        """Format event summaries into a text block for LLM compaction."""
        lines = []
        for s in summaries:
            parts = [f"[{s.get('ts', '?')}]", s.get("type", "?")]
            if s.get("agent"):
                parts.append(f"({s['agent']})")
            if s.get("tool_name"):
                parts.append(f"tool={s['tool_name']}")
            if s.get("status"):
                parts.append(f"status={s['status']}")
            if s.get("summary"):
                parts.append(s["summary"])
            lines.append(" ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _format_details_for_compaction(details: list[dict]) -> str:
        """Format event details into a text block for LLM compaction."""
        lines = []
        for d in details:
            header = f"[{d.get('ts', '?')}] {d.get('type', '?')} ({d.get('agent', '?')})"
            lines.append(header)
            if d.get("summary"):
                lines.append(f"  Summary: {d['summary']}")
            if d.get("details"):
                from .truncation import trunc
                det = trunc(d["details"], "feed.details")
                lines.append(f"  Details: {det}")
            lines.append("")
        return "\n".join(lines)
