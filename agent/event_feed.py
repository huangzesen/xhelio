"""agent/event_feed.py — Cursor-tracked event feed for agents.

Provides a pull-based model where agents call events(action="check") / events(action="details")
on demand, with built-in cursor-based deduplication. Each agent gets its own
EventFeedBuffer with an independent cursor, so repeated calls never return
duplicate events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .event_bus import EventBus

from .token_counter import count_tokens as _estimate_tokens


class EventFeedBuffer:
    """Cursor-tracked event feed for an agent. Returns only unseen events.

    Args:
        event_bus: The session EventBus instance.
        ctx_tag: Context tag for filtering (e.g. "ctx:viz_plotly").
    """

    def __init__(
        self,
        event_bus: EventBus,
        ctx_tag: str,
    ):
        self._bus = event_bus
        self._ctx_tag = ctx_tag
        self._cursor: int = 0

    def check(
        self,
        *,
        max_events: int = 50,
        event_types: list[str] | None = None,
    ) -> dict:
        """Return summaries of new events since last check.

        Args:
            max_events: Cap on returned events (default 50, max 200).
            event_types: Optional list of event type filters. If None, all
                         context-tagged events returned.
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

        return {
            "status": "success",
            "count": len(summaries),
            "events": summaries,
            "has_earlier": truncated,
        }

    def get_details(
        self,
        event_ids: list[str],
    ) -> dict:
        """Return full details for specific events by ID.

        Args:
            event_ids: List of event IDs to retrieve.
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

        return {"status": "success", "count": len(details), "events": details}

    def peek(
        self,
        *,
        max_events: int = 50,
        event_types: list[str] | None = None,
        agent_filter: str | None = None,
        since_seconds: float | None = None,
    ) -> dict:
        """Stateless cross-context read of events from any agent.

        Unlike check(), this does NOT advance the cursor and does NOT filter
        by ctx tag. It returns events that are invisible to the normal feed —
        i.e., events that lack this buffer's ctx tag.

        Args:
            max_events: Cap on returned events (default 50).
            event_types: Optional event type filter.
            agent_filter: Agent name filter. Exact match, or prefix match
                          for envoy agents (e.g. "envoy" matches "envoy:PSP").
            since_seconds: Only return events from the last N seconds.
        """
        type_set = set(event_types) if event_types else None
        # Get ALL events (no tag filter)
        events = self._bus.get_events(tags=None, types=type_set)

        # Exclude events already visible via normal check (have our ctx tag)
        ctx_tag = self._ctx_tag
        filtered = []
        for ev in events:
            if ctx_tag in ev.tags:
                continue
            # Agent filter: exact match or prefix match for envoy-style names
            if agent_filter:
                if ev.agent != agent_filter and not ev.agent.startswith(agent_filter + ":"):
                    continue
            # Time window filter
            if since_seconds is not None:
                from datetime import datetime, timezone

                try:
                    event_time = datetime.fromisoformat(ev.ts)
                    age = (datetime.now(timezone.utc) - event_time).total_seconds()
                    if age > since_seconds:
                        continue
                except (ValueError, TypeError):
                    pass
            filtered.append(ev)

        truncated = len(filtered) > max_events
        filtered = filtered[-max_events:]

        summaries = []
        for ev in filtered:
            entry = {
                "id": ev.id,
                "type": ev.type,
                "ts": ev.ts,
                "agent": ev.agent,
                "summary": ev.summary,
                "tags": sorted(ev.tags),
            }
            if ev.data.get("tool_name"):
                entry["tool_name"] = ev.data["tool_name"]
            if ev.data.get("status"):
                entry["status"] = ev.data["status"]
            if ev.data.get("label"):
                entry["label"] = ev.data["label"]
            summaries.append(entry)

        return {
            "status": "success",
            "count": len(summaries),
            "events": summaries,
            "has_earlier": truncated,
        }

    def peek_details(
        self,
        event_ids: list[str],
    ) -> dict:
        """Return full details for events by ID, regardless of ctx tags.

        Same as get_details() but explicitly tag-independent —
        works for events found via peek().

        Args:
            event_ids: List of event IDs to retrieve.
        """
        return self.get_details(event_ids)
