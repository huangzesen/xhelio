"""Short-term memory buffer for per-agent conversation history.

Subscribes to EventBus and accumulates a structured record of events
relevant to a specific agent. Append-only for prefix cache stability.

Phase 4 scope: runs in parallel with the adapter's native history.
Does NOT replace message building yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.event_bus import EventBus, SessionEvent

from agent.event_bus import USER_MESSAGE, AGENT_RESPONSE


DEFAULT_CURATED_TYPES = frozenset({USER_MESSAGE, AGENT_RESPONSE})


@dataclass(frozen=True)
class STMEntry:
    """A single entry in the STM buffer."""

    event_id: str
    ts: str
    sender: str  # from SessionEvent.agent
    target: str  # from SessionEvent.target
    event_type: str
    content: str  # formatted for LLM
    role: str  # "user" | "assistant"


class STMBuffer:
    """Per-agent short-term memory. Append-only for prefix cache stability.

    Subscribes to an EventBus and filters events by type and target relevance.
    """

    def __init__(
        self,
        agent_id: str,
        event_bus: "EventBus",
        curated_types: frozenset[str] | None = None,
    ):
        self._agent_id = agent_id
        self._bus = event_bus
        self._types = curated_types or DEFAULT_CURATED_TYPES
        self._entries: list[STMEntry] = []
        self._last_bus_idx: int = 0  # index into bus._events for incremental sync

    def sync(self) -> int:
        """Pull new matching events from bus, append to buffer.

        Returns the number of new entries added.
        """
        events = self._bus.get_events_since(self._last_bus_idx)
        added = 0
        for evt in events:
            self._last_bus_idx = max(self._last_bus_idx, 1)  # at least processed one
            if evt.type not in self._types:
                continue
            # Determine if this event is relevant to this agent
            if not self._is_relevant(evt):
                continue
            entry = self._event_to_entry(evt)
            self._entries.append(entry)
            added += 1
        # Update cursor to current bus length
        self._last_bus_idx = self._bus.event_count()
        return added

    def get_all(self) -> list[STMEntry]:
        """Full accumulated STM (for client-managed history adapters)."""
        return list(self._entries)

    def get_since(self, cursor: int) -> tuple[list[STMEntry], int]:
        """New entries since cursor (for server-side history adapters).

        Returns (new_entries, new_cursor).
        """
        new_entries = self._entries[cursor:]
        return new_entries, len(self._entries)

    def entry_count(self) -> int:
        """Number of entries in the buffer."""
        return len(self._entries)

    def _is_relevant(self, evt: "SessionEvent") -> bool:
        """Check if an event is relevant to this agent.

        An event is relevant if:
        - target matches this agent's ID (directed to us)
        - target is empty (broadcast)
        - sender is this agent (we sent it)
        """
        return (
            evt.target == self._agent_id
            or evt.target == ""
            or evt.agent == self._agent_id
        )

    def _event_to_entry(self, evt: "SessionEvent") -> STMEntry:
        """Convert a SessionEvent to an STMEntry."""
        # Determine role based on event type
        if evt.type == USER_MESSAGE:
            role = "user"
            content = evt.data.get("text", evt.summary) if evt.data else evt.summary
        else:
            role = "assistant"
            content = evt.data.get("text", evt.summary) if evt.data else evt.summary

        return STMEntry(
            event_id=evt.id,
            ts=evt.ts,
            sender=evt.agent,
            target=evt.target,
            event_type=evt.type,
            content=content,
            role=role,
        )
