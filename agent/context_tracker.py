"""Track injected context per agent to enable incremental (delta-based) injection.

Every orchestrator turn and every delegation re-injects store state, memory,
session context, and plot state into the message text.  Since all actors maintain
persistent LLM sessions, re-injecting identical context wastes tokens.

ContextTracker records what was last injected per agent and computes deltas so
that only *new or changed* context is sent.  When nothing has changed, a short
"unchanged" note is injected instead, or the section is skipped entirely.

Thread safety: snapshots are replaced atomically (dict value assignment is
atomic in CPython).  No lock is needed because each agent ID is only written
from the thread that owns the delegation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class _Snapshot:
    """What was last injected into an agent."""

    store_labels: frozenset[str] = frozenset()
    store_hash: str = ""
    memory_hash: str = ""
    pipeline_hash: str = ""
    plot_hash: str = ""
    session_hash: str = ""


class ContextTracker:
    """Tracks injected context per agent ID."""

    def __init__(self) -> None:
        self._snapshots: dict[str, _Snapshot] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest() if text else ""

    # ------------------------------------------------------------------
    # Store delta
    # ------------------------------------------------------------------

    def get_store_delta(
        self, agent_id: str, entries: list[dict],
    ) -> tuple[list[dict], list[str], str]:
        """Return NEW and REMOVED store entries since last injection.

        Returns:
            (new_entries, removed_labels, full_hash)
        """
        full_json = json.dumps(entries, default=str, sort_keys=True)
        full_hash = self._hash(full_json)

        snap = self._snapshots.get(agent_id)
        if snap and snap.store_hash == full_hash:
            return [], [], full_hash  # Nothing changed

        current_labels = frozenset(e["label"] for e in entries)

        if snap:
            new_entries = [e for e in entries if e["label"] not in snap.store_labels]
            removed_labels = sorted(snap.store_labels - current_labels)
            # If no new/removed but hash differs, metadata changed (e.g. point count)
            # — send the changed entries
            if not new_entries and not removed_labels:
                new_entries = entries  # re-send all (metadata update)
            return new_entries, removed_labels, full_hash

        return entries, [], full_hash  # First injection — send all

    # ------------------------------------------------------------------
    # Generic field change detection
    # ------------------------------------------------------------------

    def is_changed(self, agent_id: str, field_name: str, text: str) -> bool:
        """Check if a context field has changed since last injection."""
        snap = self._snapshots.get(agent_id)
        if not snap:
            return True
        return getattr(snap, f"{field_name}_hash", "") != self._hash(text)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, agent_id: str, **kwargs) -> None:
        """Record what was injected.

        Keyword args: store_entries, store_hash, memory, pipeline, plot, session.
        """
        snap = self._snapshots.get(agent_id, _Snapshot())
        if "store_entries" in kwargs:
            snap.store_labels = frozenset(
                e["label"] for e in kwargs["store_entries"]
            )
        if "store_hash" in kwargs:
            snap.store_hash = kwargs["store_hash"]
        for f in ("memory", "pipeline", "plot", "session"):
            if f in kwargs:
                setattr(snap, f"{f}_hash", self._hash(kwargs[f]))
        self._snapshots[agent_id] = snap

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, agent_id: str) -> None:
        """Reset tracking for an agent (e.g., when its session is recreated)."""
        self._snapshots.pop(agent_id, None)

    def reset_all(self) -> None:
        """Reset all tracking (e.g., when all sub-agents are reset)."""
        self._snapshots.clear()
