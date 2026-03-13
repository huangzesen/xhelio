"""Tracks in-flight work units (delegations, tool executions).

Thread-safe work unit registry with a simpler API focused on tracking
and cancellation. Completion is handled by queue message flow.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _WorkUnit:
    work_id: str
    agent_id: str
    description: str
    cancel_event: threading.Event
    started_at: float = field(default_factory=time.time)
    status: str = "active"  # active, completed, failed, cancelled
    error: str = ""


class WorkTracker:
    """Thread-safe tracker for in-flight work units."""

    def __init__(self) -> None:
        self._units: dict[str, _WorkUnit] = {}
        self._lock = threading.Lock()

    def register(
        self,
        work_id: str,
        agent_id: str,
        description: str,
        cancel_event: threading.Event,
    ) -> None:
        """Register a new in-flight work unit."""
        with self._lock:
            self._units[work_id] = _WorkUnit(
                work_id=work_id,
                agent_id=agent_id,
                description=description,
                cancel_event=cancel_event,
            )

    def mark_completed(self, work_id: str) -> None:
        """Mark a work unit as completed."""
        with self._lock:
            if work_id in self._units:
                self._units[work_id].status = "completed"

    def mark_failed(self, work_id: str, error: str) -> None:
        """Mark a work unit as failed."""
        with self._lock:
            if work_id in self._units:
                self._units[work_id].status = "failed"
                self._units[work_id].error = error

    def list_active(self) -> list[dict]:
        """Return all in-flight work units."""
        with self._lock:
            return [
                {
                    "work_id": u.work_id,
                    "agent_id": u.agent_id,
                    "description": u.description,
                    "elapsed_s": round(time.time() - u.started_at, 1),
                }
                for u in self._units.values()
                if u.status == "active"
            ]

    def cancel(self, work_id: str) -> bool:
        """Cancel a specific work unit. Returns True if found."""
        with self._lock:
            unit = self._units.get(work_id)
            if unit and unit.status == "active":
                unit.cancel_event.set()
                unit.status = "cancelled"
                return True
            return False

    def cancel_all(self) -> int:
        """Cancel all in-flight work. Returns count cancelled."""
        with self._lock:
            count = 0
            for unit in self._units.values():
                if unit.status == "active":
                    unit.cancel_event.set()
                    unit.status = "cancelled"
                    count += 1
            return count

    def clear(self) -> None:
        """Remove all completed/failed/cancelled entries."""
        with self._lock:
            self._units = {
                k: v for k, v in self._units.items() if v.status == "active"
            }
