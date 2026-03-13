"""PermissionGate — blocking user-approval callback for tool handlers.

Tool handlers call ``ctx.request_permission(request_id, action, description, command)``
which delegates to a PermissionGate instance. The gate emits a PERMISSION_REQUEST
event (sent to the frontend via SSE) and blocks until ``resolve()`` is called
(by the API layer when the user responds).

In CLI mode (no event bus), the gate auto-approves with a warning.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from agent.logging import get_logger

if TYPE_CHECKING:
    from agent.event_bus import EventBus

logger = get_logger()

# Default timeout for permission requests (5 minutes)
_DEFAULT_TIMEOUT = 300.0


class PermissionGate:
    """Blocking permission callback.

    Each pending request gets a threading.Event. The calling thread blocks
    on ``event.wait(timeout)`` until ``resolve()`` sets the result and
    signals the event.
    """

    def __init__(self, event_bus: "EventBus | None" = None):
        self._event_bus = event_bus
        self._lock = threading.Lock()
        # request_id → {"event": threading.Event, "result": dict | None}
        self._pending: dict[str, dict[str, Any]] = {}

    def __call__(
        self,
        *,
        request_id: str,
        action: str,
        description: str,
        command: str,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> dict:
        """Block until user approves/denies, or timeout.

        Returns:
            {"approved": bool, "reason": str}
        """
        event = threading.Event()
        with self._lock:
            self._pending[request_id] = {"event": event, "result": None}

        # Emit SSE event for frontend
        if self._event_bus is not None:
            from agent.event_bus import PERMISSION_REQUEST

            self._event_bus.emit(
                PERMISSION_REQUEST,
                msg=f"Permission request: {action} — {description}",
                data={
                    "request_id": request_id,
                    "action": action,
                    "description": description,
                    "command": command,
                },
            )
        else:
            # CLI mode — no way to ask the user, auto-approve with warning
            logger.warning(
                "Permission requested but no event bus available "
                "(CLI mode). Auto-approving: %s — %s",
                action,
                description,
            )
            with self._lock:
                self._pending.pop(request_id, None)
            return {"approved": True, "reason": "auto-approved (no UI)"}

        # Block until resolved or timeout
        signaled = event.wait(timeout=timeout)

        with self._lock:
            entry = self._pending.pop(request_id, None)

        if not signaled or entry is None or entry["result"] is None:
            return {"approved": False, "reason": "Permission request timed out"}

        return entry["result"]

    def resolve(self, request_id: str, *, approved: bool, reason: str = "") -> None:
        """Unblock a pending permission request with the user's decision."""
        with self._lock:
            entry = self._pending.get(request_id)
            if entry is None:
                return  # Already resolved or unknown — no-op
            entry["result"] = {"approved": approved, "reason": reason}
            entry["event"].set()
