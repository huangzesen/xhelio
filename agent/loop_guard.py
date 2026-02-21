"""
Loop guard for agent tool-call loops.

Prevents infinite loops caused by Gemini repeatedly calling the same
tools (or alternating between two tools) without making progress.

Three layers of protection:
1. Hard total-call limit (absolute cap on tool invocations)
2. Subset-based duplicate detection (catches exact repeats)
3. Cycle detection (catches A->B->A->B... alternating patterns)
"""

import json


def make_call_key(tool_name: str, tool_args: dict) -> tuple:
    """Create a deterministic hashable key for a tool call.

    Uses JSON serialization with sorted keys to avoid inconsistencies
    from Protobuf Struct ordering or dict repr variations.
    """
    try:
        args_str = json.dumps(tool_args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args_str = str(sorted(tool_args.items()))
    return (tool_name, args_str)


class LoopGuard:
    """Prevents infinite tool-call loops in agent conversations.

    Usage:
        guard = LoopGuard(max_total_calls=10, max_iterations=5)

        while True:
            reason = guard.check_iteration()
            if reason:
                break

            # ... extract function_calls from response ...

            call_keys = {make_call_key(fc.name, dict(fc.args)) for fc in calls}
            reason = guard.check_calls(call_keys)
            if reason:
                break

            # ... execute tools ...

            guard.record_calls(call_keys)

            # ... send results back ...
    """

    def __init__(self, max_total_calls: int = 10, max_iterations: int = 5):
        self.max_total_calls = max_total_calls
        self.max_iterations = max_iterations
        self.total_calls = 0
        self.iteration = 0
        self.previous_calls: set = set()
        self._recent_batches: list = []

    def check_iteration(self) -> str | None:
        """Increment iteration counter and check limit. Returns stop reason or None."""
        self.iteration += 1
        if self.iteration > self.max_iterations:
            return f"iteration limit ({self.max_iterations}) reached"
        return None

    def check_calls(self, call_keys: set) -> str | None:
        """Check if proposed calls should proceed. Call BEFORE executing tools.

        Returns stop reason string or None to continue.
        """
        if not call_keys:
            return None

        # Hard limit on total calls
        if self.total_calls + len(call_keys) > self.max_total_calls:
            return (
                f"total call limit ({self.max_total_calls}) reached "
                f"after {self.total_calls} calls"
            )

        # Exact duplicate: all proposed calls were already made before
        if call_keys.issubset(self.previous_calls):
            return "duplicate tool calls detected"

        # Cycle detection: if this exact batch appeared in the last 3 batches
        batch_key = frozenset(call_keys)
        if batch_key in self._recent_batches[-3:]:
            return "cycling pattern detected"

        return None

    def record_calls(self, call_keys: set) -> None:
        """Record that a batch of calls was executed. Call AFTER executing tools."""
        self.total_calls += len(call_keys)
        self.previous_calls.update(call_keys)
        self._recent_batches.append(frozenset(call_keys))
        # Keep bounded
        if len(self._recent_batches) > 8:
            self._recent_batches = self._recent_batches[-8:]
