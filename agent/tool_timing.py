import time
from datetime import datetime, timezone


def stamp_tool_result(result: dict, elapsed_ms: int) -> dict:
    """Inject _ts and _elapsed_ms into a tool result dict (in-place)."""
    result["_ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result["_elapsed_ms"] = elapsed_ms
    return result


class ToolTimer:
    """Context manager for timing tool execution."""
    def __init__(self):
        self._start = 0.0
        self.elapsed_ms = 0
    
    def __enter__(self):
        self._start = time.monotonic()
        return self
    
    def __exit__(self, *exc):
        self.elapsed_ms = int((time.monotonic() - self._start) * 1000)
        return False
