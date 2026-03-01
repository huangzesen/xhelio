"""
Shared LLM utilities used by OrchestratorAgent, Actor, and PlannerAgent.

All functions are stateless (operate on passed-in state dicts,
event bus, etc.).
"""

import contextvars
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from .llm import LLMResponse
from .event_bus import get_event_bus, DEBUG, LLM_CALL, TOKEN_USAGE, THINKING
from .truncation import trunc

# Model context window limits for overflow detection
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gemini-3-flash": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "o1": 200_000,
    "o3": 200_000,
}


def get_context_limit(model_name: str) -> int:
    """Return context window size for a model (longest prefix match), or 0 if unknown."""
    if not model_name:
        return 0
    best, best_len = 0, 0
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if model_name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = limit, len(prefix)
    return best


# LLM API call timeout thresholds (seconds)
_LLM_WARN_INTERVAL = 20      # log a warning every N seconds while waiting
_LLM_RETRY_TIMEOUT = 180     # abandon call and retry after this
_LLM_MAX_RETRIES = 2         # max retries before giving up


class _CancelledDuringLLM(Exception):
    """Raised when user cancellation is detected during an LLM API wait."""


def _is_stale_interaction_error(exc: Exception) -> bool:
    """Return True if the error indicates a stale/expired Interactions API session."""
    msg = str(exc).lower()
    return "interaction" in msg and ("not found" in msg or "invalid" in msg or "expired" in msg)


def send_with_timeout(
    chat,
    message,
    timeout_pool: ThreadPoolExecutor,
    cancel_event: threading.Event | None,
    retry_timeout: float,
    agent_name: str,
    logger,
) -> LLMResponse:
    """Send a message to the LLM with periodic warnings and retry on timeout.

    Shared implementation used by OrchestratorAgent, Actor, and PlannerAgent.

    - Warns every _LLM_WARN_INTERVAL seconds while waiting.
    - After retry_timeout seconds, abandons the call and retries.
    - Retries up to _LLM_MAX_RETRIES times before raising.
    - Checks cancel_event between each poll interval so cancellation
      is responsive even during long LLM waits.
    """
    last_exc = None
    for attempt in range(1 + _LLM_MAX_RETRIES):
        future: Future = timeout_pool.submit(chat.send, message)
        t0 = time.monotonic()
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    future.cancel()
                    get_event_bus().emit(DEBUG, agent=agent_name, level="info", msg=f"[{agent_name}] LLM call cancelled by user")
                    raise _CancelledDuringLLM()

                elapsed = time.monotonic() - t0
                remaining = retry_timeout - elapsed
                if remaining <= 0:
                    break
                wait = min(_LLM_WARN_INTERVAL, remaining)
                try:
                    return future.result(timeout=wait)
                except TimeoutError:
                    elapsed = time.monotonic() - t0
                    if elapsed >= retry_timeout:
                        break
                    get_event_bus().emit(LLM_CALL, agent=agent_name, level="warning", msg=f"[{agent_name}] LLM API not responding after {elapsed:.0f}s (attempt {attempt + 1})...")

            elapsed = time.monotonic() - t0
            future.cancel()
            last_exc = TimeoutError(
                f"LLM API call timed out after {elapsed:.0f}s"
            )
            if attempt < _LLM_MAX_RETRIES:
                get_event_bus().emit(LLM_CALL, agent=agent_name, level="warning", msg=f"[{agent_name}] LLM API timed out after {elapsed:.0f}s, retrying ({attempt + 1}/{_LLM_MAX_RETRIES})...")
            else:
                get_event_bus().emit(LLM_CALL, agent=agent_name, level="error", msg=f"[{agent_name}] LLM API timed out after {elapsed:.0f}s, no retries left")
        except _CancelledDuringLLM:
            raise
        except Exception:
            raise

    raise last_exc


def send_with_timeout_stream(
    chat,
    message,
    timeout_pool: ThreadPoolExecutor,
    cancel_event: threading.Event | None,
    retry_timeout: float,
    agent_name: str,
    logger,
    on_chunk=None,
) -> LLMResponse:
    """Like ``send_with_timeout`` but uses ``chat.send_stream()`` for incremental text.

    ``on_chunk`` is called from the thread-pool thread as text deltas arrive.
    Since ``EventBus.emit()`` is thread-safe, callers can safely emit events
    from within ``on_chunk``.
    """
    last_exc = None
    for attempt in range(1 + _LLM_MAX_RETRIES):
        future: Future = timeout_pool.submit(chat.send_stream, message, on_chunk)
        t0 = time.monotonic()
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    future.cancel()
                    get_event_bus().emit(DEBUG, agent=agent_name, level="info", msg=f"[{agent_name}] LLM call cancelled by user")
                    raise _CancelledDuringLLM()

                elapsed = time.monotonic() - t0
                remaining = retry_timeout - elapsed
                if remaining <= 0:
                    break
                wait = min(_LLM_WARN_INTERVAL, remaining)
                try:
                    return future.result(timeout=wait)
                except TimeoutError:
                    elapsed = time.monotonic() - t0
                    if elapsed >= retry_timeout:
                        break
                    get_event_bus().emit(LLM_CALL, agent=agent_name, level="warning", msg=f"[{agent_name}] LLM API not responding after {elapsed:.0f}s (attempt {attempt + 1})...")

            elapsed = time.monotonic() - t0
            future.cancel()
            last_exc = TimeoutError(
                f"LLM API call timed out after {elapsed:.0f}s"
            )
            if attempt < _LLM_MAX_RETRIES:
                get_event_bus().emit(LLM_CALL, agent=agent_name, level="warning", msg=f"[{agent_name}] LLM API timed out after {elapsed:.0f}s, retrying ({attempt + 1}/{_LLM_MAX_RETRIES})...")
            else:
                get_event_bus().emit(LLM_CALL, agent=agent_name, level="error", msg=f"[{agent_name}] LLM API timed out after {elapsed:.0f}s, no retries left")
        except _CancelledDuringLLM:
            raise
        except Exception:
            raise

    raise last_exc


def track_llm_usage(
    response: LLMResponse,
    token_state: dict,
    agent_name: str,
    last_tool_context: str,
    *,
    system_tokens: int = 0,
    tools_tokens: int = 0,
):
    """Accumulate token usage from an LLMResponse and emit via EventBus.

    Shared implementation used by OrchestratorAgent, Actor, and PlannerAgent.

    Args:
        response: The LLMResponse to extract usage from.
        token_state: Mutable dict with keys 'input', 'output', 'thinking',
            'cached', 'api_calls'. Updated in-place.
        agent_name: Label for log messages.
        last_tool_context: Tool context string for the token log.
        system_tokens: Approximate token count of the system prompt (0 = unknown).
        tools_tokens: Approximate token count of tool declarations (0 = unknown).
    """
    usage = response.usage
    call_input = usage.input_tokens
    call_output = usage.output_tokens
    call_thinking = usage.thinking_tokens
    call_cached = usage.cached_tokens
    token_state["input"] += call_input
    token_state["output"] += call_output
    token_state["thinking"] += call_thinking
    token_state["cached"] += call_cached
    token_state["api_calls"] += 1

    # Compute history tokens when decomposition is available
    has_decomp = system_tokens > 0 or tools_tokens > 0
    history_tokens = max(0, call_input - system_tokens - tools_tokens) if has_decomp else 0

    # Build log message
    if has_decomp:
        decomp_str = f" (sys:{system_tokens} tools:{tools_tokens} hist:{history_tokens})"
    else:
        decomp_str = ""

    get_event_bus().emit(
        TOKEN_USAGE,
        agent=agent_name,
        level="debug",
        msg=(
            f"[Tokens] {agent_name} in:{call_input}{decomp_str} out:{call_output} "
            f"think:{call_thinking} | cum_in:{token_state['input']} "
            f"cum_out:{token_state['output']} calls:{token_state['api_calls']}"
        ),
        data={
            "agent_name": agent_name,
            "tool_context": trunc(last_tool_context, "console.outcome") if last_tool_context else "unknown",
            "input_tokens": call_input,
            "output_tokens": call_output,
            "thinking_tokens": call_thinking,
            "cached_tokens": call_cached,
            "system_tokens": system_tokens,
            "tools_tokens": tools_tokens,
            "history_tokens": history_tokens,
            "cumulative_input": token_state["input"],
            "cumulative_output": token_state["output"],
            "cumulative_thinking": token_state["thinking"],
            "cumulative_cached": token_state["cached"],
            "api_calls": token_state["api_calls"],
        },
    )
    # Always emit thinking events so SSE listener can forward to frontend
    for thought in response.thoughts:
        get_event_bus().emit(THINKING, agent=agent_name, level="debug", msg=f"[Thinking] {thought}", data={"text": thought})


def execute_tools_batch(
    function_calls: list,
    tool_executor,
    parallel_safe_tools: set[str],
    parallel_enabled: bool,
    max_workers: int,
    agent_name: str,
    logger,
) -> list[tuple[str | None, str, dict, dict]]:
    """Execute tool calls, parallelizing when all are in the safe set.

    Shared implementation used by OrchestratorAgent.

    Returns list of (tool_call_id, tool_name, tool_args, result) in original order.
    """
    parsed = [
        (
            getattr(fc, "id", None),
            fc.name,
            fc.args if isinstance(fc.args, dict) else (dict(fc.args) if fc.args else {}),
        )
        for fc in function_calls
    ]

    all_safe = (
        parallel_enabled
        and len(parsed) > 1
        and all(name in parallel_safe_tools for _, name, _ in parsed)
    )

    if not all_safe:
        return [
            (tc_id, name, args, tool_executor(name, args, tc_id))
            for tc_id, name, args in parsed
        ]

    get_event_bus().emit(DEBUG, agent=agent_name, level="debug", msg=f"[{agent_name}] Parallel: {len(parsed)} tools concurrently")
    workers = min(len(parsed), max_workers)
    results_by_idx: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(contextvars.copy_context().run, tool_executor, name, args, tc_id): idx
            for idx, (tc_id, name, args) in enumerate(parsed)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_by_idx[idx] = future.result()
            except Exception as e:
                results_by_idx[idx] = {
                    "status": "error",
                    "message": f"Parallel execution error: {e}",
                }

    return [
        (parsed[i][0], parsed[i][1], parsed[i][2], results_by_idx[i])
        for i in range(len(parsed))
    ]


def build_outcome_summary(tool_outcomes: list[dict]) -> str:
    """Build a concise text summary from structured tool outcomes.

    Used by the orchestrator's _execute_task fallback when the LLM
    produces no text.
    """
    parts = []
    for o in tool_outcomes:
        s = f"{o['tool']}={o['status']}"
        if o.get("label"):
            s += f"(label={o['label']}"
            if o.get("num_points"):
                s += f", {o['num_points']} pts"
            if o.get("units"):
                s += f", {o['units']}"
            s += ")"
        elif o.get("message") and o["status"] == "error":
            msg = trunc(str(o["message"]), "context.task_outcome_error")
            s += f"({msg})"
        parts.append(s)
    return "Outcomes: " + "; ".join(parts)
