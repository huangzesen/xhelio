"""
Reusable tool-calling loop for agents that manage their own chat sessions.

Extracted from BaseSubAgent.process_request() so that PlannerAgent (which
keeps a persistent chat across rounds) can also run tool calls without
inheriting from BaseSubAgent.
"""

import threading

from .llm import LLMAdapter
from .loop_guard import LoopGuard, make_call_key
from .event_bus import get_event_bus, DEBUG, TOOL_CALL, TOOL_ERROR_LOG


def extract_text_from_response(response) -> str:
    """Extract concatenated text from an LLMResponse.

    Returns empty string if no text is found.
    """
    return response.text or ""


def _extract_tool_calls(response):
    """Extract tool calls from an LLMResponse.

    Returns list of ToolCall objects with .name and .args attributes.
    """
    return response.tool_calls


def run_tool_loop(
    chat,
    response,
    tool_executor,
    adapter: LLMAdapter,
    agent_name: str = "Agent",
    max_total_calls: int = 10,
    max_iterations: int = 5,
    track_usage=None,
    collect_tool_results: dict = None,
    cancel_event: threading.Event | None = None,
    send_fn=None,
):
    """Run a tool-calling loop on an existing chat session.

    Keeps sending tool results back to the model until it stops issuing
    function calls (or a guard limit is hit).

    Args:
        chat: An active ChatSession.
        response: The initial LLMResponse.
        tool_executor: ``(tool_name: str, tool_args: dict) -> dict`` callable.
        adapter: LLMAdapter instance for building tool result messages.
        agent_name: Label for log messages.
        max_total_calls: Hard cap on total tool invocations.
        max_iterations: Hard cap on loop iterations.
        track_usage: Optional ``(response) -> None`` callback for token accounting.
        collect_tool_results: Optional dict to collect raw tool results.
            If provided, results are stored as ``{tool_name: [result, ...]}``
            so callers can inspect what the tools returned (e.g. parameter lists).
        send_fn: Optional callable ``(message) -> response`` for sending messages.
            If provided, used instead of ``chat.send``.  Allows callers
            to inject timeout/retry wrappers.

    Returns:
        The final response after tools stop.
    """
    if send_fn is None:
        send_fn = chat.send

    guard = LoopGuard(max_total_calls=max_total_calls, max_iterations=max_iterations)
    consecutive_errors = 0

    while True:
        stop_reason = guard.check_iteration()
        if stop_reason:
            get_event_bus().emit(DEBUG, agent=agent_name, level="debug", msg=f"[{agent_name}] Tool loop stopping: {stop_reason}")
            break

        if cancel_event and cancel_event.is_set():
            get_event_bus().emit(DEBUG, agent=agent_name, level="info", msg=f"[{agent_name}] Tool loop interrupted by user")
            break

        function_calls = _extract_tool_calls(response)
        if not function_calls:
            break

        # Check for loops / duplicates / cycling
        call_keys = set()
        for fc in function_calls:
            args = fc.args if isinstance(fc.args, dict) else (dict(fc.args) if fc.args else {})
            call_keys.add(make_call_key(fc.name, args))
        stop_reason = guard.check_calls(call_keys)
        if stop_reason:
            get_event_bus().emit(DEBUG, agent=agent_name, level="debug", msg=f"[{agent_name}] Tool loop stopping: {stop_reason}")
            break

        # Execute each tool
        function_responses = []
        all_errors = True
        for fc in function_calls:
            tool_name = fc.name
            tool_args = fc.args if isinstance(fc.args, dict) else (dict(fc.args) if fc.args else {})
            get_event_bus().emit(TOOL_CALL, agent=agent_name, level="debug", msg=f"[{agent_name}] Tool: {tool_name}({tool_args})", data={"tool_name": tool_name, "tool_args": tool_args})

            result = tool_executor(tool_name, tool_args)

            # Collect raw results for callers that need them
            if collect_tool_results is not None:
                collect_tool_results.setdefault(tool_name, []).append({
                    "args": tool_args,
                    "result": result,
                })

            if result.get("status") != "error":
                all_errors = False
            else:
                get_event_bus().emit(TOOL_ERROR_LOG, agent=agent_name, level="warning", msg=f"[{agent_name}] Tool error: {result.get('message', '')}")

            function_responses.append(
                adapter.make_tool_result_message(
                    tool_name, result, tool_call_id=getattr(fc, "id", None)
                )
            )

        guard.record_calls(call_keys)

        # Bail after 2 consecutive all-error rounds
        if all_errors:
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        if consecutive_errors >= 2:
            get_event_bus().emit(TOOL_ERROR_LOG, agent=agent_name, level="warning", msg=f"[{agent_name}] {consecutive_errors} consecutive error rounds, stopping")
            break

        # Feed results back to the model
        get_event_bus().emit(DEBUG, agent=agent_name, level="debug", msg=f"[{agent_name}] Sending {len(function_responses)} tool result(s) back...")
        # Set tool context on the agent (if track_usage is a bound method)
        if track_usage:
            agent_obj = getattr(track_usage, "__self__", None)
            if agent_obj and hasattr(agent_obj, "_last_tool_context"):
                tool_names = [fc.name for fc in function_calls]
                agent_obj._last_tool_context = "+".join(tool_names)
        response = send_fn(function_responses)
        if track_usage:
            track_usage(response)

    return response
