"""agent/event_formatters.py — Per-event-type (summary, details) formatters.

Each formatter receives (agent, data, children) and returns a
(summary, details) tuple.  summary is a short one-liner (<=120 chars)
that sub-agents can reason about; details is multi-line context.

The format_event() function is the single entry point used by
EventBus.emit() and EventBus.span() to auto-generate summary/details
from the event type and its data dict.
"""

from __future__ import annotations

from typing import Callable

from .truncation import child_summaries

# ---- Formatter registry ----

FORMATTERS: dict[str, Callable] = {}


def register(event_type: str):
    """Decorator to register a formatter for an event type."""
    def decorator(fn):
        FORMATTERS[event_type] = fn
        return fn
    return decorator


def format_event(
    event_type: str,
    agent: str,
    data: dict | None,
    children: list | None = None,
) -> tuple[str, str]:
    """Produce (summary, details) for an event.

    Falls back to _default_formatter for unregistered types.
    """
    fn = FORMATTERS.get(event_type, _default_formatter)
    return fn(agent, data or {}, children or [])


def _default_formatter(agent: str, data: dict, children: list) -> tuple[str, str]:
    """Fallback: use msg from data if available, else event type."""
    msg = data.get("_msg", "")
    if msg:
        return (msg, msg)
    return ("", "")


# ---- Helpers ----

def _first_output(data: dict) -> str:
    outputs = data.get("outputs", [])
    if isinstance(outputs, dict):
        return str(next(iter(outputs))) if outputs else "(no label)"
    return str(outputs[0]) if outputs else "(no label)"


# =====================================================================
# Conversation / session
# =====================================================================

@register("user_message")
def _fmt_user_message(agent: str, data: dict, children: list) -> tuple[str, str]:
    text = data.get("text", "")
    return (f"User: {text}", text)


@register("agent_response")
def _fmt_agent_response(agent: str, data: dict, children: list) -> tuple[str, str]:
    text = data.get("text", "")
    return (f"Agent: {text}", text)


@register("session_start")
def _fmt_session_start(agent: str, data: dict, children: list) -> tuple[str, str]:
    sid = data.get("session_id", "")
    return ("Session started", f"Session ID: {sid}" if sid else "Session started")


@register("session_end")
def _fmt_session_end(agent: str, data: dict, children: list) -> tuple[str, str]:
    return ("Session ended", "Session ended")


# =====================================================================
# Tool lifecycle (orchestrator tools)
# =====================================================================

@register("data_fetched")
def _fmt_data_fetched(agent: str, data: dict, children: list) -> tuple[str, str]:
    args = data.get("args", {})
    label = _first_output(data)
    status = data.get("status", "success")
    error = data.get("error")

    if error or status == "error":
        summary = f"data fetch FAILED: {error or 'unknown error'}"
        details = f"Error: {error}\nLabel: {label or '(unknown)'}"
        return (summary, details)

    parts = [f"data_fetched -> {label}"]
    info_parts = []
    if data.get("n_pts"):
        info_parts.append(f"{data['n_pts']} pts")
    if data.get("n_cols"):
        info_parts.append(f"{data['n_cols']} cols")
    col_names = data.get("col_names")
    if col_names:
        info_parts.append(f"({', '.join(str(c) for c in col_names)})")
    if data.get("units"):
        info_parts.append(data["units"])
    time_start = data.get("time_start") or args.get("time_start", "")
    time_end = data.get("time_end") or args.get("time_end", "")
    if time_start and time_end:
        info_parts.append(f"{time_start} to {time_end}")

    if info_parts:
        parts.append(": " + ", ".join(info_parts))
    summary = "".join(parts)

    detail_lines = [f"Label: {label}"]
    if time_start:
        detail_lines.append(f"Time: {time_start} to {time_end}")
    if data.get("nan_pct") is not None:
        detail_lines.append(f"NaN%: {data['nan_pct']}")
    child_info = child_summaries(children)
    if child_info:
        detail_lines.append(f"Sub-events:\n{child_info}")
    details = "\n".join(detail_lines)

    return (summary, details)


@register("data_computed")
def _fmt_data_computed(agent: str, data: dict, children: list) -> tuple[str, str]:
    args = data.get("args", {})
    label = _first_output(data)
    status = data.get("status", "success")
    error = data.get("error")

    if error or status == "error":
        summary = f"run_code FAILED: {error or 'unknown error'}"
        code = args.get("code", "")
        sources = data.get("inputs", [])
        details = f"Error: {error}\nCode: {code}\nSources: {', '.join(str(s) for s in sources)}"
        return (summary, details)

    parts = [f"run_code -> {label}"]
    info = []
    if data.get("n_pts"):
        info.append(f"{data['n_pts']} pts")
    if data.get("units"):
        info.append(data["units"])
    sources = data.get("inputs", [])
    if sources:
        info.append(f"from [{', '.join(str(s) for s in sources)}]")
    if info:
        parts.append(": " + ", ".join(info))
    summary = "".join(parts)

    detail_lines = [f"Label: {label}"]
    code = args.get("code", "")
    if code:
        detail_lines.append(f"Code: {code}")
    if sources:
        detail_lines.append(f"Sources: {', '.join(str(s) for s in sources)}")
    details = "\n".join(detail_lines)

    return (summary, details)


@register("data_created")
def _fmt_data_created(agent: str, data: dict, children: list) -> tuple[str, str]:
    args = data.get("args", {})
    label = _first_output(data)
    n_pts = data.get("n_pts", "")
    pts_info = f": {n_pts} pts" if n_pts else ""
    summary = f"run_code -> {label}{pts_info}"

    detail_lines = [f"Label: {label}"]
    if args.get("code"):
        detail_lines.append(f"Code: {args['code']}")
    if args.get("description"):
        detail_lines.append(f"Description: {args['description']}")
    if data.get("units"):
        detail_lines.append(f"Units: {data['units']}")
    details = "\n".join(detail_lines)

    return (summary, details)


@register("render_executed")
def _fmt_render_executed(agent: str, data: dict, children: list) -> tuple[str, str]:
    status = data.get("status", "success")
    error = data.get("error")

    if error or status == "error":
        summary = f"render_plotly_json FAILED: {error or 'unknown error'}"
        details = f"Error: {error}"
        inputs = data.get("inputs", [])
        if inputs:
            details += f"\nAttempted labels: {', '.join(str(i) for i in inputs)}"
        return (summary, details)

    n_panels = data.get("n_panels", "(unknown)")
    inputs = data.get("inputs", [])
    label_str = ", ".join(str(i) for i in inputs) if inputs else "(none)"
    summary = f"render_plotly_json -> {n_panels} panel(s), traces: [{label_str}]"

    detail_lines = [f"Panels: {n_panels}", f"Data labels: {', '.join(str(i) for i in inputs) if inputs else '(none)'}"]
    if data.get("op_id"):
        detail_lines.append(f"Op ID: {data['op_id']}")
    details = "\n".join(detail_lines)

    return (summary, details)


@register("mpl_render_executed")
def _fmt_mpl_render_executed(agent: str, data: dict, children: list) -> tuple[str, str]:
    status = data.get("status", "success")
    error = data.get("error")

    if error or status == "error":
        summary = f"generate_mpl_script FAILED: {error or 'unknown error'}"
        details = f"Error: {error}"
        return (summary, details)

    inputs = data.get("inputs", [])
    label_str = ", ".join(str(i) for i in inputs) if inputs else "(none)"
    summary = f"generate_mpl_script -> traces: [{label_str}]"

    detail_lines = [f"Data labels: {label_str}"]
    if data.get("op_id"):
        detail_lines.append(f"Op ID: {data['op_id']}")
    details = "\n".join(detail_lines)

    return (summary, details)


@register("jsx_render_executed")
def _fmt_jsx_render_executed(agent: str, data: dict, children: list) -> tuple[str, str]:
    status = data.get("status", "success")
    error = data.get("error")

    if error or status == "error":
        summary = f"generate_jsx_component FAILED: {error or 'unknown error'}"
        details = f"Error: {error}"
        return (summary, details)

    inputs = data.get("inputs", [])
    label_str = ", ".join(str(i) for i in inputs) if inputs else "(none)"
    summary = f"generate_jsx_component -> traces: [{label_str}]"

    detail_lines = [f"Data labels: {label_str}"]
    if data.get("op_id"):
        detail_lines.append(f"Op ID: {data['op_id']}")
    details = "\n".join(detail_lines)

    return (summary, details)


@register("render_error")
def _fmt_render_error(agent: str, data: dict, children: list) -> tuple[str, str]:
    error = data.get("error", data.get("_msg", "unknown error"))
    summary = f"render_plotly_json FAILED: {error}"
    details = f"Error: {error}"
    return (summary, details)


@register("plot_action")
def _fmt_plot_action(agent: str, data: dict, children: list) -> tuple[str, str]:
    args = data.get("args", {})
    action = args.get("action", data.get("action", "(unknown action)"))
    summary = f"manage_plot -> {action}"
    detail_lines = [f"Action: {action}"]
    if args.get("filename"):
        detail_lines.append(f"Filename: {args['filename']}")
    if args.get("format"):
        detail_lines.append(f"Format: {args['format']}")
    details = "\n".join(detail_lines)
    return (summary, details)


@register("custom_op_failure")
def _fmt_custom_op_failure(agent: str, data: dict, children: list) -> tuple[str, str]:
    args = data.get("args", {})
    error = data.get("error", "unknown")
    code = args.get("code", "")
    sources = data.get("inputs", [])
    summary = f"run_code FAILED: {error}"
    detail_lines = [f"Error: {error}"]
    if code:
        detail_lines.append(f"Code: {code}")
    if sources:
        detail_lines.append(f"Sources: {', '.join(str(s) for s in sources)}")
    details = "\n".join(detail_lines)
    return (summary, details)


# =====================================================================
# Tool call/result (orchestrator-level)
# =====================================================================

@register("tool_call")
def _fmt_tool_call(agent: str, data: dict, children: list) -> tuple[str, str]:
    tool_name = data.get("tool_name", "(unknown tool)")
    tool_args = data.get("tool_args", {})
    # Build key args summary (first 3 simple args)
    key_args = []
    for k, v in list(tool_args.items())[:3]:
        key_args.append(f"{k}={v}")
    args_str = ", ".join(key_args)
    summary = f"Calling {tool_name}({args_str})"
    details = f"Tool: {tool_name}\nArgs: {tool_args}"
    return (summary, details)


@register("tool_result")
def _fmt_tool_result(agent: str, data: dict, children: list) -> tuple[str, str]:
    tool_name = data.get("tool_name", "(unknown tool)")
    status = data.get("status", "(unknown)")
    result = data.get("tool_result", {})
    brief = ""
    if isinstance(result, dict) and result.get("status") != "error":
        # Brief result info for actionable summaries
        if result.get("count") is not None:
            brief = f"{result['count']} entries"
        elif result.get("label"):
            brief = str(result["label"])
        elif result.get("labels"):
            brief = ", ".join(str(l) for l in result["labels"][:3])
        if result.get("num_points"):
            brief = f"{brief}, {result['num_points']} pts" if brief else f"{result['num_points']} pts"
    elif isinstance(result, dict) and result.get("status") == "error":
        brief = str(result.get("message", ""))
    summary = f"{tool_name} -> {status}"
    if brief:
        summary += f" ({brief})"
    details = f"Tool: {tool_name}\nArgs: {data.get('tool_args', {})}\nStatus: {status}"
    if brief:
        details += f"\nResult: {brief}"
    return (summary, details)


@register("tool_error")
def _fmt_tool_error(agent: str, data: dict, children: list) -> tuple[str, str]:
    tool_name = data.get("tool_name", "(unknown tool)")
    error = data.get("error", "unknown")
    summary = f"{tool_name} FAILED: {error}"
    details = f"Tool: {tool_name}\nError: {error}"
    return (summary, details)


@register("tool_call_log")
def _fmt_tool_call_log(agent: str, data: dict, children: list) -> tuple[str, str]:
    tool_name = data.get("tool_name", "(unknown tool)")
    summary = f"[log] Calling {tool_name}"
    return (summary, summary)


@register("tool_result_log")
def _fmt_tool_result_log(agent: str, data: dict, children: list) -> tuple[str, str]:
    tool_name = data.get("tool_name", "(unknown tool)")
    status = data.get("status", "(unknown)")
    summary = f"[log] {tool_name} -> {status}"
    return (summary, summary)


@register("tool_error_log")
def _fmt_tool_error_log(agent: str, data: dict, children: list) -> tuple[str, str]:
    tool_name = data.get("tool_name", "(unknown tool)")
    error = data.get("error", "unknown")
    summary = f"[log] {tool_name} FAILED: {error}"
    return (summary, summary)




@register("sub_agent_error")
def _fmt_sub_agent_error(agent: str, data: dict, children: list) -> tuple[str, str]:
    error = data.get("error", data.get("_msg", "unknown"))
    summary = f"{agent} ERROR: {error}"
    details = f"Agent: {agent}\nError: {error}"
    if data.get("tool_name"):
        details += f"\nTool: {data['tool_name']}"
    return (summary, details)


# =====================================================================
# Routing / delegation
# =====================================================================

@register("delegation")
def _fmt_delegation(agent: str, data: dict, children: list) -> tuple[str, str]:
    target = data.get("target", data.get("_msg", "(unknown agent)"))
    request_text = data.get("request", "")
    if request_text:
        summary = f"Delegating to {target}: {request_text}"
    else:
        summary = f"Delegating to {target}"
    details = f"Target: {target}"
    if request_text:
        details += f"\nRequest: {request_text}"
    child_info = child_summaries(children)
    if child_info:
        details += f"\nSub-events:\n{child_info}"
    return (summary, details)


@register("delegation_done")
def _fmt_delegation_done(agent: str, data: dict, children: list) -> tuple[str, str]:
    target = data.get("target", data.get("_msg", "(unknown agent)"))
    status = data.get("status", "")
    outcome = data.get("outcome", "")
    summary = f"{target} finished"
    if status:
        summary += f" ({status})"
    if outcome:
        summary += f": {outcome}"
    details = f"Target: {target}"
    if status:
        details += f"\nStatus: {status}"
    if outcome:
        details += f"\nOutcome: {outcome}"
    return (summary, details)


@register("delegation_async_started")
def _fmt_delegation_async_started(agent: str, data: dict, children: list) -> tuple[str, str]:
    target = data.get("target", "(unknown agent)")
    summary = f"Async: started {target}"
    return (summary, f"Target: {target}")


@register("delegation_async_completed")
def _fmt_delegation_async_completed(agent: str, data: dict, children: list) -> tuple[str, str]:
    target = data.get("target", "(unknown agent)")
    summary = f"Async: {target} completed"
    return (summary, f"Target: {target}")


# =====================================================================
# Data source sub-events (children of DATA_FETCHED span)
# =====================================================================

@register("high_nan")
def _fmt_high_nan(agent: str, data: dict, children: list) -> tuple[str, str]:
    pct = data.get("nan_pct", "(unknown)")
    label = data.get("label", "(no label)")
    summary = f"High NaN: {label} ({pct}%)"
    details = f"Label: {label}\nNaN percentage: {pct}%"
    return (summary, details)


# =====================================================================
# Planning
# =====================================================================

@register("plan_created")
def _fmt_plan_created(agent: str, data: dict, children: list) -> tuple[str, str]:
    plan = data.get("plan")
    if plan:
        tasks = plan.get("tasks", [])
        n = len(tasks)
        summary = f"Plan created: {n} tasks"
        lines = [data.get("_msg", summary)]
        reasoning = plan.get("reasoning", "")
        if reasoning:
            lines.append(f"Reasoning: {reasoning}")
        for i, t in enumerate(tasks, 1):
            desc = t.get("description", "?")
            mission = t.get("mission", "")
            instr = t.get("instruction", "")
            ds = t.get("candidate_datasets")
            ds_str = f" datasets=[{', '.join(ds)}]" if ds else ""
            lines.append(f"  {i}. [{mission}] {desc}{ds_str}")
            if instr:
                lines.append(f"     instruction: {instr}")
        details = "\n".join(lines)
    else:
        n = data.get("n_tasks", "(unknown)")
        summary = f"Plan created: {n} tasks"
        details = data.get("_msg", summary)
    return (summary, details)


@register("plan_task")
def _fmt_plan_task(agent: str, data: dict, children: list) -> tuple[str, str]:
    plan = data.get("plan")
    if plan:
        tasks = plan.get("tasks", [])
        round_num = data.get("round", "?")
        summary = f"Plan round {round_num}: {len(tasks)} tasks"
        lines = [summary]
        reasoning = plan.get("reasoning", "")
        if reasoning:
            lines.append(f"Reasoning: {reasoning}")
        for i, t in enumerate(tasks, 1):
            desc = t.get("description", "?")
            mission = t.get("mission", "")
            instr = t.get("instruction", "")
            ds = t.get("candidate_datasets")
            ds_str = f" datasets=[{', '.join(ds)}]" if ds else ""
            lines.append(f"  {i}. [{mission}] {desc}{ds_str}")
            if instr:
                lines.append(f"     instruction: {instr}")
        details = "\n".join(lines)
    else:
        task = data.get("task", data.get("_msg", "?"))
        summary = f"Plan task: {task}"
        details = str(task)
    return (summary, details)


@register("plan_completed")
def _fmt_plan_completed(agent: str, data: dict, children: list) -> tuple[str, str]:
    summary = "Plan completed"
    details = data.get("_msg", summary)
    return (summary, details)


# =====================================================================
# LLM
# =====================================================================

@register("thinking")
def _fmt_thinking(agent: str, data: dict, children: list) -> tuple[str, str]:
    text = data.get("text", data.get("_msg", ""))
    summary = f"Thinking: {text}"
    return (summary, text)


@register("llm_call")
def _fmt_llm_call(agent: str, data: dict, children: list) -> tuple[str, str]:
    msg = data.get("_msg", f"LLM call from {agent}")
    summary = msg
    details = f"Agent: {agent}\n{msg}"
    return (summary, details)


@register("llm_response")
def _fmt_llm_response(agent: str, data: dict, children: list) -> tuple[str, str]:
    msg = data.get("_msg", f"LLM response for {agent}")
    summary = msg
    details = f"Agent: {agent}\n{msg}"
    return (summary, details)


# =====================================================================
# Token usage
# =====================================================================

@register("token_usage")
def _fmt_token_usage(agent: str, data: dict, children: list) -> tuple[str, str]:
    a = data.get("agent_name", agent)
    inp = data.get("input_tokens", 0)
    out = data.get("output_tokens", 0)
    cum_in = data.get("cumulative_input", 0)
    cum_out = data.get("cumulative_output", 0)
    sys_t = data.get("system_tokens", 0)
    tools_t = data.get("tools_tokens", 0)
    hist_t = data.get("history_tokens", 0)

    if sys_t or tools_t:
        summary = f"Tokens: {a} in:{inp} (sys:{sys_t} tools:{tools_t} hist:{hist_t}) out:{out} (cum: {cum_in}/{cum_out})"
    else:
        summary = f"Tokens: {a} in:{inp} out:{out} (cum: {cum_in}/{cum_out})"

    detail_lines = [
        f"Agent: {a}",
        f"Input tokens: {inp}",
    ]
    if sys_t or tools_t:
        detail_lines.append(f"  System: {sys_t}")
        detail_lines.append(f"  Tools: {tools_t}")
        detail_lines.append(f"  History: {hist_t}")
    detail_lines.append(f"Output tokens: {out}")
    detail_lines.append(f"Cumulative input: {cum_in}")
    detail_lines.append(f"Cumulative output: {cum_out}")
    thinking = data.get("thinking_tokens")
    if thinking:
        detail_lines.append(f"Thinking tokens: {thinking}")
    cached = data.get("cached_tokens")
    if cached:
        detail_lines.append(f"Cached tokens: {cached}")
    details = "\n".join(detail_lines)
    return (summary, details)


# =====================================================================
# Memory
# =====================================================================

@register("memory_extraction_start")
def _fmt_memory_extraction_start(agent: str, data: dict, children: list) -> tuple[str, str]:
    n = data.get("n_events", "(unknown)")
    scopes = data.get("scopes", [])
    scope_str = ", ".join(str(s) for s in scopes) if scopes else "all"
    summary = f"Memory extraction started ({n} events, scopes: [{scope_str}])"
    details = f"Events: {n}\nScopes: {scope_str}"
    return (summary, details)


@register("memory_extraction_done")
def _fmt_memory_extraction_done(agent: str, data: dict, children: list) -> tuple[str, str]:
    actions = data.get("actions", {})
    add = actions.get("add", actions.get("added", 0))
    drop = actions.get("drop", actions.get("dropped", 0))
    edit = actions.get("edit", actions.get("edited", 0))
    summary = f"Memory extraction done: +{add} -{drop} ~{edit}"
    return (summary, summary)


@register("memory_extraction_error")
def _fmt_memory_extraction_error(agent: str, data: dict, children: list) -> tuple[str, str]:
    error = data.get("error", data.get("_msg", "unknown"))
    summary = f"Memory extraction error: {error}"
    return (summary, f"Error: {error}")


@register("memory_injected")
def _fmt_memory_injected(agent: str, data: dict, children: list) -> tuple[str, str]:
    count = data.get("memory_count", 0)
    scope = data.get("scope", "?")
    summary = f"Memory refreshed: {count} memories injected (scope={scope})"
    return (summary, summary)


@register("memory_action")
def _fmt_memory_action(agent: str, data: dict, children: list) -> tuple[str, str]:
    action = data.get("action", "(unknown action)")
    content = data.get("content", "")

    if action == "add":
        mtype = data.get("type", "(unknown type)")
        summary = f"Memory add: {mtype} -- {content}"
        return (summary, f"Action: add\nType: {mtype}\nContent: {content}")
    elif action == "edit":
        mem_id = data.get("id", "(unknown id)")
        summary = f"Memory edit: {mem_id} -- {content}"
        return (summary, f"Action: edit\nID: {mem_id}\nContent: {content}")
    elif action == "drop":
        mem_id = data.get("id", "(unknown id)")
        summary = f"Memory drop: {mem_id}"
        return (summary, f"Action: drop\nID: {mem_id}")
    elif action == "discard_pipeline":
        render_op_id = data.get("render_op_id", "(unknown id)")
        summary = f"Memory discard pipeline: {render_op_id}"
        return (summary, f"Action: discard_pipeline\nRender Op ID: {render_op_id}")
    else:
        # Fallback for unknown actions — use the raw msg if available
        msg = data.get("_msg", f"Memory {action}")
        return (msg, msg)


@register("memory_summary")
def _fmt_memory_summary(agent: str, data: dict, children: list) -> tuple[str, str]:
    text = data.get("text", data.get("_msg", ""))
    summary = f"Memory summary: {text[:100]}" if len(text) > 100 else f"Memory summary: {text}"
    return (summary, text)


@register("pipeline_registered")
def _fmt_pipeline_registered(agent: str, data: dict, children: list) -> tuple[str, str]:
    summary = data.get("_msg", "Pipeline registered")
    return (summary, summary)


# =====================================================================
# Insight
# =====================================================================

@register("insight_feedback")
def _fmt_insight_feedback(agent: str, data: dict, children: list) -> tuple[str, str]:
    passed = data.get("passed", True)
    text = data.get("text", data.get("_msg", ""))
    status = "pass" if passed else "fail"
    summary = f"Insight feedback: {status}"
    if text:
        summary += f" -- {text}"
    return (summary, text or summary)


# =====================================================================
# Context compaction (message-level)
# =====================================================================

@register("context_compaction")
def _fmt_context_compaction(agent: str, data: dict, children: list) -> tuple[str, str]:
    before = data.get("before_tokens", "?")
    after = data.get("after_tokens", "?")
    window = data.get("context_window", "?")
    summary = f"Context compacted: {before}\u2192{after}/{window} tokens"
    details = f"Agent: {agent}\nBefore: {before} tokens\nAfter: {after} tokens\nContext window: {window} tokens"
    return (summary, details)


# =====================================================================
# Control center (turnless orchestrator)
# =====================================================================

@register("work_registered")
def _fmt_work_registered(agent: str, data: dict, children: list) -> tuple[str, str]:
    _msg = data.get("_msg", "Work registered")
    summary = _msg
    return (summary, _msg)


@register("work_cancelled")
def _fmt_work_cancelled(agent: str, data: dict, children: list) -> tuple[str, str]:
    _msg = data.get("_msg", "Work cancelled")
    summary = _msg
    return (summary, _msg)


@register("user_amendment")
def _fmt_user_amendment(agent: str, data: dict, children: list) -> tuple[str, str]:
    _msg = data.get("_msg", "User amendment")
    summary = _msg
    return (summary, _msg)


@register("round_start")
def _fmt_round_start(agent: str, data: dict, children: list) -> tuple[str, str]:
    return ("Round started", "Round started")


@register("round_end")
def _fmt_round_end(agent: str, data: dict, children: list) -> tuple[str, str]:
    usage = data.get("round_token_usage", {})
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    summary = f"Round complete (in:{inp} out:{out})"
    return (summary, summary)


# =====================================================================
# Other / catch-all
# =====================================================================

@register("progress")
def _fmt_progress(agent: str, data: dict, children: list) -> tuple[str, str]:
    msg = data.get("_msg", "")
    return (msg if msg else "Progress", msg)


@register("debug")
def _fmt_debug(agent: str, data: dict, children: list) -> tuple[str, str]:
    msg = data.get("_msg", "")
    return (msg if msg else "Debug", msg)


@register("recovery")
def _fmt_recovery(agent: str, data: dict, children: list) -> tuple[str, str]:
    msg = data.get("_msg", "")
    return (msg if msg else "Recovery", msg)


# =====================================================================
# Eureka (scientific findings)
# =====================================================================

@register("eureka_extraction_start")
def _fmt_eureka_extraction_start(agent: str, data: dict, children: list) -> tuple[str, str]:
    summary = "Eureka extraction started"
    return (summary, summary)


@register("eureka_extraction_done")
def _fmt_eureka_extraction_done(agent: str, data: dict, children: list) -> tuple[str, str]:
    n = data.get("n_findings", 0)
    summary = f"Eureka extraction done: {n} finding(s)"
    return (summary, summary)


@register("eureka_extraction_error")
def _fmt_eureka_extraction_error(agent: str, data: dict, children: list) -> tuple[str, str]:
    error = data.get("error", data.get("_msg", "unknown"))
    summary = f"Eureka extraction error: {error}"
    return (summary, f"Error: {error}")


@register("eureka_finding")
def _fmt_eureka_finding(agent: str, data: dict, children: list) -> tuple[str, str]:
    title = data.get("title", "(untitled)")
    confidence = data.get("confidence", "?")
    summary = f"[Eureka] {title} (confidence: {confidence})"
    details = f"Title: {title}\nObservation: {data.get('observation', '')}\nHypothesis: {data.get('hypothesis', '')}\nConfidence: {confidence}"
    return (summary, details)


# =============================================================================
# Registry protocol adapter
# =============================================================================


class _EventFormatterRegistryAdapter:
    name = "events.formatters"
    description = "Per-event-type formatter functions for EventBus.span()"

    def get(self, key: str):
        return FORMATTERS.get(key)

    def list_all(self) -> dict:
        return dict(FORMATTERS)


EVENT_FORMATTER_REGISTRY = _EventFormatterRegistryAdapter()
from agent.registry_protocol import register_registry  # noqa: E402
register_registry(EVENT_FORMATTER_REGISTRY)
