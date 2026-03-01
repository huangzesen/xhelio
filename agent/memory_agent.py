"""
MemoryAgent — think-then-act memory extraction with full session context.

Single MemoryAgent sees the full session context (curated EventBus events
and active memories) and outputs concrete actions (add/edit/drop).
Consolidation happens organically — the agent merges entries as part of normal
operation. Only memories for active scopes are loaded.

Called periodically during the session via _maybe_extract_memories().
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import config
from .llm import LLMAdapter
from .memory import Memory, MemoryStore, generate_tags, MEMORY_TOKEN_BUDGET
from .token_counter import count_tokens as estimate_tokens
from .model_fallback import get_active_model
from .event_bus import EventBus, get_event_bus, MEMORY_ACTION, TOKEN_USAGE, PIPELINE_REGISTERED
from .truncation import trunc, trunc_items

# Valid scope pattern
_VALID_SCOPE_RE = re.compile(r"^(generic|visualization|data_ops|mission:\w+)$")

# Valid types
_VALID_TYPES = {"preference", "summary", "pitfall", "reflection"}


# ---- Priority-based memory event curation registry ----

CURATED_EVENTS_TOKEN_BUDGET = 50_000

P0_CRITICAL = 0  # Errors & failures
P1_HIGH = 1      # User intent
P2_MEDIUM = 2    # Outcomes
P3_LOW = 3       # Routing & context
P4_FILL = 4      # Fill-in: tool lifecycle, data bookkeeping


@dataclass(frozen=True)
class MemoryCurationEntry:
    """Registry entry controlling how an event type is curated for memory."""
    event_type: str
    priority: int
    format_fn: Callable[[dict], str]
    filter_fn: Callable[[dict], bool] | None = None


# ---- Formatter functions ----

def _fmt_sub_agent_error(ev: dict) -> str:
    agent_name = ev.get("agent", "?")
    tool_name = ev.get("tool_name", "")
    error = ev.get("error", "")
    if tool_name and error:
        return f"  [{agent_name}] ERROR in {tool_name}: {error}"
    msg = ev.get("msg", ev.get("_msg", ""))
    if msg:
        return f"  [{agent_name}] ERROR: {msg}"
    if error:
        return f"  [{agent_name}] ERROR: {error}"
    return ""


def _fmt_tool_error(ev: dict) -> str:
    # Shape 1: from _execute_tool_safe — {"tool_name", "error"}
    tool_name = ev.get("tool_name", "")
    error = ev.get("error", "")
    if tool_name and error:
        return f"  {tool_name} ERROR: {error}"
    # Shape 2: from log_error() — {"short", "context"}
    short = ev.get("short", "")
    context = ev.get("context", "")
    if short:
        line = f"  tool_error: {short}"
        if context:
            line += f"\n    context: {context}"
        return line
    if error:
        return f"  tool_error: {error}"
    msg = ev.get("msg", ev.get("_msg", ""))
    if msg:
        return f"  tool_error: {msg}"
    return ""


def _fmt_custom_op_failure(ev: dict) -> str:
    args = ev.get("args", {})
    desc = args.get("description", "?")
    code = args.get("code", "")
    error = ev.get("error", "")
    line = f"  custom_operation({desc}) -> FAILED"
    if error:
        line += f": {error}"
    if code:
        line += f"\n    code: {code}"
    return line


def _fmt_user_message(ev: dict) -> str:
    text = ev.get("text", ev.get("msg", ev.get("_msg", "")))
    if text:
        return f"  [User] {text}"
    return ""


def _fmt_user_amendment(ev: dict) -> str:
    text = ev.get("text", ev.get("msg", ev.get("_msg", "")))
    if text:
        return f"  [User amendment] {text}"
    return ""


def _fmt_work_cancelled(ev: dict) -> str:
    msg = ev.get("msg", ev.get("_msg", ""))
    if msg:
        return f"  [Work cancelled] {msg}"
    scope = ev.get("scope", "")
    count = ev.get("count", "")
    if scope or count:
        return f"  [Work cancelled] scope={scope} count={count}"
    return ""


def _fmt_agent_response(ev: dict) -> str:
    text = ev.get("text", ev.get("msg", ev.get("_msg", "")))
    if text:
        return f"  [Agent] {text}"
    return ""


def _fmt_data_fetched(ev: dict) -> str:
    args = ev.get("args", {})
    ds = args.get("dataset_id", "?")
    param = args.get("parameter_id", "?")
    status = ev.get("status", "")
    error = ev.get("error", "")
    nan_pct = ev.get("nan_percentage", 0)
    line = f"  fetch_data({ds}/{param}) -> {status}"
    if error:
        line += f" ERROR: {error}"
    if nan_pct and nan_pct > 25:
        line += f" [NaN: {nan_pct:.0f}%]"
    return line


def _fmt_data_computed(ev: dict) -> str:
    args = ev.get("args", {})
    desc = args.get("description", "?")
    label = args.get("output_label", "?")
    code = args.get("code", "")
    status = ev.get("status", "")
    error = ev.get("error", "")
    outputs = ev.get("outputs", [])
    line = f"  custom_operation({desc}) -> {label} [{status}]"
    if error:
        line += f" ERROR: {error}"
    if code:
        line += f"\n    code: {code}"
    if outputs:
        line += f" [outputs: {', '.join(str(o) for o in outputs)}]"
    return line


def _fmt_render_executed(ev: dict) -> str:
    status = ev.get("status", "")
    error = ev.get("error", "")
    args = ev.get("args", {})
    figure_json = args.get("figure_json", "")
    line = f"  render_plotly_json -> {status}"
    if error:
        line += f" ERROR: {error}"
    if figure_json:
        fj_str = json.dumps(figure_json, default=str) if isinstance(figure_json, dict) else str(figure_json)
        line += f"\n    figure_json: {fj_str}"
    return line


def _fmt_insight_feedback(ev: dict) -> str:
    text = ev.get("text", ev.get("msg", ev.get("_msg", "")))
    passed = ev.get("passed", True)
    verdict = "PASS" if passed else "NEEDS_IMPROVEMENT"
    if text:
        return f"  [Figure Review] {verdict}: {text}"
    return f"  [Figure Review] {verdict}"


def _fmt_thinking(ev: dict) -> str:
    agent_name = ev.get("agent", "")
    text = ev.get("text", ev.get("msg", ev.get("_msg", "")))
    if not text:
        return ""
    prefix = f"  [{agent_name} thinking]" if agent_name else "  [Thinking]"
    return f"{prefix} {text}"


def _fmt_delegation(ev: dict) -> str:
    agent_name = ev.get("agent", "")
    msg = ev.get("msg", ev.get("_msg", ""))
    return f"  delegation({agent_name}): {msg}"


def _fmt_delegation_done(ev: dict) -> str:
    agent_name = ev.get("agent", "")
    msg = ev.get("msg", ev.get("_msg", ""))
    return f"  delegation_done({agent_name}): {msg}"


def _fmt_delegation_async_completed(ev: dict) -> str:
    tool = ev.get("tool_name", ev.get("tool", "?"))
    work_unit_id = ev.get("work_unit_id", "?")
    return f"  async_completed({tool}): {work_unit_id}"


def _fmt_tool_call(ev: dict) -> str:
    tool_name = ev.get("tool_name", "?")
    tool_args = ev.get("tool_args", {})
    args_str = json.dumps(tool_args, default=str) if tool_args else ""
    if args_str:
        return f"  [Tool Call] {tool_name}({args_str})"
    return f"  [Tool Call] {tool_name}()"


def _fmt_tool_result(ev: dict) -> str:
    tool_name = ev.get("tool_name", "?")
    status = ev.get("status", "")
    return f"  [Tool Result] {tool_name} -> {status}"


def _fmt_sub_agent_tool(ev: dict) -> str:
    agent_name = ev.get("agent", "?")
    tool_name = ev.get("tool_name", "?")
    tool_result = ev.get("tool_result", {})
    status = tool_result.get("status", "") if isinstance(tool_result, dict) else ""
    return f"  [{agent_name}] {tool_name} -> {status}"


def _fmt_data_created(ev: dict) -> str:
    args = ev.get("args", {})
    desc = args.get("description", "?")
    status = ev.get("status", "")
    outputs = ev.get("outputs", [])
    line = f"  store_dataframe({desc}) -> {status}"
    if outputs:
        line += f" [outputs: {', '.join(str(o) for o in outputs)}]"
    return line


def _fmt_plot_action(ev: dict) -> str:
    args = ev.get("args", {})
    action = args.get("action", "?")
    status = ev.get("status", "")
    return f"  manage_plot({action}) -> {status}"


def _fmt_catchall(ev: dict) -> str:
    """Generic formatter for any event type not explicitly registered."""
    event_type = ev.get("event", "?")
    agent_name = ev.get("agent", "")
    msg = ev.get("msg", ev.get("_msg", ""))
    status = ev.get("status", "")
    error = ev.get("error", "")
    parts = [f"  [{event_type}]"]
    if agent_name:
        parts.append(f"({agent_name})")
    if msg:
        parts.append(msg)
    if status:
        parts.append(f"-> {status}")
    if error:
        parts.append(f"ERROR: {error}")
    line = " ".join(parts)
    # Append any data keys that look informative (skip internal/huge ones)
    _SKIP_KEYS = {"event", "agent", "msg", "_msg", "status", "error", "args",
                  "figure_json", "tool_args", "tool_result", "tags"}
    extras = {k: v for k, v in ev.items() if k not in _SKIP_KEYS and v}
    if extras:
        line += f" {json.dumps(extras, default=str)}"
    return line


# ---- Filter functions ----

def _filter_data_fetched(ev: dict) -> bool:
    """Return False for already_loaded and routine success with NaN <= 25%."""
    args = ev.get("args", {})
    if args.get("already_loaded"):
        return False
    status = ev.get("status", "")
    error = ev.get("error", "")
    if status == "success" and not error:
        nan_pct = ev.get("nan_percentage", 0)
        if not nan_pct or nan_pct <= 25:
            return False
    return True


# ---- Registry ----

MEMORY_CURATION_REGISTRY: list[MemoryCurationEntry] = [
    # P0: Errors
    MemoryCurationEntry("sub_agent_error",   P0_CRITICAL, _fmt_sub_agent_error),
    MemoryCurationEntry("tool_error",        P0_CRITICAL, _fmt_tool_error),
    MemoryCurationEntry("custom_op_failure", P0_CRITICAL, _fmt_custom_op_failure),
    # P1: User intent
    MemoryCurationEntry("user_message",      P1_HIGH, _fmt_user_message),
    MemoryCurationEntry("user_amendment",    P1_HIGH, _fmt_user_amendment),
    MemoryCurationEntry("work_cancelled",    P1_HIGH, _fmt_work_cancelled),
    # P2: Outcomes & reasoning
    MemoryCurationEntry("thinking",          P2_MEDIUM, _fmt_thinking),
    MemoryCurationEntry("agent_response",    P2_MEDIUM, _fmt_agent_response),
    MemoryCurationEntry("data_fetched",      P1_HIGH, _fmt_data_fetched, _filter_data_fetched),
    MemoryCurationEntry("data_computed",     P1_HIGH, _fmt_data_computed),
    MemoryCurationEntry("render_executed",   P1_HIGH, _fmt_render_executed),
    MemoryCurationEntry("insight_feedback",  P2_MEDIUM, _fmt_insight_feedback),
    # P3: Routing
    MemoryCurationEntry("delegation",                P3_LOW, _fmt_delegation),
    MemoryCurationEntry("delegation_done",           P3_LOW, _fmt_delegation_done),
    MemoryCurationEntry("delegation_async_completed", P3_LOW, _fmt_delegation_async_completed),
    # P4: Fill-in (tool lifecycle, data bookkeeping)
    MemoryCurationEntry("tool_call",       P4_FILL, _fmt_tool_call),
    MemoryCurationEntry("tool_result",     P4_FILL, _fmt_tool_result),
    MemoryCurationEntry("sub_agent_tool",  P4_FILL, _fmt_sub_agent_tool),
    MemoryCurationEntry("data_created",    P4_FILL, _fmt_data_created),
    MemoryCurationEntry("plot_action",     P4_FILL, _fmt_plot_action),
]

_CURATION_INDEX: dict[str, MemoryCurationEntry] = {e.event_type: e for e in MEMORY_CURATION_REGISTRY}
MEMORY_RELEVANT_TYPES: set[str] = {e.event_type for e in MEMORY_CURATION_REGISTRY}

# Number of recent raw events that get their own budget half
RECENT_EVENTS_WINDOW = 300


def _fill_budget(
    candidates: list[tuple[int, int, str]],
    budget: int,
) -> tuple[list[tuple[int, str]], int]:
    """Fill a token budget from a list of (priority, index, text) candidates.

    Groups by priority tier, fills greedily within each tier (chronological
    order). When an event doesn't fit, breaks from that tier and tries the next.

    Returns (accepted, used_tokens) where accepted is list of (original_index, text).
    """
    tiers: dict[int, list[tuple[int, str]]] = {}
    for priority, idx, text in candidates:
        tiers.setdefault(priority, []).append((idx, text))

    accepted: list[tuple[int, str]] = []
    used_tokens = 0

    for priority in sorted(tiers.keys()):
        for idx, text in tiers[priority]:
            tokens = estimate_tokens(text)
            if used_tokens + tokens > budget:
                break
            used_tokens += tokens
            accepted.append((idx, text))

    return accepted, used_tokens


def _validate_scopes(scopes_raw) -> list[str]:
    """Normalize and validate a scopes value (str or list) → list[str]."""
    if isinstance(scopes_raw, str):
        scopes_raw = [scopes_raw]
    if not isinstance(scopes_raw, list):
        return ["generic"]
    valid = [s for s in scopes_raw if isinstance(s, str) and _VALID_SCOPE_RE.match(s)]
    return valid or ["generic"]


@dataclass
class MemoryContext:
    """Full session context for the MemoryAgent."""
    events: list[dict]              # All memory-tagged EventBus events (curated)
    active_memories: list[dict]     # [{id, type, scopes, content}, ...]
    active_scopes: list[str]
    token_budget: int = MEMORY_TOKEN_BUDGET
    total_memory_tokens: int = 0
    pipeline_candidates: list[dict] = field(default_factory=list)  # Pipeline candidates for curation


class MemoryAgent:
    """Think-then-act memory extractor.

    Sees full session context every time, reasons about what changed,
    and outputs concrete actions (add/edit/drop) on scoped memories.

    Usage::

        agent = MemoryAgent(adapter, model_name, memory_store)
        actions = agent.run(context)  # returns list of executed actions
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        memory_store: MemoryStore,
        verbose: bool = False,
        session_id: str = "",
        event_bus: Optional[EventBus] = None,
        pipeline_store=None,
    ):
        self.adapter = adapter
        self.model_name = model_name
        self.memory_store = memory_store
        self.verbose = verbose
        self.session_id = session_id
        self._bus = event_bus or get_event_bus()
        self._pipeline_store = pipeline_store

        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._total_cached_tokens = 0
        self._api_calls = 0

    def get_token_usage(self) -> dict:
        """Return cumulative token usage for this agent."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "thinking_tokens": self._total_thinking_tokens,
            "cached_tokens": self._total_cached_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens + self._total_thinking_tokens,
            "api_calls": self._api_calls,
            "ctx_system_tokens": 0,
            "ctx_tools_tokens": 0,
            "ctx_history_tokens": 0,
            "ctx_total_tokens": 0,
        }

    def _track_usage(self, response):
        """Accumulate token usage from an LLMResponse."""
        usage = response.usage
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._total_thinking_tokens += usage.thinking_tokens
        self._total_cached_tokens += usage.cached_tokens
        self._api_calls += 1

        self._bus.emit(
            TOKEN_USAGE,
            agent="MemoryAgent",
            level="debug",
            msg=(
                f"[Tokens] MemoryAgent in:{usage.input_tokens} "
                f"out:{usage.output_tokens}"
            ),
            data={
                "agent_name": "MemoryAgent",
                "tool_context": "memory_extraction",
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "thinking_tokens": usage.thinking_tokens,
                "cached_tokens": usage.cached_tokens,
                "cumulative_input": self._total_input_tokens,
                "cumulative_output": self._total_output_tokens,
                "cumulative_thinking": self._total_thinking_tokens,
                "cumulative_cached": self._total_cached_tokens,
                "api_calls": self._api_calls,
            },
        )

    def run(self, context: MemoryContext) -> list[dict]:
        """Think-then-act: analyze full context, return executed actions."""
        if not context.events:
            return []  # Nothing to analyze

        try:
            prompt = self._build_prompt(context)
            actual_model = get_active_model(self.model_name)
            response = self.adapter.generate(
                model=actual_model,
                contents=prompt,
                temperature=0.2,
            )
            self._track_usage(response)
            text = (response.text or "").strip()
            actions = self._parse_actions(text)
            executed = self._execute_actions(actions)
            return executed
        except Exception as e:
            self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                           msg=f"[MemoryAgent] run() failed: {e}")
            return []

    # ---- Curated events builder ----

    @staticmethod
    def build_curated_events(
        raw_events: list[dict],
        token_budget: int = CURATED_EVENTS_TOKEN_BUDGET,
    ) -> list[str]:
        """Filter and format EventBus events for the memory agent LLM prompt.

        Uses the priority-based MEMORY_CURATION_REGISTRY to decide which events
        to include. Events are formatted by their registered formatter, filtered
        by their optional filter function, and fit into a token budget ordered
        by priority (P0 Critical → P3 Low).

        The budget is split 50/50 between older events and recent events.
        The last RECENT_EVENTS_WINDOW (300) raw events get half the budget;
        everything before that gets the other half. This ensures recent
        activity is always well-represented even in long sessions. Each
        half is filled independently using the same priority logic.

        No truncation is applied — full text for all fields. Known-huge fields
        (figure_json) are excluded at the formatter level.

        Args:
            raw_events: Raw event dicts (keys: event, agent, msg, plus data fields).
            token_budget: Maximum estimated tokens for the curated output.

        Returns:
            Chronological list of formatted event strings that fit the budget.
        """
        # Pass 1: Look up registry entry → filter → format → collect (priority, index, text)
        # Events with a registered entry use their priority/formatter.
        # Unregistered event types fall through to _fmt_catchall at P4.
        candidates: list[tuple[int, int, str]] = []  # (priority, original_index, text)
        for idx, ev in enumerate(raw_events):
            event_type = ev.get("event", "")
            entry = _CURATION_INDEX.get(event_type)
            if entry is not None:
                # Apply filter if present
                if entry.filter_fn is not None and not entry.filter_fn(ev):
                    continue
                text = entry.format_fn(ev)
                priority = entry.priority
            else:
                text = _fmt_catchall(ev)
                priority = P4_FILL
            if not text:
                continue
            candidates.append((priority, idx, text))

        if not candidates:
            return []

        # Pass 2: Split by raw event index — last 300 raw events get their own budget
        # Old half fills first; unused tokens roll into the recent half.
        half_budget = token_budget // 2
        recent_cutoff = max(0, len(raw_events) - RECENT_EVENTS_WINDOW)

        old_half = [(p, i, t) for p, i, t in candidates if i < recent_cutoff]
        recent_half = [(p, i, t) for p, i, t in candidates if i >= recent_cutoff]

        old_accepted, old_used = _fill_budget(old_half, half_budget)
        recent_budget = half_budget + (half_budget - old_used)
        recent_accepted, _ = _fill_budget(recent_half, recent_budget)

        # Pass 3: Merge and re-sort by original index → chronological order
        accepted = old_accepted + recent_accepted
        accepted.sort(key=lambda x: x[0])

        return [text for _, text in accepted]

    # ---- Prompt building ----

    def _build_prompt(self, context: MemoryContext) -> str:
        """Build the single LLM prompt with full session context."""
        sections = []

        # Active memories with full context: reviews, version history, access stats
        injected_ids = self.memory_store._last_injected_ids
        if context.active_memories:
            sections.append("## Current Memories")
            for m in context.active_memories:
                mid = m.get("id", "?")
                mtype = m.get("type", "?")
                scopes = m.get("scopes", ["generic"])
                content = m.get("content", "")
                version = m.get("version", 1)
                access_count = m.get("access_count", 0)
                created = m.get("created_at", "")[:10]  # date only
                injected_tag = " [INJECTED]" if mid in injected_ids else ""

                header = f"  [{mid}] ({mtype}, {scopes}, v{version}, used {access_count}x, created {created}){injected_tag}"
                sections.append(f"{header}\n    {content}")

                # Reviews from consuming agents
                reviews = m.get("reviews", [])
                if reviews:
                    for rv in reviews:
                        agent = rv.get("agent", "unknown")
                        feedback = rv.get("feedback", "")
                        rv_date = rv.get("date", "")[:10]
                        sections.append(f"    Review by {agent} ({rv_date}): {feedback}")

                # Previous versions (edit history)
                prev_versions = m.get("previous_versions", [])
                if prev_versions:
                    for pv in prev_versions:
                        pv_ver = pv.get("version", "?")
                        pv_content = pv.get("content", "")
                        pv_date = pv.get("date", "")[:10]
                        sections.append(f"    Previous v{pv_ver} ({pv_date}): {pv_content}")
        else:
            sections.append("## Current Memories\n  (none)")

        # Chronological session activity (conversation + ops + routing interleaved)
        if context.events:
            sections.append("\n## Session Activity")
            for line in context.events:
                sections.append(line)

        # Pipeline candidates section
        if context.pipeline_candidates:
            sections.append("\n## Pipeline Candidates")
            sections.append(
                "The following data pipelines have not yet been curated.\n"
                "Decide which are worth registering as reusable pipelines."
            )
            for cand in context.pipeline_candidates:
                op_id = cand.get("render_op_id", "?")
                step_count = cand.get("step_count", 0)
                vanilla = "vanilla" if cand.get("is_vanilla") else "non-vanilla"
                scopes_str = ", ".join(cand.get("scopes", [])) or "unknown"
                sections.append(
                    f"\n### Candidate [{op_id}] — {step_count} steps, "
                    f"{vanilla}, scopes: {scopes_str}"
                )
                for i, step in enumerate(cand.get("steps", []), 1):
                    tool = step.get("tool", "?")
                    parts = [f"  {i}. {tool}:"]
                    if tool == "fetch_data":
                        ds = step.get("dataset_id", "")
                        param = step.get("parameter_id", "")
                        label = step.get("output_label", "")
                        parts.append(f"{ds}.{param}" if param else ds)
                        if label:
                            parts.append(f'→ "{label}"')
                    elif tool in ("custom_operation", "store_dataframe"):
                        desc = step.get("description", "")
                        code = step.get("code", "")
                        label = step.get("output_label", "")
                        if desc:
                            parts.append(desc)
                        if code:
                            parts.append(f"(code: {code})")
                        if label:
                            parts.append(f'→ "{label}"')
                    elif tool == "render_plotly_json":
                        inputs = step.get("inputs", [])
                        parts.append(f"inputs={inputs}")
                    elif tool == "manage_plot":
                        action = step.get("action", "")
                        parts.append(action)
                    else:
                        label = step.get("output_label", "")
                        if label:
                            parts.append(f'→ "{label}"')
                    sections.append(" ".join(parts))

                # Show existing feedback from the saved pipeline file (if registered)
                feedback = cand.get("feedback", [])
                if feedback:
                    sections.append("  User feedback:")
                    for fb in feedback:
                        fb_date = fb.get("timestamp", "")[:10]
                        fb_comment = fb.get("comment", "")
                        sections.append(f'  - "{fb_comment}" ({fb_date})')

            # Inject valid mission IDs for scopes
            from knowledge.mission_prefixes import get_all_canonical_ids
            valid_missions = ", ".join(get_all_canonical_ids())
            sections.append(
                f'\nValid mission IDs for scopes (use "mission:<ID>" format):\n'
                f"{valid_missions}"
            )

        # Token budget warning
        if context.total_memory_tokens > context.token_budget * 0.8:
            pct = context.total_memory_tokens / context.token_budget * 100
            sections.append(
                f"\n⚠ Memory is at {pct:.0f}% of token budget ({context.total_memory_tokens}/{context.token_budget}). "
                f"Consider dropping or consolidating less useful entries."
            )

        context_block = "\n".join(sections)

        return f"""You are a memory management agent. Analyze the session context below and decide what memories to add, edit, or drop.

Active scopes: {context.active_scopes}

{context_block}

Respond with a JSON array of actions. Each action is an object:
- {{"action": "add", "type": "<preference|pitfall|reflection|summary>", "scopes": ["<scope>", ...], "content": "<text>"}}
- {{"action": "edit", "id": "<memory_id>", "content": "<updated text>"}}
- {{"action": "drop", "id": "<memory_id>"}}
- {{"action": "register_pipeline", "render_op_id": "<op_id>", "name": "<pipeline name>", "description": {{"source": "<what the user was doing>", "rationale": "<why this is worth saving>", "use_cases": "<what analyses this enables>"}}, "scopes": ["mission:<ID>", ...], "tags": ["<keyword>", ...]}}
- {{"action": "discard_pipeline", "render_op_id": "<op_id>"}}

Content format per type:
- **preference**: 1-2 sentences capturing a user habit or style choice.
- **pitfall**: Use this structure:
    Trigger: <what situation or action caused the issue>
    Problem: <what went wrong>
    Fix: <how to avoid or resolve it>
- **reflection**: Use this structure:
    Trigger: <what situation or action caused the issue>
    Problem: <what went wrong>
    Fix: <how to avoid or resolve it>
- **summary**: Use this structure:
    Data: <what datasets/instruments were used>
    Analysis: <what was done>
    Finding: <key results or observations>

Rules:
- Only add genuinely new information not already captured in Current Memories
- Use "edit" when an existing memory needs minor updates
- Valid scopes: "generic", "visualization", "data_ops", "mission:<ID>" (uppercase mission ID)
- Each entry can have multiple scopes when it spans domains
- Return empty array [] if nothing worth remembering
- Sub-agent errors (lines with "[AgentName] ERROR") are strong candidates for "pitfall" memories — especially API constraints, invalid parameters, coordinate frame issues, or other recurring failure patterns that agents should avoid in future sessions

Pipeline rules (only relevant when Pipeline Candidates are listed above):
- Only register pipelines that represent reusable, non-trivial workflows
- Pipelines marked "vanilla" (simple fetch+render, no transforms) are usually NOT worth registering
- If a pipeline is NOT worth registering, use "discard_pipeline" to mark it so it won't be shown again
- "scopes" must use "mission:<ID>" format with IDs from the valid missions list above
- "tags" are free-form keywords: analysis techniques, phenomena, data types (e.g., "magnetic-field", "spectral-analysis", "time-series")
- "description" must be a JSON object with three keys:
  - "source": What the user was doing when this pipeline was created (1 sentence)
  - "rationale": Why this is worth registering — what non-trivial processing it performs (1-2 sentences)
  - "use_cases": What scientific questions or analyses this pipeline enables (1-2 sentences)
- You MUST decide on each candidate: either register_pipeline or discard_pipeline — do not leave candidates unprocessed

Consolidation policy — BE CONSERVATIVE:
- **Default stance: preserve existing memories.** Do not drop or merge unless there is strong evidence.
- Only "drop" + "add" (merge) when entries are **clearly redundant** (nearly identical content covering the same topic).
- Only "edit" or "drop" a memory when one of these conditions is met:
  1. The memory's **current version** has collected **3+ reviews** and the evidence clearly supports a change, OR
  2. The memory's **current version** has **terrible reviews** (average ≤ 2 stars) indicating it is actively harmful.
- If a memory has few or no reviews on its current version, **leave it alone** — it hasn't been tested enough to judge.
- When in doubt, keep entries separate rather than merging them. Granularity is preferred over compression.

Using reviews and history:
- Each memory may have **reviews** from consuming agents (star ratings 1-5 with structured comments). Star meanings: 5=prevented mistake, 4=useful context, 3=relevant but no impact, 2=irrelevant, 1=misleading. Comments may include criticism and suggestions.
- Reviews shown are the **10 most recent** across the memory's entire version history. This prevents bias from outdated feedback on earlier versions and reflects current agent sentiment.
- Pay attention to which reviews are for the **current version** vs older versions — only current-version reviews should drive edit/drop decisions.
- Each memory may have **previous versions** showing how it evolved over past edits.
- **access_count** shows how often a memory was injected into agent prompts.
- Use all of the above to inform your add/edit/drop decisions.

Respond with JSON array only, no markdown fencing."""

    # ---- Response parsing ----

    def _parse_actions(self, text: str) -> list[dict]:
        """Parse JSON actions from LLM response."""
        if not text:
            return []
        # Strip markdown fencing
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3].strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return []
        except (json.JSONDecodeError, ValueError) as e:
            self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                           msg=f"[MemoryAgent] Failed to parse actions: {e}")
            return []

    # ---- Action execution ----

    def _execute_actions(self, actions: list[dict]) -> list[dict]:
        """Validate and execute a list of action dicts. Returns executed actions."""
        executed = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_type = action.get("action", "")
            try:
                if action_type == "add":
                    if self._execute_add(action):
                        executed.append(action)
                        self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                            msg=f"[MemoryAgent] ADD {action.get('type', '?')}: {trunc(action.get('content', ''), 'console.summary')}",
                            data={"action": "add", "type": action.get("type"),
                                  "scopes": action.get("scopes", []),
                                  "content": action.get("content", "")})
                elif action_type == "edit":
                    if self._execute_edit(action):
                        executed.append(action)
                        self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                            msg=f"[MemoryAgent] EDIT {action.get('id', '?')}: {trunc(action.get('content', ''), 'console.summary')}",
                            data={"action": "edit", "id": action.get("id"),
                                  "content": action.get("content", "")})
                elif action_type == "drop":
                    if self._execute_drop(action):
                        executed.append(action)
                        self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                            msg=f"[MemoryAgent] DROP {action.get('id', '?')}",
                            data={"action": "drop", "id": action.get("id")})
                elif action_type == "register_pipeline":
                    result = self._execute_register_pipeline(action)
                    if result:
                        executed.append({**action, **result})
                        self._bus.emit(PIPELINE_REGISTERED, agent="MemoryAgent", level="info",
                            msg=f"[MemoryAgent] REGISTER_PIPELINE {result.get('pipeline_id', '?')}: {action.get('name', '')}",
                            data={"action": "register_pipeline", **result})
                elif action_type == "discard_pipeline":
                    if self._execute_discard_pipeline(action):
                        executed.append(action)
                        self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                            msg=f"[MemoryAgent] DISCARD_PIPELINE {action.get('render_op_id', '?')}",
                            data={"action": "discard_pipeline", "render_op_id": action.get("render_op_id")})
                else:
                    self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                        msg=f"[MemoryAgent] Unknown action type: {action_type}")
            except Exception as e:
                self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                    msg=f"[MemoryAgent] Action failed: {action_type} — {e}")

        # Single save after all actions
        if executed:
            self.memory_store.save()
            self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                msg=f"[MemoryAgent] Executed {len(executed)} actions")

        return executed

    def _execute_add(self, action: dict) -> bool:
        """Execute an 'add' action. Returns True if successful."""
        content = action.get("content", "").strip()
        if not content:
            return False
        mtype = action.get("type", "preference")
        if mtype not in _VALID_TYPES:
            return False
        scopes = _validate_scopes(action.get("scopes", ["generic"]))

        tags = generate_tags(content, scopes)
        self.memory_store.add_no_save(Memory(
            type=mtype,
            scopes=scopes,
            content=content,
            source="extracted",
            source_session=self.session_id,
            tags=tags,
        ))
        return True

    def _execute_edit(self, action: dict) -> bool:
        """Execute an 'edit' action (supersede pattern). Returns True if successful."""
        entry_id = action.get("id", "")
        content = action.get("content", "").strip()
        if not entry_id or not content:
            return False

        old = self.memory_store.get_by_id(entry_id)
        if old is None or old.archived:
            return False

        tags = generate_tags(content, old.scopes)
        new_memory = Memory(
            type=old.type,
            scopes=old.scopes,
            content=content,
            source="extracted",
            source_session=self.session_id,
            supersedes=entry_id,
            version=old.version + 1,
            tags=tags,
        )
        # Archive old, add new (no individual save — batched)
        old.archived = True
        self.memory_store.add_no_save(new_memory)
        return True

    def _execute_drop(self, action: dict) -> bool:
        """Execute a 'drop' action (archive). Returns True if successful."""
        entry_id = action.get("id", "")
        if not entry_id:
            return False

        entry = self.memory_store.get_by_id(entry_id)
        if entry is None or entry.archived:
            return False

        entry.archived = True
        self.memory_store.embeddings.invalidate()
        return True

    @staticmethod
    def _session_id_from_op_id(render_op_id: str) -> str | None:
        """Extract session ID from a scoped op ID like 'session_id:op_NNN'."""
        if ":" in render_op_id:
            return render_op_id.rsplit(":", 1)[0]
        return None

    def _set_pipeline_status_on_disk(self, render_op_id: str, status: str) -> bool:
        """Set pipeline_status on a render op, persisting to the correct session.

        For the current session, updates the in-memory log.
        For past sessions, loads the operations.json, updates, and saves back.
        """
        from data_ops.operations_log import get_operations_log, OperationsLog

        # Try in-memory log first (current session)
        ops_log = get_operations_log()
        if ops_log.set_pipeline_status(render_op_id, status):
            return True

        # Not in current session — load the past session's file
        source_sid = self._session_id_from_op_id(render_op_id)
        if not source_sid:
            return False

        ops_file = config.get_data_dir() / "sessions" / source_sid / "operations.json"
        if not ops_file.exists():
            return False

        try:
            past_log = OperationsLog(session_id=source_sid)
            past_log.load_from_file(ops_file)
            if past_log.set_pipeline_status(render_op_id, status):
                past_log.save_to_file(ops_file)
                return True
        except Exception:
            pass
        return False

    def _execute_register_pipeline(self, action: dict) -> Optional[dict]:
        """Execute a 'register_pipeline' action.

        Extracts the pipeline from the source session's operations log
        using render_op_id, saves to disk, and registers in PipelineStore.

        The source session is extracted from the scoped render_op_id
        (format: ``session_id:op_NNN``), so pipelines from any session
        can be registered.

        Accepts ``scopes`` (``"mission:<ID>"`` format) and ``tags``
        (free-form keywords).  Scopes are validated against the canonical
        mission list and converted to a ``missions`` list for PipelineEntry.

        Returns dict with pipeline_id and registration info, or None on failure.
        """
        from knowledge.mission_prefixes import get_all_canonical_ids

        render_op_id = action.get("render_op_id", "")
        name = action.get("name", "Untitled Pipeline")
        raw_desc = action.get("description", "")
        if isinstance(raw_desc, dict):
            description = (
                f"Source: {raw_desc.get('source', '')}\n"
                f"Rationale: {raw_desc.get('rationale', '')}\n"
                f"Use cases: {raw_desc.get('use_cases', '')}"
            )
        else:
            description = str(raw_desc)

        # Parse scopes → validated missions list
        valid_missions = set(get_all_canonical_ids())
        scopes = action.get("scopes", [])
        llm_missions = []
        for s in scopes:
            if s.startswith("mission:"):
                mid = s.split(":", 1)[1]
                if mid in valid_missions:
                    llm_missions.append(mid)

        tags = action.get("tags", [])

        if not render_op_id:
            return None

        # Determine which session owns this pipeline
        source_sid = self._session_id_from_op_id(render_op_id) or self.session_id
        if not source_sid:
            return None

        if self._pipeline_store is None:
            self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                msg="[MemoryAgent] Cannot register pipeline: no pipeline store")
            return None

        try:
            from data_ops.pipeline import SavedPipeline

            pipeline = SavedPipeline.from_session(
                source_sid,
                render_op_id=render_op_id,
                name=name,
                description=description,
                tags=tags,
            )

            # 1. Structural validation
            issues = pipeline.validate()
            if issues:
                self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                    msg=f"[MemoryAgent] Pipeline validation failed: {'; '.join(issues)}")
                self._set_pipeline_status_on_disk(render_op_id, "discarded")
                return None

            # 2. Test-replay with original time range
            t_start, t_end = pipeline.time_range_original
            if t_start and t_end:
                self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                    msg=f"[MemoryAgent] Test-replaying pipeline '{name}' ({t_start} to {t_end})...")
                try:
                    test_result = pipeline.execute(t_start, t_end)
                    if test_result.errors:
                        error_summary = "; ".join(
                            f"{e['tool']}({e['op_id']}): {e['error']}"
                            for e in test_result.errors
                        )
                        self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                            msg=f"[MemoryAgent] Pipeline test-replay failed: {error_summary}")
                        self._set_pipeline_status_on_disk(render_op_id, "discarded")
                        return None
                except Exception as e:
                    self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                        msg=f"[MemoryAgent] Pipeline test-replay exception: {e}")
                    self._set_pipeline_status_on_disk(render_op_id, "discarded")
                    return None
                self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                    msg=f"[MemoryAgent] Pipeline test-replay passed "
                        f"({test_result.steps_completed}/{test_result.steps_total} steps)")
            else:
                self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                    msg="[MemoryAgent] No time_range_original; skipping test-replay")

            pipeline.save()

            # Register in store (handles family dedup)
            entry = self._pipeline_store.register(
                pipeline,
                llm_missions=llm_missions or None,
                llm_tags=tags or None,
            )

            # Mark the render op as registered so it won't appear as a candidate again
            self._set_pipeline_status_on_disk(render_op_id, "registered")

            return {
                "pipeline_id": pipeline.id,
                "registered": entry is not None,
                "family_variants": len(entry.variant_ids) if entry else 0,
            }
        except Exception as e:
            self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                msg=f"[MemoryAgent] Pipeline registration failed: {e}")
            return None

    def _execute_discard_pipeline(self, action: dict) -> bool:
        """Execute a 'discard_pipeline' action.

        Marks the render op as discarded so it won't appear as a candidate again.
        Works for both current and past session ops.

        Returns True if the render op was found and marked.
        """
        render_op_id = action.get("render_op_id", "")
        if not render_op_id:
            return False

        return self._set_pipeline_status_on_disk(render_op_id, "discarded")

