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
from typing import Optional

import config
from .llm import LLMAdapter
from .memory import Memory, MemoryStore, generate_tags, estimate_tokens, MEMORY_TOKEN_BUDGET
from .model_fallback import get_active_model
from .event_bus import EventBus, get_event_bus, MEMORY_ACTION, TOKEN_USAGE

# Valid scope pattern
_VALID_SCOPE_RE = re.compile(r"^(generic|visualization|data_ops|mission:\w+)$")

# Valid types
_VALID_TYPES = {"preference", "summary", "pitfall", "reflection"}


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
    ):
        self.adapter = adapter
        self.model_name = model_name
        self.memory_store = memory_store
        self.verbose = verbose
        self.session_id = session_id
        self._bus = event_bus or get_event_bus()

        # Token usage tracking (same pattern as BaseSubAgent)
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
    def build_curated_events(raw_events: list[dict]) -> list[str]:
        """Filter memory-tagged EventBus events to signal-only entries for the LLM prompt.

        Takes raw event dicts (with keys: event, agent, msg, plus data fields)
        and returns a chronological list of concise one-line summaries.

        Excludes:
        - data_fetched with already_loaded: True
        - data_fetched success with no anomalies (routine loading)
        - plot_action with action: reset
        - render_executed args (figure_json is huge)
        - debug events, thinking, llm_call, llm_response (noise)

        Includes:
        - user_message / agent_response — conversation turns
        - data_computed — description, output label, status, error, code snippet
        - render_executed — status + error only
        - delegation / delegation_done — agent routing
        - data_created — description + status
        - data_fetched errors — dataset/param + error message
        - data_fetched with anomalies (high NaN)
        - custom_op_failure — error details
        """
        lines = []
        for ev in raw_events:
            event_type = ev.get("event", "")
            args = ev.get("args", {})
            status = ev.get("status", "")
            error = ev.get("error")
            outputs = ev.get("outputs", [])

            if event_type == "user_message":
                text = ev.get("text", ev.get("msg", ""))
                if text:
                    lines.append(f"  [User] {text[:300]}")
                continue

            elif event_type == "agent_response":
                text = ev.get("text", ev.get("msg", ""))
                if text:
                    lines.append(f"  [Agent] {text[:300]}")
                continue

            elif event_type == "data_fetched":
                # Skip already-loaded (cache hit)
                if args.get("already_loaded"):
                    continue
                # Skip routine success (no anomalies)
                if status == "success" and not error:
                    nan_pct = ev.get("nan_percentage", 0)
                    if not nan_pct or nan_pct <= 25:
                        continue
                ds = args.get("dataset_id", "?")
                param = args.get("parameter_id", "?")
                line = f"  fetch_data({ds}/{param}) → {status}"
                if error:
                    line += f" ERROR: {error[:150]}"
                nan_pct = ev.get("nan_percentage", 0)
                if nan_pct and nan_pct > 25:
                    line += f" [NaN: {nan_pct:.0f}%]"

            elif event_type == "data_computed":
                desc = args.get("description", "?")
                label = args.get("output_label", "?")
                code = args.get("code", "")
                line = f"  custom_operation({desc}) → {label} [{status}]"
                if error:
                    line += f" ERROR: {error[:150]}"
                if code:
                    line += f"\n    code: {code[:200]}"

            elif event_type == "custom_op_failure":
                desc = args.get("description", "?")
                code = args.get("code", "")
                line = f"  custom_operation({desc}) → error"
                if error:
                    line += f" ERROR: {error[:150]}"
                elif code:
                    line += f"\n    code: {code[:200]}"

            elif event_type == "render_executed":
                line = f"  render_plotly_json → {status}"
                if error:
                    line += f" ERROR: {error[:150]}"

            elif event_type == "plot_action":
                action = args.get("action", "")
                if action == "reset":
                    continue
                line = f"  manage_plot({action}) → {status}"

            elif event_type == "delegation":
                agent_name = ev.get("agent", "")
                msg = ev.get("msg", "")
                line = f"  delegation({agent_name}): {msg[:150]}"

            elif event_type == "delegation_done":
                agent_name = ev.get("agent", "")
                msg = ev.get("msg", "")
                line = f"  delegation_done({agent_name}): {msg[:150]}"

            elif event_type == "data_created":
                desc = args.get("description", "?")
                line = f"  store_dataframe({desc}) → {status}"

            else:
                # Skip other event types (debug, thinking, llm_call, etc.)
                continue

            if outputs:
                line += f" [outputs: {', '.join(str(o) for o in outputs[:5])}]"
            lines.append(line)

        return lines

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
- Use "drop" + "add" for consolidation (merge similar entries into one)
- Use "edit" when an existing memory needs minor updates
- Valid scopes: "generic", "visualization", "data_ops", "mission:<ID>" (uppercase mission ID)
- Each entry can have multiple scopes when it spans domains
- Return empty array [] if nothing worth remembering

Using reviews and history:
- Each memory may have **reviews** from consuming agents (star ratings 1-5 with structured comments). Star meanings: 5=prevented mistake, 4=useful context, 3=relevant but no impact, 2=irrelevant, 1=misleading. Comments may include criticism and suggestions.
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
                            msg=f"[MemoryAgent] ADD {action.get('type', '?')}: {action.get('content', '')[:120]}",
                            data={"action": "add", "type": action.get("type"),
                                  "scopes": action.get("scopes", []),
                                  "content": action.get("content", "")})
                elif action_type == "edit":
                    if self._execute_edit(action):
                        executed.append(action)
                        self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                            msg=f"[MemoryAgent] EDIT {action.get('id', '?')}: {action.get('content', '')[:120]}",
                            data={"action": "edit", "id": action.get("id"),
                                  "content": action.get("content", "")})
                elif action_type == "drop":
                    if self._execute_drop(action):
                        executed.append(action)
                        self._bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                            msg=f"[MemoryAgent] DROP {action.get('id', '?')}",
                            data={"action": "drop", "id": action.get("id")})
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

