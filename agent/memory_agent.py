"""
MemoryAgent — tool-calling memory extraction as a SubAgent.

The MemoryAgent sees the full console log (same events the user sees),
can drill into event details via tool calls, and emits memory actions
(add/edit/drop) through tools. Its final text response becomes a session
summary event.

Extends SubAgent for persistent thread, inbox pattern, and standard tool loop.
Called periodically during the session via _maybe_extract_memories().
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import config
from .memory import Memory, MemoryStore, generate_tags, MEMORY_TOKEN_BUDGET
from .sub_agent import SubAgent, Message
from .event_bus import EventBus, SessionEvent, get_event_bus, MEMORY_ACTION, MEMORY_SUMMARY, PIPELINE_REGISTERED
from .turn_limits import get_limit
from .truncation import trunc
from .logging import get_logger

logger = get_logger()

# Valid scope pattern
_VALID_SCOPE_RE = re.compile(r"^(generic|visualization|data_ops|envoy:\w+)$")

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
    console_events: list[SessionEvent]  # All console-tagged events
    active_scopes: list[str]
    total_memory_tokens: int = 0
    pipeline_candidates: list[dict] = field(default_factory=list)


class MemoryAgent(SubAgent):
    """Tool-calling memory extractor as a proper SubAgent.

    Sees the full console log, drills into event details via tools,
    and emits memory actions (add/edit/drop) through tool calls.
    The final text response becomes a session summary.

    Usage::

        agent = MemoryAgent(adapter, model_name, memory_store)
        agent.start()
        actions = agent.run(context)  # wraps SubAgent.send()
    """

    _PARALLEL_SAFE_TOOLS = {
        "get_event_details",
    }

    def __init__(
        self,
        service,
        memory_store: MemoryStore,
        verbose: bool = False,
        session_id: str = "",
        event_bus: Optional[EventBus] = None,
        pipeline_store=None,
    ):
        from .memory_tools import get_memory_tools

        # MemoryAgent manages the memory store directly — it reads ALL
        # memories for its system prompt, not scoped injection.
        self.memory_store = memory_store
        self._pipeline_store = pipeline_store
        self._memory_session_id = session_id

        # Per-cycle state (set during _pre_request, used by tool handlers)
        self._current_events: list[SessionEvent] = []
        self._current_scopes: list[str] = ["generic"]
        self._executed_actions: list[dict] = []

        super().__init__(
            agent_id="MemoryAgent",
            service=service,
            agent_type="memory",
            tool_executor=self._route_tool,
            system_prompt="",  # Built dynamically in _pre_request
            tool_schemas=get_memory_tools(),
            event_bus=event_bus,
            # No SubAgent core memory injection — MemoryAgent injects
            # ALL memories directly into its own system prompt.
            memory_store=None,
            memory_scope="",
        )

    # ------------------------------------------------------------------
    # SubAgent hooks
    # ------------------------------------------------------------------

    def _get_guard_limits(self) -> tuple[int, int, int]:
        """Memory-specific turn limits."""
        return (
            get_limit("memory.max_total_calls"),
            get_limit("sub_agent.dup_free_passes"),
            get_limit("sub_agent.dup_hard_block"),
        )

    def _pre_request(self, msg) -> str:
        """Build context prompt from MemoryContext stored in msg.content."""
        if not isinstance(msg.content, dict):
            return msg.content if isinstance(msg.content, str) else json.dumps(msg.content)

        context = msg.content.get("_context")
        if context is None:
            return json.dumps(msg.content)

        self._current_events = context.console_events
        self._current_scopes = context.active_scopes
        self._executed_actions = []

        # Force fresh chat each cycle — system prompt includes ALL current
        # memories which change between cycles. MemoryAgent doesn't need
        # cross-cycle persistent context.
        self._chat = None
        self.system_prompt = self._build_memory_system_prompt(context)

        return self._build_user_message(context)

    def _post_request(self, msg, result: dict) -> None:
        """Persist executed actions, emit summary."""
        # Single save after all tool-call mutations
        if self._executed_actions:
            self.memory_store.save()
            self._event_bus.emit(
                MEMORY_ACTION, agent="MemoryAgent", level="info",
                msg=f"[MemoryAgent] Executed {len(self._executed_actions)} actions",
            )

        # Final text = session summary
        summary_text = result.get("text", "").strip()
        if summary_text:
            self._event_bus.emit(
                MEMORY_SUMMARY,
                agent="MemoryAgent",
                level="info",
                msg=summary_text,
                data={"text": summary_text},
            )

        self._current_events = []

    # ------------------------------------------------------------------
    # Convenience wrapper for orchestrator compatibility
    # ------------------------------------------------------------------

    def run(self, context: MemoryContext) -> list[dict]:
        """Run memory extraction — convenience wrapper around SubAgent.send().

        Args:
            context: MemoryContext with console events, scopes, etc.

        Returns:
            List of executed action dicts.
        """
        if not context.console_events:
            return []

        self._executed_actions = []
        msg_content = {"_context": context}
        result = self.send(msg_content, sender="orchestrator", timeout=180.0)

        return list(self._executed_actions)

    # ------------------------------------------------------------------
    # Tool routing
    # ------------------------------------------------------------------

    def _route_tool(self, name: str, args: dict, tool_call_id=None) -> dict:
        """Route tool calls to internal handlers."""
        try:
            if name == "get_event_details":
                return self._tool_get_event_details(args)
            elif name == "add_memory":
                return self._tool_add_memory(args, self._executed_actions)
            elif name == "edit_memory":
                return self._tool_edit_memory(args, self._executed_actions)
            elif name == "drop_memory":
                return self._tool_drop_memory(args, self._executed_actions)
            elif name == "register_pipeline":
                return self._tool_register_pipeline(args, self._executed_actions)
            elif name == "discard_pipeline":
                return self._tool_discard_pipeline(args, self._executed_actions)
            else:
                return {"status": "error", "message": f"Unknown tool: {name}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_memory_system_prompt(self, context: MemoryContext) -> str:
        """Build the system prompt with role, rules, and ALL current memories."""
        sections = []

        sections.append(
            "You are a memory management agent. You see the full console log of a session "
            "and decide what long-term memories to add, edit, or drop.\n"
            "\nYou have tools to inspect event details and manage memories. "
            "Review the session log, drill into interesting events if needed, "
            "then use the memory tools to record anything worth remembering. "
            "When you're done, respond with a brief text summary of the session "
            "(2-4 sentences covering what the user did and key outcomes)."
        )

        sections.append(f"\nActive scopes: {context.active_scopes}")

        # Memory type formats
        sections.append("""
Content format per memory type:
- **preference**: 1-2 sentences capturing a user habit or style choice.
- **pitfall**: Trigger: <situation> / Problem: <what went wrong> / Fix: <how to avoid>
- **reflection**: Trigger: <situation> / Problem: <what went wrong> / Fix: <how to avoid>
- **summary**: Data: <datasets used> / Analysis: <what was done> / Finding: <key results>

Rules:
- Only add genuinely new information not already in Current Memories
- Use edit_memory when an existing memory needs minor updates
- Valid scopes: "generic", "visualization", "data_ops", "envoy:<ID>" (uppercase mission ID)
- Each entry can have multiple scopes when it spans domains
- Sub-agent errors are strong candidates for "pitfall" memories
- Be conservative with drops — only drop when clearly wrong or harmful

Pipeline rules (only when Pipeline Candidates are listed):
- Only register non-trivial reusable workflows
- Vanilla fetch+render pipelines are usually NOT worth registering — discard them
- You MUST decide on each candidate: register or discard""")

        # Consolidation policy
        sections.append("""
Consolidation policy — BE CONSERVATIVE:
- Default: preserve existing memories. Do not drop or merge without strong evidence.
- Only drop+add (merge) when entries are clearly redundant (near-identical content).
- Only edit/drop when a memory has 3+ reviews with clear evidence, or terrible reviews (avg <= 2 stars).
- If a memory has few or no reviews, leave it alone.""")

        # ALL current memories — not filtered by scope
        injected_ids = self.memory_store._last_injected_ids
        all_memories = [m for m in self.memory_store.get_enabled() if m.type != "review"]

        if all_memories:
            sections.append("\n## Current Memories")
            for m in all_memories:
                injected_tag = " [INJECTED]" if m.id in injected_ids else ""
                header = (
                    f"  [{m.id}] ({m.type}, {m.scopes}, v{m.version}, "
                    f"used {m.access_count}x, created {m.created_at[:10]}){injected_tag}"
                )
                sections.append(f"{header}\n    {m.content}")

                # Attach review feedback
                reviews = self.memory_store.get_recent_reviews_for_lineage(m.id, n=10)
                if reviews:
                    for r in reviews:
                        agent_tag = next(
                            (
                                t
                                for t in r.tags
                                if t
                                and not t.startswith("review:")
                                and not t.startswith("stars:")
                            ),
                            "",
                        )
                        sections.append(
                            f"    Review by {agent_tag} ({r.created_at[:10]}): {r.content}"
                        )

                # Attach version history
                if m.supersedes:
                    prev_id = m.supersedes
                    seen = set()
                    while prev_id and prev_id not in seen:
                        seen.add(prev_id)
                        prev = self.memory_store.get_by_id(prev_id)
                        if prev is None:
                            break
                        sections.append(
                            f"    Previous v{prev.version} ({prev.created_at[:10]}): {prev.content}"
                        )
                        prev_id = prev.supersedes
        else:
            sections.append("\n## Current Memories\n  (none)")

        # Token budget warning
        if context.total_memory_tokens > MEMORY_TOKEN_BUDGET * 0.8:
            pct = context.total_memory_tokens / MEMORY_TOKEN_BUDGET * 100
            sections.append(
                f"\n!! Memory is at {pct:.0f}% of token budget "
                f"({context.total_memory_tokens}/{MEMORY_TOKEN_BUDGET}). "
                f"Consider dropping or consolidating less useful entries."
            )

        return "\n".join(sections)

    def _build_user_message(self, context: MemoryContext) -> str:
        """Build the user message with session log and pipeline candidates."""
        sections = []

        # Session log — numbered summaries
        sections.append("## Session Log\n")
        for i, event in enumerate(context.console_events):
            agent_prefix = f"({event.agent}) " if event.agent else ""
            sections.append(f"  [{i}] {agent_prefix}{event.summary}")

        # Pipeline candidates
        if context.pipeline_candidates:
            sections.append("\n## Pipeline Candidates")
            sections.append(
                "The following data pipelines have not yet been curated. "
                "Decide each one: register_pipeline or discard_pipeline."
            )
            for cand in context.pipeline_candidates:
                op_id = cand.get("render_op_id", "?")
                step_count = cand.get("step_count", 0)
                vanilla = "vanilla" if cand.get("is_vanilla") else "non-vanilla"
                scopes_str = ", ".join(cand.get("scopes", [])) or "unknown"
                sections.append(
                    f"\n### Candidate [{op_id}] -- {step_count} steps, "
                    f"{vanilla}, scopes: {scopes_str}"
                )
                for j, step in enumerate(cand.get("steps", []), 1):
                    tool = step.get("tool", "?")
                    parts = [f"  {j}. {tool}:"]
                    if tool == "fetch_data":
                        ds = step.get("dataset_id", "")
                        param = step.get("parameter_id", "")
                        label = step.get("output_label", "")
                        parts.append(f"{ds}.{param}" if param else ds)
                        if label:
                            parts.append(f'-> "{label}"')
                    elif tool in ("custom_operation", "store_dataframe"):
                        desc = step.get("description", "")
                        code = step.get("code", "")
                        label = step.get("output_label", "")
                        if desc:
                            parts.append(desc)
                        if code:
                            parts.append(f"(code: {code})")
                        if label:
                            parts.append(f'-> "{label}"')
                    elif tool == "render_plotly_json":
                        inputs = step.get("inputs", [])
                        parts.append(f"inputs={inputs}")
                    elif tool == "manage_plot":
                        action = step.get("action", "")
                        parts.append(action)
                    else:
                        label = step.get("output_label", "")
                        if label:
                            parts.append(f'-> "{label}"')
                    sections.append(" ".join(parts))

                feedback = cand.get("feedback", [])
                if feedback:
                    sections.append("  User feedback:")
                    for fb in feedback:
                        fb_date = fb.get("timestamp", "")[:10]
                        fb_comment = fb.get("comment", "")
                        sections.append(f'  - "{fb_comment}" ({fb_date})')

            # Valid mission IDs
            from knowledge.mission_prefixes import get_all_canonical_ids
            valid_missions = ", ".join(get_all_canonical_ids())
            sections.append(
                f'\nValid mission IDs for scopes (use "envoy:<ID>" format):\n'
                f"{valid_missions}"
            )

        sections.append(
            "\nAnalyze the session log above. Use get_event_details to inspect "
            "any events that need closer examination. Then use the memory tools "
            "to record anything worth remembering. Finally, respond with a brief "
            "session summary."
        )

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_get_event_details(self, args: dict) -> dict:
        """Return full details for an event by index."""
        idx = args.get("event_index", -1)
        if idx < 0 or idx >= len(self._current_events):
            return {
                "status": "error",
                "message": f"Invalid event index {idx}. Valid range: 0-{len(self._current_events) - 1}",
            }
        event = self._current_events[idx]
        return {
            "status": "ok",
            "index": idx,
            "type": event.type,
            "agent": event.agent,
            "summary": event.summary,
            "details": event.details,
            "level": event.level,
            "timestamp": event.ts,
        }

    def _tool_add_memory(self, args: dict, executed: list[dict]) -> dict:
        """Add a new memory entry."""
        action = {
            "action": "add",
            "type": args.get("type", "preference"),
            "scopes": args.get("scopes", ["generic"]),
            "content": args.get("content", ""),
        }
        if self._execute_add(action):
            executed.append(action)
            self._event_bus.emit(
                MEMORY_ACTION, agent="MemoryAgent", level="info",
                msg=f"[MemoryAgent] ADD {action['type']}: {trunc(action['content'], 'console.summary')}",
                data={"action": "add", "type": action["type"],
                      "scopes": action["scopes"], "content": action["content"]},
            )
            return {"status": "ok", "message": f"Added {action['type']} memory."}
        return {"status": "error", "message": "Failed to add memory. Check type and content."}

    def _tool_edit_memory(self, args: dict, executed: list[dict]) -> dict:
        """Edit an existing memory."""
        action = {
            "action": "edit",
            "id": args.get("memory_id", ""),
            "content": args.get("content", ""),
        }
        if self._execute_edit(action):
            executed.append(action)
            self._event_bus.emit(
                MEMORY_ACTION, agent="MemoryAgent", level="info",
                msg=f"[MemoryAgent] EDIT {action['id']}: {trunc(action['content'], 'console.summary')}",
                data={"action": "edit", "id": action["id"], "content": action["content"]},
            )
            return {"status": "ok", "message": f"Edited memory {action['id']}."}
        return {"status": "error", "message": f"Failed to edit memory {action['id']}. Check ID exists and is not archived."}

    def _tool_drop_memory(self, args: dict, executed: list[dict]) -> dict:
        """Drop (archive) a memory."""
        action = {
            "action": "drop",
            "id": args.get("memory_id", ""),
        }
        if self._execute_drop(action):
            executed.append(action)
            self._event_bus.emit(
                MEMORY_ACTION, agent="MemoryAgent", level="info",
                msg=f"[MemoryAgent] DROP {action['id']}",
                data={"action": "drop", "id": action["id"]},
            )
            return {"status": "ok", "message": f"Dropped memory {action['id']}."}
        return {"status": "error", "message": f"Failed to drop memory {action['id']}. Check ID exists and is not already archived."}

    def _tool_register_pipeline(self, args: dict, executed: list[dict]) -> dict:
        """Register a pipeline candidate."""
        action = {
            "action": "register_pipeline",
            "render_op_id": args.get("render_op_id", ""),
            "name": args.get("name", "Untitled Pipeline"),
            "description": args.get("description", {}),
            "scopes": args.get("scopes", []),
            "tags": args.get("tags", []),
        }
        result = self._execute_register_pipeline(action)
        if result:
            executed.append({**action, **result})
            self._event_bus.emit(
                PIPELINE_REGISTERED, agent="MemoryAgent", level="info",
                msg=f"[MemoryAgent] REGISTER_PIPELINE {result.get('pipeline_id', '?')}: {action['name']}",
                data={"action": "register_pipeline", **result},
            )
            return {"status": "ok", "pipeline_id": result.get("pipeline_id"), "message": f"Registered pipeline '{action['name']}'."}
        return {"status": "error", "message": "Failed to register pipeline. Check render_op_id and validation."}

    def _tool_discard_pipeline(self, args: dict, executed: list[dict]) -> dict:
        """Discard a pipeline candidate."""
        action = {
            "action": "discard_pipeline",
            "render_op_id": args.get("render_op_id", ""),
        }
        if self._execute_discard_pipeline(action):
            executed.append(action)
            self._event_bus.emit(
                MEMORY_ACTION, agent="MemoryAgent", level="info",
                msg=f"[MemoryAgent] DISCARD_PIPELINE {action['render_op_id']}",
                data={"action": "discard_pipeline", "render_op_id": action["render_op_id"]},
            )
            return {"status": "ok", "message": f"Discarded pipeline {action['render_op_id']}."}
        return {"status": "error", "message": f"Failed to discard pipeline {action['render_op_id']}."}

    # ------------------------------------------------------------------
    # Action execution (kept from original)
    # ------------------------------------------------------------------

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
            source_session=self._memory_session_id,
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
            source_session=self._memory_session_id,
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

    # ------------------------------------------------------------------
    # Pipeline helpers (kept from original)
    # ------------------------------------------------------------------

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

        Accepts ``scopes`` (``"envoy:<ID>"`` format) and ``tags``
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
            if s.startswith("envoy:"):
                mid = s.split(":", 1)[1]
                if mid in valid_missions:
                    llm_missions.append(mid)

        tags = action.get("tags", [])

        if not render_op_id:
            return None

        # Determine which session owns this pipeline
        source_sid = self._session_id_from_op_id(render_op_id) or self._memory_session_id
        if not source_sid:
            return None

        if self._pipeline_store is None:
            self._event_bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
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
                self._event_bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                    msg=f"[MemoryAgent] Pipeline validation failed: {'; '.join(issues)}")
                self._set_pipeline_status_on_disk(render_op_id, "discarded")
                return None

            # 2. Test-replay with original time range
            t_start, t_end = pipeline.time_range_original
            if t_start and t_end:
                self._event_bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                    msg=f"[MemoryAgent] Test-replaying pipeline '{name}' ({t_start} to {t_end})...")
                try:
                    test_result = pipeline.execute(t_start, t_end)
                    if test_result.errors:
                        error_summary = "; ".join(
                            f"{e['tool']}({e['op_id']}): {e['error']}"
                            for e in test_result.errors
                        )
                        self._event_bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                            msg=f"[MemoryAgent] Pipeline test-replay failed: {error_summary}")
                        self._set_pipeline_status_on_disk(render_op_id, "discarded")
                        return None
                except Exception as e:
                    self._event_bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
                        msg=f"[MemoryAgent] Pipeline test-replay exception: {e}")
                    self._set_pipeline_status_on_disk(render_op_id, "discarded")
                    return None
                self._event_bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="info",
                    msg=f"[MemoryAgent] Pipeline test-replay passed "
                        f"({test_result.steps_completed}/{test_result.steps_total} steps)")
            else:
                self._event_bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
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
            self._event_bus.emit(MEMORY_ACTION, agent="MemoryAgent", level="warning",
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
