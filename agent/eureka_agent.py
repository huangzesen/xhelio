"""
EurekaAgent — Persistent scientific advisor that scans session assets,
proposes findings (eurekas), and suggests actionable follow-ups.

Extends SubAgent for persistent chat, inbox pattern, and standard tool loop.
Three logical phases run in a single persistent ChatSession:
  1. Think: call investigation tools to inspect data, figures, history
  2. Propose: produce structured scientific findings (eurekas)
  3. Suggest: propose actionable follow-ups linked to findings
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import config
from .eureka_store import EurekaStore, EurekaEntry, EurekaSuggestion
from .event_bus import (
    EventBus,
    EUREKA_FINDING,
    EUREKA_SUGGESTION,
    TEXT_DELTA,
    USER_MESSAGE,
    TOOL_CALL,
    TOOL_RESULT,
    RENDER_EXECUTED,
)
from .llm import FunctionSchema
from .logging import get_logger
from .sub_agent import SubAgent, Message
from .turn_limits import get_limit
from knowledge.prompt_builder import build_eureka_prompt
from data_ops.store import get_store

if TYPE_CHECKING:
    from .memory import MemoryStore

logger = get_logger()


# -- Internal tool schemas (not in tool_registry.json) --

_INTERNAL_TOOL_SCHEMAS = [
    FunctionSchema(
        name="get_session_figure",
        description="Exports current Plotly figure to PNG bytes, returns as base64-encoded image for multimodal analysis.",
        parameters={"type": "object", "properties": {}},
    ),
    FunctionSchema(
        name="read_session_history",
        description="Returns curated EventBus events (data operations, renders, user messages).",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of events to return.",
                    "default": 20,
                }
            },
        },
    ),
    FunctionSchema(
        name="read_memories",
        description="Returns active memories relevant to current session.",
        parameters={"type": "object", "properties": {}},
    ),
    FunctionSchema(
        name="read_eureka_history",
        description="Returns this session's previous eureka findings and suggestions. Use this to build on your prior observations.",
        parameters={"type": "object", "properties": {}},
    ),
    FunctionSchema(
        name="submit_eureka",
        description=(
            "Submit a scientific finding. Call once per finding (max 3 per cycle). "
            "Each call stores the finding and returns its index for linking suggestions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short descriptive title of the finding.",
                },
                "observation": {
                    "type": "string",
                    "description": "What you observed in the data.",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "A plausible physical explanation.",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of evidence strings (data labels, time ranges, visual features).",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level 0.0-1.0 (minimum 0.3).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Categorization tags (e.g. solar_wind, anomaly, correlation).",
                },
            },
            "required": ["title", "observation"],
        },
    ),
    FunctionSchema(
        name="submit_suggestion",
        description=(
            "Submit an actionable follow-up suggestion. Call exactly 3 times per cycle. "
            "Categories: fetch new data, run analysis/computation, or create visualization."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action type: fetch_data, compute, or visualize.",
                    "enum": ["fetch_data", "compute", "visualize"],
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of what to do.",
                },
                "details": {
                    "type": "string",
                    "description": "Rationale: why this suggestion matters, what it would reveal.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Action-specific parameters (mission, dataset, timerange, etc.).",
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level.",
                    "enum": ["high", "medium", "low"],
                },
                "linked_eureka_index": {
                    "type": "integer",
                    "description": "0-based index of the eureka finding this relates to. Omit if unlinked.",
                },
            },
            "required": ["action", "description", "details"],
        },
    ),
]

# Internal tool names for routing
_INTERNAL_TOOL_NAMES = frozenset(t.name for t in _INTERNAL_TOOL_SCHEMAS)


class EurekaAgent(SubAgent):
    """A SubAgent specialized for scientific discovery with persistent context.

    Three-phase cycle (think → propose → suggest) runs in a single persistent
    ChatSession via the standard SubAgent tool loop. The LLM calls investigation
    tools (think), then calls submit_eureka/submit_suggestion tools to report
    findings and suggestions (propose + suggest).
    """

    _PARALLEL_SAFE_TOOLS = {
        "list_fetched_data",
        "preview_data",
        "describe_data",
        "events",
        "get_session_figure",
        "read_session_history",
        "read_memories",
        "read_eureka_history",
        "delegate_to_insight",
    }

    def __init__(
        self,
        service,
        tool_executor,
        *,
        event_bus: EventBus | None = None,
        memory_store: Optional["MemoryStore"] = None,
        memory_scope: str = "eureka",
        orchestrator_ref=None,
    ):
        from .tools import get_function_schemas

        # Build combined tool schemas: external tools + internal tools
        external_tool_names = [
            "list_fetched_data",
            "preview_data",
            "describe_data",
            "events",
            "delegate_to_insight",
        ]
        external_schemas = get_function_schemas(names=external_tool_names)
        all_schemas = external_schemas + list(_INTERNAL_TOOL_SCHEMAS)

        self.orchestrator = orchestrator_ref
        self.eureka_store = EurekaStore()
        self._session_id: str = "unknown"
        self._external_tool_executor = tool_executor
        self._pending_eurekas: list[dict] = []
        self._pending_suggestions: list[dict] = []

        super().__init__(
            agent_id="EurekaAgent",
            service=service,
            agent_type="eureka",
            tool_executor=self._route_tool,
            system_prompt=build_eureka_prompt(),
            tool_schemas=all_schemas,
            event_bus=event_bus,
            memory_store=memory_store,
            memory_scope=memory_scope,
        )

    def _get_guard_limits(self) -> tuple[int, int, int]:
        """Eureka-specific turn limits."""
        return (
            get_limit("eureka.max_total_calls"),
            get_limit("sub_agent.dup_free_passes"),
            get_limit("sub_agent.dup_hard_block"),
        )

    def _pre_request(self, msg) -> str:
        """Extract session_id from dict content, reset accumulators."""
        # Reset per-cycle accumulators
        self._pending_eurekas = []
        self._pending_suggestions = []

        # Extract session_id and message text from the context dict
        if isinstance(msg.content, dict):
            self._session_id = msg.content.get("session_id", "unknown")
            return msg.content.get("message", json.dumps(msg.content))
        return msg.content if isinstance(msg.content, str) else json.dumps(msg.content)

    def _post_request(self, msg, result: dict) -> None:
        """Store and emit eureka findings/suggestions, emit final text."""
        # Store and emit whatever was submitted via tool calls
        self._store_and_emit({
            "eurekas": self._pending_eurekas,
            "suggestions": self._pending_suggestions,
        })

        if not self._pending_eurekas and not self._pending_suggestions:
            logger.warning(
                "[EurekaAgent] Cycle produced 0 findings and 0 suggestions"
            )

        # Emit final text as user-facing commentary
        final_text = result.get("text", "")
        if final_text:
            self._event_bus.emit(
                TEXT_DELTA,
                agent="EurekaAgent",
                level="info",
                msg=f"[EurekaAgent] {final_text}",
                data={"text": final_text + "\n\n", "commentary": True},
            )

    def _route_tool(self, name: str, args: dict, tool_call_id=None) -> dict:
        """Route tools: internal tools handled locally, others via orchestrator."""
        if name in _INTERNAL_TOOL_NAMES:
            try:
                if name == "get_session_figure":
                    return self._tool_get_session_figure()
                elif name == "read_session_history":
                    return self._tool_read_session_history(args.get("limit", 20))
                elif name == "read_memories":
                    return self._tool_read_memories()
                elif name == "read_eureka_history":
                    return self._tool_read_eureka_history()
                elif name == "submit_eureka":
                    return self._tool_submit_eureka(args)
                elif name == "submit_suggestion":
                    return self._tool_submit_suggestion(args)
                return {"status": "error", "message": f"Unknown internal tool: {name}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        # External tools: delegate to orchestrator's tool executor
        return self._external_tool_executor(name, args, tool_call_id)

    # ------------------------------------------------------------------
    # Output tool handlers (submit_eureka / submit_suggestion)
    # ------------------------------------------------------------------

    def _tool_submit_eureka(self, args: dict) -> dict:
        """Accumulate a eureka finding submitted via tool call."""
        if len(self._pending_eurekas) >= 3:
            return {"status": "error", "message": "Maximum 3 eurekas per cycle reached."}
        self._pending_eurekas.append(args)
        return {"status": "ok", "index": len(self._pending_eurekas) - 1}

    def _tool_submit_suggestion(self, args: dict) -> dict:
        """Accumulate a suggestion submitted via tool call."""
        self._pending_suggestions.append(args)
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Internal tool implementations
    # ------------------------------------------------------------------

    def _tool_get_session_figure(self) -> Dict[str, Any]:
        """Exports current Plotly figure to PNG bytes, returns as base64-encoded image."""
        image_bytes = self.orchestrator.get_latest_figure_png()
        if image_bytes is None:
            return {"status": "error", "message": "No figure currently rendered"}

        try:
            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            return {
                "mime_type": "image/png",
                "data": b64_data,
                "msg": "Figure exported successfully.",
            }
        except Exception as e:
            return {"status": "error", "message": f"Figure PNG export failed: {e}"}

    def _tool_read_session_history(self, limit: int) -> List[str]:
        """Returns curated EventBus events."""
        relevant_types = {USER_MESSAGE, TOOL_CALL, TOOL_RESULT, RENDER_EXECUTED}
        target_tools = {
            "fetch_data",
            "run_code",
            "render_plotly_json",
            "manage_plot",
            "delegate_to_envoy",
            "delegate_to_viz",
            "delegate_to_data_ops",
        }

        all_events = self._event_bus.get_events()
        curated = []
        for ev in all_events:
            if ev.type not in relevant_types:
                continue

            if ev.type in {TOOL_CALL, TOOL_RESULT}:
                tool_name = ev.data.get("tool", "") if ev.data else ""
                if tool_name not in target_tools:
                    continue

            from datetime import datetime

            ts = ""
            if ev.ts:
                try:
                    dt = datetime.fromisoformat(ev.ts.replace("Z", "+00:00"))
                    ts = dt.strftime("%H:%M:%S")
                except (ValueError, TypeError):
                    ts = str(ev.ts)
            if ev.type == USER_MESSAGE:
                curated.append(f"[{ts}] USER: {ev.msg}")
            elif ev.type == TOOL_CALL:
                tool = ev.data.get("tool", "")
                curated.append(f"[{ts}] CALL: {tool}")
            elif ev.type == TOOL_RESULT:
                tool = ev.data.get("tool", "")
                status = ev.data.get("status", "")
                curated.append(f"[{ts}] RESULT: {tool} ({status})")
            elif ev.type == RENDER_EXECUTED:
                curated.append(f"[{ts}] RENDER: Plotly figure produced")

        return curated[-limit:]

    def _tool_read_memories(self) -> List[Dict[str, Any]]:
        """Returns active memories relevant to current session."""
        if not self._memory_store:
            return []

        memories = self._memory_store.get_enabled()
        return [
            {"id": m.id, "type": m.type, "content": m.content, "scopes": m.scopes}
            for m in memories[:20]
        ]

    def _tool_read_eureka_history(self) -> Dict[str, Any]:
        """Returns this session's previous findings and suggestions."""
        return self.eureka_store.get_session_history(self._session_id)

    # ------------------------------------------------------------------
    # Storage and emission
    # ------------------------------------------------------------------

    def _store_and_emit(self, parsed: Dict[str, Any]) -> None:
        """Store parsed eurekas and suggestions, emit events."""
        timestamp = datetime.now().isoformat()
        eureka_ids = []  # Track IDs for linking suggestions

        # Process eurekas
        for e in parsed.get("eurekas", [])[:3]:  # Hard limit per cycle
            try:
                if not e.get("title") or not e.get("observation"):
                    continue

                entry = EurekaEntry(
                    id=str(uuid.uuid4()),
                    session_id=self._session_id,
                    timestamp=timestamp,
                    title=e["title"],
                    observation=e["observation"],
                    hypothesis=e.get("hypothesis", ""),
                    evidence=e.get("evidence", []),
                    confidence=float(e.get("confidence", 0.5)),
                    tags=e.get("tags", []),
                    status="proposed",
                )
                self.eureka_store.add(entry)
                eureka_ids.append(entry.id)

                self._event_bus.emit(
                    EUREKA_FINDING,
                    agent="EurekaAgent",
                    level="info",
                    msg=f"[Eureka] {entry.title}",
                    data=asdict(entry),
                )
            except Exception as ex:
                logger.warning(f"Failed to process eureka entry: {ex}")

        # Process suggestions
        for s in parsed.get("suggestions", []):
            try:
                if not s.get("action") or not s.get("description"):
                    continue

                # Link to the eureka ID by index
                linked_idx = s.get("linked_eureka_index") if s.get("linked_eureka_index") is not None else s.get("linked_eureka_id", 0)
                linked_id = ""
                if isinstance(linked_idx, int) and 0 <= linked_idx < len(eureka_ids):
                    linked_id = eureka_ids[linked_idx]

                rationale = s.get("details") or s.get("rationale", "")
                suggestion = EurekaSuggestion(
                    id=str(uuid.uuid4()),
                    session_id=self._session_id,
                    timestamp=timestamp,
                    action=s["action"],
                    description=s["description"],
                    rationale=rationale,
                    parameters=s.get("parameters", {}),
                    priority=s.get("priority", "medium"),
                    linked_eureka_id=linked_id,
                    status="proposed",
                )
                self.eureka_store.add_suggestion(suggestion)

                self._event_bus.emit(
                    EUREKA_SUGGESTION,
                    agent="EurekaAgent",
                    level="info",
                    msg=f"[Eureka Suggestion] {suggestion.description}",
                    data=asdict(suggestion),
                )
            except Exception as ex:
                logger.warning(f"Failed to process eureka suggestion: {ex}")

    def build_context_message(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the context dict to send via send().

        Called by the orchestrator to prepare the message content.
        """
        asset_keys = context.get("data_store_keys", [])
        has_figure = context.get("has_figure", False)
        recent_msgs = context.get("recent_messages", [])

        lines = [
            f"Session ID: {context.get('session_id', 'unknown')}",
            f"Active Data Store Keys: {', '.join(asset_keys) if asset_keys else 'None'}",
            f"Current Figure Available: {'Yes' if has_figure else 'No'}",
        ]
        if recent_msgs:
            lines.append("\nRecent User Messages:")
            for m in recent_msgs:
                lines.append(f"- {m}")

        lines.append(
            "\nInspect the session and report any scientific findings or interesting patterns. "
            "Build on your previous observations if any."
        )

        return {
            "session_id": context.get("session_id", "unknown"),
            "message": "\n".join(lines),
        }
