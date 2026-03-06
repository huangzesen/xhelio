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
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import config
from .eureka_store import EurekaStore, EurekaEntry, EurekaSuggestion
from .event_bus import (
    EventBus,
    EUREKA_FINDING,
    EUREKA_SUGGESTION,
    USER_MESSAGE,
    TOOL_CALL,
    TOOL_RESULT,
    RENDER_EXECUTED,
)
from .llm import LLMAdapter, FunctionSchema
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
]

# Internal tool names for routing
_INTERNAL_TOOL_NAMES = frozenset(t.name for t in _INTERNAL_TOOL_SCHEMAS)


class EurekaAgent(SubAgent):
    """A SubAgent specialized for scientific discovery with persistent context.

    Three-phase cycle (think → propose → suggest) runs in a single persistent
    ChatSession via the standard SubAgent tool loop. The LLM calls investigation
    tools (think), then produces structured JSON output with eurekas and
    suggestions (propose + suggest).
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
        adapter: LLMAdapter,
        model_name: str,
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

        super().__init__(
            agent_id="EurekaAgent",
            adapter=adapter,
            model_name=model_name,
            tool_executor=self._route_tool,
            system_prompt=build_eureka_prompt(),
            tool_schemas=all_schemas,
            event_bus=event_bus,
            memory_store=memory_store,
            memory_scope=memory_scope,
        )

    def _handle_request(self, msg: Message) -> None:
        """Override to use eureka-specific turn limits and post-process results."""
        from .loop_guard import LoopGuard

        self._guard = LoopGuard(
            max_total_calls=get_limit("eureka.max_total_calls"),
            dup_free_passes=get_limit("sub_agent.dup_free_passes"),
            dup_hard_block=get_limit("sub_agent.dup_hard_block"),
        )

        # Extract session_id and message text from the context dict
        if isinstance(msg.content, dict):
            self._session_id = msg.content.get("session_id", "unknown")
            content = msg.content.get("message", json.dumps(msg.content))
        else:
            content = msg.content

        # Prepend current time (matches SubAgent._handle_request pattern)
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        content = f"[Current time: {current_time}]\n\n{content}"

        response = self._llm_send(content)
        result = self._process_response(response)

        # Post-process: parse eurekas + suggestions from result text
        parsed = self._parse_output(result.get("text", ""))
        self._store_and_emit(parsed)

        self._deliver_result(msg, result)

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
                return {"status": "error", "message": f"Unknown internal tool: {name}"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        # External tools: delegate to orchestrator's tool executor
        return self._external_tool_executor(name, args, tool_call_id)

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
            "custom_operation",
            "store_dataframe",
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
    # Output parsing and storage
    # ------------------------------------------------------------------

    def _parse_output(self, text: str) -> Dict[str, Any]:
        """Parse structured JSON (eurekas + suggestions) from LLM text response."""
        if not text:
            return {"eurekas": [], "suggestions": []}

        try:
            # Try to find a JSON object with "eurekas" key
            match = re.search(
                r'\{[^{}]*"eurekas"\s*:\s*\[.*?\](?:[^{}]*"suggestions"\s*:\s*\[.*?\])?[^{}]*\}',
                text,
                re.DOTALL,
            )
            if match:
                data = json.loads(match.group(0))
                return {
                    "eurekas": data.get("eurekas", []),
                    "suggestions": data.get("suggestions", []),
                }

            # Try parsing the whole text as JSON
            data = json.loads(text)
            if isinstance(data, dict):
                return {
                    "eurekas": data.get("eurekas", []),
                    "suggestions": data.get("suggestions", []),
                }
            if isinstance(data, list):
                return {"eurekas": data, "suggestions": []}
        except Exception:
            pass

        # Try to find just an array (backward compat)
        try:
            json_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if json_match:
                arr = json.loads(json_match.group(0))
                return {"eurekas": arr, "suggestions": []}
        except Exception:
            pass

        return {"eurekas": [], "suggestions": []}

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
                linked_idx = s.get("linked_eureka_id", 0)
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
