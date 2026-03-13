"""MemoryAgent — long-term memory extraction and injection (BaseAgent subclass)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from uuid import uuid4

from .base_agent import BaseAgent
from .llm.base import FunctionSchema

if TYPE_CHECKING:
    from .llm import LLMService
    from .memory import MemoryStore
    from .session_context import SessionContext


# -- manage_memory tool schema (private to MemoryAgent) --

_MANAGE_MEMORY_SCHEMA = FunctionSchema(
    name="manage_memory",
    description=(
        "Manage long-term memories: add new memories, edit existing ones, "
        "drop outdated ones, or list all current memories."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "edit", "drop", "list"],
                "description": "The action to perform.",
            },
            "content": {
                "type": "string",
                "description": "Memory content text (required for add/edit).",
            },
            "type": {
                "type": "string",
                "enum": ["preference", "summary", "pitfall", "reflection"],
                "description": "Memory type (required for add).",
            },
            "scopes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Scopes for this memory, e.g. ['generic'], "
                    "['visualization'], ['envoy:PSP']. Required for add."
                ),
            },
            "id": {
                "type": "string",
                "description": "Memory ID (required for edit/drop).",
            },
        },
        "required": ["action"],
    },
)


class MemoryAgent(BaseAgent):
    """Memory agent for long-term memory extraction and injection."""

    agent_type = "memory"

    def __init__(
        self,
        service: LLMService,
        session_ctx: SessionContext | None = None,
        *,
        memory_store: MemoryStore | None = None,
        **kwargs,
    ):
        # Build system prompt with current memories injected
        system_prompt = "You are a memory management specialist."
        try:
            from knowledge.prompt_builder import build_memory_system_prompt
            system_prompt = build_memory_system_prompt(memory_store)
        except (ImportError, AttributeError):
            pass

        super().__init__(
            agent_id=f"memory:{uuid4().hex[:6]}",
            service=service,
            system_prompt=system_prompt,
            session_ctx=session_ctx,
            **kwargs,
        )

        self._memory_store = memory_store

        # Action counters — reset before each extraction cycle, read after
        self._action_counts: dict[str, int] = {"add": 0, "edit": 0, "drop": 0}

        # Memory agent doesn't need intrinsic tools (vision, web_search)
        self._local_tools.pop("vision", None)
        self._local_tools.pop("web_search", None)
        self._tool_schemas = [
            s for s in self._tool_schemas
            if s.name not in ("vision", "web_search")
        ]

        # Register manage_memory as a private local tool
        self._local_tools["manage_memory"] = self._handle_manage_memory
        self._tool_schemas.append(_MANAGE_MEMORY_SCHEMA)

    def reset_action_counts(self) -> None:
        """Reset action counters — call before each extraction cycle."""
        self._action_counts = {"add": 0, "edit": 0, "drop": 0}

    def get_action_counts(self) -> dict[str, int]:
        """Return action counts from the most recent extraction cycle."""
        return dict(self._action_counts)

    def _handle_manage_memory(self, ctx: Any, args: dict, caller: Any) -> dict:
        """Handle manage_memory tool calls."""
        action = args.get("action")

        if action == "list":
            if self._memory_store is None:
                return {"status": "error", "message": "No memory store available."}
            memories = self._memory_store.get_all()
            return {
                "status": "ok",
                "memories": [
                    {
                        "id": m.id,
                        "type": m.type,
                        "content": m.content,
                        "scopes": m.scopes,
                        "tags": m.tags,
                        "created": m.created,
                    }
                    for m in memories
                ],
            }

        if action in ("add", "edit", "drop"):
            if self._memory_store is None:
                return {"status": "error", "message": "No memory store available."}

            store_action = {"action": action}
            if action == "add":
                store_action["content"] = args.get("content", "")
                store_action["type"] = args.get("type", "preference")
                store_action["scopes"] = args.get("scopes", ["generic"])
            elif action == "edit":
                store_action["id"] = args.get("id", "")
                store_action["content"] = args.get("content", "")
            elif action == "drop":
                store_action["id"] = args.get("id", "")

            try:
                count = self._memory_store.execute_actions([store_action])
                if count > 0:
                    self._action_counts[action] = self._action_counts.get(action, 0) + count
                return {"status": "ok", "actions_applied": count}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return {"status": "error", "message": f"Unknown action: {action}"}
