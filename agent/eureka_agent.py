"""EurekaAgent — automated discovery and insight generation (BaseAgent subclass)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from .base_agent import BaseAgent
from .llm.base import FunctionSchema

if TYPE_CHECKING:
    from .eureka_store import EurekaStore
    from .llm import LLMService
    from .session_context import SessionContext


# -- submit_finding tool schema (private to EurekaAgent) --

_SUBMIT_FINDING_SCHEMA = FunctionSchema(
    name="submit_finding",
    description=(
        "Submit a scientific finding — an anomaly, correlation, deviation, "
        "timing coincidence, or structural pattern observed in the data."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the finding.",
            },
            "observation": {
                "type": "string",
                "description": "What was observed in the data.",
            },
            "hypothesis": {
                "type": "string",
                "description": "Proposed explanation for the observation.",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Supporting evidence (data labels, time ranges, visual features).",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level from 0.0 to 1.0.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Categorization tags (e.g. 'anomaly', 'correlation').",
            },
        },
        "required": ["title", "observation", "hypothesis", "evidence", "confidence", "tags"],
    },
)


# -- submit_suggestion tool schema (private to EurekaAgent) --

_SUBMIT_SUGGESTION_SCHEMA = FunctionSchema(
    name="submit_suggestion",
    description=(
        "Submit an actionable follow-up suggestion — a concrete next step "
        "for the user to explore, such as fetching more data, computing a "
        "transform, or creating a visualization."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["fetch_data", "visualize", "compute", "zoom", "compare"],
                "description": "Type of action to suggest.",
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of what to do.",
            },
            "rationale": {
                "type": "string",
                "description": "Why this suggestion matters.",
            },
            "parameters": {
                "type": "object",
                "description": "Action-specific parameters.",
            },
            "priority": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Priority level.",
            },
            "linked_eureka_id": {
                "type": "string",
                "description": "ID of the finding this suggestion aims to validate (empty string if none).",
            },
        },
        "required": ["action", "description", "rationale", "parameters", "priority", "linked_eureka_id"],
    },
)


class EurekaAgent(BaseAgent):
    """Eureka agent for automated discovery and insight generation.

    Has intrinsic tools (vision, web_search) from BaseAgent plus
    eureka-specific local tools (submit_finding, submit_suggestion).
    """

    agent_type = "eureka"

    def __init__(
        self,
        service: LLMService,
        session_ctx: SessionContext | None = None,
        *,
        eureka_store: EurekaStore | None = None,
        session_id: str = "",
        **kwargs,
    ):
        system_prompt = "You are an automated discovery specialist."
        try:
            from knowledge.prompt_builder import build_eureka_system_prompt
            system_prompt = build_eureka_system_prompt()
        except (ImportError, AttributeError):
            pass

        super().__init__(
            agent_id=f"eureka:{uuid4().hex[:6]}",
            service=service,
            system_prompt=system_prompt,
            session_ctx=session_ctx,
            **kwargs,
        )

        self._eureka_store = eureka_store
        self._session_id = session_id

        # Register real tools
        self._local_tools["submit_finding"] = self._handle_submit_finding
        self._local_tools["submit_suggestion"] = self._handle_submit_suggestion
        self._tool_schemas.append(_SUBMIT_FINDING_SCHEMA)
        self._tool_schemas.append(_SUBMIT_SUGGESTION_SCHEMA)

    def _handle_submit_finding(self, ctx: Any, args: dict, caller: Any) -> dict:
        """Handle submit_finding tool calls."""
        if self._eureka_store is None:
            return {"status": "error", "message": "No eureka store available."}

        from .eureka_store import EurekaEntry

        entry = EurekaEntry(
            id=uuid4().hex[:12],
            session_id=self._session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            title=args.get("title", ""),
            observation=args.get("observation", ""),
            hypothesis=args.get("hypothesis", ""),
            evidence=args.get("evidence", []),
            confidence=float(args.get("confidence", 0.5)),
            tags=args.get("tags", []),
            status="active",
        )

        self._eureka_store.add(entry)
        return {"status": "ok", "eureka_id": entry.id}

    def _handle_submit_suggestion(self, ctx: Any, args: dict, caller: Any) -> dict:
        """Handle submit_suggestion tool calls."""
        if self._eureka_store is None:
            return {"status": "error", "message": "No eureka store available."}

        from .eureka_store import EurekaSuggestion

        suggestion = EurekaSuggestion(
            id=uuid4().hex[:12],
            session_id=self._session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=args.get("action", ""),
            description=args.get("description", ""),
            rationale=args.get("rationale", ""),
            parameters=args.get("parameters", {}),
            priority=args.get("priority", "medium"),
            linked_eureka_id=args.get("linked_eureka_id", ""),
            status="proposed",
        )

        self._eureka_store.add_suggestion(suggestion)
        return {"status": "ok", "suggestion_id": suggestion.id}
