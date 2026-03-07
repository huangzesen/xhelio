"""LLMService — single entry point between backend and LLM providers.

See docs/plans/2026-03-06-llm-service-design.md for design rationale.
"""

from __future__ import annotations

import uuid
from typing import Any

from .base import (
    ChatSession,
    FunctionSchema,
    LLMAdapter,
    LLMResponse,
)
from .interface import LLMInterface, ToolResultBlock

def _generate_session_id() -> str:
    """Generate a unique xhelio session ID."""
    return f"xh_{uuid.uuid4().hex[:12]}"


class LLMService:
    """Single entry point between backend and LLM providers.

    Responsibilities:
    - Adapter factory: constructs the right adapter from config
    - Session registry: assigns xhelio session IDs, tracks active sessions
    - One-shot gateway: routes generate() through the same tracking path
    - Token accounting: centralizes per-session usage tracking via interface

    Does NOT:
    - Wrap ChatSession.send() — backend calls that directly
    - Handle fallback/retry — errors surface to the backend
    - Add business logic — pure delegation + bookkeeping
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._provider = provider.lower()
        self._model = model
        self._adapter = self._create_adapter(api_key, base_url)
        self._sessions: dict[str, ChatSession] = {}

    def _create_adapter(self, api_key: str | None, base_url: str | None) -> LLMAdapter:
        # Build kwargs, omitting None values so adapters fall back to env vars
        key_kw: dict = {"api_key": api_key} if api_key is not None else {}
        url_kw: dict = {"base_url": base_url} if base_url is not None else {}

        if self._provider == "gemini":
            from .gemini_adapter import GeminiAdapter
            return GeminiAdapter(**key_kw)
        elif self._provider == "anthropic":
            from .anthropic_adapter import AnthropicAdapter
            return AnthropicAdapter(**key_kw, **url_kw)
        elif self._provider == "openai":
            from .openai_adapter import OpenAIAdapter
            return OpenAIAdapter(**key_kw, **url_kw)
        elif self._provider == "minimax":
            from .minimax_adapter import MiniMaxAdapter
            return MiniMaxAdapter(**key_kw, **url_kw)
        else:
            raise ValueError(f"Unknown provider: {self._provider!r}")

    @property
    def adapter(self) -> LLMAdapter:
        """Direct adapter access (escape hatch for incremental migration)."""
        return self._adapter

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    # --- Session management ---

    def create_session(
        self,
        system_prompt: str,
        tools: list[FunctionSchema] | None = None,
        *,
        model: str | None = None,
        thinking: str = "default",
        agent_type: str = "",
        tracked: bool = True,
        interaction_id: str | None = None,
        json_schema: dict | None = None,
        force_tool_call: bool = False,
    ) -> ChatSession:
        """Start a new multi-turn conversation.

        Returns a ChatSession with a .session_id assigned.
        """
        session_model = model or self._model
        chat = self._adapter.create_chat(
            model=session_model,
            system_prompt=system_prompt,
            tools=tools,
            thinking=thinking,
            interaction_id=interaction_id,
            json_schema=json_schema,
            force_tool_call=force_tool_call,
        )
        if tracked:
            chat.session_id = _generate_session_id()
            chat._agent_type = agent_type
            chat._tracked = True
            self._sessions[chat.session_id] = chat
        else:
            chat.session_id = ""
            chat._tracked = False
        return chat

    def resume_session(self, saved_state: dict) -> ChatSession:
        """Restore a session from a saved state dict."""
        session_id = saved_state.get("session_id", "")
        messages = saved_state.get("messages", [])
        metadata = saved_state.get("metadata", {})

        interface = LLMInterface.from_dict(messages)

        # Determine interaction_id: check metadata first (where core.py stores it),
        # then fall back to scanning provider_data on assistant messages.
        interaction_id = metadata.get("interaction_id")
        if not interaction_id:
            for entry in reversed(interface.entries):
                if entry.role == "assistant" and entry.provider_data:
                    interaction_id = entry.provider_data.get("interaction_id")
                    if interaction_id:
                        break

        chat = self._adapter.create_chat(
            model=self._model,
            system_prompt=interface.current_system_prompt or "",
            tools=None,  # Adapter reconstructs tools from interface.current_tools
            interface=interface,
            interaction_id=interaction_id,
        )
        chat.session_id = session_id or _generate_session_id()
        chat._agent_type = metadata.get("agent_type", "")
        chat._tracked = metadata.get("tracked", True)
        if chat._tracked:
            self._sessions[chat.session_id] = chat
        return chat

    def get_session(self, session_id: str) -> ChatSession | None:
        """Look up an active session by ID."""
        return self._sessions.get(session_id)

    # --- One-shot generation ---

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        json_schema: dict | None = None,
        max_output_tokens: int | None = None,
        tracked: bool = True,
    ) -> LLMResponse:
        """Single-turn generation.

        If tracked=True (default): logs usage event.
        If tracked=False: fire-and-forget, no logging.
        """
        gen_model = model or self._model
        response = self._adapter.generate(
            model=gen_model,
            contents=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_schema=json_schema,
            max_output_tokens=max_output_tokens,
        )
        # TODO: emit usage tracking event when tracked=True
        return response

    # --- Tool results ---

    def make_tool_result(
        self, tool_name: str, result: dict, *, tool_call_id: str | None = None,
    ) -> ToolResultBlock:
        """Build a canonical ToolResultBlock."""
        return self._adapter.make_tool_result_message(
            tool_name, result, tool_call_id=tool_call_id,
        )
