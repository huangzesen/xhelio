"""LLMService — single entry point between backend and LLM providers.

See docs/plans/2026-03-06-llm-service-design.md for design rationale.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from .base import (
    ChatSession,
    FunctionSchema,
    LLMAdapter,
    LLMResponse,
)
from .interface import ChatInterface, ToolResultBlock
from config import get_api_key

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
        provider_config: dict | None = None,
    ) -> None:
        self._provider = provider.lower()
        self._model = model
        self._base_url = base_url
        self._config = provider_config or {}
        self._adapters: dict[tuple[str, str | None], LLMAdapter] = {}
        self._adapter_lock = threading.Lock()
        self._adapters[(self._provider, base_url)] = self._create_adapter(self._provider, api_key, base_url)
        self._sessions: dict[str, ChatSession] = {}

    def _create_adapter(self, provider: str, api_key: str | None, base_url: str | None) -> LLMAdapter:
        # Build kwargs, omitting None values so adapters fall back to env vars
        key_kw: dict = {"api_key": api_key} if api_key is not None else {}
        url_kw: dict = {"base_url": base_url} if base_url is not None else {}

        p = provider.lower()
        if p == "gemini":
            from .gemini.adapter import GeminiAdapter
            return GeminiAdapter(**key_kw)
        elif p == "anthropic":
            from .anthropic.adapter import AnthropicAdapter
            return AnthropicAdapter(**key_kw, **url_kw)
        elif p == "openai":
            from .openai.adapter import OpenAIAdapter
            return OpenAIAdapter(**key_kw, **url_kw)
        elif p == "minimax":
            from .minimax.adapter import MiniMaxAdapter
            return MiniMaxAdapter(**key_kw, **url_kw)
        elif p == "grok":
            from .grok.adapter import GrokAdapter
            return GrokAdapter(**key_kw)
        elif p == "deepseek":
            from .deepseek.adapter import DeepSeekAdapter
            return DeepSeekAdapter(**key_kw)
        elif p == "qwen":
            from .qwen.adapter import QwenAdapter
            return QwenAdapter(**key_kw)
        elif p == "kimi":
            from .kimi.adapter import create_kimi_adapter
            defaults = self._get_provider_defaults(p)
            compat = defaults.get("api_compat", "openai") if defaults else "openai"
            return create_kimi_adapter(**key_kw, api_compat=compat, **url_kw)
        elif p == "glm":
            from .glm.adapter import GLMAdapter
            return GLMAdapter(**key_kw)
        elif p == "custom":
            from .custom.adapter import create_custom_adapter
            defaults = self._get_provider_defaults(p)
            compat = defaults.get("api_compat", "openai") if defaults else "openai"
            ws = defaults.get("supports_web_search", False) if defaults else False
            vis = defaults.get("supports_vision", False) if defaults else False
            return create_custom_adapter(
                **key_kw, api_compat=compat, supports_web_search=ws,
                supports_vision=vis, **url_kw,
            )
        else:
            raise ValueError(f"Unknown provider: {provider!r}")

    # --- Adapter cache ---

    def get_adapter(self, provider: str, base_url: str | None = None) -> LLMAdapter:
        """Return cached adapter for *provider* + *base_url*, creating one on demand.

        The cache is keyed by ``(provider, base_url)`` so the same provider
        with different base URLs (e.g. OpenRouter vs local vLLM) gets separate
        adapter instances.

        Raises RuntimeError if the API key for *provider* is not configured.
        """
        provider = provider.lower()
        cache_key = (provider, base_url)

        # Fast path — no lock needed for reads of an already-cached adapter
        if cache_key in self._adapters:
            return self._adapters[cache_key]
        if base_url is None and (provider, None) in self._adapters:
            return self._adapters[(provider, None)]

        # Slow path — lock to prevent duplicate adapter creation
        with self._adapter_lock:
            # Double-check after acquiring lock
            if cache_key in self._adapters:
                return self._adapters[cache_key]
            if base_url is None and (provider, None) in self._adapters:
                return self._adapters[(provider, None)]

            # Need to create a new adapter — check API key first
            api_key = get_api_key(provider)
            if api_key is None:
                raise RuntimeError(
                    f"API key for provider {provider!r} is not configured. "
                    f"Set the appropriate environment variable or .env entry."
                )

            # For on-demand adapters without explicit base_url, check provider defaults
            effective_base_url = base_url
            if effective_base_url is None:
                defaults = self._get_provider_defaults(provider)
                effective_base_url = defaults.get("base_url") if defaults else None
            adapter = self._create_adapter(provider, api_key, effective_base_url)
            self._adapters[cache_key] = adapter
            return adapter

    # --- Capability routing ---

    def web_search(self, query: str) -> LLMResponse:
        """Web search — routed to configured web_search_provider."""
        provider_name = self._config.get("web_search_provider")
        if provider_name is None:
            return LLMResponse(text="")
        try:
            adapter = self.get_adapter(provider_name)
        except RuntimeError:
            return LLMResponse(text="")
        defaults = self._get_provider_defaults(provider_name)
        model = defaults.get("model", "") if defaults else ""
        return adapter.web_search(query, model=model)

    def make_multimodal_message(
        self, text: str, image_bytes: bytes, mime_type: str = "image/png"
    ) -> dict | None:
        """Vision — routed to configured vision_provider."""
        provider_name = self._config.get("vision_provider")
        if provider_name is None:
            return None
        try:
            adapter = self.get_adapter(provider_name)
        except RuntimeError:
            return None
        return adapter.make_multimodal_message(text, image_bytes, mime_type)

    @staticmethod
    def _get_provider_defaults(provider_name: str) -> dict | None:
        """Get DEFAULTS for a provider, reusing config's pre-loaded cache."""
        import config as cfg
        return cfg._PROVIDER_DEFAULTS.get(provider_name)

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
        provider: str | None = None,
        interface: "ChatInterface | None" = None,
    ) -> ChatSession:
        """Start a new multi-turn conversation.

        Returns a ChatSession with a .session_id assigned.
        If *interface* is provided, restores an existing conversation history.
        """
        adapter = self.get_adapter(provider) if provider else self.get_adapter(self._provider, self._base_url)
        session_model = model or self._model
        chat = adapter.create_chat(
            model=session_model,
            system_prompt=system_prompt,
            tools=tools,
            thinking=thinking,
            interaction_id=interaction_id,
            json_schema=json_schema,
            force_tool_call=force_tool_call,
            interface=interface,
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

    def resume_session(self, saved_state: dict, *, thinking: str = "high") -> ChatSession:
        """Restore a session from a saved state dict."""
        session_id = saved_state.get("session_id", "")
        messages = saved_state.get("messages", [])
        metadata = saved_state.get("metadata", {})

        interface = ChatInterface.from_dict(messages)

        # Restore tools from interface so adapters can build provider-specific format
        tools = FunctionSchema.from_dicts(interface.current_tools)

        chat = self.get_adapter(self._provider, self._base_url).create_chat(
            model=self._model,
            system_prompt=interface.current_system_prompt or "",
            tools=tools,
            interface=interface,
            thinking=thinking,
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
        provider: str | None = None,
    ) -> LLMResponse:
        """Single-turn generation.

        If tracked=True (default): logs usage event.
        If tracked=False: fire-and-forget, no logging.
        """
        adapter = self.get_adapter(provider) if provider else self.get_adapter(self._provider, self._base_url)
        gen_model = model or self._model
        response = adapter.generate(
            model=gen_model,
            contents=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_schema=json_schema,
            max_output_tokens=max_output_tokens,
        )
        if tracked and response.usage:
            from agent.event_bus import get_event_bus, TOKEN_USAGE
            usage = response.usage
            get_event_bus().emit(
                TOKEN_USAGE,
                agent="generate",
                level="debug",
                msg=(
                    f"[Tokens] generate in:{usage.input_tokens} "
                    f"out:{usage.output_tokens} think:{usage.thinking_tokens}"
                ),
                data={
                    "agent_name": "generate",
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "thinking_tokens": usage.thinking_tokens,
                    "cached_tokens": usage.cached_tokens,
                },
            )
        return response

    # --- Tool results ---

    def make_tool_result(
        self, tool_name: str, result: dict, *, tool_call_id: str | None = None,
        provider: str | None = None,
    ) -> ToolResultBlock:
        """Build a canonical ToolResultBlock."""
        adapter = self.get_adapter(provider) if provider else self.get_adapter(self._provider, self._base_url)
        return adapter.make_tool_result_message(
            tool_name, result, tool_call_id=tool_call_id,
        )
