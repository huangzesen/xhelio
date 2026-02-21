"""Gemini adapter — wraps all google-genai SDK calls.

This is the **only** module in the project that imports ``google.genai``.
All other agent code talks to Gemini through the :class:`GeminiAdapter` and
:class:`GeminiChatSession` interfaces defined here.
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import errors as genai_errors, types

from .base import (
    ChatSession,
    FunctionSchema,
    LLMAdapter,
    LLMResponse,
    ToolCall,
    UsageMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_function_declarations(
    tools: list[FunctionSchema] | None,
) -> list[types.FunctionDeclaration] | None:
    """Convert our FunctionSchema list to Gemini FunctionDeclaration list."""
    if not tools:
        return None
    return [
        types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
        )
        for t in tools
    ]


def _parse_response(raw) -> LLMResponse:
    """Parse a raw Gemini response into a provider-agnostic LLMResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thoughts: list[str] = []

    candidates = getattr(raw, "candidates", None) or []
    if candidates:
        content = candidates[0].content
        if content and content.parts:
            for part in content.parts:
                if getattr(part, "thought", False) and hasattr(part, "text") and part.text:
                    thoughts.append(part.text)
                elif hasattr(part, "function_call") and part.function_call and part.function_call.name:
                    tool_calls.append(ToolCall(
                        name=part.function_call.name,
                        args=dict(part.function_call.args) if part.function_call.args else {},
                    ))
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

    # Token usage
    meta = getattr(raw, "usage_metadata", None)
    usage = UsageMetadata(
        input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        thinking_tokens=getattr(meta, "thoughts_token_count", 0) or 0,
        cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
    ) if meta else UsageMetadata()

    return LLMResponse(
        text="\n".join(text_parts) if text_parts else "",
        tool_calls=tool_calls,
        usage=usage,
        thoughts=thoughts,
        raw=raw,
    )


def _supports_thinking(model: str) -> bool:
    """Return True if the model supports thinking config (Gemini 3+)."""
    # Match model names like "gemini-3-flash-preview", "gemini-3-pro", etc.
    # Gemini 2.x (including 2.5-flash-preview) does NOT support thinking.
    parts = model.lower().replace("models/", "").split("-")
    if len(parts) >= 2 and parts[0] == "gemini":
        try:
            major = int(parts[1].split(".")[0])
            return major >= 3
        except (ValueError, IndexError):
            pass
    return False


def _thinking_config(level: str) -> types.ThinkingConfig | None:
    """Build a Gemini ThinkingConfig from a normalized level string.

    Returns None if thinking is disabled ("off").
    """
    if level == "off":
        return None
    level_upper = level.upper() if level != "default" else "LOW"
    return types.ThinkingConfig(include_thoughts=True, thinking_level=level_upper)


# ---------------------------------------------------------------------------
# GeminiChatSession
# ---------------------------------------------------------------------------

class GeminiChatSession(ChatSession):
    """Wraps a ``genai`` chat session."""

    def __init__(self, chat):
        self._chat = chat

    def send(self, message) -> LLMResponse:
        """Send a message (text or list of tool-result Parts) and parse the response."""
        raw = self._chat.send_message(message)
        return _parse_response(raw)

    def get_history(self) -> list[dict]:
        """Return serializable history dicts (for session persistence)."""
        return [
            content.model_dump(exclude_none=True)
            for content in self._chat.get_history()
        ]

    @property
    def raw_chat(self):
        """Escape hatch — the underlying ``genai`` chat object."""
        return self._chat


# ---------------------------------------------------------------------------
# GeminiAdapter
# ---------------------------------------------------------------------------

class GeminiAdapter(LLMAdapter):
    """Adapter that wraps all ``google-genai`` SDK calls."""

    def __init__(self, api_key: str, timeout_ms: int = 300_000):
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(),
            ),
        )
        # Registry of cache ingredients for auto-recreation on expiry.
        # Keyed by cache name → {model, system_prompt, tools, ttl}.
        self._cache_registry: dict[str, dict] = {}
        # Alias map: old expired cache name → current live name.
        self._cache_aliases: dict[str, str] = {}

    # -- LLMAdapter interface --------------------------------------------------

    def create_cache(
        self,
        model: str,
        system_prompt: str,
        tools: list[FunctionSchema] | None = None,
        ttl: str = "86400s",
    ) -> str:
        """Create a Gemini cached content object.

        Caches the system prompt and tool declarations so they don't count
        as fresh input tokens on every API call (75% discount).

        Args:
            model: Model identifier (must match the model used in create_chat).
            system_prompt: System instruction to cache.
            tools: Tool schemas to cache alongside the system prompt.
            ttl: Time-to-live (default 24 hours).

        Returns:
            The cache resource name (e.g. ``"cachedContents/abc123"``).
        """
        cache_config: dict[str, Any] = {
            "system_instruction": system_prompt,
            "ttl": ttl,
        }
        fds = _build_function_declarations(tools)
        if fds:
            cache_config["tools"] = [types.Tool(function_declarations=fds)]

        cache = self._client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(**cache_config),
        )
        self._cache_registry[cache.name] = {
            "model": model,
            "system_prompt": system_prompt,
            "tools": tools,
            "ttl": ttl,
        }
        return cache.name

    def ensure_cache(self, cache_name: str) -> str | None:
        """Validate a cache is still alive; recreate from stored ingredients if expired.

        Returns the valid cache name (may differ from input if recreated),
        or None if the cache cannot be restored. Callers may hold stale names;
        an internal alias map redirects them to the latest recreation.
        """
        if not cache_name:
            return None
        # Follow alias chain (old expired name → current live name)
        resolved = cache_name
        while resolved in self._cache_aliases:
            resolved = self._cache_aliases[resolved]
        # Quick check — is the cache still alive?
        try:
            self._client.caches.get(name=resolved)
            return resolved
        except Exception:
            pass
        # Expired — try to recreate from stored ingredients
        ingredients = self._cache_registry.get(resolved)
        if not ingredients:
            return None
        try:
            new_name = self.create_cache(**ingredients)
            # Alias: old name(s) → new name
            self._cache_aliases[resolved] = new_name
            if cache_name != resolved:
                self._cache_aliases[cache_name] = new_name
            return new_name
        except Exception:
            return None

    def create_chat(
        self,
        model: str,
        system_prompt: str,
        tools: list[FunctionSchema] | None = None,
        *,
        json_schema: dict | None = None,
        force_tool_call: bool = False,
        history: list[dict] | None = None,
        thinking: str = "default",
        cached_content: str | None = None,
    ) -> GeminiChatSession:
        # Validate cache before using it
        if cached_content:
            cached_content = self.ensure_cache(cached_content)

        # Build GenerateContentConfig
        config_kwargs: dict[str, Any] = {}

        if cached_content:
            # System prompt + tools are in the cache — use cached_content instead
            config_kwargs["cached_content"] = cached_content
        else:
            config_kwargs["system_instruction"] = system_prompt

        # Only send thinking_config for Gemini 3+ models.
        # Per-call `thinking` param indicates the tier: "high" = smart model
        # (orchestrator/planner), "low" = sub-agent. Config overrides per tier.
        if _supports_thinking(model):
            from config import GEMINI_THINKING_MODEL, GEMINI_THINKING_SUB_AGENT
            if thinking in ("high", "default"):
                effective = GEMINI_THINKING_MODEL
            else:
                effective = GEMINI_THINKING_SUB_AGENT
            tc = _thinking_config(effective)
            if tc is not None:
                config_kwargs["thinking_config"] = tc

        # Tools — skip if already in the cache
        if not cached_content:
            fds = _build_function_declarations(tools)
            if fds:
                config_kwargs["tools"] = [types.Tool(function_declarations=fds)]
        # tool_config is NOT cached — but Gemini rejects it with cached_content
        if force_tool_call and tools and not cached_content:
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
                )

        # JSON schema enforcement
        if json_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = json_schema

        config = types.GenerateContentConfig(**config_kwargs)

        # Create the chat
        create_kwargs: dict[str, Any] = {"model": model, "config": config}
        if history:
            create_kwargs["history"] = history

        chat = self._client.chats.create(**create_kwargs)
        return GeminiChatSession(chat)

    def generate(
        self,
        model: str,
        contents: str | list,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        json_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        config_kwargs: dict[str, Any] = {}
        if system_prompt is not None:
            config_kwargs["system_instruction"] = system_prompt
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens
        if json_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = json_schema

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        raw = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return _parse_response(raw)

    def make_tool_result_message(
        self, tool_name: str, result: dict, *, tool_call_id: str | None = None
    ) -> Any:
        # Gemini doesn't use tool_call_id — it matches by name.
        return types.Part.from_function_response(
            name=tool_name,
            response={"result": result},
        )

    def is_quota_error(self, exc: Exception) -> bool:
        if isinstance(exc, genai_errors.ClientError):
            return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)
        return False

    def delete_cache(self, cache_name: str) -> None:
        """Delete a cached content object to stop storage charges."""
        self._cache_registry.pop(cache_name, None)
        # Clean up any aliases pointing to this cache
        self._cache_aliases = {k: v for k, v in self._cache_aliases.items() if v != cache_name}
        self._client.caches.delete(name=cache_name)

    # -- Gemini-specific methods (not in ABC) ----------------------------------

    def google_search(self, query: str, model: str) -> LLMResponse:
        """Execute a Google Search grounded generation call.

        This is Gemini-specific (uses GoogleSearch tool).
        """
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        raw = self._client.models.generate_content(
            model=model,
            contents=query,
            config=config,
        )
        return _parse_response(raw)

    def generate_multimodal(
        self, model: str, contents: list, **kwargs
    ) -> LLMResponse:
        """Generate with multimodal content (e.g. image/PDF bytes + text).

        This is an escape hatch for document extraction which sends
        ``types.Part.from_bytes(...)`` directly.
        """
        config = types.GenerateContentConfig(**kwargs) if kwargs else None
        raw = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return _parse_response(raw)

    @staticmethod
    def make_bytes_part(data: bytes, mime_type: str) -> Any:
        """Create a Gemini Part from raw bytes (for document/image input)."""
        return types.Part.from_bytes(data=data, mime_type=mime_type)

    @property
    def client(self):
        """Escape hatch — the underlying ``genai.Client``."""
        return self._client
