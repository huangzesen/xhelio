"""Provider-agnostic types and abstract base class for LLM adapters.

All agent code should depend on these types, never on provider-specific SDKs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single function/tool invocation extracted from the LLM response.

    Attributes:
        name: Tool/function name.
        args: Parsed arguments dict.
        id: Provider-assigned call ID (e.g. ``call_xxxxx`` for OpenAI,
            ``toolu_xxxxx`` for Anthropic).  None for Gemini which doesn't
            use explicit tool-call IDs.
    """
    name: str
    args: dict
    id: str | None = None


@dataclass
class UsageMetadata:
    """Normalized token counts."""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class LLMResponse:
    """Provider-agnostic response from an LLM call.

    Attributes:
        text: Concatenated text output (excludes thinking text).
        tool_calls: Extracted function/tool calls.
        usage: Token usage for this call.
        thoughts: List of thinking/reasoning text blocks (for verbose logging).
        raw: The original provider-specific response object. Use for escape
            hatches (e.g. Gemini grounding metadata, multimodal parts).
    """
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: UsageMetadata = field(default_factory=UsageMetadata)
    thoughts: list[str] = field(default_factory=list)
    raw: Any = None


@dataclass
class FunctionSchema:
    """Wraps a tool/function schema dict for type clarity.

    The ``parameters`` dict is already JSON-schema-shaped and provider-agnostic.
    """
    name: str
    description: str
    parameters: dict


# ---------------------------------------------------------------------------
# ChatSession ABC
# ---------------------------------------------------------------------------

class ChatSession(ABC):
    """Abstract multi-turn chat session."""

    @abstractmethod
    def send(self, message) -> LLMResponse:
        """Send a user message or tool results and return the model response.

        ``message`` can be:
        - A string (user text message)
        - A list of tool-result objects (provider-specific, built via
          ``LLMAdapter.make_tool_result_message()``)
        """

    @abstractmethod
    def get_history(self) -> list[dict]:
        """Return serializable conversation history for session persistence."""


# ---------------------------------------------------------------------------
# LLMAdapter ABC
# ---------------------------------------------------------------------------

class LLMAdapter(ABC):
    """Abstract interface that every LLM provider adapter must implement."""

    @abstractmethod
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
    ) -> ChatSession:
        """Create a new multi-turn chat session.

        Args:
            model: Model identifier (e.g. ``"gemini-3-flash-preview"``).
            system_prompt: System instruction for the session.
            tools: Tool/function schemas available to the model.
            json_schema: If set, enforce JSON output conforming to this schema.
            force_tool_call: If True, force the model to call a tool (Gemini
                ``mode="ANY"``).
            history: Previously serialized conversation turns to restore.
            thinking: Thinking level — ``"low"``, ``"high"``, or ``"default"``
                (adapter decides).
            cached_content: Provider-specific cached content identifier.
                For Gemini, this is the ``CachedContent.name`` from
                ``create_cache()``.  When set, the system prompt and tools
                are served from the cache (75% input token discount).
                Ignored by providers that don't support explicit caching.
        """

    @abstractmethod
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
        """One-shot generation (no chat history).

        Used for memory analysis, follow-up suggestions, document extraction,
        and other single-turn calls.
        """

    @abstractmethod
    def make_tool_result_message(
        self, tool_name: str, result: dict, *, tool_call_id: str | None = None
    ) -> Any:
        """Build a provider-specific tool result object.

        Returned value is passed into ``ChatSession.send()`` as part of a list
        of tool results.

        Args:
            tool_name: The name of the tool that was called.
            result: The result dict returned by the tool executor.
            tool_call_id: Provider-assigned tool-call ID from ``ToolCall.id``.
                Required by OpenAI/Anthropic; ignored by Gemini.
        """

    @abstractmethod
    def is_quota_error(self, exc: Exception) -> bool:
        """Return True if ``exc`` represents a quota/rate-limit error (429)."""
