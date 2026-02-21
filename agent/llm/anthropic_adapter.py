"""Anthropic adapter — wraps the ``anthropic`` SDK for Claude models.

This is the **only** module that imports the ``anthropic`` package.

Key Anthropic API differences from OpenAI/Gemini:
- System prompt is a separate ``system`` parameter, not a message.
- Strict user/assistant alternation required — consecutive same-role messages
  must be merged.
- Tool results are sent inside a ``user`` message with ``tool_result`` blocks.
- Thinking/extended thinking is controlled via a ``thinking`` parameter with
  a token budget.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import anthropic

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

def _build_tools(schemas: list[FunctionSchema] | None) -> list[dict] | None:
    """Convert FunctionSchema list to Anthropic tool format."""
    if not schemas:
        return None
    return [
        {
            "name": s.name,
            "description": s.description,
            "input_schema": s.parameters,
        }
        for s in schemas
    ]


def _parse_response(raw) -> LLMResponse:
    """Parse an Anthropic Messages response into a provider-agnostic LLMResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thoughts: list[str] = []

    for block in raw.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                name=block.name,
                args=block.input if isinstance(block.input, dict) else {},
                id=block.id,
            ))
        elif block.type == "thinking":
            thinking_text = getattr(block, "thinking", None)
            if thinking_text:
                thoughts.append(thinking_text)

    # Token usage
    usage = UsageMetadata()
    if raw.usage:
        usage = UsageMetadata(
            input_tokens=getattr(raw.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw.usage, "output_tokens", 0) or 0,
        )

    return LLMResponse(
        text="\n".join(text_parts) if text_parts else "",
        tool_calls=tool_calls,
        usage=usage,
        thoughts=thoughts,
        raw=raw,
    )


def _ensure_alternation(messages: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages to satisfy Anthropic's alternation rule.

    Anthropic requires strict user/assistant alternation. If two consecutive
    messages have the same role, merge their content.
    """
    if not messages:
        return messages

    merged: list[dict] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]
            # Merge content — both could be str or list
            prev_content = prev.get("content", "")
            new_content = msg.get("content", "")

            # Normalize to list form for merging
            if isinstance(prev_content, str):
                prev_list = [{"type": "text", "text": prev_content}] if prev_content else []
            else:
                prev_list = list(prev_content)

            if isinstance(new_content, str):
                new_list = [{"type": "text", "text": new_content}] if new_content else []
            else:
                new_list = list(new_content)

            combined = prev_list + new_list
            prev["content"] = combined
        else:
            merged.append(dict(msg))

    return merged


def _response_to_messages(raw) -> list[dict]:
    """Convert an Anthropic response into message dicts for the history."""
    result: dict[str, Any] = {"role": "assistant", "content": []}

    for block in raw.content:
        if block.type == "text":
            result["content"].append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result["content"].append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input if isinstance(block.input, dict) else {},
            })
        elif block.type == "thinking":
            # Include thinking blocks so history round-trips correctly
            result["content"].append({
                "type": "thinking",
                "thinking": getattr(block, "thinking", ""),
                # Anthropic requires a signature for thinking blocks in history
                "signature": getattr(block, "signature", ""),
            })

    if not result["content"]:
        result["content"] = [{"type": "text", "text": ""}]

    return [result]


# ---------------------------------------------------------------------------
# AnthropicChatSession
# ---------------------------------------------------------------------------

class AnthropicChatSession(ChatSession):
    """Client-managed chat session for the Anthropic Messages API.

    Maintains a message list and ensures strict user/assistant alternation.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: dict | None,
        extra_kwargs: dict,
    ):
        self._client = client
        self._model = model
        self._system = system_prompt
        self._messages = messages
        self._tools = tools
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs

    def send(self, message) -> LLMResponse:
        """Send a user message (str) or tool results (list of dicts).

        For tool results, ``message`` is a list of dicts, each built by
        :meth:`AnthropicAdapter.make_tool_result_message`. These get wrapped
        in a single user message with all tool_result blocks.
        """
        if isinstance(message, str):
            self._messages.append({"role": "user", "content": message})
        elif isinstance(message, list):
            # Tool results — wrap in a single user message
            self._messages.append({
                "role": "user",
                "content": message,
            })
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

        # Ensure alternation before sending
        clean_messages = _ensure_alternation(self._messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": clean_messages,
            "max_tokens": self._extra_kwargs.pop("max_tokens", 8192),
            **self._extra_kwargs,
        }
        if self._system:
            kwargs["system"] = self._system
        if self._tools:
            kwargs["tools"] = self._tools
            if self._tool_choice:
                kwargs["tool_choice"] = self._tool_choice

        raw = self._client.messages.create(**kwargs)

        # Append the assistant response to history
        assistant_msgs = _response_to_messages(raw)
        self._messages.extend(assistant_msgs)

        return _parse_response(raw)

    def get_history(self) -> list[dict]:
        """Return the message list for session persistence."""
        return list(self._messages)


# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------

class AnthropicAdapter(LLMAdapter):
    """Adapter that wraps the ``anthropic`` SDK for Claude models."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_ms: int = 300_000,
    ):
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_ms / 1000.0,
        )

    # -- LLMAdapter interface --------------------------------------------------

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
        cached_content: str | None = None,  # ignored — Anthropic uses automatic caching
    ) -> AnthropicChatSession:
        messages: list[dict] = []
        if history:
            messages.extend(history)

        anthropic_tools = _build_tools(tools)
        tool_choice: dict | None = None
        if force_tool_call and anthropic_tools:
            tool_choice = {"type": "any"}

        # JSON schema enforcement via tool-based structured output
        if json_schema is not None and anthropic_tools is None:
            anthropic_tools = []
        if json_schema is not None:
            schema_tool_name = json_schema.get("title", "structured_output")
            anthropic_tools.append({
                "name": schema_tool_name,
                "description": "Return the structured response matching the required schema.",
                "input_schema": json_schema,
            })
            tool_choice = {"type": "tool", "name": schema_tool_name}

        extra_kwargs: dict[str, Any] = {}

        # Thinking/extended thinking
        if thinking == "high":
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 16384,
            }
            extra_kwargs["max_tokens"] = 24576  # must be > budget
        elif thinking == "low":
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 2048,
            }
            extra_kwargs["max_tokens"] = 10240

        return AnthropicChatSession(
            client=self._client,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=anthropic_tools,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )

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
        messages: list[dict] = []
        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        elif isinstance(contents, list):
            messages.append({"role": "user", "content": contents})
        else:
            messages.append({"role": "user", "content": str(contents)})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens or 8192,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature

        if json_schema is not None:
            tools = [{
                "name": json_schema.get("title", "structured_output"),
                "description": "Return the structured response.",
                "input_schema": json_schema,
            }]
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {
                "type": "tool",
                "name": json_schema.get("title", "structured_output"),
            }

        raw = self._client.messages.create(**kwargs)
        return _parse_response(raw)

    def make_tool_result_message(
        self, tool_name: str, result: dict, *, tool_call_id: str | None = None
    ) -> dict:
        """Build an Anthropic tool_result content block.

        Returns a dict that gets collected into a list and wrapped in a
        ``{"role": "user", "content": [...]}`` message by the session.
        """
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id or f"toolu_{uuid.uuid4().hex[:24]}",
            "content": json.dumps(result, default=str),
        }

    def is_quota_error(self, exc: Exception) -> bool:
        """Check if the exception is an Anthropic rate-limit error."""
        return isinstance(exc, anthropic.RateLimitError)

    # -- Convenience properties ------------------------------------------------

    @property
    def client(self):
        """Escape hatch — the underlying ``anthropic.Anthropic`` client."""
        return self._client
