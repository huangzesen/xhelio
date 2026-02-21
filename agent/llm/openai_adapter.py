"""OpenAI adapter — wraps the ``openai`` SDK for OpenAI and compatible APIs.

Covers: OpenAI, DeepSeek, Qwen (Alibaba), Kimi (Moonshot), MiniMax, Mistral,
xAI Grok, Together AI, Groq, Fireworks, Ollama, vLLM, and any other provider
exposing an OpenAI-compatible ``/chat/completions`` endpoint.

This is the **only** module that imports the ``openai`` package.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import openai

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
    """Convert FunctionSchema list to OpenAI tool format."""
    if not schemas:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            },
        }
        for s in schemas
    ]


def _parse_tool_calls(raw_tool_calls) -> list[ToolCall]:
    """Parse OpenAI tool calls into our ToolCall dataclass."""
    if not raw_tool_calls:
        return []
    result = []
    for tc in raw_tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        result.append(ToolCall(
            name=tc.function.name,
            args=args,
            id=tc.id,
        ))
    return result


def _parse_response(raw) -> LLMResponse:
    """Parse a raw OpenAI ChatCompletion into a provider-agnostic LLMResponse."""
    if not raw.choices:
        return LLMResponse(raw=raw)

    choice = raw.choices[0]
    message = choice.message

    text = message.content or ""
    tool_calls = _parse_tool_calls(message.tool_calls)

    # Extract thinking/reasoning (OpenAI o-series models put reasoning in
    # a separate field or content block; the SDK exposes it via
    # message.reasoning_content when available)
    thoughts: list[str] = []
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        thoughts.append(reasoning)

    # Token usage
    usage = UsageMetadata()
    if raw.usage:
        usage = UsageMetadata(
            input_tokens=raw.usage.prompt_tokens or 0,
            output_tokens=raw.usage.completion_tokens or 0,
            thinking_tokens=getattr(raw.usage, "completion_tokens_details", None)
            and getattr(raw.usage.completion_tokens_details, "reasoning_tokens", 0)
            or 0,
        )

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=usage,
        thoughts=thoughts,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# OpenAIChatSession
# ---------------------------------------------------------------------------

class OpenAIChatSession(ChatSession):
    """Client-managed chat session for OpenAI-compatible APIs.

    Unlike Gemini's SDK-managed sessions, OpenAI requires the client to
    maintain and send the full message list on every request.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | None,
        extra_kwargs: dict,
    ):
        self._client = client
        self._model = model
        self._messages = messages
        self._tools = tools
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs

    def send(self, message) -> LLMResponse:
        """Send a user message (str) or tool results (list of dicts).

        For tool results, ``message`` is a list of dicts, each built by
        :meth:`OpenAIAdapter.make_tool_result_message`.
        """
        if isinstance(message, str):
            self._messages.append({"role": "user", "content": message})
        elif isinstance(message, list):
            # Tool results — each is a dict with role/tool_call_id/content.
            # We also need the assistant message with the tool_calls that
            # triggered these results, but that was already appended when
            # we parsed the previous response.
            self._messages.extend(message)
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            **self._extra_kwargs,
        }
        if self._tools:
            kwargs["tools"] = self._tools
            if self._tool_choice:
                kwargs["tool_choice"] = self._tool_choice
        raw = self._client.chat.completions.create(**kwargs)

        # Append the assistant response to the message list
        assistant_msg = self._response_to_message(raw)
        self._messages.append(assistant_msg)

        return _parse_response(raw)

    def get_history(self) -> list[dict]:
        """Return the message list for session persistence."""
        return list(self._messages)

    @staticmethod
    def _response_to_message(raw) -> dict:
        """Convert an OpenAI ChatCompletion response to a message dict for history."""
        choice = raw.choices[0] if raw.choices else None
        if not choice:
            return {"role": "assistant", "content": ""}
        msg = choice.message
        result: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            result["content"] = msg.content
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        if not msg.content and not msg.tool_calls:
            result["content"] = ""
        return result


# ---------------------------------------------------------------------------
# OpenAIAdapter
# ---------------------------------------------------------------------------

class OpenAIAdapter(LLMAdapter):
    """Adapter that wraps the ``openai`` SDK for OpenAI and compatible APIs."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout_ms: int = 300_000,
    ):
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        kwargs["timeout"] = timeout_ms / 1000.0  # openai SDK uses seconds
        self._client = openai.OpenAI(**kwargs)

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
        cached_content: str | None = None,  # ignored — OpenAI uses automatic caching
    ) -> OpenAIChatSession:
        # Start with system message + any restored history
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)

        openai_tools = _build_tools(tools)
        tool_choice: str | None = None
        if force_tool_call and openai_tools:
            tool_choice = "required"

        # Extra kwargs for the completions call
        extra_kwargs: dict[str, Any] = {}

        # JSON schema enforcement (OpenAI Structured Outputs)
        if json_schema is not None:
            extra_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "response"),
                    "strict": True,
                    "schema": json_schema,
                },
            }

        # Reasoning effort for o-series models
        if thinking != "default":
            extra_kwargs["reasoning_effort"] = (
                "high" if thinking == "high" else "low"
            )

        return OpenAIChatSession(
            client=self._client,
            model=model,
            messages=messages,
            tools=openai_tools,
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
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # contents can be a string or a list of content blocks
        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        elif isinstance(contents, list):
            messages.append({"role": "user", "content": contents})
        else:
            messages.append({"role": "user", "content": str(contents)})

        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens

        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "response"),
                    "strict": True,
                    "schema": json_schema,
                },
            }

        raw = self._client.chat.completions.create(**kwargs)
        return _parse_response(raw)

    def make_tool_result_message(
        self, tool_name: str, result: dict, *, tool_call_id: str | None = None
    ) -> dict:
        """Build an OpenAI tool-result message dict.

        OpenAI requires ``tool_call_id`` to match the original tool call.
        If not provided, generates a placeholder ID (may cause issues with
        some strict providers).
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id or f"call_{uuid.uuid4().hex[:24]}",
            "content": json.dumps(result, default=str),
        }

    def is_quota_error(self, exc: Exception) -> bool:
        """Check if the exception is an OpenAI rate-limit error."""
        return isinstance(exc, openai.RateLimitError)

    # -- Convenience properties ------------------------------------------------

    @property
    def client(self):
        """Escape hatch — the underlying ``openai.OpenAI`` client."""
        return self._client
