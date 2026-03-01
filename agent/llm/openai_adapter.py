"""OpenAI adapter — wraps the ``openai`` SDK for OpenAI and compatible APIs.

Covers: OpenAI, DeepSeek, Qwen (Alibaba), Kimi (Moonshot), MiniMax, Mistral,
xAI Grok, Together AI, Groq, Fireworks, Ollama, vLLM, and any other provider
exposing an OpenAI-compatible ``/chat/completions`` endpoint.

This is the **only** module that imports the ``openai`` package.
"""

from __future__ import annotations

import base64
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
        result.append(
            ToolCall(
                name=tc.function.name,
                args=args,
                id=tc.id,
            )
        )
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
        cached = getattr(raw.usage, "prompt_tokens_details", None)
        cached_tokens = getattr(cached, "cached_tokens", 0) if cached else 0
        usage = UsageMetadata(
            input_tokens=raw.usage.prompt_tokens or 0,
            output_tokens=raw.usage.completion_tokens or 0,
            thinking_tokens=getattr(raw.usage, "completion_tokens_details", None)
            and getattr(raw.usage.completion_tokens_details, "reasoning_tokens", 0)
            or 0,
            cached_tokens=cached_tokens,
        )

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=usage,
        thoughts=thoughts,
        raw=raw,
    )


def _parse_responses_api_response(raw) -> LLMResponse:
    """Parse a raw OpenAI Responses API response into a provider-agnostic LLMResponse."""
    text_parts = []
    tool_calls = []
    thoughts = []

    for item in raw.output or []:
        if item.type == "message":
            for block in item.content or []:
                if block.type == "output_text":
                    text_parts.append(block.text)
        elif item.type == "function_call":
            try:
                args = json.loads(item.arguments) if item.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(name=item.name, args=args, id=item.call_id))
        elif item.type == "reasoning":
            for summary in getattr(item, "summary", None) or []:
                if getattr(summary, "type", None) == "summary_text":
                    thoughts.append(summary.text)

    # Token usage
    usage = UsageMetadata()
    if raw.usage:
        cached = getattr(raw.usage, "input_tokens_details", None)
        cached_tokens = getattr(cached, "cached_tokens", 0) if cached else 0
        usage = UsageMetadata(
            input_tokens=getattr(raw.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw.usage, "output_tokens", 0) or 0,
            thinking_tokens=getattr(raw.usage, "output_tokens_details", None)
            and getattr(raw.usage.output_tokens_details, "reasoning_tokens", 0)
            or 0,
            cached_tokens=cached_tokens,
        )

    return LLMResponse(
        text="\n".join(text_parts),
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
        elif isinstance(message, dict):
            # Pre-built message (e.g., multimodal from make_multimodal_message)
            self._messages.append(message)
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

    def update_tools(self, tools: list[FunctionSchema] | None) -> None:
        """Replace the tool schemas for subsequent calls in this session."""
        self._tools = _build_tools(tools) if tools else None

    def update_system_prompt(self, system_prompt: str) -> None:
        """Replace the system prompt for subsequent calls in this session."""
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = system_prompt
        else:
            # Prepend if not present (rare)
            self._messages.insert(0, {"role": "system", "content": system_prompt})

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

    def send_stream(self, message, on_chunk=None) -> LLMResponse:
        """Send a streaming request."""
        if isinstance(message, str):
            self._messages.append({"role": "user", "content": message})
        elif isinstance(message, dict):
            self._messages.append(message)
        elif isinstance(message, list):
            self._messages.extend(message)
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._extra_kwargs,
        }
        if self._tools:
            kwargs["tools"] = self._tools
            if self._tool_choice:
                kwargs["tool_choice"] = self._tool_choice

        text_parts = []
        _pending_tools = {}
        usage = UsageMetadata()

        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            if not chunk.choices:
                if chunk.usage:
                    cached = getattr(chunk.usage, "prompt_tokens_details", None)
                    cached_tokens = getattr(cached, "cached_tokens", 0) if cached else 0
                    usage = UsageMetadata(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        cached_tokens=cached_tokens,
                    )
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                text_parts.append(delta.content)
                if on_chunk:
                    on_chunk(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in _pending_tools:
                        _pending_tools[idx] = {
                            "id": tc.id or "",
                            "name": (tc.function.name if tc.function else "") or "",
                            "args_json": "",
                        }
                    if tc.id and not _pending_tools[idx]["id"]:
                        _pending_tools[idx]["id"] = tc.id
                    if (
                        tc.function
                        and tc.function.name
                        and not _pending_tools[idx]["name"]
                    ):
                        _pending_tools[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        _pending_tools[idx]["args_json"] += tc.function.arguments

        # Finalize tool calls
        tool_calls = []
        for idx in sorted(_pending_tools):
            pt = _pending_tools[idx]
            try:
                args = json.loads(pt["args_json"]) if pt["args_json"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(name=pt["name"], args=args, id=pt["id"]))

        # Append assistant message to history
        text = "".join(text_parts)
        assistant_msg = {"role": "assistant"}
        if text:
            assistant_msg["content"] = text
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                }
                for tc in tool_calls
            ]
        if not text and not tool_calls:
            assistant_msg["content"] = ""
        self._messages.append(assistant_msg)

        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage, raw=None)


# ---------------------------------------------------------------------------
# OpenAIResponsesSession
# ---------------------------------------------------------------------------


class OpenAIResponsesSession(ChatSession):
    """Session backed by OpenAI's Responses API with server-side state."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        instructions: str,
        tools: list[dict] | None,
        tool_choice: str | None,
        extra_kwargs: dict,
        previous_response_id: str | None = None,
        compact_threshold: int | None = None,
    ):
        self._client = client
        self._model = model
        self._instructions = instructions
        self._tools = tools
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs
        self._response_id: str | None = previous_response_id
        self._compact_threshold = compact_threshold

    def _convert_input(self, message) -> list[dict]:
        """Convert messages to Responses API input format."""
        if isinstance(message, str):
            return [{"role": "user", "content": message}]
        elif isinstance(message, dict):
            return [message]
        elif isinstance(message, list):
            items = []
            for item in message:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "function_call_output"
                ):
                    items.append(item)
                elif isinstance(item, dict) and item.get("role") == "tool":
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": item["tool_call_id"],
                            "output": item["content"],
                        }
                    )
                else:
                    items.append(item)
            return items
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

    def send(self, message) -> LLMResponse:
        """Send a user message (str) or tool results (list of dicts)."""
        input_items = self._convert_input(message)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            **self._extra_kwargs,
        }
        if self._instructions:
            kwargs["instructions"] = self._instructions
        if self._tools:
            kwargs["tools"] = self._tools
            if self._tool_choice:
                kwargs["tool_choice"] = self._tool_choice
        if self._response_id:
            kwargs["previous_response_id"] = self._response_id
        if self._compact_threshold:
            kwargs["context_management"] = [
                {"type": "compaction", "compact_threshold": self._compact_threshold}
            ]

        raw = self._client.responses.create(**kwargs)
        self._response_id = raw.id
        return _parse_responses_api_response(raw)

    def send_stream(self, message, on_chunk=None) -> LLMResponse:
        """Send a streaming request."""
        input_items = self._convert_input(message)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            "stream": True,
            **self._extra_kwargs,
        }
        if self._instructions:
            kwargs["instructions"] = self._instructions
        if self._tools:
            kwargs["tools"] = self._tools
            if self._tool_choice:
                kwargs["tool_choice"] = self._tool_choice
        if self._response_id:
            kwargs["previous_response_id"] = self._response_id
        if self._compact_threshold:
            kwargs["context_management"] = [
                {"type": "compaction", "compact_threshold": self._compact_threshold}
            ]

        text_parts, tool_calls, thoughts = [], [], []
        response_id = None
        _pending_tool = None
        usage = UsageMetadata()

        stream = self._client.responses.create(**kwargs)
        for event in stream:
            if event.type == "response.output_text.delta":
                text_parts.append(event.delta)
                if on_chunk:
                    on_chunk(event.delta)
            elif event.type == "response.function_call_arguments.delta":
                if _pending_tool:
                    _pending_tool["args_json"] += event.delta
            elif event.type == "response.output_item.added":
                if getattr(event.item, "type", None) == "function_call":
                    _pending_tool = {
                        "call_id": event.item.call_id,
                        "name": event.item.name,
                        "args_json": "",
                    }
            elif event.type == "response.output_item.done":
                if (
                    _pending_tool
                    and getattr(event.item, "type", None) == "function_call"
                ):
                    try:
                        args = (
                            json.loads(_pending_tool["args_json"])
                            if _pending_tool["args_json"]
                            else {}
                        )
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(
                        ToolCall(
                            name=_pending_tool["name"],
                            args=args,
                            id=_pending_tool["call_id"],
                        )
                    )
                    _pending_tool = None
            elif event.type == "response.completed":
                response_id = event.response.id
                if event.response.usage:
                    cached = getattr(event.response.usage, "input_tokens_details", None)
                    cached_tokens = getattr(cached, "cached_tokens", 0) if cached else 0
                    usage = UsageMetadata(
                        input_tokens=getattr(event.response.usage, "input_tokens", 0)
                        or 0,
                        output_tokens=getattr(event.response.usage, "output_tokens", 0)
                        or 0,
                        thinking_tokens=getattr(
                            event.response.usage, "output_tokens_details", None
                        )
                        and getattr(
                            event.response.usage.output_tokens_details,
                            "reasoning_tokens",
                            0,
                        )
                        or 0,
                        cached_tokens=cached_tokens,
                    )

        self._response_id = response_id
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            thoughts=thoughts,
            raw=None,
        )

    def get_history(self) -> list[dict]:
        """Return minimal state for session persistence (server-side)."""
        return [{"_response_id": self._response_id}]

    @property
    def session_resume_id(self) -> str | None:
        """Return the response ID for session resumption."""
        return self._response_id


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
        use_responses: bool = False,
    ):
        self.base_url = base_url
        self._use_responses = use_responses
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
        interaction_id: str | None = None,  # ignored — Gemini-specific
    ) -> ChatSession:
        # Check config for whether to use Responses API
        try:
            from config import get as config_get

            use_responses = config_get("providers.openai.use_responses_api", True)
        except ImportError:
            use_responses = self._use_responses

        # Only use Responses API for actual OpenAI (not compatible providers)
        if use_responses and not self.base_url:
            return self._create_responses_session(
                model,
                system_prompt,
                tools,
                json_schema,
                force_tool_call,
                history,
                thinking,
            )

        # Fallback: Chat Completions for compatible providers
        return self._create_completions_session(
            model, system_prompt, tools, json_schema, force_tool_call, history, thinking
        )

    def _create_responses_session(
        self,
        model: str,
        system_prompt: str,
        tools: list[FunctionSchema] | None = None,
        json_schema: dict | None = None,
        force_tool_call: bool = False,
        history: list[dict] | None = None,
        thinking: str = "default",
    ) -> OpenAIResponsesSession:
        # Extract previous_response_id from history if resuming
        previous_response_id = None
        if history:
            for item in history:
                if isinstance(item, dict) and "_response_id" in item:
                    previous_response_id = item["_response_id"]
                    break

        openai_tools = _build_tools(tools)
        tool_choice: str | None = None
        if force_tool_call and openai_tools:
            tool_choice = "required"

        extra_kwargs: dict[str, Any] = {}

        if json_schema is not None:
            extra_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "response"),
                    "strict": True,
                    "schema": json_schema,
                },
            }

        if thinking != "default":
            extra_kwargs["reasoning_effort"] = "high" if thinking == "high" else "low"

        # Get compact threshold from config
        compact_threshold = None
        try:
            from config import get as config_get

            compact_threshold = config_get("providers.openai.compact_threshold", 100000)
        except ImportError:
            pass

        return OpenAIResponsesSession(
            client=self._client,
            model=model,
            instructions=system_prompt,
            tools=openai_tools,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
            previous_response_id=previous_response_id,
            compact_threshold=compact_threshold,
        )

    def _create_completions_session(
        self,
        model: str,
        system_prompt: str,
        tools: list[FunctionSchema] | None = None,
        json_schema: dict | None = None,
        force_tool_call: bool = False,
        history: list[dict] | None = None,
        thinking: str = "default",
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
            extra_kwargs["reasoning_effort"] = "high" if thinking == "high" else "low"

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

        Returns Responses API format if using Responses API, otherwise
        returns Chat Completions format.
        """
        # Determine if we're using Responses API at call time
        try:
            from config import get as config_get

            use_responses = config_get("providers.openai.use_responses_api", True)
        except ImportError:
            use_responses = self._use_responses

        # Only use Responses API for actual OpenAI (not compatible providers)
        if use_responses and not self.base_url:
            return {
                "type": "function_call_output",
                "call_id": tool_call_id or f"call_{uuid.uuid4().hex[:24]}",
                "output": json.dumps(result, default=str),
            }
        return {
            "role": "tool",
            "tool_call_id": tool_call_id or f"call_{uuid.uuid4().hex[:24]}",
            "content": json.dumps(result, default=str),
        }

    def make_multimodal_message(
        self, text: str, image_bytes: bytes, mime_type: str = "image/png"
    ) -> dict:
        """Build an OpenAI multimodal message (image_url + text content blocks).

        Returns a pre-built user message dict that can be passed directly
        to OpenAIChatSession.send() via the dict path.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": text},
            ],
        }

    def is_quota_error(self, exc: Exception) -> bool:
        """Check if the exception is an OpenAI rate-limit error."""
        return isinstance(exc, openai.RateLimitError)

    # -- Convenience properties ------------------------------------------------

    @property
    def client(self):
        """Escape hatch — the underlying ``openai.OpenAI`` client."""
        return self._client
