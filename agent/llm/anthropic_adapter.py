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

import base64
import json
import uuid
from collections.abc import Callable
from typing import Any

import anthropic

import config
from agent.logging import get_logger

logger = get_logger()

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


def _build_tools(
    schemas: list[FunctionSchema] | None, *, cache_tools: bool = False
) -> list[dict] | None:
    """Convert FunctionSchema list to Anthropic tool format."""
    if not schemas:
        return None
    tools = [
        {
            "name": s.name,
            "description": s.description,
            "input_schema": s.parameters,
        }
        for s in schemas
    ]
    if cache_tools and tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


def _build_system_with_cache(system_prompt: str) -> list[dict]:
    """Build system prompt as cached content blocks for Anthropic."""
    if not system_prompt:
        return []
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
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
            tool_calls.append(
                ToolCall(
                    name=block.name,
                    args=block.input if isinstance(block.input, dict) else {},
                    id=block.id,
                )
            )
        elif block.type == "thinking":
            thinking_text = getattr(block, "thinking", None)
            if thinking_text:
                thoughts.append(thinking_text)

    # Token usage — includes cache metrics
    # Anthropic's input_tokens only counts tokens AFTER the last cache
    # breakpoint.  The true total is: input_tokens + cache_read + cache_write.
    # We normalise here so the rest of the system sees the same semantics as
    # OpenAI (prompt_tokens = total) and Gemini (prompt_token_count = total).
    usage = UsageMetadata()
    if raw.usage:
        cache_read = getattr(raw.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(raw.usage, "cache_creation_input_tokens", 0) or 0
        raw_input = getattr(raw.usage, "input_tokens", 0) or 0
        usage = UsageMetadata(
            input_tokens=raw_input + cache_read + cache_write,
            output_tokens=getattr(raw.usage, "output_tokens", 0) or 0,
            thinking_tokens=getattr(raw.usage, "thinking_tokens", 0) or 0,
            cached_tokens=cache_read,
        )
        if cache_read or cache_write:
            logger.debug(
                "Anthropic cache: read=%d write=%d uncached=%d total_input=%d",
                cache_read,
                cache_write,
                raw_input,
                usage.input_tokens,
            )

    return LLMResponse(
        text="\n".join(text_parts) if text_parts else "",
        tool_calls=tool_calls,
        usage=usage,
        thoughts=thoughts,
        raw=raw,
    )


def _parse_search_response(raw) -> LLMResponse:
    """Parse an Anthropic response that used web search tool."""
    text_parts: list[str] = []
    for block in raw.content:
        if block.type == "text":
            text_parts.append(block.text)

    usage = UsageMetadata()
    if raw.usage:
        cache_read = getattr(raw.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(raw.usage, "cache_creation_input_tokens", 0) or 0
        raw_input = getattr(raw.usage, "input_tokens", 0) or 0
        usage = UsageMetadata(
            input_tokens=raw_input + cache_read + cache_write,
            output_tokens=getattr(raw.usage, "output_tokens", 0) or 0,
            thinking_tokens=getattr(raw.usage, "thinking_tokens", 0) or 0,
            cached_tokens=cache_read,
        )

    return LLMResponse(
        text="\n".join(text_parts),
        usage=usage,
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
                prev_list = (
                    [{"type": "text", "text": prev_content}] if prev_content else []
                )
            else:
                prev_list = list(prev_content)

            if isinstance(new_content, str):
                new_list = (
                    [{"type": "text", "text": new_content}] if new_content else []
                )
            else:
                new_list = list(new_content)

            combined = prev_list + new_list
            prev["content"] = combined
        else:
            merged.append(dict(msg))

    return merged


def _filter_invalid_tool_results(messages: list[dict]) -> list[dict]:
    """Filter out tool_results with invalid tool_use_id, replacing with explanatory text.

    Validates each tool_result.tool_use_id against valid tool_use.id in history.
    If invalid, replaces with factual text explaining the mismatch to the LLM.

    This prevents Anthropic 400 error "tool call result does not follow tool call"
    by catching mismatches before sending to the API.
    """
    # Step 1: Collect all valid tool_use IDs from assistant messages
    valid_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tid = block.get("id")
                    if tid:
                        valid_ids.add(tid)

    # Step 2: Filter tool_results in user messages
    cleaned: list[dict] = []
    for msg in messages:
        if msg.get("role") != "user":
            cleaned.append(msg)
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        new_content: list[dict] = []
        for block in content:
            # Keep non-tool_result blocks as-is
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                new_content.append(block)
                continue

            tool_use_id = block.get("tool_use_id")
            original_content = block.get("content", "")

            # Keep valid tool_results as-is
            if tool_use_id and tool_use_id in valid_ids:
                new_content.append(block)
                continue

            # Invalid ID - replace with explanatory text
            tool_name = block.get("name", "unknown")
            result_preview = str(original_content)[:200]

            new_content.append({
                "type": "text",
                "text": (
                    f"[Tool {tool_name} result ignored — tool_use_id={tool_use_id} does not match "
                    f"any tool call in history. Original result: {result_preview}]"
                ),
            })

        if new_content:
            cleaned.append({**msg, "content": new_content})

    return cleaned


def _response_to_messages(raw) -> list[dict]:
    """Convert an Anthropic response into message dicts for the history."""
    result: dict[str, Any] = {"role": "assistant", "content": []}

    for block in raw.content:
        if block.type == "text":
            result["content"].append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result["content"].append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input if isinstance(block.input, dict) else {},
                }
            )
        elif block.type == "thinking":
            # Include thinking blocks so history round-trips correctly
            result["content"].append(
                {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                    # Anthropic requires a signature for thinking blocks in history
                    "signature": getattr(block, "signature", ""),
                }
            )

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
        system_prompt: str | list[dict],
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: dict | None,
        extra_kwargs: dict,
        client_kwargs: dict | None = None,
    ):
        self._client = client
        self._model = model
        self._system = system_prompt
        self._messages = messages
        self._tools = tools
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs
        self._client_kwargs = client_kwargs or {}

        # Context window for compaction
        from agent.llm_utils import get_context_limit
        self._context_window = get_context_limit(model)

    def _build_request_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._extra_kwargs.get("max_tokens", 8192),
            **self._extra_kwargs,
        }
        if self._system:
            kwargs["system"] = self._system
        if self._tools:
            kwargs["tools"] = self._tools
            if self._tool_choice:
                kwargs["tool_choice"] = self._tool_choice
        return kwargs

    def send(self, message) -> LLMResponse:
        """Send a user message (str) or tool results (list of dicts).

        For tool results, ``message`` is a list of dicts, each built by
        :meth:`AnthropicAdapter.make_tool_result_message`. These get wrapped
        in a single user message with all tool_result blocks.

        The user message is only committed to history after a successful API
        call, preventing duplicate messages on retry.
        """
        # Build candidate messages without mutating self._messages
        candidate = list(self._messages)
        if isinstance(message, str):
            candidate.append({"role": "user", "content": message})
        elif isinstance(message, dict):
            candidate.append(message)
        elif isinstance(message, list):
            candidate.append({"role": "user", "content": message})
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

        clean_messages = _ensure_alternation(candidate)
        clean_messages = _filter_invalid_tool_results(clean_messages)
        kwargs = self._build_request_kwargs(clean_messages)

        try:
            raw = self._client.messages.create(**kwargs)
        except Exception as api_err:
            # DEBUG: Save messages on error 2013
            err_str = str(api_err)
            if "tool call result does not follow" in err_str or "2013" in err_str:
                import os
                import tempfile
                try:
                    debug_dir = os.path.join(tempfile.gettempdir(), "xhelio_desync_debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    import time as ts
                    debug_file = os.path.join(debug_dir, f"error_{int(ts.time()*1000)}.json")
                    with open(debug_file, "w") as f:
                        json.dump({
                            "error": err_str,
                            "candidate_messages": candidate,
                            "clean_messages": clean_messages,
                            "kwargs": {k: v for k, v in kwargs.items() if k != "messages"},
                        }, f, indent=2, default=str)
                    logger.warning(f"DEBUG: Saved desync error details to {debug_file}")
                except Exception as debug_err:
                    logger.warning(f"DEBUG: Failed to save desync info: {debug_err}")
            raise

        # Commit: adopt candidate and append the assistant response
        self._messages = candidate
        assistant_msgs = _response_to_messages(raw)
        self._messages.extend(assistant_msgs)

        return _parse_response(raw)

    def send_stream(
        self,
        message,
        on_chunk: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Streaming send. User message committed to history only after success."""
        # Build candidate messages without mutating self._messages
        candidate = list(self._messages)
        if isinstance(message, str):
            candidate.append({"role": "user", "content": message})
        elif isinstance(message, dict):
            candidate.append(message)
        elif isinstance(message, list):
            candidate.append({"role": "user", "content": message})
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

        clean_messages = _ensure_alternation(candidate)
        clean_messages = _filter_invalid_tool_results(clean_messages)
        kwargs = self._build_request_kwargs(clean_messages)

        text_parts, tool_calls, thoughts = [], [], []
        _pending_tool = None
        _thinking_parts: list[str] = []

        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        _pending_tool = {
                            "id": block.id,
                            "name": block.name,
                            "args_json": "",
                        }
                    elif block and getattr(block, "type", None) == "thinking":
                        _thinking_parts = []
                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        t = getattr(delta, "text", "")
                        if t:
                            text_parts.append(t)
                            if on_chunk:
                                on_chunk(t)
                    elif dtype == "thinking_delta":
                        t = getattr(delta, "thinking", "")
                        if t:
                            _thinking_parts.append(t)
                    elif dtype == "input_json_delta":
                        partial = getattr(delta, "partial_json", "")
                        if partial and _pending_tool is not None:
                            _pending_tool["args_json"] += partial
                elif etype == "content_block_stop":
                    if _thinking_parts:
                        thoughts.append("".join(_thinking_parts))
                        _thinking_parts = []
                    if _pending_tool is not None:
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
                                id=_pending_tool["id"],
                            )
                        )
                        _pending_tool = None

            final_message = stream.get_final_message()

        # Extract usage from final message (includes cache metrics)
        # Same normalisation as _parse_response — see comment there.
        usage = UsageMetadata()
        if final_message and final_message.usage:
            u = final_message.usage
            cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
            raw_input = getattr(u, "input_tokens", 0) or 0
            usage = UsageMetadata(
                input_tokens=raw_input + cache_read + cache_write,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                thinking_tokens=getattr(u, "thinking_tokens", 0) or 0,
                cached_tokens=cache_read,
            )
            if cache_read or cache_write:
                logger.debug(
                    "Anthropic cache (stream): read=%d write=%d uncached=%d total_input=%d",
                    cache_read,
                    cache_write,
                    raw_input,
                    usage.input_tokens,
                )

        # Commit: adopt candidate and append assistant response to history
        self._messages = candidate
        if final_message:
            self._messages.extend(_response_to_messages(final_message))

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            thoughts=thoughts,
            raw=final_message,
        )

    def commit_tool_results(self, tool_results: list) -> None:
        """Append tool results to history without an API call."""
        if tool_results:
            self._messages.append({"role": "user", "content": tool_results})

    def get_history(self) -> list[dict]:
        """Return the message list for session persistence."""
        return list(self._messages)

    def update_tools(self, tools: list[FunctionSchema] | None) -> None:
        """Replace the tool schemas for subsequent calls in this session."""
        self._tools = _build_tools(tools, cache_tools=True) if tools else None

    def update_system_prompt(self, system_prompt: str) -> None:
        """Replace the system prompt for subsequent calls in this session."""
        self._system = _build_system_with_cache(system_prompt)

    def reset(self) -> None:
        """Create a truly fresh session instance while preserving state.

        Reconstructs a new AnthropicChatSession with a fresh HTTP client
        and copies all attributes onto self, giving a clean connection and
        fresh internal state.
        """
        if self._client_kwargs:
            new_client = anthropic.Anthropic(**self._client_kwargs)
            new_session = AnthropicChatSession(
                client=new_client,
                model=self._model,
                system_prompt=self._system,
                messages=list(self._messages),
                tools=self._tools,
                tool_choice=self._tool_choice,
                extra_kwargs=self._extra_kwargs,
                client_kwargs=self._client_kwargs,
            )
            self.__dict__.update(new_session.__dict__)

    # -- Context compaction ---------------------------------------------------

    def context_window(self) -> int:
        return self._context_window

    def estimate_context_tokens(self) -> int:
        """Estimate total tokens in current context."""
        from agent.token_counter import count_tokens
        total = 0
        # System prompt
        if self._system:
            if isinstance(self._system, str):
                total += count_tokens(self._system)
            elif isinstance(self._system, list):
                total += count_tokens(json.dumps(self._system, default=str))
        # Tool definitions
        if self._tools:
            total += count_tokens(json.dumps(self._tools, default=str))
        # Messages
        if self._messages:
            total += count_tokens(json.dumps(self._messages, default=str))
        return total

    def compact(self, summarizer: Callable[[str], str]) -> bool:
        """Compact older messages by summarizing them.

        Keeps the last 3 complete turns (6+ messages) intact, summarizes
        everything before that. Tool-use/tool-result pairs are never split.

        Returns True if compaction happened.
        """
        if len(self._messages) < 8:
            return False

        # Find safe compaction boundary — walk backward keeping last 3 turns.
        # A "turn" is a user message + assistant response (with possible
        # tool_use/tool_result exchanges between them).
        boundary = self._find_compaction_boundary(keep_turns=3)
        if boundary is None or boundary < 2:
            return False

        # Format older messages for summarization
        older = self._messages[:boundary]
        text_parts = []
        for msg in older:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(f"[{role}] {content}")
            elif isinstance(content, list):
                for block in content:
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(f"[{role}] {block.get('text', '')}")
                    elif btype == "tool_use":
                        text_parts.append(
                            f"[{role}] tool_use: {block.get('name', '?')}({json.dumps(block.get('input', {}), default=str)[:200]})"
                        )
                    elif btype == "tool_result":
                        result_text = str(block.get("content", ""))[:300]
                        text_parts.append(f"[{role}] tool_result: {result_text}")
                    elif btype == "thinking":
                        pass  # Drop thinking blocks — large, not actionable
        raw_text = "\n".join(text_parts)

        if not raw_text.strip():
            return False

        summary = summarizer(raw_text)
        if not summary:
            return False

        # Replace older messages with summary + recent messages
        recent = self._messages[boundary:]
        self._messages = [
            {"role": "user", "content": f"[Previous conversation summary]\n{summary}"},
            {"role": "assistant", "content": "Understood. I have the context from the previous conversation."},
        ] + recent

        return True

    def _find_compaction_boundary(self, keep_turns: int = 3) -> int | None:
        """Find the index to split messages at, keeping `keep_turns` complete turns.

        A turn boundary is between a complete assistant response (no pending
        tool_use) and the next user message. Never splits tool_use/tool_result
        pairs.

        Returns the index into self._messages, or None if not enough turns.
        """
        messages = self._messages
        n = len(messages)

        # Walk backward counting turn boundaries
        turns_found = 0
        i = n - 1
        while i >= 0 and turns_found < keep_turns:
            msg = messages[i]
            role = msg.get("role", "")

            if role == "user":
                # Check if the previous message (assistant) has tool_use blocks
                # that would be paired with tool_results in this user message
                content = msg.get("content", "")
                has_tool_result = False
                if isinstance(content, list):
                    has_tool_result = any(
                        b.get("type") == "tool_result" for b in content
                    )

                if not has_tool_result:
                    # This is a clean user message — this is a turn boundary
                    turns_found += 1

            i -= 1

        # The boundary is at the start of the kept section
        boundary = i + 1
        if boundary <= 0:
            return None

        # Verify boundary doesn't split a tool_use/tool_result pair
        # If the message at boundary is a user message with tool_results,
        # move boundary earlier to include the preceding assistant tool_use
        if boundary < n:
            msg = messages[boundary]
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list) and any(
                    b.get("type") == "tool_result" for b in content
                ):
                    # Move back to include the assistant message with tool_use
                    boundary = max(0, boundary - 1)
                    if boundary <= 0:
                        return None

        return boundary


# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------


class AnthropicAdapter(LLMAdapter):
    """Adapter that wraps the ``anthropic`` SDK for Claude models."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout_ms: int = 300_000,
    ):
        self._base_url = base_url
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_ms / 1000.0,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client_kwargs = dict(kwargs)  # store for session reset
        self._client = anthropic.Anthropic(**kwargs)

    @staticmethod
    def _resolve_thinking_budget(thinking: str) -> int | None:
        """Resolve thinking tier to budget tokens using config."""
        if thinking == "default" or thinking is None:
            return None

        if thinking == "high":
            tier = config._provider_get("thinking_model", "high")
        elif thinking == "low":
            tier = config._provider_get("thinking_sub_agent", "low")
        elif thinking == "insight":
            tier = config._provider_get("thinking_insight", "low")
        else:
            return None

        if tier == "high":
            return 16384
        elif tier in ("low", "medium"):
            return 2048
        return None

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
    ) -> AnthropicChatSession:
        messages: list[dict] = []
        if history:
            messages.extend(history)

        anthropic_tools = _build_tools(tools, cache_tools=True)
        tool_choice: dict | None = None
        if force_tool_call and anthropic_tools:
            tool_choice = {"type": "any"}

        # JSON schema enforcement via tool-based structured output
        if json_schema is not None and anthropic_tools is None:
            anthropic_tools = []
        if json_schema is not None:
            schema_tool_name = json_schema.get("title", "structured_output")
            anthropic_tools.append(
                {
                    "name": schema_tool_name,
                    "description": "Return the structured response matching the required schema.",
                    "input_schema": json_schema,
                }
            )
            tool_choice = {"type": "tool", "name": schema_tool_name}

        extra_kwargs: dict[str, Any] = {}

        # Thinking/extended thinking
        thinking_budget = self._resolve_thinking_budget(thinking)
        if thinking_budget is not None:
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            extra_kwargs["max_tokens"] = max(
                thinking_budget * 2, thinking_budget + 8192
            )

        return AnthropicChatSession(
            client=self._client,
            model=model,
            system_prompt=_build_system_with_cache(system_prompt),
            messages=messages,
            tools=anthropic_tools,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
            client_kwargs=self._client_kwargs,
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
            tools = [
                {
                    "name": json_schema.get("title", "structured_output"),
                    "description": "Return the structured response.",
                    "input_schema": json_schema,
                }
            ]
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

    def make_multimodal_message(
        self, text: str, image_bytes: bytes, mime_type: str = "image/png"
    ) -> dict:
        """Build an Anthropic multimodal message (image + text content blocks).

        Returns a pre-built user message dict that can be passed directly
        to AnthropicChatSession.send() via the dict path.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64,
                    },
                },
                {"type": "text", "text": text},
            ],
        }

    def is_quota_error(self, exc: Exception) -> bool:
        """Check if the exception is an Anthropic rate-limit error."""
        return isinstance(exc, anthropic.RateLimitError)

    def web_search(self, query: str, model: str) -> LLMResponse:
        """Execute a web search via Anthropic's native web search tool."""
        import config as cfg

        if not cfg._provider_get("web_search", False):
            return LLMResponse(text="")

        if self._base_url:
            return LLMResponse(text="")

        try:
            raw = self._client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": query}],
                tools=[
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5,
                    }
                ],
            )
            return _parse_search_response(raw)
        except Exception as e:
            logger.warning("Anthropic web search failed: %s", e)
            return LLMResponse(text="")

    # -- Convenience properties ------------------------------------------------

    @property
    def client(self):
        """Escape hatch — the underlying ``anthropic.Anthropic`` client."""
        return self._client
