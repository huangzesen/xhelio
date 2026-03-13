"""Agent factory and utilities.

This module is the entry point for creating agents. The actual agent logic
is in base_agent.py and orchestrator_agent.py. This module provides:

- ``create_agent()`` — factory for creating a fully configured OrchestratorAgent
- ``OrchestratorAgent`` — re-exported for backward compatibility
- ``_sanitize_for_json()`` — NaN/Inf safety for LLM tool results
- ``_extract_turns()`` — extract user/agent text from adapter-specific history
"""

from __future__ import annotations

import math

import config
from config import get_api_key
from .llm import LLMService
from .session_lifecycle import Session
from .orchestrator_agent import OrchestratorAgent  # re-export
from .turn_limits import get_limit


# ------------------------------------------------------------------
# History extraction utilities
# ------------------------------------------------------------------

_USER_ROLES = {"user"}
_AGENT_ROLES = {"model", "assistant"}


def _extract_turns(history_entries: list, *, max_text: int | None = None) -> list[str]:
    """Extract user/agent text turns from adapter-specific history formats.

    Works with Gemini (role="model", parts=[{text}]),
    OpenAI (role="assistant", content=str), and
    Anthropic (role="assistant", content=str|list).
    """
    turns = []
    for content in history_entries:
        if isinstance(content, dict):
            role = content.get("role", "")
        else:
            role = getattr(content, "role", "")

        if role in _USER_ROLES:
            label = "User"
        elif role in _AGENT_ROLES:
            label = "Agent"
        else:
            continue

        # Extract text: try Gemini-style parts first, then OpenAI/Anthropic content
        text = None
        if isinstance(content, dict):
            parts = content.get("parts")
            if parts:
                for part in parts:
                    t = (
                        part.get("text")
                        if isinstance(part, dict)
                        else getattr(part, "text", None)
                    )
                    if t:
                        text = t
                        break
            if not text:
                c = content.get("content", "")
                if isinstance(c, str):
                    text = c
                elif isinstance(c, list):
                    for block in c:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text")
                            break
        else:
            parts = getattr(content, "parts", None) or []
            for part in parts:
                t = getattr(part, "text", None)
                if t:
                    text = t
                    break
            if not text:
                text = getattr(content, "content", None)
                if isinstance(text, list):
                    text = None

        if text:
            limit = max_text if max_text is not None else get_limit("context.turn_text")
            turns.append(f"{label}: {text[:limit]}")
    return turns


# ------------------------------------------------------------------
# JSON safety
# ------------------------------------------------------------------


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for JSON safety.

    Gemini's API rejects function_response containing NaN or Inf values
    (400 INVALID_ARGUMENT). This ensures all tool results are safe.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ------------------------------------------------------------------
# Agent factory
# ------------------------------------------------------------------


def _create_llm_service() -> LLMService:
    """Create the LLM service based on config."""
    provider, model, base_url = config.resolve_agent_model("orchestrator")
    api_key = get_api_key(provider)
    caps = config.resolve_capabilities()
    return LLMService(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url or config.LLM_BASE_URL,
        provider_config=caps,
    )


def create_agent(
    verbose: bool = False,
    gui_mode: bool = False,
    model: str | None = None,
    defer_chat: bool = False,
    web_mode: bool = False,
) -> OrchestratorAgent:
    """Factory function to create a new OrchestratorAgent.

    Args:
        verbose: If True, enable verbose logging.
        gui_mode: If True, enable GUI mode.
        model: Model name override.
        defer_chat: If True, skip creating the initial LLM chat session.
        web_mode: If True, enable web/API mode.

    Returns:
        Configured OrchestratorAgent backed by a full SessionContext.
    """
    service = _create_llm_service()
    session = Session(service=service, web_mode=web_mode)
    ctx = session.start()

    agent = OrchestratorAgent(
        session_ctx=ctx,
        service=service,
        cancel_event=ctx.cancel_event,
    )

    # Connect EurekaHooks inbox injector to orchestrator
    if ctx.eureka_hooks is not None:
        from .base_agent import _make_message, MSG_USER_INPUT

        def _inject_eureka_msg(text: str):
            msg = _make_message(MSG_USER_INPUT, "eureka_mode", text)
            agent.inbox.put(msg)

        ctx.eureka_hooks._inbox_injector = _inject_eureka_msg

    # Connect MemoryHooks callbacks to the orchestrator's internals
    if ctx.memory_hooks is not None:
        def _inject_prompt(prompt: str):
            agent.system_prompt = prompt
            if agent._chat is not None:
                agent._chat.update_system_prompt(prompt)

        def _restart_session():
            if agent._chat is None:
                return
            from .prompts import get_system_prompt
            interface = agent._chat.interface
            memory_section = ctx.memory_store.format_for_injection(
                scope="generic", include_review_instruction=False
            )
            base_prompt = get_system_prompt()
            new_prompt = f"{base_prompt}\n\n{memory_section}" if memory_section else base_prompt
            interface.add_system(new_prompt)
            agent._chat = ctx.service.create_session(
                system_prompt=new_prompt,
                tools=agent._tool_schemas,
                model=agent.model_name,
                thinking="high",
                tracked=False,
                interface=interface,
            )

        ctx.memory_hooks._prompt_injector = _inject_prompt
        ctx.memory_hooks._session_restarter = _restart_session

    return agent
