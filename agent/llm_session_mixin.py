"""Shared LLM session behavior used by both OrchestratorAgent and SubAgent.

Extracted to eliminate duplication between core.py and sub_agent.py.
"""
from __future__ import annotations

from .truncation import trunc


COMPACTION_PROMPT = (
    "Summarize the following conversation history concisely for an AI agent "
    "that needs to continue the session. Preserve:\n"
    "- ALL errors, failures, and their details verbatim\n"
    "- Key decisions and user preferences\n"
    "- Data that was fetched/computed and current state\n"
    "- Tool calls that produced important results\n\n"
    "Drop thinking blocks and routine acknowledgments. "
    "Output ONLY the summary, no commentary.\n\n"
    "Conversation history:\n"
)


def summarize_tool_calls(history: list[dict]) -> str:
    """Extract tool call names and args from the last assistant turn.

    Works with adapter-specific history format (list of role/content dicts).
    """
    parts = []
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            break
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "?")
                args = block.get("input", {})
                args_str = ", ".join(
                    f"{k}={repr(v)[:80]}" for k, v in args.items()
                )
                parts.append(f"- {name}({args_str})")
    return "\n".join(reversed(parts)) if parts else "(no tool calls found)"


def summarize_tool_results(message) -> str:
    """Extract tool result summaries from the message that failed to send."""
    if isinstance(message, str):
        return trunc(message, "history.result")
    parts = []
    items = message if isinstance(message, list) else [message]
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content", item)
        if isinstance(content, str):
            parts.append(f"- {content[:300]}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_result":
                        tool_id = block.get("tool_use_id", "?")
                        text = block.get("content", "")
                        if isinstance(text, str):
                            parts.append(f"- [{tool_id}]: {text[:300]}")
                        elif isinstance(text, list):
                            for sub in text:
                                if (
                                    isinstance(sub, dict)
                                    and sub.get("type") == "text"
                                ):
                                    parts.append(
                                        f"- [{tool_id}]: {sub.get('text', '')[:300]}"
                                    )
    return "\n".join(parts) if parts else "(tool results not available)"
