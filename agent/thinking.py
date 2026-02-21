"""Utilities for thought extraction and thinking-token tracking.

Works with both LLMResponse objects (from the adapter layer) and raw
Gemini responses (for backward compatibility during migration).
"""


def extract_thoughts(response) -> list[str]:
    """Extract thinking text from a response.

    Accepts either an ``LLMResponse`` (has ``.thoughts``) or a raw Gemini
    response (has ``.candidates[0].content.parts``).
    """
    # LLMResponse path
    if hasattr(response, "thoughts"):
        return list(response.thoughts)

    # Raw Gemini response fallback
    thoughts = []
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return thoughts
    content = candidates[0].content
    if not content:
        return thoughts
    for part in content.parts:
        if getattr(part, "thought", False) and hasattr(part, "text") and part.text:
            thoughts.append(part.text)
    return thoughts


def get_thinking_tokens(response) -> int:
    """Extract thinking token count from a response (0 if unavailable).

    Accepts either an ``LLMResponse`` (has ``.usage.thinking_tokens``) or a
    raw Gemini response (has ``.usage_metadata.thoughts_token_count``).
    """
    # LLMResponse path
    usage = getattr(response, "usage", None)
    if usage and hasattr(usage, "thinking_tokens"):
        return usage.thinking_tokens or 0

    # Raw Gemini response fallback
    meta = getattr(response, "usage_metadata", None)
    if meta:
        return getattr(meta, "thoughts_token_count", 0) or 0
    return 0
