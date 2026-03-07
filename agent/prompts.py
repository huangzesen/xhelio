"""
System prompts and response formatting for the agent.

The system prompt is dynamically generated from the mission catalog
via knowledge/prompt_builder.py — no hardcoded mission or dataset tables.
"""

import threading

from knowledge.prompt_builder import build_system_prompt
from knowledge.prompt_loader import invalidate_cache as _invalidate_prompt_files

# Lazy: built on first call to get_system_prompt(), not at import time.
_SYSTEM_PROMPT_TEMPLATE: str | None = None
_template_lock = threading.Lock()


def _ensure_template() -> str:
    """Build the template on first access (thread-safe)."""
    global _SYSTEM_PROMPT_TEMPLATE
    if _SYSTEM_PROMPT_TEMPLATE is None:
        with _template_lock:
            if _SYSTEM_PROMPT_TEMPLATE is None:
                _SYSTEM_PROMPT_TEMPLATE = build_system_prompt(include_catalog=False)
    return _SYSTEM_PROMPT_TEMPLATE


def invalidate_system_prompt_cache() -> None:
    """Force rebuild of the cached system prompt template.

    Called after background mission loading completes so that the
    routing table reflects newly loaded missions.
    """
    global _SYSTEM_PROMPT_TEMPLATE
    with _template_lock:
        _SYSTEM_PROMPT_TEMPLATE = None
    _invalidate_prompt_files()


def get_system_prompt() -> str:
    """Return the system prompt (static, no date placeholder)."""
    return _ensure_template()



