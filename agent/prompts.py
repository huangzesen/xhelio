"""
System prompts and response formatting for the agent.

The system prompt is dynamically generated from the mission catalog
via knowledge/prompt_builder.py — no hardcoded mission or dataset tables.
"""

from datetime import datetime

import threading

from knowledge.prompt_builder import build_system_prompt

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


def get_system_prompt(gui_mode: bool = False) -> str:
    """Return the system prompt with current date.

    Args:
        gui_mode: If True, the orchestrator knows GUI mode is active (passed
            through to the visualization agent, not appended to orchestrator prompt).
    """
    template = _ensure_template()
    return template.replace("{today}", datetime.now().strftime("%Y-%m-%d"))



