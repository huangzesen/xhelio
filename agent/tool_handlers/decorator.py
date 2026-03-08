"""Decorator for self-registering tool handlers.

Usage:
    @tool_handler("my_tool_name")
    def handle_my_tool(orch, args):
        return {"result": "ok"}

All decorated handlers are collected in _HANDLERS and can be
merged into the main TOOL_REGISTRY via get_all_handlers().
"""

from __future__ import annotations

from typing import Callable

_HANDLERS: dict[str, Callable] = {}


def tool_handler(tool_name: str):
    """Register a function as the handler for a tool.

    Raises ValueError if the tool name is already registered via decorator.
    """
    def decorator(fn):
        if tool_name in _HANDLERS:
            raise ValueError(f"Duplicate @tool_handler for '{tool_name}'")
        _HANDLERS[tool_name] = fn
        return fn
    return decorator


def get_all_handlers() -> dict[str, Callable]:
    """Return all decorator-registered handlers."""
    return dict(_HANDLERS)
