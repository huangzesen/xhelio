"""
Automatic model fallback for LLM API calls.

When any model hits quota or rate limits (429 RESOURCE_EXHAUSTED), all agents
switch to the fallback model for the remainder of the session.
"""

import threading

from .event_bus import get_event_bus, DEBUG

# Session-level flag: once set, ALL subsequent API calls use the fallback model.
_fallback_active = False
_fallback_model = None
_lock = threading.Lock()


def activate_fallback(fallback_model: str) -> None:
    """Activate fallback mode — all future calls use *fallback_model*."""
    global _fallback_active, _fallback_model
    with _lock:
        _fallback_active = True
        _fallback_model = fallback_model
    get_event_bus().emit(DEBUG, agent="Fallback", level="warning", msg=f"[Fallback] Activated — all models switching to {fallback_model}")


def reset_fallback() -> None:
    """Deactivate fallback mode — revert to using configured models.

    Called when the user explicitly changes model settings via config,
    since the new models may not have the same quota issues.
    """
    global _fallback_active, _fallback_model
    with _lock:
        was_active = _fallback_active
        _fallback_active = False
        _fallback_model = None
    if was_active:
        get_event_bus().emit(DEBUG, agent="Fallback", level="info", msg="[Fallback] Reset — reverting to configured models")


def is_fallback_active() -> bool:
    return _fallback_active


def get_active_model(requested_model: str) -> str:
    """Return the model to actually use: fallback if active, else requested."""
    with _lock:
        if _fallback_active and _fallback_model:
            return _fallback_model
    return requested_model


def is_quota_error(exc: Exception, adapter=None) -> bool:
    """Return True if *exc* is a 429 / RESOURCE_EXHAUSTED error.

    Delegates to the adapter if provided, otherwise falls back to checking
    the exception message string (provider-agnostic heuristic).
    """
    if adapter is not None:
        return adapter.is_quota_error(exc)
    # Fallback heuristic — works for most providers
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg
