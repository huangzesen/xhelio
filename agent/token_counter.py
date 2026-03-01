"""agent/token_counter.py — Centralized token counting using Gemini's local tokenizer.

Uses google-genai's LocalTokenizer (backed by sentencepiece) for exact Gemini
token counts with no API calls.  Falls back to len(text) // 4 if sentencepiece
is unavailable.

Public API:
    count_tokens(text: str) -> int
"""

from __future__ import annotations

import logging
import threading
import warnings

_lock = threading.Lock()
_tokenizer = None          # lazily initialized LocalTokenizer
_fallback_warned = False    # log fallback warning only once

logger = logging.getLogger(__name__)

# Suppress ExperimentalWarning from google-genai (fires on import and on calls)
try:
    from google.genai._common import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
except ImportError:
    pass

# Model used for tokenization — gemini-2.0-flash is always supported and
# shares the same tokenizer as other Gemini models.
_TOKENIZER_MODEL = "gemini-2.0-flash"


def _init_tokenizer():
    """Lazy-init the LocalTokenizer.  Thread-safe, called at most once."""
    global _tokenizer, _fallback_warned
    if _tokenizer is not None:
        return

    with _lock:
        # Double-check after acquiring lock
        if _tokenizer is not None:
            return
        try:
            from google.genai.local_tokenizer import LocalTokenizer
            _tokenizer = LocalTokenizer(model_name=_TOKENIZER_MODEL)
        except Exception as exc:
            if not _fallback_warned:
                logger.warning(
                    "LocalTokenizer unavailable (%s), falling back to len//4 heuristic",
                    exc,
                )
                _fallback_warned = True


def count_tool_tokens(schemas: list) -> int:
    """Count tokens in a list of FunctionSchema tool declarations.

    Serializes each schema's name, description, and parameters to JSON,
    then counts tokens using the local tokenizer.
    """
    if not schemas:
        return 0
    import json
    blob = json.dumps([
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in schemas
    ])
    return count_tokens(blob)


def count_tokens(text: str) -> int:
    """Count tokens in *text* using the Gemini local tokenizer.

    Returns an exact count when sentencepiece is available, otherwise
    falls back to ``len(text) // 4``.
    """
    if not text:
        return 0

    _init_tokenizer()

    if _tokenizer is not None:
        try:
            return _tokenizer.count_tokens(text).total_tokens
        except Exception:
            pass

    # Fallback heuristic
    return len(text) // 4
