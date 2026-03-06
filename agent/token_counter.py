"""agent/token_counter.py — Centralized token counting using Gemini's local tokenizer.

Uses google-genai's LocalTokenizer (backed by sentencepiece) for exact Gemini
token counts with no API calls.

Public API:
    count_tokens(text: str) -> int
"""

from __future__ import annotations

import threading
import warnings

from .logging import get_logger

_lock = threading.Lock()
_tokenizer = None  # lazily initialized LocalTokenizer

logger = get_logger()

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
    global _tokenizer
    if _tokenizer is not None:
        return

    with _lock:
        # Double-check after acquiring lock
        if _tokenizer is not None:
            return
        from google.genai.local_tokenizer import LocalTokenizer

        _tokenizer = LocalTokenizer(model_name=_TOKENIZER_MODEL)


def count_tool_tokens(schemas: list) -> int:
    """Count tokens in a list of FunctionSchema tool declarations.

    Serializes each schema's name, description, and parameters to JSON,
    then counts tokens using the local tokenizer.
    """
    if not schemas:
        return 0
    import json

    blob = json.dumps(
        [
            {"name": s.name, "description": s.description, "parameters": s.parameters}
            for s in schemas
        ]
    )
    return count_tokens(blob)


def count_tokens(text: str) -> int:
    """Count tokens in *text* using the Gemini local tokenizer.

    Returns an exact count using the LocalTokenizer.
    Raises an exception if the tokenizer is unavailable.
    """
    if not text:
        return 0

    _init_tokenizer()

    return _tokenizer.count_tokens(text).total_tokens
