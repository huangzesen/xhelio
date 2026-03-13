"""
Matplotlib script utilities — data label extraction.

Provides regex extraction of data labels from load_data() calls
in LLM-generated matplotlib scripts.

Script validation and execution are handled by data_ops/sandbox.py.
"""

from __future__ import annotations

import re


# ---- Data Label Extraction ----

_LOAD_DATA_PATTERN = re.compile(r'load_data\(\s*["\']([^"\']+)["\']\s*\)')


def extract_data_labels(code: str) -> list[str]:
    """Extract data labels from load_data("label") calls in matplotlib scripts.

    Args:
        code: Python code string to scan.

    Returns:
        List of unique data labels found (order preserved).
    """
    seen = set()
    labels = []
    for match in _LOAD_DATA_PATTERN.finditer(code):
        label = match.group(1)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels
