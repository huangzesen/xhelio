"""Load and cache prompt sections from knowledge/prompts/ markdown files.

Provides a simple file-loading and assembly layer. Each markdown file is
read once per process (via lru_cache) and multiple sections are joined
into a single prompt string.

Functions:
    load_section(path) — Read a markdown file relative to knowledge/prompts/.
    assemble(sections, **substitutions) — Load + join + substitute placeholders.
    invalidate_cache() — Clear the file cache (for hot-reload or testing).
"""

import functools
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@functools.lru_cache(maxsize=128)
def load_section(path: str) -> str:
    """Load a markdown file relative to knowledge/prompts/.

    The result is cached — each file is read at most once per process.

    Args:
        path: Relative path within knowledge/prompts/ (e.g., "_shared/domain_rules.md").

    Returns:
        File contents as a string, with trailing whitespace stripped.

    Raises:
        FileNotFoundError: If the markdown file does not exist.
    """
    return (_PROMPTS_DIR / path).read_text(encoding="utf-8").rstrip()


def assemble(sections: list[str], **substitutions) -> str:
    """Load sections by path, concatenate, and apply {placeholder} substitutions.

    Each section path is loaded via load_section(), the results are joined
    with double-newline separators, and then any keyword substitutions are
    applied via str.format_map().

    Args:
        sections: List of paths relative to knowledge/prompts/.
        **substitutions: Keyword arguments for {placeholder} replacement.

    Returns:
        Assembled prompt string with substitutions applied.

    Raises:
        KeyError: If a {placeholder} in the text has no matching substitution.
        FileNotFoundError: If any section file does not exist.
    """
    parts = [load_section(s) for s in sections]
    text = "\n\n".join(parts)
    if substitutions:
        text = text.format_map(substitutions)
    return text


def invalidate_cache():
    """Clear the file cache.

    Call this after modifying markdown files at runtime (hot-reload)
    or in test setup to ensure fresh reads.
    """
    load_section.cache_clear()
