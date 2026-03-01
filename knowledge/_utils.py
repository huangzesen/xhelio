"""Shared utility functions for the knowledge package."""


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge *patch* into *base* (mutates *base* in place).

    - If both values are dicts, merge recursively.
    - Otherwise the patch value replaces the base value.

    Returns *base* for convenience.
    """
    for key, value in patch.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base
