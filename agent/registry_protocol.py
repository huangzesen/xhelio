"""Registry protocol and meta-registry for unified discovery of all registries."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Registry(Protocol):
    """Structural protocol that any registry in the codebase can satisfy.

    Implementors need only provide:
      - name: str          — unique identifier for the registry
      - description: str   — human-readable purpose
      - get(key) -> Any    — lookup by key, returns None on miss
      - list_all() -> dict — returns all entries
    """

    name: str
    description: str

    def get(self, key: str) -> Any:
        ...

    def list_all(self) -> dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Meta-registry: a central index of all Registry instances
# ---------------------------------------------------------------------------

_META_REGISTRY: dict[str, Registry] = {}


def register_registry(registry: Registry) -> None:
    """Register a registry instance.

    Idempotent: re-registering the *same* object is a no-op (safe under
    module reload).  Raises ValueError only when a *different* registry
    attempts to claim an already-taken name.
    """
    existing = _META_REGISTRY.get(registry.name)
    if existing is not None:
        if existing is registry:
            return  # same object — idempotent
        raise ValueError(
            f"Registry '{registry.name}' is already registered"
        )
    _META_REGISTRY[registry.name] = registry


def list_registries() -> dict[str, Registry]:
    """Return a shallow copy of all registered registries."""
    return dict(_META_REGISTRY)


def get_registry(name: str) -> Registry | None:
    """Look up a registry by name.  Returns None if not found."""
    return _META_REGISTRY.get(name)


def _reset_for_testing() -> dict[str, Registry]:
    """Snapshot and clear the meta-registry.  Returns the snapshot for restore.

    Usage in tests::

        saved = _reset_for_testing()
        try:
            ...
        finally:
            _META_REGISTRY.clear()
            _META_REGISTRY.update(saved)
    """
    saved = dict(_META_REGISTRY)
    _META_REGISTRY.clear()
    return saved
