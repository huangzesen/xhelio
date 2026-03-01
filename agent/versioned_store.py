"""
VersionedStore — generic base class for persistent knowledge stores with versioning.

Provides:
- Monotonic version numbers per entry lineage
- Supersedes chains for safe consolidation
- In-file archival instead of cold storage files
- Consistent atomic persistence with schema migration
- Embedding-based + tag-based search
"""

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, Optional, runtime_checkable, Protocol

import numpy as np

from .event_bus import get_event_bus, DEBUG

T = TypeVar("T")


@runtime_checkable
class VersionedEntry(Protocol):
    """Structural type for versioned entries."""

    id: str
    created_at: str
    tags: list[str]
    version: int
    supersedes: str
    archived: bool


class _EmbeddingWrapper:
    """Lightweight wrapper to give arbitrary entries a .content attribute for MemoryEmbeddings."""

    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class VersionedStore(ABC, Generic[T]):
    """Base class for persistent versioned knowledge stores.

    Subclasses must implement abstract methods for serialization, migration,
    and entry introspection. Concrete methods provide shared logic for
    persistence, archival, search, and version tracking.
    """

    def __init__(self, path: Path):
        self.path = path
        self._entries: list[T] = []
        self._save_lock = threading.Lock()
        self._mutation_epoch: int = 0
        # Lazy import to avoid circular dependency (memory.py → versioned_store.py)
        from .memory import MemoryEmbeddings
        self.embeddings = MemoryEmbeddings()

    @property
    def mutation_epoch(self) -> int:
        """Monotonic counter incremented on every mutation (add/remove/update)."""
        return self._mutation_epoch

    # ---- Abstract methods (subclass MUST implement) ----

    @property
    @abstractmethod
    def _schema_version(self) -> int:
        """Return current schema version number."""

    @abstractmethod
    def _deserialize_entry(self, raw: dict) -> T:
        """Convert a raw dict to a typed entry."""

    @abstractmethod
    def _serialize_entry(self, entry: T) -> dict:
        """Convert a typed entry to a dict for JSON serialization."""

    @abstractmethod
    def _migrate(self, data: dict, from_version: int) -> dict:
        """Migrate file data from from_version to current schema version.

        Must handle all intermediate versions (e.g., v1→v2→v3→v4).
        Returns the migrated data dict.
        """

    @abstractmethod
    def _estimate_entry_tokens(self, entry: T) -> int:
        """Estimate token count for a single entry."""

    @abstractmethod
    def _entry_search_text(self, entry: T) -> str:
        """Return text to embed for similarity search."""

    @abstractmethod
    def _get_entry_tags(self, entry: T) -> list[str]:
        """Return tags for tag-based search fallback."""

    @abstractmethod
    def _get_entry_id(self, entry: T) -> str:
        """Return the unique ID of an entry."""

    @abstractmethod
    def _is_archived(self, entry: T) -> bool:
        """Check if an entry is archived."""

    @abstractmethod
    def _set_archived(self, entry: T, archived: bool) -> None:
        """Set the archived flag on an entry."""

    @abstractmethod
    def _get_version(self, entry: T) -> int:
        """Return the version number of an entry."""

    @abstractmethod
    def _get_supersedes(self, entry: T) -> str:
        """Return the ID of the entry this one supersedes ('' if original)."""

    # ---- Overridable hooks (with defaults) ----

    def _serialize_file(self, entries: list[T]) -> dict:
        """Serialize the full file content. Override for custom top-level keys."""
        return {
            "schema_version": self._schema_version,
            "entries": [self._serialize_entry(e) for e in entries],
        }

    def _parse_file(self, data: dict) -> list[T]:
        """Parse entries from loaded file data. Override for custom structure."""
        raw_entries = data.get("entries", [])
        result = []
        for raw in raw_entries:
            try:
                result.append(self._deserialize_entry(raw))
            except (TypeError, KeyError) as e:
                get_event_bus().emit(DEBUG, agent=self.__class__.__name__, level="warning", msg=f"[{self.__class__.__name__}] Skipping malformed entry: {e}")
        return result

    def _on_load(self, data: dict) -> None:
        """Extract store-level metadata from loaded data. Override as needed."""

    def _similarity_threshold(self) -> float:
        """Minimum cosine similarity for embedding-based search. Override to tune."""
        return 0.55

    # ---- Persistence ----

    def load(self) -> None:
        """Load entries from disk, migrating old schema if needed."""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)

            schema_version = data.get("schema_version", 1)
            self._on_load(data)

            if schema_version < self._schema_version:
                data = self._migrate(data, schema_version)

            self._entries = self._parse_file(data)

            needs_save = schema_version < self._schema_version
            needs_save |= self._post_load_fixup()

            if needs_save:
                self.save()

            self._rebuild_embeddings()

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            get_event_bus().emit(DEBUG, agent=self.__class__.__name__, level="warning", msg=f"[{self.__class__.__name__}] Could not load {self.path}: {e}")
            self._entries = []

    def _post_load_fixup(self) -> bool:
        """Hook for post-load fixups. Returns True if save needed. Override as needed."""
        return False

    def save(self) -> None:
        """Atomically save entries to disk (write temp + rename).

        Serialized with a lock so concurrent background threads don't
        race on the same temp file.
        """
        with self._save_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            data = self._serialize_file(self._entries)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self.path)

    # ---- CRUD ----

    def _add_entry(self, entry: T, save: bool = True) -> None:
        """Append an entry and optionally save. Invalidates embeddings."""
        self._entries.append(entry)
        self._mutation_epoch += 1
        self.embeddings.invalidate()
        if save:
            self.save()

    def _update_entry(self, entry_id: str, new_entry: T) -> None:
        """Archive old entry, add new one with supersedes chain."""
        for e in self._entries:
            if self._get_entry_id(e) == entry_id and not self._is_archived(e):
                self._set_archived(e, True)
                break
        # _add_entry increments _mutation_epoch
        self._add_entry(new_entry)

    def _remove_entry(self, entry_id: str) -> bool:
        """Archive an entry by ID. Returns True if found."""
        for e in self._entries:
            if self._get_entry_id(e) == entry_id and not self._is_archived(e):
                self._set_archived(e, True)
                self._mutation_epoch += 1
                self.embeddings.invalidate()
                self.save()
                return True
        return False

    # ---- Queries ----

    def get_active(self) -> list[T]:
        """Return all non-archived entries."""
        return [e for e in self._entries if not self._is_archived(e)]

    def get_archived(self) -> list[T]:
        """Return all archived entries."""
        return [e for e in self._entries if self._is_archived(e)]

    def get_all_including_archived(self) -> list[T]:
        """Return all entries (active + archived)."""
        return list(self._entries)

    def get_by_id(self, entry_id: str) -> Optional[T]:
        """Find an entry by ID (searches all, including archived)."""
        for e in self._entries:
            if self._get_entry_id(e) == entry_id:
                return e
        return None

    def get_version_history(self, entry_id: str) -> list[T]:
        """Get the full supersedes chain for an entry.

        Returns entries in reverse chronological order (newest first).
        """
        chain = []
        current_id = entry_id

        # First find all entries in the chain going backward
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            entry = self.get_by_id(current_id)
            if entry is None:
                break
            chain.append(entry)
            current_id = self._get_supersedes(entry)

        return chain

    # ---- Token estimation ----

    def total_active_tokens(self) -> int:
        """Sum of estimated tokens for all active entries."""
        return sum(self._estimate_entry_tokens(e) for e in self.get_active())

    def is_over_budget(self, budget: int) -> bool:
        """Check if active entries exceed a token budget."""
        return self.total_active_tokens() > budget

    # ---- Bulk operations ----

    def replace_active(self, new_entries: list[T]) -> None:
        """Archive all active entries, add new ones."""
        for e in self._entries:
            if not self._is_archived(e):
                self._set_archived(e, True)
        self._entries.extend(new_entries)
        self._mutation_epoch += 1
        self.embeddings.invalidate()
        self._rebuild_embeddings()
        self.save()

    def purge_archived(self) -> int:
        """Permanently remove all archived entries. Returns count removed."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if not self._is_archived(e)]
        removed = before - len(self._entries)
        if removed > 0:
            self.save()
        return removed

    # ---- Search ----

    def search(
        self,
        query: str,
        limit: int = 20,
        include_archived: bool = False,
    ) -> list[T]:
        """Search entries using embedding similarity or tag fallback.

        Args:
            query: Search text.
            limit: Max results.
            include_archived: If True, also search archived entries.

        Returns:
            List of entries sorted by relevance (highest first).
        """
        if not query or not query.strip():
            return []

        if include_archived:
            candidates = list(self._entries)
        else:
            candidates = self.get_active()

        if not candidates:
            return []

        if self.embeddings.available:
            return self._search_by_embeddings(query, candidates, limit)
        return self._search_by_tags(query, candidates, limit)

    def _search_by_embeddings(
        self, query: str, candidates: list[T], limit: int,
    ) -> list[T]:
        """Rank candidates by cosine similarity to query embedding."""
        try:
            contents = [self._entry_search_text(e) for e in candidates]
            all_texts = [query] + contents
            embs = np.array(list(self.embeddings._model.embed(all_texts)))
            query_emb = embs[0]
            candidate_embs = embs[1:]
            sims = candidate_embs @ query_emb

            threshold = self._similarity_threshold()
            scored = [
                (float(sims[i]), candidates[i])
                for i in range(len(candidates))
                if sims[i] > threshold
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [e for _, e in scored[:limit]]
        except Exception:
            return self._search_by_tags(query, candidates, limit)

    def _search_by_tags(
        self, query: str, candidates: list[T], limit: int,
    ) -> list[T]:
        """Rank candidates by tag overlap + substring match."""
        import re
        from .memory import _STOP_WORDS

        query_tokens = set(re.split(r"[^a-zA-Z0-9_]+", query.lower()))
        query_tokens -= _STOP_WORDS
        query_tokens.discard("")
        query_lower = query.lower()

        if not query_tokens and not query_lower.strip():
            return []

        scored: list[tuple[float, T]] = []
        for e in candidates:
            score = self._score_entry(e, query_tokens, query_lower)
            if score > 0:
                scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def _score_entry(self, entry: T, query_tokens: set[str], query_lower: str) -> float:
        """Score an entry against a query. Higher = more relevant. Override for custom scoring."""
        score = 0.0
        tags = self._get_entry_tags(entry)
        if tags:
            tag_set = set(tags)
            tag_matches = len(query_tokens & tag_set)
            score += tag_matches * 2.0
        search_text = self._entry_search_text(entry)
        if query_lower in search_text.lower():
            score += 1.0
        return score

    # ---- Embedding helpers ----

    def _rebuild_embeddings(self) -> None:
        """Rebuild embedding index from active entries."""
        active = self.get_active()
        if active:
            wrappers = [_EmbeddingWrapper(self._entry_search_text(e)) for e in active]
            self.embeddings.build(wrappers)
        else:
            self.embeddings.invalidate()
