"""
Persistent library of successful custom operations.

Saves non-trivial (5+ line) custom_operation code to a JSON file so the LLM
can reuse proven transformations in future sessions.  Deduplicates by
normalized description, tracks usage counts, and evicts least-used entries
when the library hits a configurable cap.

Storage: ``~/.xhelio/custom_ops_library.json``
"""

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.event_bus import get_event_bus, DEBUG

DEFAULT_MAX_ENTRIES = 50
_LIBRARY_FILENAME = "custom_ops_library.json"


class OpsLibrary:
    """Thread-safe, JSON-persisted library of reusable custom operations."""

    def __init__(self, path: Optional[Path] = None, max_entries: Optional[int] = None):
        if path is None:
            from config import get_data_dir
            path = get_data_dir() / _LIBRARY_FILENAME
        self._path = path
        if max_entries is None:
            try:
                import config
                max_entries = config.get("ops_library_max_entries", DEFAULT_MAX_ENTRIES)
            except Exception:
                max_entries = DEFAULT_MAX_ENTRIES
        self._max_entries = max_entries
        self._entries: list[dict] = []
        self._version: int = 1
        self._lock = threading.Lock()
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load library from disk. Silently starts empty if file is missing."""
        with self._lock:
            if self._path.exists():
                try:
                    with open(self._path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._version = data.get("version", 1)
                    self._entries = data.get("entries", [])
                except (json.JSONDecodeError, OSError) as exc:
                    get_event_bus().emit(DEBUG, agent="data_ops", level="warning", msg=f"[OpsLibrary] Failed to load {self._path}: {exc}")
                    self._entries = []

    def save(self) -> None:
        """Atomic write: tmp file + rename."""
        with self._lock:
            self._save_unlocked()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_or_update(
        self,
        description: str,
        code: str,
        source_labels: list[str],
        units: str = "",
        session_id: str = "",
    ) -> dict:
        """Add a new entry or update an existing one (dedup by description).

        Returns the entry dict (new or updated).
        """
        norm = self._normalize_description(description)
        now = datetime.now(timezone.utc).isoformat()
        source_type = self._infer_source_type(source_labels, code)
        num_sources = len(source_labels)

        with self._lock:
            # Look for existing entry with same normalized description
            for entry in self._entries:
                if self._normalize_description(entry["description"]) == norm:
                    entry["code"] = code
                    entry["num_sources"] = num_sources
                    entry["source_type"] = source_type
                    entry["units"] = units
                    entry["use_count"] = entry.get("use_count", 1) + 1
                    entry["last_used_at"] = now
                    entry["session_id"] = session_id
                    self._save_unlocked()
                    return dict(entry)

            # Evict if at capacity
            if len(self._entries) >= self._max_entries:
                self._evict_one_unlocked()

            entry = {
                "id": uuid.uuid4().hex[:8],
                "description": description,
                "code": code,
                "num_sources": num_sources,
                "source_type": source_type,
                "units": units,
                "use_count": 1,
                "created_at": now,
                "last_used_at": now,
                "session_id": session_id,
            }
            self._entries.append(entry)
            self._save_unlocked()
            return dict(entry)

    def record_reuse(self, entry_id: str) -> bool:
        """Bump use_count for an entry by ID. Returns True if found."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for entry in self._entries:
                if entry["id"] == entry_id:
                    entry["use_count"] = entry.get("use_count", 1) + 1
                    entry["last_used_at"] = now
                    self._save_unlocked()
                    return True
        return False

    def get_entry_by_id(self, entry_id: str) -> dict | None:
        """Lookup a single entry by its 8-char hex ID."""
        with self._lock:
            for entry in self._entries:
                if entry["id"] == entry_id:
                    return dict(entry)
        return None

    def find_matching_code(self, code: str) -> dict | None:
        """Find an entry with exactly matching code (after strip)."""
        needle = code.strip()
        with self._lock:
            for entry in self._entries:
                if entry["code"].strip() == needle:
                    return dict(entry)
        return None

    def get_top_entries(self, limit: int = 20) -> list[dict]:
        """Return top entries sorted by use_count desc, then last_used_at desc."""
        with self._lock:
            sorted_entries = sorted(
                self._entries,
                key=lambda e: (e.get("use_count", 1), e.get("last_used_at", "")),
                reverse=True,
            )
            return [dict(e) for e in sorted_entries[:limit]]

    def build_prompt_section(self) -> str:
        """Build markdown for injection into the DataOps think-phase prompt.

        Returns empty string if no entries exist.
        """
        entries = self.get_top_entries(limit=20)
        if not entries:
            return ""

        lines = [
            "## Saved Operations Library",
            "",
            "Previously successful operations (5+ lines, non-trivial). Suggest reusing",
            "these when applicable (adapt df_SUFFIX / da_SUFFIX variable names to match",
            "current data labels).",
            "",
        ]
        for entry in entries:
            eid = entry["id"]
            desc = entry["description"]
            num = entry["num_sources"]
            stype = entry["source_type"]
            units = entry.get("units", "")
            count = entry.get("use_count", 1)

            lines.append(f"### [{eid}] {desc}")
            meta_parts = [f"Sources: {num} {stype}"]
            if units:
                meta_parts.append(f"Units: {units}")
            meta_parts.append(f"Used {count} time{'s' if count != 1 else ''}")
            lines.append(f"- {' | '.join(meta_parts)}")
            lines.append("```python")
            lines.append(entry["code"])
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_unlocked(self) -> None:
        """Save without acquiring the lock (caller must hold it)."""
        data = {
            "version": self._version,
            "max_entries": self._max_entries,
            "entries": self._entries,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self._path)

    def _evict_one_unlocked(self) -> None:
        """Remove the entry with lowest use_count (tiebreak: oldest last_used_at)."""
        if not self._entries:
            return
        victim = min(
            self._entries,
            key=lambda e: (e.get("use_count", 1), e.get("last_used_at", "")),
        )
        self._entries.remove(victim)

    @staticmethod
    def _normalize_description(desc: str) -> str:
        """Lowercase, collapse whitespace, strip [from ...] refs."""
        desc = re.sub(r'\[from [a-f0-9]{8}\]', '', desc)
        desc = desc.lower().strip()
        desc = re.sub(r'\s+', ' ', desc)
        return desc

    @staticmethod
    def _infer_source_type(source_labels: list[str], code: str) -> str:
        """Infer whether sources are DataFrames or DataArrays from the code."""
        has_da = bool(re.search(r'\bda_', code))
        has_df = bool(re.search(r'\bdf\b|\bdf_', code))
        if has_da and has_df:
            return "mixed"
        if has_da:
            return "da"
        return "df"


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_library: Optional[OpsLibrary] = None
_library_lock = __import__("threading").Lock()


def get_ops_library() -> OpsLibrary:
    """Return the global OpsLibrary singleton."""
    global _library
    if _library is None:
        with _library_lock:
            if _library is None:
                _library = OpsLibrary()
    return _library


def reset_ops_library() -> None:
    """Reset the global singleton (for testing)."""
    global _library
    _library = None
