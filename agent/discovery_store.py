"""
Discovery Memory — persistent scientific knowledge learned from browsing NASA datasets.

Separate from operational memory (preferences, pitfalls). Discoveries are:
- Stored flat with a session_id field
- Not auto-injected into prompts (only accessible via search_discoveries tool)
- Stored in ~/.xhelio/discoveries.json with in-file archival

Each discovery includes pipeline provenance: the exact tool calls that led to the finding.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import config
from .memory import MemoryEmbeddings, estimate_tokens, generate_tags
from .versioned_store import VersionedStore
from .event_bus import get_event_bus, DEBUG

SCHEMA_VERSION = 2


@dataclass
class Discovery:
    """A single scientific discovery entry."""

    id: str = field(default_factory=lambda: f"disc_{uuid.uuid4().hex[:8]}")
    summary: str = ""          # 1-2 sentence abstract
    content: str = ""          # Full detailed knowledge (can be long)
    tags: list[str] = field(default_factory=list)
    missions: list[str] = field(default_factory=list)   # e.g. ["PSP"]
    datasets: list[str] = field(default_factory=list)   # e.g. ["PSP_SWP_SPC_L3I"]
    pipeline: list[dict] = field(default_factory=list)   # [{tool, args, output_label}]
    reasoning: str = ""        # Why this knowledge matters
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_agent: str = ""     # "orchestrator", "mission:PSP", etc.

    # v2 schema fields
    session_id: str = ""       # which session discovered this
    version: int = 1           # monotonic per entry lineage
    supersedes: str = ""       # ID of entry this replaces
    archived: bool = False     # True = old version or consolidated-away


class DiscoveryStore(VersionedStore[Discovery]):
    """Manages scientific discoveries persisted as JSON with in-file archival.

    Inherits VersionedStore for versioning, archival, and search.
    Adds discovery-specific logic: session grouping, mission filtering.
    """

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = config.get_data_dir() / "discoveries.json"
        self._metadata: dict = {}  # session-level metadata
        super().__init__(path)
        self.load()

    # ---- VersionedStore abstract implementations ----

    @property
    def _schema_version(self) -> int:
        return SCHEMA_VERSION

    def _deserialize_entry(self, raw: dict) -> Discovery:
        known = {k for k in Discovery.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in known}
        return Discovery(**filtered)

    def _serialize_entry(self, entry: Discovery) -> dict:
        return asdict(entry)

    def _migrate(self, data: dict, from_version: int) -> dict:
        if from_version < 2:
            # v1 → v2: flatten session-grouped format to flat entries
            sessions = data.get("sessions", {})
            flat_entries = []
            metadata = {}

            for session_id, session_data in sessions.items():
                metadata[session_id] = {
                    "created_at": session_data.get("created_at", ""),
                    "updated_at": session_data.get("updated_at", ""),
                }
                for d in session_data.get("discoveries", []):
                    d["session_id"] = session_id
                    d.setdefault("version", 1)
                    d.setdefault("supersedes", "")
                    d.setdefault("archived", False)
                    flat_entries.append(d)

            data["entries"] = flat_entries
            data["metadata"] = metadata
            data.pop("sessions", None)
            data["schema_version"] = SCHEMA_VERSION

        return data

    def _estimate_entry_tokens(self, entry: Discovery) -> int:
        return (
            estimate_tokens(entry.summary)
            + estimate_tokens(entry.content)
            + estimate_tokens(entry.reasoning)
        )

    def _entry_search_text(self, entry: Discovery) -> str:
        return f"{entry.summary} {entry.content[:500]}"

    def _get_entry_tags(self, entry: Discovery) -> list[str]:
        return entry.tags

    def _get_entry_id(self, entry: Discovery) -> str:
        return entry.id

    def _is_archived(self, entry: Discovery) -> bool:
        return entry.archived

    def _set_archived(self, entry: Discovery, archived: bool) -> None:
        entry.archived = archived

    def _get_version(self, entry: Discovery) -> int:
        return entry.version

    def _get_supersedes(self, entry: Discovery) -> str:
        return entry.supersedes

    def _similarity_threshold(self) -> float:
        return 0.50  # lower threshold than memory (discoveries are richer)

    # ---- VersionedStore hooks ----

    def _serialize_file(self, entries: list[Discovery]) -> dict:
        return {
            "schema_version": self._schema_version,
            "metadata": self._metadata,
            "entries": [self._serialize_entry(e) for e in entries],
        }

    def _parse_file(self, data: dict) -> list[Discovery]:
        raw_entries = data.get("entries", [])
        result = []
        for raw in raw_entries:
            try:
                result.append(self._deserialize_entry(raw))
            except (TypeError, KeyError) as e:
                get_event_bus().emit(DEBUG, agent="Discovery", level="warning", msg=f"[Discovery] Skipping malformed entry: {e}")
        return result

    def _on_load(self, data: dict) -> None:
        self._metadata = data.get("metadata", {})

    def _post_load_fixup(self) -> bool:
        """Import old numbered discovery files if they exist."""
        return self._import_numbered_files()

    def _import_numbered_files(self) -> bool:
        """Merge old discoveries_1.json, discoveries_2.json, etc. into main file."""
        stem = self.path.stem      # "discoveries"
        suffix = self.path.suffix  # ".json"
        parent = self.path.parent
        existing_ids = {d.id for d in self._entries}
        imported = 0
        idx = 1

        while True:
            p = parent / f"{stem}_{idx}{suffix}"
            if not p.exists():
                break
            try:
                with open(p, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                sessions = old_data.get("sessions", {})
                for session_id, session_data in sessions.items():
                    for d in session_data.get("discoveries", []):
                        d["session_id"] = session_id
                        d.setdefault("version", 1)
                        d.setdefault("supersedes", "")
                        d.setdefault("archived", False)
                        known = {k for k in Discovery.__dataclass_fields__}
                        filtered = {k: v for k, v in d.items() if k in known}
                        try:
                            disc = Discovery(**filtered)
                        except (TypeError, KeyError):
                            continue
                        if disc.id not in existing_ids:
                            self._entries.append(disc)
                            existing_ids.add(disc.id)
                            imported += 1
            except (json.JSONDecodeError, OSError):
                pass
            idx += 1

        if imported > 0:
            get_event_bus().emit(DEBUG, agent="Discovery", level="debug", msg=f"[Discovery] Imported {imported} entries from numbered files")
            return True
        return False

    # ---- Score override for mission bonus ----

    def _score_entry(self, entry: Discovery, query_tokens: set[str], query_lower: str) -> float:
        score = super()._score_entry(entry, query_tokens, query_lower)
        # Mission match bonus
        for m in entry.missions:
            if m.lower() in query_lower:
                score += 1.5
        return score

    # ---- CRUD ----

    def add(self, session_id: str, discovery: Discovery) -> None:
        """Append a discovery to a session group and save."""
        if not discovery.tags:
            discovery.tags = generate_tags(
                f"{discovery.summary} {discovery.content}",
                ["generic"],
            )
        discovery.session_id = session_id

        # Update session metadata
        now = datetime.now().isoformat()
        if session_id not in self._metadata:
            self._metadata[session_id] = {
                "created_at": now,
                "updated_at": now,
            }
        self._metadata[session_id]["updated_at"] = now

        self._add_entry(discovery)

    # ---- Queries ----

    def get_all_flat(self) -> list[Discovery]:
        """Return all active (non-archived) discoveries as a flat list."""
        return self.get_active()

    def get_session_discoveries(self, session_id: str) -> list[Discovery]:
        """Return active discoveries for a specific session."""
        return [
            d for d in self._entries
            if not d.archived and d.session_id == session_id
        ]

    def total_tokens(self) -> int:
        """Estimate total tokens across all active discoveries."""
        return self.total_active_tokens()

    # ---- Session operations ----

    def replace_session(self, session_id: str, new_discoveries: list[Discovery]) -> None:
        """Archive old session entries, add new ones."""
        # Archive existing entries for this session
        for d in self._entries:
            if not d.archived and d.session_id == session_id:
                d.archived = True

        # Add new entries
        now = datetime.now().isoformat()
        for d in new_discoveries:
            if not d.tags:
                d.tags = generate_tags(f"{d.summary} {d.content}", ["generic"])
            d.session_id = session_id
        self._metadata[session_id] = {"created_at": now, "updated_at": now}
        self._entries.extend(new_discoveries)
        self.embeddings.invalidate()
        self._rebuild_embeddings()
        self.save()

    def replace_all_sessions(self, new_sessions: dict[str, list[Discovery]]) -> None:
        """Archive all active entries, add new ones grouped by session."""
        for d in self._entries:
            if not d.archived:
                d.archived = True

        now = datetime.now().isoformat()
        for session_id, discoveries in new_sessions.items():
            self._metadata[session_id] = {"created_at": now, "updated_at": now}
            for d in discoveries:
                if not d.tags:
                    d.tags = generate_tags(f"{d.summary} {d.content}", ["generic"])
                d.session_id = session_id
                self._entries.append(d)

        self.embeddings.invalidate()
        self._rebuild_embeddings()
        self.save()

    # ---- Search ----

    def search(
        self,
        query: str,
        limit: int = 5,
        mission: str | None = None,
    ) -> list[Discovery]:
        """Search discoveries by embedding similarity or tag fallback.

        Args:
            query: Search text.
            limit: Max results.
            mission: Optional mission filter (e.g. "PSP").

        Returns:
            List of Discovery objects sorted by relevance.
        """
        if not query or not query.strip():
            return []

        candidates = self.get_active()
        if mission:
            candidates = [
                d for d in candidates
                if mission.upper() in [m.upper() for m in d.missions]
            ]
        if not candidates:
            return []

        # Try embedding-based search
        if self.embeddings.available:
            return self._search_by_embeddings(query, candidates, limit)

        # Fallback to tag-based search
        return self._search_by_tags(query, candidates, limit)
