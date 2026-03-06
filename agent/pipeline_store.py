"""
Pipeline Store — persistent index of saved pipelines with semantic search.

Provides a searchable metadata index on top of the existing saved pipeline
files in ~/.xhelio/pipelines/.  Each PipelineEntry is a lightweight
metadata record (name, tags, datasets, missions); the actual
pipeline DAG stays in its own JSON file managed by SavedPipeline.

Follows the VersionedStore pattern for versioning, archival,
embedding-based search, and tag fallback.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
from .memory import MemoryEmbeddings, generate_tags
from .token_counter import count_tokens as estimate_tokens
from .versioned_store import VersionedStore
from .event_bus import get_event_bus, DEBUG

SCHEMA_VERSION = 2


def _gen_entry_id() -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"pl_{date}_{uuid.uuid4().hex[:8]}"


@dataclass
class PipelineEntry:
    """Metadata record for a saved pipeline."""

    id: str = field(default_factory=_gen_entry_id)
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)   # e.g. ["AC_H2_MFI.BGSEc"]
    missions: list[str] = field(default_factory=list)    # e.g. ["ACE"]
    step_count: int = 0
    source_session_id: str = ""
    pipeline_file: str = ""     # filename in pipelines/ dir
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Family dedup fields
    family_id: str = ""                                   # SHA-256 appropriation fingerprint
    variant_ids: list[str] = field(default_factory=list)  # all pl_ IDs sharing this family

    # VersionedStore required fields
    version: int = 1
    supersedes: str = ""
    archived: bool = False


class PipelineStore(VersionedStore[PipelineEntry]):
    """Manages pipeline metadata with in-file archival and search.

    Inherits VersionedStore for versioning, archival, and search.
    Adds pipeline-specific logic: mission/dataset filtering and context
    injection.  Registration is driven by the MemoryAgent (LLM-judged).
    """

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = config.get_data_dir() / "pipeline_store.json"
        super().__init__(path)
        self.load()

    # ---- VersionedStore abstract implementations ----

    @property
    def _schema_version(self) -> int:
        return SCHEMA_VERSION

    def _deserialize_entry(self, raw: dict) -> PipelineEntry:
        known = {k for k in PipelineEntry.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in known}
        return PipelineEntry(**filtered)

    def _serialize_entry(self, entry: PipelineEntry) -> dict:
        return asdict(entry)

    def _migrate(self, data: dict, from_version: int) -> dict:
        if from_version < 2:
            # v1 → v2: add family_id and variant_ids fields
            for entry in data.get("entries", []):
                if "family_id" not in entry:
                    entry["family_id"] = ""
                if "variant_ids" not in entry:
                    entry["variant_ids"] = [entry["id"]]
            data["schema_version"] = 2
        return data

    def _estimate_entry_tokens(self, entry: PipelineEntry) -> int:
        return (
            estimate_tokens(entry.name)
            + estimate_tokens(entry.description)
            + estimate_tokens(" ".join(entry.tags))
        )

    def _entry_search_text(self, entry: PipelineEntry) -> str:
        return (
            f"{entry.name} {entry.description} "
            f"{' '.join(entry.tags)} "
            f"{' '.join(entry.datasets)} "
            f"{' '.join(entry.missions)}"
        )

    def _get_entry_tags(self, entry: PipelineEntry) -> list[str]:
        return entry.tags

    def _get_entry_id(self, entry: PipelineEntry) -> str:
        return entry.id

    def _is_archived(self, entry: PipelineEntry) -> bool:
        return entry.archived

    def _set_archived(self, entry: PipelineEntry, archived: bool) -> None:
        entry.archived = archived

    def _get_version(self, entry: PipelineEntry) -> int:
        return entry.version

    def _get_supersedes(self, entry: PipelineEntry) -> str:
        return entry.supersedes

    def _similarity_threshold(self) -> float:
        return 0.50  # pipeline descriptions are rich

    def _post_load_fixup(self) -> bool:
        """Backfill empty family_id fields from actual pipeline files."""
        from data_ops.pipeline import SavedPipeline, appropriation_fingerprint

        changed = False
        for entry in self._entries:
            if not entry.family_id:
                try:
                    pipeline = SavedPipeline.load(entry.id)
                    entry.family_id = appropriation_fingerprint(pipeline.steps)
                    if not entry.variant_ids:
                        entry.variant_ids = [entry.id]
                    changed = True
                except (FileNotFoundError, Exception):
                    # Pipeline file missing or unreadable — leave empty
                    pass
        return changed

    def _find_by_family_id(self, family_id: str) -> Optional[PipelineEntry]:
        """Find an active entry with the given family_id."""
        for entry in self._entries:
            if not entry.archived and entry.family_id == family_id:
                return entry
        return None

    # ---- Registration ----

    def register(
        self,
        pipeline,
        llm_missions: list[str] | None = None,
        llm_tags: list[str] | None = None,
    ) -> Optional[PipelineEntry]:
        """Create or update a PipelineEntry from a SavedPipeline.

        Registration is driven by the MemoryAgent — the LLM has already
        decided this pipeline is worth persisting.  The store handles
        family dedup (identical appropriation fingerprints are grouped
        under a single entry).

        Args:
            pipeline: A SavedPipeline instance (already saved to disk).
            llm_missions: Mission IDs provided by the LLM (already validated).
                If provided, used directly instead of auto-extraction.
            llm_tags: Free-form keyword tags from the LLM.
                If provided, merged with any auto-generated tags.

        Returns:
            The PipelineEntry (new or existing family match).
        """
        from data_ops.pipeline import appropriation_fingerprint
        from knowledge.mission_prefixes import match_dataset_to_mission, get_canonical_id

        # 1. Compute family fingerprint
        family_id = appropriation_fingerprint(pipeline.steps)

        # 3. Extract datasets and missions
        datasets = []
        for step in pipeline.steps:
            if step["tool"] == "fetch_data":
                ds = step["params"].get("dataset_id", "")
                param = step["params"].get("parameter_id", "")
                if ds:
                    label = f"{ds}.{param}" if param else ds
                    datasets.append(label)

        if llm_missions is not None:
            missions_set: set[str] = set(llm_missions)
        else:
            missions_set = set()
            for d in datasets:
                dataset_id = d.split(".")[0]
                stem, _ = match_dataset_to_mission(dataset_id)
                if stem:
                    missions_set.add(get_canonical_id(stem))

        # Tags: merge LLM-provided tags with pipeline.tags / auto-generated
        tags = list(pipeline.tags) if pipeline.tags else []
        if llm_tags:
            tags_set = set(tags)
            for t in llm_tags:
                if t not in tags_set:
                    tags.append(t)
                    tags_set.add(t)
        if not tags:
            tag_text = f"{pipeline.name} {pipeline.description} {' '.join(datasets)}"
            tags = generate_tags(tag_text, ["pipeline"])

        # 4. Check for existing family
        existing = self._find_by_family_id(family_id)
        if existing is not None:
            # Append variant ID if not already tracked
            if pipeline.id not in existing.variant_ids:
                existing.variant_ids.append(pipeline.id)
            existing.updated_at = datetime.now().isoformat()
            # Update description if new one is richer
            if len(pipeline.description or "") > len(existing.description or ""):
                existing.description = pipeline.description
            # Merge any new tags
            existing_tags_set = set(existing.tags)
            for t in tags:
                if t not in existing_tags_set:
                    existing.tags.append(t)
                    existing_tags_set.add(t)
            self.embeddings.invalidate()
            self.save()
            get_event_bus().emit(
                DEBUG, agent="PipelineStore", level="debug",
                msg=f"[PipelineStore] Added variant {pipeline.id} to family "
                    f"{existing.id} ({len(existing.variant_ids)} variants)",
            )
            return existing

        # 5. No existing family — archive any old entry with same pipeline ID
        old = self.get_by_id(pipeline.id)
        if old and not self._is_archived(old):
            self._set_archived(old, True)

        entry = PipelineEntry(
            id=pipeline.id,
            name=pipeline.name,
            description=pipeline.description or "",
            tags=tags,
            datasets=sorted(datasets),
            missions=sorted(missions_set),
            step_count=len(pipeline.steps),
            source_session_id=getattr(pipeline, "source_session_id", ""),
            pipeline_file=f"{pipeline.id}.json",
            family_id=family_id,
            variant_ids=[pipeline.id],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self._add_entry(entry)
        get_event_bus().emit(
            DEBUG, agent="PipelineStore", level="debug",
            msg=f"[PipelineStore] Registered pipeline '{entry.name}' ({entry.id}), "
                f"{entry.step_count} steps, datasets={entry.datasets}",
        )
        return entry

    # ---- Search with mission/dataset pre-filtering ----

    def search(
        self,
        query: str = "",
        limit: int = 10,
        mission: str | None = None,
        dataset: str | None = None,
        include_archived: bool = False,
    ) -> list[PipelineEntry]:
        """Search pipelines by query, mission, or dataset.

        Args:
            query: Natural language search text.
            limit: Max results.
            mission: Optional mission filter (e.g., "ACE").
            dataset: Optional dataset substring filter (e.g., "AC_H2_MFI").
            include_archived: If True, also search archived entries.

        Returns:
            List of PipelineEntry objects sorted by relevance.
        """
        if include_archived:
            candidates = list(self._entries)
        else:
            candidates = self.get_active()

        # Pre-filter by mission
        if mission:
            candidates = [
                e for e in candidates
                if mission.upper() in [m.upper() for m in e.missions]
            ]

        # Pre-filter by dataset substring
        if dataset:
            dataset_lower = dataset.lower()
            candidates = [
                e for e in candidates
                if any(dataset_lower in d.lower() for d in e.datasets)
            ]

        if not candidates:
            return []

        # If no query text, sort by updated_at desc
        if not query or not query.strip():
            candidates.sort(key=lambda e: e.updated_at, reverse=True)
            return candidates[:limit]

        # Embedding-based or tag-based search
        if self.embeddings.available:
            return self._search_by_embeddings(query, candidates, limit)
        return self._search_by_tags(query, candidates, limit)

    # ---- Context injection ----

    def get_for_injection(self, limit: int = 15) -> list[PipelineEntry]:
        """Return top entries for passive context injection.

        Sorted by updated_at desc (most recently created/modified first).

        Args:
            limit: Maximum entries to return.

        Returns:
            List of PipelineEntry.
        """
        active = self.get_active()
        if not active:
            return []

        active.sort(key=lambda e: e.updated_at, reverse=True)
        return active[:limit]

