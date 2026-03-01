"""
Long-term memory system for cross-session user preferences and session summaries.

Stores memories in ~/.xhelio/memory.json. Memories are automatically
extracted at session boundaries and injected into future conversations.

Memory types:
    - "preference": User habits, plot styles, missions of interest
    - "summary": Brief summaries of past analysis sessions
    - "pitfall": Lessons learned from errors or unexpected behavior
    - "reflection": Procedural knowledge learned from errors (Reflexion pattern)
    - "review": Star rating + comment on another memory (review_of links to target)

Scopes (multi-scope: each memory can belong to multiple scopes):
    - "generic": General operational advice
    - "mission:<ID>": Mission-specific knowledge (e.g., "mission:PSP")
    - "visualization": Plotting and rendering knowledge
    - "data_ops": Data transformation and computation knowledge
"""

import json
import re
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import config
from .versioned_store import VersionedStore
from .event_bus import get_event_bus, DEBUG

# Global token budget for memory injection
MEMORY_TOKEN_BUDGET = config.get("memory_token_budget", 100000)


_GITIGNORE_CONTENT = (
    "# Track memory.json and pipelines/\n"
    "*\n"
    "!.gitignore\n"
    "!memory.json\n"
    "!pipelines/\n"
    "!pipelines/*.json\n"
)


def _git_commit_data(repo_dir: Path) -> None:
    """Auto-commit data files (memory.json + pipelines/) to git.

    Lazily initializes a git repo in ``repo_dir`` (~/.xhelio/) if needed.
    Never raises — all errors are logged and swallowed.
    """
    run_kwargs = dict(cwd=repo_dir, capture_output=True, timeout=10)

    try:
        # Lazy git init
        gitignore = repo_dir / ".gitignore"
        if not (repo_dir / ".git").exists():
            subprocess.run(["git", "init"], **run_kwargs, check=True)
            subprocess.run(
                ["git", "config", "user.email", "helion@local"],
                **run_kwargs, check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Helion Memory"],
                **run_kwargs, check=True,
            )
            gitignore.write_text(_GITIGNORE_CONTENT)
        elif gitignore.exists() and "!pipelines/" not in gitignore.read_text():
            # Existing repo but .gitignore doesn't include pipelines/ — rewrite
            gitignore.write_text(_GITIGNORE_CONTENT)

        # Stage and commit
        subprocess.run(
            ["git", "add", "memory.json", "pipelines/", ".gitignore"],
            **run_kwargs, check=True,
        )
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            **run_kwargs,
        )
        if result.returncode != 0:
            # There are staged changes to commit
            from datetime import datetime as _dt
            timestamp = _dt.now().isoformat(timespec="seconds")
            subprocess.run(
                ["git", "commit", "-m", f"data: auto-save {timestamp}"],
                **run_kwargs, check=True,
            )
    except Exception as exc:
        get_event_bus().emit(
            DEBUG, agent="Memory", level="debug",
            msg=f"[Memory] git commit failed: {exc}",
        )


from .token_counter import count_tokens


def estimate_tokens(text: str) -> int:
    """Count tokens in *text* using the Gemini local tokenizer."""
    return count_tokens(text)


def estimate_memory_tokens(memory: "Memory") -> int:
    """Count tokens for a single memory rendered as '- {content}'."""
    return count_tokens(f"- {memory.content}")

# Current schema version for memory.json
SCHEMA_VERSION = 7

# Stop words for tag generation
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "up",
    "that", "this", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "they", "them", "their", "what", "which", "who",
    "whom", "these", "those", "am", "also", "still", "use", "using",
    "used", "make", "makes", "made", "get", "gets", "got", "set",
})


def generate_tags(content: str, scopes: list[str] | str = "generic") -> list[str]:
    """Extract keyword tags from memory content.

    Simple NLP: lowercase, split on non-alphanumeric, remove stop words,
    keep tokens >= 2 chars, deduplicate, add scope-based tags.

    Args:
        content: The memory content text.
        scopes: A list of scopes (or a single scope string for backward compat).
    """
    # Normalize single scope string to list
    if isinstance(scopes, str):
        scopes = [scopes]

    # Tokenize: split on non-alphanumeric characters
    tokens = re.split(r"[^a-zA-Z0-9_]+", content.lower())
    # Filter: remove stop words and short tokens
    tags = list(dict.fromkeys(
        t for t in tokens if t and len(t) >= 2 and t not in _STOP_WORDS
    ))
    # Add scope-based tags for all scopes
    for s in scopes:
        if s.startswith("mission:"):
            mission_tag = s.split(":", 1)[1].lower()
            if mission_tag not in tags:
                tags.insert(0, mission_tag)
        elif s != "generic":
            if s not in tags:
                tags.insert(0, s)
    return tags


class MemoryEmbeddings:
    """Embedding-based similarity for memory content using fastembed.

    Follows the same pattern as knowledge/catalog_search.py:
    - Same model: BAAI/bge-small-en-v1.5 via fastembed
    - Lazy init: only loads model on first use
    - Graceful fallback: all methods return 0/empty if fastembed unavailable

    The fastembed model is shared across all instances (class-level singleton)
    to avoid repeated initialization overhead (~55ms per instance).
    Per-instance ``_available`` can override to force-disable for testing.
    """

    _shared_model = None          # class-level singleton
    _shared_available: bool | None = None  # None = not checked yet

    def __init__(self):
        self._embeddings: np.ndarray | None = None
        self._contents: list[str] = []
        self._available: bool | None = None  # per-instance override (None = use shared)

    def _ensure_model(self) -> bool:
        """Lazy-init the shared embedding model. Returns True if available."""
        # Per-instance override (e.g. for testing fallback paths)
        if self._available is not None:
            return self._available
        if MemoryEmbeddings._shared_available is not None:
            return MemoryEmbeddings._shared_available
        try:
            from fastembed import TextEmbedding
            MemoryEmbeddings._shared_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            MemoryEmbeddings._shared_available = True
        except Exception:
            MemoryEmbeddings._shared_available = False
            get_event_bus().emit(DEBUG, agent="Memory", level="debug", msg="[MemoryEmbeddings] fastembed unavailable, falling back to tag-based methods")
        return MemoryEmbeddings._shared_available

    def build(self, memories) -> None:
        """Embed all memory content strings and cache the matrix.

        Accepts objects with a .content attribute (Memory objects or _EmbeddingWrapper).
        """
        if not self._ensure_model():
            return
        self._contents = [m.content for m in memories if m.content]
        if not self._contents:
            self._embeddings = None
            return
        self._embeddings = np.array(list(MemoryEmbeddings._shared_model.embed(self._contents)))

    def embed_query(self, text: str) -> np.ndarray | None:
        """Embed a single query string. Returns None if unavailable."""
        if not self._ensure_model():
            return None
        return np.array(list(MemoryEmbeddings._shared_model.embed([text])))[0]

    def cosine_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all stored embeddings.

        Returns array of similarity scores, or empty array if no embeddings.
        """
        if self._embeddings is None or query_emb is None:
            return np.array([])
        return self._embeddings @ query_emb

    def pairwise_max_similarity(self, text: str, others: list[str]) -> float:
        """Max cosine similarity between text and a list of other strings.

        Returns 0.0 if fastembed unavailable or inputs are empty.
        """
        if not others or not text:
            return 0.0
        if not self._ensure_model():
            return 0.0
        try:
            all_texts = [text] + others
            embs = np.array(list(MemoryEmbeddings._shared_model.embed(all_texts)))
            query_emb = embs[0]
            other_embs = embs[1:]
            sims = other_embs @ query_emb
            return float(np.max(sims))
        except Exception:
            return 0.0

    def invalidate(self) -> None:
        """Clear cached embeddings (call after add/remove)."""
        self._embeddings = None
        self._contents = []

    @property
    def available(self) -> bool:
        """Check if fastembed is available (triggers lazy init)."""
        return self._ensure_model()


@dataclass
class Memory:
    """A single memory entry."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: str = "preference"  # "preference", "summary", "pitfall", or "reflection"
    scopes: list[str] = field(default_factory=lambda: ["generic"])  # multi-scope list
    content: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_session: str = ""
    enabled: bool = True

    # New fields (v2 schema — all have defaults for backward compat)
    source: str = "extracted"     # "extracted" | "reflected" | "user_explicit" | "consolidated"
    tags: list[str] = field(default_factory=list)
    supersedes: str = ""          # ID of memory this replaces
    access_count: int = 0         # times retrieved for injection
    last_accessed: str = ""       # ISO timestamp

    # v4 schema fields
    version: int = 1              # monotonic per entry lineage
    archived: bool = False        # True = old version or consolidated-away

    # v7 schema field: reviews are stored as separate Memory entries
    review_of: str = ""   # non-empty only for type="review" — ID of the memory being reviewed


class MemoryStore(VersionedStore[Memory]):
    """Manages long-term memories persisted as JSON.

    Inherits VersionedStore for versioning, archival, and search.
    Adds memory-specific logic: global toggle, scoped queries, prompt building.
    """

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = config.get_data_dir() / "memory.json"
        self._global_enabled: bool = True
        self._last_injected_ids: dict[str, str] = {}  # memory_id → consuming agent name
        super().__init__(path)
        self.load()

    def save(self) -> None:
        super().save()
        _git_commit_data(self.path.parent)

    # ---- VersionedStore abstract implementations ----

    @property
    def _schema_version(self) -> int:
        return SCHEMA_VERSION

    def _deserialize_entry(self, raw: dict) -> Memory:
        # Filter to only known fields
        known = {k for k in Memory.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in known}
        return Memory(**filtered)

    def _serialize_entry(self, entry: Memory) -> dict:
        return asdict(entry)

    def _migrate(self, data: dict, from_version: int) -> dict:
        raw_memories = data.get("memories", [])

        # v1 → v2: backfill new fields
        if from_version < 2:
            for m in raw_memories:
                m.setdefault("source", "extracted")
                m.setdefault("tags", [])
                m.setdefault("supersedes", "")
                m.setdefault("access_count", 0)
                m.setdefault("last_accessed", "")

        # v2 → v3: scope (str) → scopes (list)
        if from_version < 3:
            for m in raw_memories:
                if "scope" in m and "scopes" not in m:
                    m["scopes"] = [m.pop("scope")]
                elif "scopes" not in m:
                    m["scopes"] = ["generic"]
                m.pop("scope", None)

        # v3 → v4: add version and archived fields
        if from_version < 4:
            for m in raw_memories:
                m.setdefault("version", 1)
                m.setdefault("archived", False)

        # v4 → v5: add reviews list (intermediate step, converted to single review in v6)
        if from_version < 5:
            for m in raw_memories:
                m.setdefault("reviews", [])

        # v5 → v6: remove confidence, convert reviews list → single review
        if from_version < 6:
            for m in raw_memories:
                m.pop("confidence", None)
                reviews = m.pop("reviews", [])
                m["review"] = reviews[-1] if reviews else None

        # v6 → v7: convert inline review dict → standalone review Memory entry
        if from_version < 7:
            new_review_entries = []
            for m in raw_memories:
                review = m.pop("review", None)
                m["review_of"] = ""  # non-review memories have empty review_of
                if review and isinstance(review, dict):
                    new_review_entries.append({
                        "id": uuid.uuid4().hex[:12],
                        "type": "review",
                        "scopes": list(m.get("scopes", ["generic"])),
                        "content": f"{review.get('stars', 3)}★ {review.get('comment', '')}",
                        "created_at": review.get("created_at", m.get("created_at", "")),
                        "source_session": review.get("session_id", ""),
                        "enabled": True,
                        "source": "extracted",
                        "tags": [f"review:{m['id']}", review.get("agent", ""), f"stars:{review.get('stars', 3)}"],
                        "supersedes": "",
                        "access_count": 0,
                        "last_accessed": "",
                        "version": 1,
                        "archived": False,
                        "review_of": m["id"],
                    })
            raw_memories.extend(new_review_entries)

        data["memories"] = raw_memories
        data["schema_version"] = SCHEMA_VERSION
        return data

    def _estimate_entry_tokens(self, entry: Memory) -> int:
        return estimate_memory_tokens(entry)

    def _entry_search_text(self, entry: Memory) -> str:
        return entry.content

    def _get_entry_tags(self, entry: Memory) -> list[str]:
        return entry.tags

    def _get_entry_id(self, entry: Memory) -> str:
        return entry.id

    def _is_archived(self, entry: Memory) -> bool:
        return entry.archived

    def _set_archived(self, entry: Memory, archived: bool) -> None:
        entry.archived = archived

    def _get_version(self, entry: Memory) -> int:
        return entry.version

    def _get_supersedes(self, entry: Memory) -> str:
        return entry.supersedes

    # ---- VersionedStore hooks ----

    def _serialize_file(self, entries: list[Memory]) -> dict:
        return {
            "schema_version": self._schema_version,
            "global_enabled": self._global_enabled,
            "memories": [self._serialize_entry(e) for e in entries],
        }

    def _parse_file(self, data: dict) -> list[Memory]:
        raw_memories = data.get("memories", [])
        result = []
        for raw in raw_memories:
            try:
                result.append(self._deserialize_entry(raw))
            except (TypeError, KeyError) as e:
                get_event_bus().emit(DEBUG, agent="Memory", level="warning", msg=f"[Memory] Skipping malformed entry: {e}")
        return result

    def _on_load(self, data: dict) -> None:
        self._global_enabled = data.get("global_enabled", True)

    def _post_load_fixup(self) -> bool:
        """Generate tags for memories that have none."""
        needs_save = False

        # Retroactively generate tags
        for m in self._entries:
            if not m.tags:
                m.tags = generate_tags(m.content, m.scopes)
                needs_save = True

        return needs_save

    # ---- CRUD (delegates to VersionedStore) ----

    def add(self, memory: Memory) -> None:
        """Add a memory and save. Auto-generates tags if empty."""
        if not memory.tags:
            memory.tags = generate_tags(memory.content, memory.scopes)
        self._add_entry(memory)

    def add_no_save(self, memory: Memory) -> None:
        """Add a memory in-memory only (no disk write). Call save() later."""
        if not memory.tags:
            memory.tags = generate_tags(memory.content, memory.scopes)
        self._add_entry(memory, save=False)

    def remove(self, memory_id: str) -> bool:
        """Archive a memory by ID. Returns True if found."""
        return self._remove_entry(memory_id)

    def toggle(self, memory_id: str, enabled: bool) -> bool:
        """Toggle a memory's enabled state. Returns True if found."""
        for m in self._entries:
            if m.id == memory_id:
                m.enabled = enabled
                self._mutation_epoch += 1
                self.save()
                return True
        return False

    def replace_all(self, memories: list[Memory]) -> None:
        """Replace all memories with a new list and save."""
        self._entries = memories
        self._mutation_epoch += 1
        self._rebuild_embeddings()
        self.save()

    def execute_actions(self, actions: list[dict], session_id: str = "") -> int:
        """Apply add/edit/drop actions from the MemoryAgent. Returns count of successful actions.

        Validates each action and delegates to existing CRUD. Single save() at end.
        """
        import re
        _valid_scope_re = re.compile(r"^(generic|visualization|data_ops|mission:\w+)$")
        _valid_types = {"preference", "summary", "pitfall", "reflection"}

        def _validate_scopes(scopes_raw) -> list[str]:
            if isinstance(scopes_raw, str):
                scopes_raw = [scopes_raw]
            if not isinstance(scopes_raw, list):
                return ["generic"]
            valid = [s for s in scopes_raw if isinstance(s, str) and _valid_scope_re.match(s)]
            return valid or ["generic"]

        count = 0
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_type = action.get("action", "")

            try:
                if action_type == "add":
                    content = action.get("content", "").strip()
                    mtype = action.get("type", "preference")
                    if not content or mtype not in _valid_types:
                        continue
                    scopes = _validate_scopes(action.get("scopes", ["generic"]))
                    tags = generate_tags(content, scopes)
                    self.add_no_save(Memory(
                        type=mtype, scopes=scopes, content=content,
                        source="extracted",
                        source_session=session_id, tags=tags,
                    ))
                    count += 1

                elif action_type == "edit":
                    entry_id = action.get("id", "")
                    content = action.get("content", "").strip()
                    if not entry_id or not content:
                        continue
                    old = self.get_by_id(entry_id)
                    if old is None or old.archived:
                        continue
                    tags = generate_tags(content, old.scopes)
                    new_memory = Memory(
                        type=old.type, scopes=old.scopes, content=content,
                        source="extracted",
                        source_session=session_id, supersedes=entry_id,
                        version=old.version + 1, tags=tags,
                    )
                    old.archived = True
                    self.add_no_save(new_memory)
                    count += 1

                elif action_type == "drop":
                    entry_id = action.get("id", "")
                    if not entry_id:
                        continue
                    entry = self.get_by_id(entry_id)
                    if entry is None or entry.archived:
                        continue
                    entry.archived = True
                    self.embeddings.invalidate()
                    count += 1

            except Exception as e:
                get_event_bus().emit(DEBUG, agent="Memory", level="debug", msg=f"[Memory] execute_actions: {action_type} failed — {e}")

        if count > 0:
            # Bump epoch for any direct archival mutations not via _add_entry/_remove_entry
            self._mutation_epoch += 1
            self.save()
        return count

    def get_superseded_ids(self) -> set[str]:
        """Return the set of all IDs that are superseded by another entry."""
        return {m.supersedes for m in self._entries if m.supersedes}

    def get_truly_archived(self) -> list[Memory]:
        """Return archived entries that have no successor (dropped/deleted).

        Versioned predecessors (entries whose ID appears in another entry's
        ``supersedes`` field) are excluded — they are old versions of still-living
        memories, not truly retired.
        """
        superseded = self.get_superseded_ids()
        return [m for m in self._entries if m.archived and m.id not in superseded]

    def clear_all(self) -> int:
        """Remove all memories. Returns count removed."""
        count = len(self._entries)
        self._entries = []
        if count > 0:
            self._mutation_epoch += 1
        self.save()
        return count

    # ---- Global toggle ----

    def toggle_global(self, enabled: bool) -> None:
        """Enable or disable the entire memory system."""
        self._global_enabled = enabled
        self._mutation_epoch += 1
        self.save()

    def is_global_enabled(self) -> bool:
        """Check if the memory system is globally enabled."""
        return self._global_enabled

    # ---- Token estimation ----

    def total_tokens(self) -> int:
        """Sum of estimated tokens for all enabled active memories."""
        return sum(estimate_memory_tokens(m) for m in self._entries if m.enabled and not m.archived)

    # ---- Queries ----

    def get_all(self) -> list[Memory]:
        """Return all active (non-archived) memories."""
        return [m for m in self._entries if not m.archived]

    def get_enabled(self) -> list[Memory]:
        """Return only enabled active memories."""
        return [m for m in self._entries if m.enabled and not m.archived]

    def get_pitfalls_by_scope(self, scope: str) -> list[Memory]:
        """Return enabled pitfalls matching the given scope."""
        return [
            m for m in self._entries
            if m.enabled and not m.archived and m.type == "pitfall" and scope in m.scopes
        ]

    def get_scoped_pitfall_texts(self, scope: str) -> list[str]:
        """Return content strings for enabled pitfalls matching scope."""
        return [m.content for m in self.get_pitfalls_by_scope(scope)]

    # ---- Review queries ----

    def get_review_for(self, memory_id: str, agent: str | None = None) -> Memory | None:
        """Return an active review for a given memory, optionally filtered by agent.

        If ``agent`` is given, returns the review from that specific agent.
        Otherwise returns the first active review found.
        """
        for m in self._entries:
            if m.review_of == memory_id and m.type == "review" and not m.archived:
                if agent is None or agent in m.tags:
                    return m
        return None

    def get_reviews_for(self, memory_id: str) -> list[Memory]:
        """Return all active (non-archived) reviews for a given memory."""
        return [
            m for m in self._entries
            if m.review_of == memory_id and m.type == "review" and not m.archived
        ]

    def get_reviews(self) -> list[Memory]:
        """Return all active (non-archived) review memories."""
        return [m for m in self._entries if m.type == "review" and not m.archived]

    def get_all_reviews_for_lineage(self, memory_id: str) -> list[Memory]:
        """Return all reviews (including archived) across the entire version chain of a memory.

        Walks the supersedes chain to collect all version IDs, then returns
        every review (active or archived) whose review_of matches any of them.
        """
        # Collect all IDs in the version lineage
        lineage_ids: set[str] = set()
        current_id: str | None = memory_id
        while current_id and current_id not in lineage_ids:
            lineage_ids.add(current_id)
            entry = self.get_by_id(current_id)
            if entry is None:
                break
            current_id = entry.supersedes or None
        # Collect all reviews (active + archived) targeting any version in the lineage
        return [
            m for m in self._entries
            if m.type == "review" and m.review_of in lineage_ids
        ]

    def get_recent_reviews_for_lineage(self, memory_id: str, n: int = 10) -> list[Memory]:
        """Return the N most recent reviews across the entire version chain of a memory.

        Reuses get_all_reviews_for_lineage() and returns only the latest N
        reviews sorted by created_at descending. Useful for consolidation where
        only recent sentiment matters.
        """
        all_reviews = self.get_all_reviews_for_lineage(memory_id)
        all_reviews.sort(key=lambda r: r.created_at, reverse=True)
        return all_reviews[:n]

    # ---- Search ----

    def search(
        self,
        query: str,
        mem_type: str | None = None,
        scope: str | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        """Search memories (active + archived) using embedding similarity or tag fallback.

        Tries embedding-based cosine similarity first. Falls back to tag-based
        search if fastembed is unavailable.
        Returns Memory objects sorted by relevance (highest first).
        """
        if not query or not query.strip():
            return []

        # Collect candidate memories with filters applied
        candidates = self._collect_search_candidates(mem_type, scope)
        if not candidates:
            return []

        # Try embedding-based search first
        if self.embeddings.available:
            return self._search_by_embeddings(query, candidates, limit)

        # Fallback to tag-based search
        return self._search_by_tags(query, candidates, limit)

    def _collect_search_candidates(
        self, mem_type: str | None, scope: str | None,
    ) -> list[Memory]:
        """Collect all memories (active + archived) matching type/scope filters."""
        candidates = []
        for m in self._entries:
            if mem_type and m.type != mem_type:
                continue
            if scope and scope not in m.scopes:
                continue
            candidates.append(m)
        return candidates

    def _search_by_embeddings(
        self, query: str, candidates: list[Memory], limit: int,
    ) -> list[Memory]:
        """Rank candidates by cosine similarity to query embedding."""
        try:
            contents = [m.content for m in candidates]
            all_texts = [query] + contents
            embs = np.array(list(self.embeddings._model.embed(all_texts)))
            query_emb = embs[0]
            candidate_embs = embs[1:]
            sims = candidate_embs @ query_emb

            scored = [
                (float(sims[i]), candidates[i])
                for i in range(len(candidates))
                if sims[i] > 0.55  # minimum similarity threshold (bge-small has ~0.5 baseline)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [m for _, m in scored[:limit]]
        except Exception:
            # Fallback to tags on any embedding failure
            return self._search_by_tags(query, candidates, limit)

    def _search_by_tags(
        self, query: str, candidates: list[Memory], limit: int,
    ) -> list[Memory]:
        """Rank candidates by tag overlap + substring match (original logic)."""
        query_tokens = set(re.split(r"[^a-zA-Z0-9_]+", query.lower()))
        query_tokens -= _STOP_WORDS
        query_tokens.discard("")
        query_lower = query.lower()

        if not query_tokens and not query_lower.strip():
            return []

        scored: list[tuple[float, Memory]] = []
        for m in candidates:
            score = self._score_memory(m, query_tokens, query_lower)
            if score > 0:
                scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    @staticmethod
    def _score_memory(
        memory: Memory,
        query_tokens: set[str],
        query_lower: str,
    ) -> float:
        """Score a memory against a query. Higher = more relevant."""
        score = 0.0
        # Tag overlap scoring (×2 per match)
        if memory.tags:
            tag_set = set(memory.tags)
            tag_matches = len(query_tokens & tag_set)
            score += tag_matches * 2.0
        # Content substring match (×1)
        if query_lower in memory.content.lower():
            score += 1.0
        return score

    # ---- Prompt building ----

    @staticmethod
    def _agent_name_for_scope(scope: str) -> str:
        """Map a scope string to the consuming agent's display name."""
        if scope == "visualization":
            return "VizAgent"
        elif scope == "data_ops":
            return "DataOpsAgent"
        elif scope.startswith("mission:"):
            return f"MissionAgent[{scope.split(':', 1)[1]}]"
        return "OrchestratorAgent"

    def format_for_injection(
        self,
        scope: str = "generic",
        include_summaries: bool = False,
        include_review_instruction: bool = True,
        active_missions: set[str] | None = None,
    ) -> str:
        """Build a unified memory section for injection into any agent prompt.

        Produces identical structure for orchestrator and sub-agents:
        ``[CONTEXT FROM LONG-TERM MEMORY] ... [END MEMORY CONTEXT]``.

        Args:
            scope: Filter memories where ``scope in m.scopes``.
            include_summaries: Include ``### Past Sessions`` (summaries).
                Only the orchestrator sets this to True.
            include_review_instruction: Append the review footer.
                Set to False for inline completion.
            active_missions: Set of active mission IDs (e.g. ``{"ACE", "WIND"}``).
                When provided, memories that have a ``mission:<ID>`` scope are
                only included if that mission is in this set.  Memories without
                any mission scope are unaffected.  Useful for filtering
                irrelevant mission-specific memories from the DataOps agent.

        Returns:
            Formatted section string, or ``""`` if nothing to inject.
        """
        if not self._global_enabled:
            return ""

        enabled = self.get_enabled()
        if not enabled:
            return ""

        # Filter by scope, exclude review-type memories (they're metadata, not operational knowledge)
        matching = [m for m in enabled if scope in m.scopes and m.type != "review"]

        # When active_missions is set, exclude memories scoped to inactive missions
        if active_missions is not None:
            filtered = []
            for m in matching:
                mission_scopes = [s for s in m.scopes if s.startswith("mission:")]
                if mission_scopes:
                    # Memory is mission-specific — only include if at least one
                    # of its mission scopes is currently active
                    if any(s.split(":", 1)[1] in active_missions for s in mission_scopes):
                        filtered.append(m)
                else:
                    # No mission scope — always include
                    filtered.append(m)
            matching = filtered
        if not matching:
            return ""

        # Bucket by type
        preferences = [m for m in matching if m.type == "preference"]
        summaries = [m for m in matching if m.type == "summary"] if include_summaries else []
        pitfalls = [m for m in matching if m.type == "pitfall"]
        reflections = [m for m in matching if m.type == "reflection"]

        if not preferences and not summaries and not pitfalls and not reflections:
            return ""

        agent_name = self._agent_name_for_scope(scope)
        now = datetime.now().isoformat()
        budget = MEMORY_TOKEN_BUDGET
        tokens_used = 0
        parts = ["[CONTEXT FROM LONG-TERM MEMORY]", "## Operational Knowledge"]
        tokens_used += sum(estimate_tokens(p) for p in parts)
        injected_ids: dict[str, str] = {}

        # Pre-compute this agent's existing reviews for each memory
        own_reviews: dict[str, str] = {}  # memory_id → review content
        for m in matching:
            review = self.get_review_for(m.id, agent=agent_name)
            if review is not None:
                own_reviews[m.id] = review.content

        def _fmt(m: "Memory", extra: str = "") -> str:
            """Format a memory line with the agent's own review appended if present."""
            line = f"- [{m.id}]{extra} {m.content}"
            review_text = own_reviews.get(m.id)
            if review_text:
                line += f"\n  Your previous review: {review_text}"
            return line

        injected_memories: list["Memory"] = []

        def _add_items(items, header_lines, formatter):
            """Add items to parts until token budget is exhausted."""
            nonlocal tokens_used
            header_added = False
            for m in items:
                line = formatter(m)
                line_tokens = estimate_tokens(line)
                if tokens_used + line_tokens > budget:
                    break
                if not header_added:
                    for h in header_lines:
                        parts.append(h)
                        tokens_used += estimate_tokens(h)
                    header_added = True
                parts.append(line)
                tokens_used += line_tokens
                injected_memories.append(m)
                injected_ids[m.id] = agent_name

        _add_items(
            preferences,
            ["", "### Preferences"],
            lambda m: _fmt(m),
        )
        _add_items(
            summaries,
            ["", "### Past Sessions"],
            lambda m: _fmt(m, f" ({m.created_at[:10]})") if m.created_at else _fmt(m),
        )
        _add_items(
            pitfalls,
            ["", "### Lessons Learned"],
            lambda m: _fmt(m),
        )
        _add_items(
            reflections,
            ["", "### Operational Reflections"],
            lambda m: _fmt(m),
        )

        # If we only have the wrapper + header, return empty
        if len(parts) <= 2:
            return ""

        if include_review_instruction:
            parts.append("")
            parts.append(
                "IMPORTANT: After completing your main task, you MUST call review_memory(memory_id, stars, comment) "
                "for at least 1 (up to 4) of the memories listed above that you have NOT previously reviewed. "
                "If you have already reviewed a memory (shown as \"Your previous review\" above), "
                "only re-review it if your opinion has substantially changed based on this session's experience. "
                "Do not re-submit the same or similar review.\n"
                "The comment MUST use this exact four-line format:\n"
                "(1) Rating: why this star count\n"
                "(2) Criticism: what's wrong or could be better\n"
                "(3) Suggestion: how to improve the memory\n"
                "(4) Comment: any extra observation"
            )

        parts.append("[END MEMORY CONTEXT]")

        # Batch-update access tracking (avoids per-item mutation race with
        # parallel mission agents that each call format_for_injection)
        with self._save_lock:
            for m in injected_memories:
                m.access_count += 1
                m.last_accessed = now
        self._last_injected_ids.update(injected_ids)
        return "\n".join(parts)

    def build_prompt_section(self, include_review_instruction: bool = True) -> str:
        """Build memory section for orchestrator prompts (thin wrapper).

        Returns empty string if disabled or no enabled memories.
        """
        return self.format_for_injection(
            scope="generic", include_summaries=True,
            include_review_instruction=include_review_instruction,
        )

    def get_scoped_memory_ids(
        self,
        scope: str,
        active_missions: set[str] | None = None,
    ) -> tuple[int, dict[str, str]]:
        """Return the current epoch and a snapshot of scoped memory IDs.

        Used by actors for diff-based incremental memory injection.

        Args:
            scope: Filter memories where ``scope in m.scopes``.
            active_missions: If provided, exclude memories scoped to inactive
                missions (same logic as ``format_for_injection``).

        Returns:
            ``(mutation_epoch, {memory_id: formatted_line})`` for enabled,
            non-archived, scope-matching memories. Reviews are excluded.
        """
        if not self._global_enabled:
            return (self._mutation_epoch, {})

        enabled = self.get_enabled()
        if not enabled:
            return (self._mutation_epoch, {})

        # Filter by scope, exclude review-type memories
        matching = [m for m in enabled if scope in m.scopes and m.type != "review"]

        # When active_missions is set, exclude memories scoped to inactive missions
        if active_missions is not None:
            filtered = []
            for m in matching:
                mission_scopes = [s for s in m.scopes if s.startswith("mission:")]
                if mission_scopes:
                    if any(s.split(":", 1)[1] in active_missions for s in mission_scopes):
                        filtered.append(m)
                else:
                    filtered.append(m)
            matching = filtered

        # Build {id: formatted_line} map, bucketed by type
        result: dict[str, str] = {}
        for m in matching:
            if m.type == "preference":
                result[m.id] = f"[Preferences] [{m.id}] {m.content}"
            elif m.type == "summary":
                date_suffix = f" ({m.created_at[:10]})" if m.created_at else ""
                result[m.id] = f"[Past Sessions] [{m.id}]{date_suffix} {m.content}"
            elif m.type == "pitfall":
                result[m.id] = f"[Lessons Learned] [{m.id}] {m.content}"
            elif m.type == "reflection":
                result[m.id] = f"[Operational Reflections] [{m.id}] {m.content}"
            else:
                result[m.id] = f"[{m.type}] [{m.id}] {m.content}"

        return (self._mutation_epoch, result)
