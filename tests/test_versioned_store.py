"""
Tests for agent.versioned_store — VersionedStore base class.

Uses a minimal _TestEntry dataclass and _TestStore subclass to test
generic behavior: load/save, schema migration, add/update/remove,
archival, version history, search, replace_active, purge_archived,
token budget.

Run with: python -m pytest tests/test_versioned_store.py -v
"""

import json
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import pytest

from agent.versioned_store import VersionedStore


@dataclass
class _TestEntry:
    """Minimal entry for testing VersionedStore."""

    id: str = ""
    content: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)
    version: int = 1
    supersedes: str = ""
    archived: bool = False


class _TestStore(VersionedStore[_TestEntry]):
    """Minimal store for testing."""

    def __init__(self, path):
        super().__init__(path)
        self.load()

    @property
    def _schema_version(self) -> int:
        return 2

    def _deserialize_entry(self, raw: dict) -> _TestEntry:
        known = {k for k in _TestEntry.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in known}
        return _TestEntry(**filtered)

    def _serialize_entry(self, entry: _TestEntry) -> dict:
        return asdict(entry)

    def _migrate(self, data: dict, from_version: int) -> dict:
        entries = data.get("entries", [])
        if from_version < 2:
            for e in entries:
                e.setdefault("version", 1)
                e.setdefault("supersedes", "")
                e.setdefault("archived", False)
        data["schema_version"] = 2
        return data

    def _estimate_entry_tokens(self, entry: _TestEntry) -> int:
        return len(entry.content) // 4

    def _entry_search_text(self, entry: _TestEntry) -> str:
        return entry.content

    def _get_entry_tags(self, entry: _TestEntry) -> list[str]:
        return entry.tags

    def _get_entry_id(self, entry: _TestEntry) -> str:
        return entry.id

    def _is_archived(self, entry: _TestEntry) -> bool:
        return entry.archived

    def _set_archived(self, entry: _TestEntry, archived: bool) -> None:
        entry.archived = archived

    def _get_version(self, entry: _TestEntry) -> int:
        return entry.version

    def _get_supersedes(self, entry: _TestEntry) -> str:
        return entry.supersedes


@pytest.fixture
def tmp_path_file():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "test_store.json"


@pytest.fixture
def store(tmp_path_file):
    return _TestStore(path=tmp_path_file)


# ---- Basic CRUD ----

class TestCRUD:
    def test_add_and_get_active(self, store):
        store._add_entry(_TestEntry(id="e1", content="hello"))
        active = store.get_active()
        assert len(active) == 1
        assert active[0].id == "e1"

    def test_add_no_save(self, store, tmp_path_file):
        store._add_entry(_TestEntry(id="e1", content="hello"), save=False)
        assert len(store.get_active()) == 1
        # Reload — should be empty since not saved
        store2 = _TestStore(path=tmp_path_file)
        assert len(store2.get_active()) == 0

    def test_remove_archives(self, store):
        store._add_entry(_TestEntry(id="e1", content="hello"))
        assert store._remove_entry("e1")
        assert len(store.get_active()) == 0
        assert len(store.get_archived()) == 1
        assert store.get_archived()[0].id == "e1"

    def test_remove_nonexistent(self, store):
        assert not store._remove_entry("nonexistent")

    def test_update_entry(self, store):
        store._add_entry(_TestEntry(id="e1", content="v1", version=1))
        new = _TestEntry(id="e2", content="v2", version=2, supersedes="e1")
        store._update_entry("e1", new)

        active = store.get_active()
        assert len(active) == 1
        assert active[0].id == "e2"
        assert active[0].version == 2

        archived = store.get_archived()
        assert len(archived) == 1
        assert archived[0].id == "e1"

    def test_get_by_id(self, store):
        store._add_entry(_TestEntry(id="e1", content="hello"))
        assert store.get_by_id("e1").content == "hello"
        assert store.get_by_id("nonexistent") is None

    def test_get_all_including_archived(self, store):
        store._add_entry(_TestEntry(id="e1", content="active"))
        store._add_entry(_TestEntry(id="e2", content="archived", archived=True))
        assert len(store.get_all_including_archived()) == 2


# ---- Persistence ----

class TestPersistence:
    def test_save_and_reload(self, tmp_path_file):
        store1 = _TestStore(path=tmp_path_file)
        store1._add_entry(_TestEntry(id="e1", content="hello"))
        store2 = _TestStore(path=tmp_path_file)
        assert len(store2.get_active()) == 1
        assert store2.get_active()[0].content == "hello"

    def test_corrupt_file_handled(self, tmp_path_file):
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path_file.write_text("not valid json")
        store = _TestStore(path=tmp_path_file)
        assert len(store.get_active()) == 0

    def test_missing_file_ok(self, tmp_path_file):
        store = _TestStore(path=tmp_path_file)
        assert len(store.get_active()) == 0

    def test_atomic_write(self, tmp_path_file):
        store = _TestStore(path=tmp_path_file)
        store._add_entry(_TestEntry(id="e1", content="test"))
        assert not tmp_path_file.with_suffix(".tmp").exists()
        assert tmp_path_file.exists()

    def test_schema_version_saved(self, tmp_path_file):
        store = _TestStore(path=tmp_path_file)
        store._add_entry(_TestEntry(id="e1", content="test"))
        data = json.loads(tmp_path_file.read_text())
        assert data["schema_version"] == 2


# ---- Schema migration ----

class TestMigration:
    def test_v1_to_v2(self, tmp_path_file):
        """Loading a v1 file adds version/supersedes/archived defaults."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        v1_data = {
            "schema_version": 1,
            "entries": [
                {"id": "e1", "content": "old entry", "created_at": "2026-01-01", "tags": ["test"]},
            ],
        }
        tmp_path_file.write_text(json.dumps(v1_data))
        store = _TestStore(path=tmp_path_file)
        assert len(store.get_active()) == 1
        e = store.get_active()[0]
        assert e.version == 1
        assert e.supersedes == ""
        assert e.archived is False

        # Re-saved with new schema
        data = json.loads(tmp_path_file.read_text())
        assert data["schema_version"] == 2


# ---- Version history ----

class TestVersionHistory:
    def test_version_chain(self, store):
        store._add_entry(_TestEntry(id="e1", content="v1", version=1))
        store._update_entry("e1", _TestEntry(id="e2", content="v2", version=2, supersedes="e1"))
        store._update_entry("e2", _TestEntry(id="e3", content="v3", version=3, supersedes="e2"))

        history = store.get_version_history("e3")
        assert len(history) == 3
        assert [e.id for e in history] == ["e3", "e2", "e1"]

    def test_version_history_single(self, store):
        store._add_entry(_TestEntry(id="e1", content="only version"))
        history = store.get_version_history("e1")
        assert len(history) == 1
        assert history[0].id == "e1"

    def test_version_history_nonexistent(self, store):
        assert store.get_version_history("nonexistent") == []


# ---- Token budget ----

class TestTokenBudget:
    def test_total_active_tokens(self, store):
        store._add_entry(_TestEntry(id="e1", content="A" * 40))
        store._add_entry(_TestEntry(id="e2", content="B" * 80))
        assert store.total_active_tokens() == (40 + 80) // 4

    def test_total_active_tokens_excludes_archived(self, store):
        store._add_entry(_TestEntry(id="e1", content="A" * 40))
        store._add_entry(_TestEntry(id="e2", content="B" * 80, archived=True))
        assert store.total_active_tokens() == 40 // 4

    def test_is_over_budget(self, store):
        store._add_entry(_TestEntry(id="e1", content="A" * 400))
        assert store.is_over_budget(10)
        assert not store.is_over_budget(1000)


# ---- Bulk operations ----

class TestBulkOps:
    def test_replace_active(self, store):
        store._add_entry(_TestEntry(id="e1", content="old"))
        store._add_entry(_TestEntry(id="e2", content="old2"))
        store.replace_active([_TestEntry(id="e3", content="new")])

        assert len(store.get_active()) == 1
        assert store.get_active()[0].id == "e3"
        assert len(store.get_archived()) == 2

    def test_purge_archived(self, store):
        store._add_entry(_TestEntry(id="e1", content="active"))
        store._add_entry(_TestEntry(id="e2", content="archived", archived=True))
        store._add_entry(_TestEntry(id="e3", content="archived2", archived=True))
        removed = store.purge_archived()
        assert removed == 2
        assert len(store.get_all_including_archived()) == 1
        assert store.get_all_including_archived()[0].id == "e1"

    def test_purge_archived_none(self, store):
        store._add_entry(_TestEntry(id="e1", content="active"))
        assert store.purge_archived() == 0


# ---- Search (tag-based) ----

class TestSearch:
    def test_search_by_tags(self, store):
        store.embeddings._available = False
        store._add_entry(_TestEntry(id="e1", content="ACE magnetic field", tags=["ace", "magnetic"]))
        store._add_entry(_TestEntry(id="e2", content="PSP solar wind", tags=["psp", "solar"]))
        results = store.search("ACE magnetic")
        assert len(results) >= 1
        assert results[0].id == "e1"

    def test_search_empty_query(self, store):
        store._add_entry(_TestEntry(id="e1", content="test"))
        assert store.search("") == []

    def test_search_no_match(self, store):
        store.embeddings._available = False
        store._add_entry(_TestEntry(id="e1", content="something", tags=["something"]))
        assert store.search("nonexistent xyz") == []

    def test_search_excludes_archived_by_default(self, store):
        store.embeddings._available = False
        store._add_entry(_TestEntry(id="e1", content="active ACE", tags=["ace"]))
        store._add_entry(_TestEntry(id="e2", content="archived ACE", tags=["ace"], archived=True))
        results = store.search("ACE")
        assert len(results) == 1
        assert results[0].id == "e1"

    def test_search_includes_archived(self, store):
        store.embeddings._available = False
        store._add_entry(_TestEntry(id="e1", content="active ACE", tags=["ace"]))
        store._add_entry(_TestEntry(id="e2", content="archived ACE", tags=["ace"], archived=True))
        results = store.search("ACE", include_archived=True)
        assert len(results) == 2

    def test_search_limit(self, store):
        store.embeddings._available = False
        for i in range(10):
            store._add_entry(_TestEntry(id=f"e{i}", content=f"test {i}", tags=["test"]))
        results = store.search("test", limit=3)
        assert len(results) == 3

    def test_search_substring_match(self, store):
        store.embeddings._available = False
        store._add_entry(_TestEntry(id="e1", content="ACE data has gaps in January", tags=[]))
        results = store.search("ACE data has gaps")
        assert len(results) == 1
