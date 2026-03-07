"""
Tests for agent.memory — MemoryStore, Memory dataclass, tag generation, search.

Run with: python -m pytest tests/test_memory.py -v
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent.memory import (
    Memory,
    MemoryStore,
    MemoryEmbeddings,
    generate_tags,
    estimate_tokens,
    estimate_memory_tokens,
    _git_commit_data,
    MEMORY_TOKEN_BUDGET,
    SCHEMA_VERSION,
)


@pytest.fixture
def tmp_path_file():
    """Provide a temporary file path for memory storage."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "memory.json"


@pytest.fixture
def store(tmp_path_file):
    """Provide a MemoryStore backed by a temp file."""
    return MemoryStore(path=tmp_path_file)


# ---- Basic CRUD ----


class TestMemoryCRUD:
    def test_add_and_get_all(self, store):
        m = Memory(type="preference", content="Prefers dark theme")
        store.add(m)
        assert len(store.get_all()) == 1
        assert store.get_all()[0].content == "Prefers dark theme"

    def test_add_auto_generates_tags(self, store):
        m = Memory(type="pitfall", content="PSP SPC fill values need filtering")
        store.add(m)
        assert store.get_all()[0].tags  # tags should be non-empty

    def test_add_no_save(self, store):
        """add_no_save adds to memory but doesn't persist to disk."""
        m = Memory(type="preference", content="Test no save")
        store.add_no_save(m)
        assert len(store.get_all()) == 1
        # Reload from disk — should be empty since we didn't save
        store2 = MemoryStore(path=store.path)
        assert len(store2.get_all()) == 0

    def test_add_no_save_then_save(self, store):
        """add_no_save followed by explicit save() persists."""
        store.add_no_save(Memory(type="preference", content="Batched"))
        store.save()
        store2 = MemoryStore(path=store.path)
        assert len(store2.get_all()) == 1
        assert store2.get_all()[0].content == "Batched"

    def test_remove(self, store):
        m = Memory(id="abc123", type="preference", content="Test")
        store.add(m)
        assert store.remove("abc123")
        assert len(store.get_all()) == 0
        # Archived entry should exist
        assert len(store.get_archived()) == 1

    def test_remove_nonexistent(self, store):
        assert not store.remove("nonexistent")

    def test_toggle(self, store):
        m = Memory(id="t1", type="preference", content="Test", enabled=True)
        store.add(m)
        store.toggle("t1", False)
        assert not store.get_all()[0].enabled

    def test_toggle_nonexistent(self, store):
        assert not store.toggle("nonexistent", True)

    def test_replace_all(self, store):
        store.add(Memory(id="old1", type="preference", content="Old A"))
        store.add(Memory(id="old2", type="summary", content="Old B"))
        new_memories = [
            Memory(id="new1", type="pitfall", content="New X"),
        ]
        store.replace_all(new_memories)
        assert len(store.get_all()) == 1
        assert store.get_all()[0].id == "new1"
        assert store.get_all()[0].content == "New X"

    def test_replace_all_persists(self, tmp_path_file):
        store1 = MemoryStore(path=tmp_path_file)
        store1.add(Memory(type="preference", content="Original"))
        store1.replace_all([Memory(id="r1", type="pitfall", content="Replaced")])
        store2 = MemoryStore(path=tmp_path_file)
        assert len(store2.get_all()) == 1
        assert store2.get_all()[0].id == "r1"

    def test_replace_all_empty(self, store):
        store.add(Memory(type="preference", content="A"))
        store.replace_all([])
        assert len(store.get_all()) == 0

    def test_clear_all(self, store):
        store.add(Memory(type="preference", content="A"))
        store.add(Memory(type="summary", content="B"))
        count = store.clear_all()
        assert count == 2
        assert len(store.get_all()) == 0

    def test_clear_empty(self, store):
        assert store.clear_all() == 0


# ---- Persistence ----


class TestPersistence:
    def test_save_and_reload(self, tmp_path_file):
        store1 = MemoryStore(path=tmp_path_file)
        store1.add(Memory(id="p1", type="preference", content="Dark theme"))
        store1.add(Memory(id="s1", type="summary", content="Analyzed ACE data"))

        # Create new store from same file
        store2 = MemoryStore(path=tmp_path_file)
        assert len(store2.get_all()) == 2
        assert store2.get_all()[0].id == "p1"
        assert store2.get_all()[1].id == "s1"

    def test_global_enabled_persists(self, tmp_path_file):
        store1 = MemoryStore(path=tmp_path_file)
        store1.toggle_global(False)

        store2 = MemoryStore(path=tmp_path_file)
        assert not store2.is_global_enabled()

    def test_corrupt_file_handled(self, tmp_path_file):
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path_file.write_text("not valid json")
        store = MemoryStore(path=tmp_path_file)
        assert len(store.get_all()) == 0

    def test_missing_file_ok(self, tmp_path_file):
        store = MemoryStore(path=tmp_path_file)
        assert len(store.get_all()) == 0
        assert store.is_global_enabled()

    def test_atomic_write(self, tmp_path_file):
        store = MemoryStore(path=tmp_path_file)
        store.add(Memory(type="preference", content="Test"))
        # Check that temp file doesn't linger
        assert not tmp_path_file.with_suffix(".tmp").exists()
        # Check that actual file exists
        assert tmp_path_file.exists()
        data = json.loads(tmp_path_file.read_text())
        assert len(data["memories"]) == 1

    def test_schema_version_saved(self, tmp_path_file):
        """save() writes schema_version to the JSON file."""
        store = MemoryStore(path=tmp_path_file)
        store.add(Memory(type="preference", content="Test"))
        data = json.loads(tmp_path_file.read_text())
        assert data["schema_version"] == SCHEMA_VERSION

    def test_new_fields_persisted(self, tmp_path_file):
        """New v2/v3/v4/v7 fields are persisted and reloaded correctly."""
        store1 = MemoryStore(path=tmp_path_file)
        m = Memory(
            type="reflection",
            scopes=["data_ops"],
            content="Test reflection",
            source="reflected",
            tags=["test", "reflection"],
            access_count=3,
            version=2,
            archived=False,
        )
        store1.add(m)

        store2 = MemoryStore(path=tmp_path_file)
        loaded = store2.get_all()[0]
        assert loaded.type == "reflection"
        assert loaded.scopes == ["data_ops"]
        assert loaded.source == "reflected"
        assert loaded.tags == ["test", "reflection"]
        assert loaded.access_count == 3
        assert loaded.version == 2
        assert loaded.archived is False


# ---- Schema migration ----


class TestSchemaMigration:
    def test_v1_to_v7_migration(self, tmp_path_file):
        """Loading a v1 file (no schema_version) backfills all defaults through v7."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        v1_data = {
            "global_enabled": True,
            "memories": [
                {
                    "id": "old1",
                    "type": "pitfall",
                    "scope": "envoy:PSP",
                    "content": "PSP SPC fill values need special handling",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "sess1",
                    "enabled": True,
                }
            ],
        }
        tmp_path_file.write_text(json.dumps(v1_data))

        store = MemoryStore(path=tmp_path_file)
        assert len(store.get_all()) == 1
        m = store.get_all()[0]
        # v2 fields
        assert m.source == "extracted"
        assert m.supersedes == ""
        assert m.access_count == 0
        assert m.last_accessed == ""
        # v3: scope → scopes
        assert m.scopes == ["envoy:PSP"]
        # v4 fields
        assert m.version == 1
        assert m.archived is False
        # v7: no confidence, review_of is empty
        assert not hasattr(m, "confidence")
        assert m.review_of == ""
        # Tags auto-generated
        assert len(m.tags) > 0
        assert "psp" in m.tags

        # File re-saved with current schema
        data = json.loads(tmp_path_file.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert "scopes" in data["memories"][0]
        assert "scope" not in data["memories"][0]
        assert "version" in data["memories"][0]
        assert "archived" in data["memories"][0]
        assert "confidence" not in data["memories"][0]
        assert "reviews" not in data["memories"][0]

    def test_v2_to_v7_migration(self, tmp_path_file):
        """Loading a v2 file migrates scope→scopes, adds v4 fields, removes confidence."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        v2_data = {
            "schema_version": 2,
            "global_enabled": True,
            "memories": [
                {
                    "id": "m1",
                    "type": "pitfall",
                    "scope": "envoy:PSP",
                    "content": "PSP SPC fill values need special handling",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "sess1",
                    "enabled": True,
                    "confidence": 0.9,
                    "source": "extracted",
                    "tags": ["psp", "spc", "fill"],
                    "supersedes": "",
                    "access_count": 2,
                    "last_accessed": "2026-01-15T00:00:00",
                }
            ],
        }
        tmp_path_file.write_text(json.dumps(v2_data))

        store = MemoryStore(path=tmp_path_file)
        assert len(store.get_all()) == 1
        m = store.get_all()[0]
        assert m.scopes == ["envoy:PSP"]
        assert m.access_count == 2
        assert m.version == 1
        assert m.archived is False
        assert m.review_of == ""

        data = json.loads(tmp_path_file.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert "confidence" not in data["memories"][0]

    def test_v3_to_v7_migration(self, tmp_path_file):
        """Loading a v3 file adds version/archived fields, removes confidence."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        v3_data = {
            "schema_version": 3,
            "global_enabled": True,
            "memories": [
                {
                    "id": "m1",
                    "type": "pitfall",
                    "scopes": ["envoy:PSP"],
                    "content": "PSP fill values",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "sess1",
                    "enabled": True,
                    "confidence": 0.9,
                    "source": "extracted",
                    "tags": ["psp", "fill"],
                    "supersedes": "",
                    "access_count": 1,
                    "last_accessed": "",
                }
            ],
        }
        tmp_path_file.write_text(json.dumps(v3_data))

        store = MemoryStore(path=tmp_path_file)
        m = store.get_all()[0]
        assert m.version == 1
        assert m.archived is False
        assert m.review_of == ""

        data = json.loads(tmp_path_file.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert "confidence" not in data["memories"][0]

    def test_v1_no_scope_field(self, tmp_path_file):
        """Old JSON without scope field loads with scopes=['generic']."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "global_enabled": True,
            "memories": [
                {
                    "id": "old1",
                    "type": "pitfall",
                    "content": "Old pitfall without scope",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "",
                    "enabled": True,
                }
            ],
        }
        tmp_path_file.write_text(json.dumps(data))
        store = MemoryStore(path=tmp_path_file)
        assert len(store.get_all()) == 1
        assert store.get_all()[0].scopes == ["generic"]


# ---- Global toggle ----


class TestGlobalToggle:
    def test_default_enabled(self, store):
        assert store.is_global_enabled()

    def test_toggle_global(self, store):
        store.toggle_global(False)
        assert not store.is_global_enabled()
        store.toggle_global(True)
        assert store.is_global_enabled()


# ---- Enabled filter ----


class TestEnabledFilter:
    def test_get_enabled(self, store):
        store.add(Memory(id="a", type="preference", content="A", enabled=True))
        store.add(Memory(id="b", type="preference", content="B", enabled=False))
        store.add(Memory(id="c", type="summary", content="C", enabled=True))
        enabled = store.get_enabled()
        assert len(enabled) == 2
        assert all(m.enabled for m in enabled)


# ---- Tag generation ----


class TestGenerateTags:
    def test_basic_tags(self):
        tags = generate_tags("PSP SPC fill values need special handling", "generic")
        assert "psp" in tags
        assert "spc" in tags
        assert "fill" in tags
        assert "values" in tags
        # Stop words should be excluded
        assert "need" not in tags

    def test_mission_scope_adds_tag(self):
        tags = generate_tags("Fill values are problematic", "envoy:PSP")
        assert tags[0] == "psp"  # mission tag should be first

    def test_visualization_scope_adds_tag(self):
        tags = generate_tags("Y-axis range setting", "visualization")
        assert "visualization" in tags

    def test_data_ops_scope(self):
        tags = generate_tags("Resample before concatenation", "data_ops")
        assert "data_ops" in tags

    def test_generic_scopes_no_extra_tag(self):
        tags = generate_tags("Some useful advice", ["generic"])
        # "generic" scope should NOT inject an extra scope tag
        assert all(t in ["useful", "advice"] for t in tags)

    def test_multi_scope_tags(self):
        """Multiple scopes should all add their tags."""
        tags = generate_tags("Check NaN before plotting", ["data_ops", "visualization"])
        assert "data_ops" in tags
        assert "visualization" in tags

    def test_multi_scope_with_mission(self):
        """Mission scope in multi-scope list adds mission tag."""
        tags = generate_tags("Handle fill values", ["envoy:PSP", "data_ops"])
        assert "psp" in tags
        assert "data_ops" in tags

    def test_string_scope_backward_compat(self):
        """Single string scope still works (backward compat)."""
        tags = generate_tags("Some advice", "data_ops")
        assert "data_ops" in tags

    def test_empty_content(self):
        tags = generate_tags("", "generic")
        assert tags == []

    def test_deduplication(self):
        tags = generate_tags("test test test duplicate duplicate", "generic")
        assert tags.count("test") == 1
        assert tags.count("duplicate") == 1

    def test_short_tokens_filtered(self):
        tags = generate_tags("a b cd ef", "generic")
        assert "a" not in tags
        assert "b" not in tags
        assert "cd" in tags
        assert "ef" in tags


# ---- Prompt building ----


class TestBuildPromptSection:
    def test_empty_returns_empty(self, store):
        assert store.build_prompt_section() == ""

    def test_disabled_returns_empty(self, store):
        store.add(Memory(type="preference", content="X"))
        store.toggle_global(False)
        assert store.build_prompt_section() == ""

    def test_preferences_only(self, store):
        store.add(Memory(type="preference", content="Prefers dark theme"))
        section = store.build_prompt_section()
        assert "### Preferences" in section
        assert "Prefers dark theme" in section
        assert "### Past Sessions" not in section

    def test_summaries_only(self, store):
        store.add(
            Memory(
                type="summary",
                content="Analyzed PSP data",
                created_at="2026-02-08T10:00:00",
            )
        )
        section = store.build_prompt_section()
        assert "### Past Sessions" in section
        assert "(2026-02-08)" in section
        assert "Analyzed PSP data" in section
        assert "### Preferences" not in section

    def test_mixed(self, store):
        store.add(Memory(type="preference", content="Prefers dark theme"))
        store.add(
            Memory(
                type="summary",
                content="Analyzed PSP data",
                created_at="2026-02-08T10:00:00",
            )
        )
        section = store.build_prompt_section()
        assert "### Preferences" in section
        assert "### Past Sessions" in section

    def test_disabled_memories_excluded(self, store):
        store.add(
            Memory(
                id="a",
                type="preference",
                content="Visible",
                enabled=True,
            )
        )
        store.add(
            Memory(
                id="b",
                type="preference",
                content="Hidden",
                enabled=False,
            )
        )
        section = store.build_prompt_section()
        assert "Visible" in section
        assert "Hidden" not in section

    def test_header_present(self, store):
        store.add(Memory(type="preference", content="Test"))
        section = store.build_prompt_section()
        assert "## Operational Knowledge" in section
        assert section.startswith("[CONTEXT FROM LONG-TERM MEMORY]")

    def test_scoped_preferences_excluded_from_orchestrator(self, store):
        """Only generic-scoped preferences appear in build_prompt_section."""
        store.add(Memory(type="preference", content="Generic pref", scopes=["generic"]))
        store.add(
            Memory(type="preference", content="PSP specific", scopes=["envoy:PSP"])
        )
        store.add(
            Memory(type="preference", content="Viz specific", scopes=["visualization"])
        )
        section = store.build_prompt_section()
        assert "Generic pref" in section
        assert "PSP specific" not in section
        assert "Viz specific" not in section

    def test_multi_scope_with_generic_included(self, store):
        """Memory with scopes=["generic", "data_ops"] appears in orchestrator prompt."""
        store.add(
            Memory(
                type="pitfall",
                scopes=["generic", "data_ops"],
                content="Always validate time ranges",
            )
        )
        section = store.build_prompt_section()
        assert "Always validate time ranges" in section

    def test_reflections_section(self, store):
        """Reflections with scopes=["generic"] appear in the prompt."""
        store.add(
            Memory(
                type="reflection",
                scopes=["generic"],
                content="Always check NaN before computing magnitude",
            )
        )
        section = store.build_prompt_section()
        assert "### Operational Reflections" in section
        assert "Always check NaN before computing magnitude" in section

    def test_scoped_reflections_excluded(self, store):
        """Non-generic reflections don't appear in orchestrator prompt."""
        store.add(
            Memory(
                type="reflection",
                scopes=["data_ops"],
                content="DataOps-specific reflection",
            )
        )
        section = store.build_prompt_section()
        assert section == ""  # No generic content

    def test_includes_memory_ids(self, store):
        """build_prompt_section includes [id] prefixes for review_memory."""
        store.add(Memory(id="pref01", type="preference", content="Dark theme"))
        store.add(Memory(id="pit01", type="pitfall", content="Check NaN"))
        section = store.build_prompt_section()
        assert "[pref01]" in section
        assert "[pit01]" in section

    def test_includes_review_instruction(self, store):
        """build_prompt_section includes review_memory instruction footer."""
        store.add(Memory(type="preference", content="Test"))
        section = store.build_prompt_section()
        assert "review_memory" in section
        assert "MUST call review_memory" in section


# ---- Pitfall prompt rendering ----


class TestPitfallPrompt:
    def test_pitfalls_only(self, store):
        store.add(
            Memory(type="pitfall", content="OMNI data may have empty CSV strings")
        )
        section = store.build_prompt_section()
        assert "### Lessons Learned" in section
        assert "OMNI data may have empty CSV strings" in section
        assert "### Preferences" not in section

    def test_pitfalls_with_preferences(self, store):
        store.add(Memory(type="preference", content="Prefers dark theme"))
        store.add(Memory(type="pitfall", content="MMS dataset IDs require @0 suffix"))
        section = store.build_prompt_section()
        assert "### Preferences" in section
        assert "### Lessons Learned" in section
        assert "MMS dataset IDs require @0 suffix" in section

    def test_disabled_pitfalls_excluded(self, store):
        store.add(
            Memory(
                id="p1",
                type="pitfall",
                content="Visible pitfall",
                enabled=True,
            )
        )
        store.add(
            Memory(
                id="p2",
                type="pitfall",
                content="Hidden pitfall",
                enabled=False,
            )
        )
        section = store.build_prompt_section()
        assert "Visible pitfall" in section
        assert "Hidden pitfall" not in section

    def test_all_four_types(self, store):
        store.add(Memory(type="preference", content="Prefers log scale"))
        store.add(
            Memory(
                type="summary",
                content="Analyzed ACE data",
                created_at="2026-02-09T10:00:00",
            )
        )
        store.add(Memory(type="pitfall", content="Rolling windows need DatetimeIndex"))
        store.add(
            Memory(
                type="reflection", scopes=["generic"], content="Check units after merge"
            )
        )
        section = store.build_prompt_section()
        assert "### Preferences" in section
        assert "### Past Sessions" in section
        assert "### Lessons Learned" in section
        assert "### Operational Reflections" in section


# ---- Archived memories (replaces cold storage tests) ----

# ---- Tag-based search ----


class TestTagBasedSearch:
    def test_search_by_tags(self, store):
        """Tag-based search scores tag matches higher than substring."""
        store.add(
            Memory(
                type="pitfall",
                scopes=["envoy:ACE"],
                content="ACE data has gaps in January",
            )
        )
        store.add(
            Memory(
                type="summary",
                content="Analyzed magnetic field data",
            )
        )
        results = store.search("ACE gaps")
        assert len(results) >= 1
        # ACE pitfall should score higher (tag matches)
        assert results[0].content == "ACE data has gaps in January"

    def test_search_with_type_filter(self, store):
        store.add(Memory(type="pitfall", content="PSP fill values"))
        store.add(Memory(type="preference", content="Prefers PSP data"))
        results = store.search("PSP", mem_type="pitfall")
        assert len(results) == 1
        assert results[0].type == "pitfall"

    def test_search_with_scope_filter(self, store):
        store.add(
            Memory(type="pitfall", scopes=["envoy:PSP"], content="PSP fill values")
        )
        store.add(
            Memory(type="pitfall", scopes=["generic"], content="General PSP advice")
        )
        results = store.search("PSP", scope="envoy:PSP")
        assert len(results) == 1
        assert "envoy:PSP" in results[0].scopes

    def test_search_multi_scope_matches_any(self, store):
        """A memory with scopes=["data_ops", "visualization"] matches search for either scope."""
        store.add(
            Memory(
                type="pitfall",
                scopes=["data_ops", "visualization"],
                content="Check NaN before plotting",
            )
        )
        results_do = store.search("NaN", scope="data_ops")
        assert len(results_do) == 1
        results_viz = store.search("NaN", scope="visualization")
        assert len(results_viz) == 1

    def test_search_includes_archived_memories(self, store):
        """Search includes both active and archived memories."""
        store.add(Memory(type="pitfall", content="Active ACE pitfall"))
        m = Memory(
            id="arch1",
            type="pitfall",
            content="Archived ACE pitfall",
            tags=["ace", "pitfall", "archived"],
            archived=True,
        )
        store._entries.append(m)
        store.save()
        results = store.search("ACE pitfall")
        assert len(results) == 2

    def test_search_no_results(self, store):
        store.add(Memory(type="preference", content="Dark theme"))
        # Force tag-based fallback (embedding models have high baseline similarity)
        store.embeddings._available = False
        results = store.search("nonexistent query xyz")
        assert len(results) == 0

    def test_search_limit(self, store):
        for i in range(10):
            store.add(
                Memory(type="pitfall", content=f"Pitfall {i} about magnetic data")
            )
        results = store.search("magnetic", limit=3)
        assert len(results) == 3

    def test_search_empty_query(self, store):
        """Empty query matches nothing (no tags to intersect)."""
        store.add(Memory(type="preference", content="Some content"))
        results = store.search("")
        assert len(results) == 0


# ---- Memory dataclass defaults ----


class TestMemoryDefaults:
    def test_default_id(self):
        m = Memory()
        assert m.id  # non-empty
        assert len(m.id) == 12

    def test_default_type(self):
        m = Memory()
        assert m.type == "preference"

    def test_default_enabled(self):
        m = Memory()
        assert m.enabled is True

    def test_default_scopes(self):
        m = Memory()
        assert m.scopes == ["generic"]

    def test_created_at_set(self):
        m = Memory()
        assert m.created_at  # non-empty ISO string

    def test_new_field_defaults(self):
        m = Memory()
        assert m.source == "extracted"
        assert m.tags == []
        assert m.supersedes == ""
        assert m.access_count == 0
        assert m.last_accessed == ""
        assert m.review_of == ""

    def test_v4_field_defaults(self):
        m = Memory()
        assert m.version == 1
        assert m.archived is False

    def test_reflection_type(self):
        m = Memory(type="reflection", scopes=["data_ops"], content="Test")
        assert m.type == "reflection"
        assert m.scopes == ["data_ops"]

    def test_multi_scope(self):
        m = Memory(type="pitfall", scopes=["data_ops", "visualization"], content="Test")
        assert m.scopes == ["data_ops", "visualization"]


# ---- Scoped pitfalls ----


class TestScopedPitfalls:
    def test_scopes_default_generic(self, store):
        """Pitfalls without explicit scopes default to ['generic']."""
        m = Memory(type="pitfall", content="Some lesson")
        store.add(m)
        assert store.get_all()[0].scopes == ["generic"]

    def test_get_pitfalls_by_scope(self, store):
        """get_pitfalls_by_scope returns only matching enabled pitfalls."""
        store.add(
            Memory(type="pitfall", content="PSP fill values", scopes=["envoy:PSP"])
        )
        store.add(Memory(type="pitfall", content="ACE data gaps", scopes=["envoy:ACE"]))
        store.add(Memory(type="pitfall", content="Generic lesson", scopes=["generic"]))
        store.add(
            Memory(type="pitfall", content="Plotly y_range", scopes=["visualization"])
        )
        store.add(
            Memory(
                id="dis",
                type="pitfall",
                content="Disabled PSP",
                scopes=["envoy:PSP"],
                enabled=False,
            )
        )

        psp = store.get_pitfalls_by_scope("envoy:PSP")
        assert len(psp) == 1
        assert psp[0].content == "PSP fill values"

        ace = store.get_pitfalls_by_scope("envoy:ACE")
        assert len(ace) == 1
        assert ace[0].content == "ACE data gaps"

        viz = store.get_pitfalls_by_scope("visualization")
        assert len(viz) == 1
        assert viz[0].content == "Plotly y_range"

        generic = store.get_pitfalls_by_scope("generic")
        assert len(generic) == 1

    def test_multi_scope_pitfall_matches_both(self, store):
        """A pitfall with multiple scopes matches queries for either scope."""
        store.add(
            Memory(
                type="pitfall",
                scopes=["data_ops", "visualization"],
                content="Always check for NaN before plotting",
            )
        )
        do_pitfalls = store.get_pitfalls_by_scope("data_ops")
        assert len(do_pitfalls) == 1
        viz_pitfalls = store.get_pitfalls_by_scope("visualization")
        assert len(viz_pitfalls) == 1
        generic_pitfalls = store.get_pitfalls_by_scope("generic")
        assert len(generic_pitfalls) == 0

    def test_build_prompt_only_generic(self, store):
        """build_prompt_section only includes generic-scoped pitfalls."""
        store.add(
            Memory(
                type="pitfall", content="Generic operational advice", scopes=["generic"]
            )
        )
        store.add(
            Memory(type="pitfall", content="PSP SPC fill values", scopes=["envoy:PSP"])
        )
        store.add(
            Memory(
                type="pitfall", content="Plotly rendering tip", scopes=["visualization"]
            )
        )

        section = store.build_prompt_section()
        assert "Generic operational advice" in section
        assert "PSP SPC fill values" not in section
        assert "Plotly rendering tip" not in section

    def test_get_scoped_pitfall_texts(self, store):
        """get_scoped_pitfall_texts returns content strings for matching scope."""
        store.add(Memory(type="pitfall", content="PSP lesson 1", scopes=["envoy:PSP"]))
        store.add(Memory(type="pitfall", content="PSP lesson 2", scopes=["envoy:PSP"]))
        store.add(Memory(type="pitfall", content="ACE lesson", scopes=["envoy:ACE"]))

        texts = store.get_scoped_pitfall_texts("envoy:PSP")
        assert texts == ["PSP lesson 1", "PSP lesson 2"]

        texts = store.get_scoped_pitfall_texts("envoy:ACE")
        assert texts == ["ACE lesson"]

        texts = store.get_scoped_pitfall_texts("mission:WIND")
        assert texts == []

    def test_backward_compat_no_scope(self, tmp_path_file):
        """Old JSON without scope field loads with scopes=['generic']."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "global_enabled": True,
            "memories": [
                {
                    "id": "old1",
                    "type": "pitfall",
                    "content": "Old pitfall without scope",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "",
                    "enabled": True,
                }
            ],
        }
        tmp_path_file.write_text(json.dumps(data))
        store = MemoryStore(path=tmp_path_file)
        assert len(store.get_all()) == 1
        assert store.get_all()[0].scopes == ["generic"]
        assert store.get_all()[0].content == "Old pitfall without scope"


# ---- format_for_injection ----


class TestFormatForInjection:
    def test_empty_store_returns_empty(self, store):
        assert store.format_for_injection() == ""

    def test_disabled_store_returns_empty(self, store):
        store.add(Memory(type="preference", content="X"))
        store.toggle_global(False)
        assert store.format_for_injection() == ""

    def test_wrapper_markers(self, store):
        """Output starts with [CONTEXT FROM LONG-TERM MEMORY] and ends with [END MEMORY CONTEXT]."""
        store.add(Memory(type="preference", content="Test"))
        section = store.format_for_injection()
        assert section.startswith("[CONTEXT FROM LONG-TERM MEMORY]")
        assert section.endswith("[END MEMORY CONTEXT]")

    def test_operational_knowledge_header(self, store):
        store.add(Memory(type="preference", content="Test"))
        section = store.format_for_injection()
        assert "## Operational Knowledge" in section

    def test_generic_scope_filtering(self, store):
        """Generic scope only returns generic-scoped memories."""
        store.add(Memory(type="preference", content="Generic pref", scopes=["generic"]))
        store.add(
            Memory(type="preference", content="PSP specific", scopes=["envoy:PSP"])
        )
        store.add(
            Memory(type="preference", content="Viz specific", scopes=["visualization"])
        )
        section = store.format_for_injection(scope="generic")
        assert "Generic pref" in section
        assert "PSP specific" not in section
        assert "Viz specific" not in section

    def test_scoped_filtering(self, store):
        """Scoped call returns only memories matching that scope."""
        store.add(Memory(type="pitfall", scopes=["envoy:PSP"], content="PSP pitfall"))
        store.add(
            Memory(type="pitfall", scopes=["data_ops"], content="DataOps pitfall")
        )
        store.add(Memory(type="preference", scopes=["envoy:PSP"], content="PSP pref"))
        store.add(
            Memory(type="reflection", scopes=["envoy:PSP"], content="PSP reflection")
        )
        section = store.format_for_injection(scope="envoy:PSP")
        assert "PSP pitfall" in section
        assert "PSP pref" in section
        assert "PSP reflection" in section
        assert "DataOps pitfall" not in section

    def test_multi_scope_memory_appears_in_both(self, store):
        """Memory with multiple scopes appears when either scope is queried."""
        store.add(
            Memory(
                type="pitfall",
                scopes=["data_ops", "visualization"],
                content="Always check for NaN before plotting",
            )
        )
        do_section = store.format_for_injection(scope="data_ops")
        assert "Always check for NaN before plotting" in do_section
        viz_section = store.format_for_injection(scope="visualization")
        assert "Always check for NaN before plotting" in viz_section

    def test_summaries_excluded_by_default(self, store):
        """Summaries are not included when include_summaries=False (default)."""
        store.add(Memory(type="summary", content="Session summary"))
        store.add(Memory(type="pitfall", content="A pitfall"))
        section = store.format_for_injection()
        assert "### Past Sessions" not in section
        assert "Session summary" not in section
        assert "A pitfall" in section

    def test_summaries_included_when_requested(self, store):
        """Summaries appear when include_summaries=True."""
        store.add(
            Memory(
                type="summary",
                content="Analyzed PSP data",
                created_at="2026-02-08T10:00:00",
            )
        )
        section = store.format_for_injection(include_summaries=True)
        assert "### Past Sessions" in section
        assert "(2026-02-08)" in section
        assert "Analyzed PSP data" in section

    def test_review_instruction_present(self, store):
        store.add(Memory(type="preference", content="Test"))
        section = store.format_for_injection(include_review_instruction=True)
        assert "review_memory" in section
        assert "MUST call review_memory" in section

    def test_review_instruction_omitted(self, store):
        store.add(Memory(type="preference", content="Test"))
        section = store.format_for_injection(include_review_instruction=False)
        assert "review_memory" not in section
        assert "MUST call review_memory" not in section

    def test_memory_ids_visible(self, store):
        store.add(Memory(id="pref01", type="preference", content="Dark theme"))
        store.add(Memory(id="pit01", type="pitfall", content="Check NaN"))
        section = store.format_for_injection()
        assert "[pref01]" in section
        assert "[pit01]" in section

    def test_last_injected_ids_tracking_generic(self, store):
        """Generic scope tracks IDs as OrchestratorAgent."""
        store.add(Memory(id="g1", type="preference", content="Test"))
        store._last_injected_ids.clear()
        store.format_for_injection(scope="generic")
        assert store._last_injected_ids.get("g1") == "OrchestratorAgent"

    def test_last_injected_ids_tracking_scoped(self, store):
        """Scoped calls track IDs with correct agent names."""
        store.add(
            Memory(
                id="viz1", type="pitfall", scopes=["visualization"], content="Viz tip"
            )
        )
        store.add(
            Memory(id="do1", type="pitfall", scopes=["data_ops"], content="DO tip")
        )
        store.add(
            Memory(id="psp1", type="pitfall", scopes=["envoy:PSP"], content="PSP tip")
        )

        store._last_injected_ids.clear()
        store.format_for_injection(scope="visualization")
        assert store._last_injected_ids.get("viz1") == "VizAgent"

        store._last_injected_ids.clear()
        store.format_for_injection(scope="data_ops")
        assert store._last_injected_ids.get("do1") == "DataOpsAgent"

        store._last_injected_ids.clear()
        store.format_for_injection(scope="envoy:PSP")
        assert store._last_injected_ids.get("psp1") == "EnvoyAgent[PSP]"

    def test_access_count_updated(self, store):
        store.add(Memory(type="pitfall", scopes=["visualization"], content="Test"))
        store.format_for_injection(scope="visualization")
        m = store.get_all()[0]
        assert m.access_count == 1
        assert m.last_accessed != ""

    def test_token_budget_respected(self, store, monkeypatch):
        """Token budget limits returned items."""
        monkeypatch.setattr("agent.memory.MEMORY_TOKEN_BUDGET", 10000)
        long_content = "Y" * 200  # ~50 tokens each
        for i in range(800):
            store.add_no_save(
                Memory(
                    id=f"b{i}",
                    type="pitfall",
                    scopes=["data_ops"],
                    content=f"{long_content} pitfall_{i}",
                )
            )
        section = store.format_for_injection(scope="data_ops")
        pref_lines = [l for l in section.split("\n") if l.startswith("- ")]
        assert pref_lines  # at least some
        assert len(pref_lines) < 800  # not all

    def test_consistent_structure_orchestrator_vs_subagent(self, store):
        """Orchestrator and sub-agent paths produce same structural elements."""
        store.add(Memory(type="preference", content="Generic pref", scopes=["generic"]))
        store.add(Memory(type="pitfall", content="Generic pitfall", scopes=["generic"]))

        orchestrator = store.format_for_injection(
            scope="generic", include_summaries=True
        )
        sub_agent = store.format_for_injection(scope="generic")

        # Both should have same structural markers
        for marker in [
            "[CONTEXT FROM LONG-TERM MEMORY]",
            "## Operational Knowledge",
            "### Preferences",
            "### Lessons Learned",
            "[END MEMORY CONTEXT]",
        ]:
            assert marker in orchestrator, f"Missing {marker} in orchestrator"
            assert marker in sub_agent, f"Missing {marker} in sub-agent"

    def test_empty_scope_returns_empty(self, store):
        """No memories for a given scope returns empty string."""
        store.add(Memory(type="pitfall", scopes=["data_ops"], content="Test"))
        assert store.format_for_injection(scope="visualization") == ""

    def test_excludes_disabled_memories(self, store):
        """Disabled memories are not included."""
        store.add(Memory(type="pitfall", scopes=["data_ops"], content="Active"))
        store.add(
            Memory(
                type="pitfall",
                scopes=["data_ops"],
                content="Disabled",
                enabled=False,
            )
        )
        section = store.format_for_injection(scope="data_ops")
        assert "Active" in section
        assert "Disabled" not in section

    def test_returns_all_types_for_scope(self, store):
        """format_for_injection returns pitfalls + reflections + preferences (not summaries by default)."""
        store.add(Memory(type="pitfall", scopes=["envoy:PSP"], content="PSP pitfall"))
        store.add(
            Memory(type="reflection", scopes=["envoy:PSP"], content="PSP reflection")
        )
        store.add(
            Memory(type="preference", scopes=["envoy:PSP"], content="PSP preference")
        )
        store.add(Memory(type="summary", scopes=["envoy:PSP"], content="PSP summary"))
        section = store.format_for_injection(scope="envoy:PSP")
        assert "PSP pitfall" in section
        assert "PSP reflection" in section
        assert "PSP preference" in section
        # summaries are not included by default
        assert "PSP summary" not in section

    def test_lessons_learned_header(self, store):
        """Pitfalls use ### Lessons Learned header (not ## Operational Knowledge subsection)."""
        store.add(Memory(type="pitfall", content="Check NaN"))
        section = store.format_for_injection()
        assert "### Lessons Learned" in section


class TestAgentNameForScope:
    def test_visualization(self):
        assert MemoryStore._agent_name_for_scope("visualization") == "VizAgent"

    def test_data_ops(self):
        assert MemoryStore._agent_name_for_scope("data_ops") == "DataOpsAgent"

    def test_mission(self):
        assert MemoryStore._agent_name_for_scope("envoy:PSP") == "EnvoyAgent[PSP]"

    def test_generic(self):
        assert MemoryStore._agent_name_for_scope("generic") == "OrchestratorAgent"

    def test_unknown_scope(self):
        assert (
            MemoryStore._agent_name_for_scope("something_else") == "OrchestratorAgent"
        )


# ---- Token estimation ----


class TestTokenEstimation:
    def test_estimate_tokens_basic(self):
        result = estimate_tokens("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_long(self):
        text = "a" * 400
        result = estimate_tokens(text)
        assert result > 0
        # Real tokenizer should give a reasonable count (not wildly off)
        assert 10 < result < 800

    def test_estimate_memory_tokens(self):
        m = Memory(type="preference", content="Dark theme preferred")
        tokens = estimate_memory_tokens(m)
        assert tokens == estimate_tokens("- Dark theme preferred")

    def test_total_tokens(self, store):
        store.add(Memory(type="preference", content="A" * 40))
        store.add(Memory(type="pitfall", content="B" * 80))
        total = store.total_tokens()
        expected = estimate_memory_tokens(
            Memory(content="A" * 40)
        ) + estimate_memory_tokens(Memory(content="B" * 80))
        assert total == expected

    def test_total_tokens_excludes_disabled(self, store):
        store.add(Memory(type="preference", content="Enabled " * 10))
        store.add(Memory(type="preference", content="Disabled " * 10, enabled=False))
        total = store.total_tokens()
        expected = estimate_memory_tokens(Memory(content="Enabled " * 10))
        assert total == expected

    def test_total_tokens_empty(self, store):
        assert store.total_tokens() == 0

    def test_total_tokens_excludes_archived(self, store):
        store.add(Memory(type="preference", content="Active " * 10))
        m = Memory(type="preference", content="Archived " * 10, archived=True)
        store._entries.append(m)
        total = store.total_tokens()
        expected = estimate_memory_tokens(Memory(content="Active " * 10))
        assert total == expected


# ---- Token-budgeted prompt building ----


class TestTokenBudgetedPrompt:
    def test_build_prompt_section_token_budget(self, store, monkeypatch):
        """build_prompt_section stops adding items when token budget is exceeded."""
        monkeypatch.setattr("agent.memory.MEMORY_TOKEN_BUDGET", 10000)
        # Add many long preferences that exceed the patched 10k budget
        # Use diverse words (~125 tokens each); 200 items × 125 = 25k >> 10k budget
        long_content = "The quick brown fox jumps over the lazy dog near a stream " * 10
        for i in range(200):
            store.add_no_save(
                Memory(type="preference", content=f"{long_content} pref_{i}")
            )
        section = store.build_prompt_section()
        # Should not contain all 200 preferences
        pref_lines = [l for l in section.split("\n") if l.startswith("- ")]
        assert pref_lines  # at least some
        assert len(pref_lines) < 200  # not all

    def test_format_for_injection_token_budget(self, store, monkeypatch):
        """Sub-agent token budget limits returned items."""
        monkeypatch.setattr("agent.memory.MEMORY_TOKEN_BUDGET", 10000)
        # Use diverse words (~125 tokens each); 800 items × 125 = 100k >> 10k budget
        long_content = "The quick brown fox jumps over the lazy dog near a stream " * 10
        for i in range(800):
            store.add_no_save(
                Memory(
                    type="pitfall",
                    scopes=["data_ops"],
                    content=f"{long_content} pitfall_{i}",
                )
            )
        section = store.format_for_injection(scope="data_ops")
        item_lines = [l for l in section.split("\n") if l.startswith("- ")]
        assert item_lines  # at least some
        assert len(item_lines) < 800  # not all


# ---- Memory Embeddings ----


class TestMemoryEmbeddings:
    def test_build_and_query(self):
        """MemoryEmbeddings can build index and compute similarity."""
        emb = MemoryEmbeddings()
        if not emb.available:
            pytest.skip("fastembed not available")
        memories = [
            Memory(content="ACE magnetic field data has gaps in January"),
            Memory(content="PSP solar wind speed measurements"),
            Memory(content="Dark theme preferred for plots"),
        ]
        emb.build(memories)
        assert emb._embeddings is not None
        assert emb._embeddings.shape[0] == 3

    def test_pairwise_max_similarity_semantic(self):
        """Semantically similar text should have high similarity."""
        emb = MemoryEmbeddings()
        if not emb.available:
            pytest.skip("fastembed not available")
        sim = emb.pairwise_max_similarity(
            "prefers RTN coordinates",
            ["likes radial-tangential-normal frame", "uses dark theme"],
        )
        assert sim > 0.6  # semantically similar

    def test_pairwise_max_similarity_empty(self):
        """Empty inputs return 0.0."""
        emb = MemoryEmbeddings()
        assert emb.pairwise_max_similarity("test", []) == 0.0
        assert emb.pairwise_max_similarity("", ["test"]) == 0.0

    def test_invalidate(self):
        """invalidate() clears cached embeddings."""
        emb = MemoryEmbeddings()
        if not emb.available:
            pytest.skip("fastembed not available")
        emb.build([Memory(content="test")])
        assert emb._embeddings is not None
        emb.invalidate()
        assert emb._embeddings is None
        assert emb._contents == []

    def test_embedding_search(self, store):
        """Embedding-based search returns semantically relevant results."""
        if not store.embeddings.available:
            pytest.skip("fastembed not available")
        store.add(
            Memory(
                type="pitfall", content="ACE magnetic field data has gaps in January"
            )
        )
        store.add(Memory(type="pitfall", content="PSP solar wind proton measurements"))
        store.add(Memory(type="preference", content="Prefers dark theme for all plots"))
        results = store.search("ACE magnetometer data issues")
        assert len(results) >= 1
        # ACE pitfall should be the top result
        assert "ACE" in results[0].content

    def test_embedding_search_fallback(self, store):
        """Falls back to tag-based search when fastembed is unavailable."""
        store.embeddings._available = False
        store.add(Memory(type="pitfall", content="ACE data has gaps in January"))
        store.add(Memory(type="pitfall", content="PSP fill values need filtering"))
        results = store.search("ACE gaps")
        assert len(results) >= 1
        assert "ACE" in results[0].content


# ---- Review as memory ----


class TestReviewAsMemory:
    def test_v7_field_defaults(self):
        """Memory has review_of='' by default."""
        m = Memory()
        assert m.review_of == ""

    def test_schema_version_is_7(self):
        assert SCHEMA_VERSION == 7

    def test_review_memory_creation(self, store):
        """A review is a Memory with type='review' and review_of linking to target."""
        target = Memory(id="tgt1", type="pitfall", content="Check NaN")
        store.add(target)
        review = Memory(
            type="review",
            review_of="tgt1",
            scopes=["generic"],
            content="4★ Caught the NaN issue",
            tags=["review:tgt1", "TestAgent", "stars:4"],
        )
        store.add(review)
        assert len(store.get_all()) == 2
        r = store.get_review_for("tgt1")
        assert r is not None
        assert r.review_of == "tgt1"
        assert "4★" in r.content

    def test_review_supersession(self, store):
        """Updating a review supersedes the old one (version control)."""
        target = Memory(id="tgt1", type="pitfall", content="Check NaN")
        store.add(target)
        r1 = Memory(
            id="rev1",
            type="review",
            review_of="tgt1",
            scopes=["generic"],
            content="3★ OK",
            tags=["review:tgt1", "Agent", "stars:3"],
        )
        store.add(r1)
        # Supersede with new review
        r1_entry = store.get_by_id("rev1")
        r1_entry.archived = True
        r2 = Memory(
            id="rev2",
            type="review",
            review_of="tgt1",
            scopes=["generic"],
            content="5★ Great",
            tags=["review:tgt1", "Agent", "stars:5"],
            supersedes="rev1",
            version=2,
        )
        store.add(r2)
        # Current review should be the new one
        current = store.get_review_for("tgt1")
        assert current is not None
        assert current.id == "rev2"
        assert current.version == 2
        assert "5★" in current.content
        # Old review is archived
        old = store.get_by_id("rev1")
        assert old.archived is True

    def test_review_version_history(self, store):
        """Review version chain is retrievable via get_version_history."""
        target = Memory(id="tgt1", type="pitfall", content="Test")
        store.add(target)
        r1 = Memory(id="rev1", type="review", review_of="tgt1", content="3★ OK")
        store.add(r1)
        r1.archived = True
        r2 = Memory(
            id="rev2",
            type="review",
            review_of="tgt1",
            content="5★ Great",
            supersedes="rev1",
            version=2,
        )
        store.add(r2)
        history = store.get_version_history("rev2")
        assert len(history) == 2
        assert history[0].id == "rev2"
        assert history[1].id == "rev1"

    def test_get_reviews_returns_all_active(self, store):
        """get_reviews() returns all active review memories."""
        store.add(Memory(id="tgt1", type="pitfall", content="A"))
        store.add(Memory(id="tgt2", type="pitfall", content="B"))
        store.add(Memory(type="review", review_of="tgt1", content="4★ Good"))
        store.add(Memory(type="review", review_of="tgt2", content="2★ Bad"))
        reviews = store.get_reviews()
        assert len(reviews) == 2

    def test_get_review_for_nonexistent(self, store):
        assert store.get_review_for("nonexistent") is None

    def test_reviews_excluded_from_injection(self, store):
        """Review-type memories are not injected as operational knowledge."""
        store.add(Memory(id="tgt1", type="pitfall", content="Check NaN"))
        store.add(Memory(type="review", review_of="tgt1", content="5★ Great"))
        section = store.format_for_injection()
        assert "Check NaN" in section
        assert "5★ Great" not in section

    def test_review_serialization_roundtrip(self, tmp_path_file):
        """Review memory is persisted and reloaded correctly."""
        store1 = MemoryStore(path=tmp_path_file)
        store1.add(Memory(id="tgt1", type="pitfall", content="Test"))
        store1.add(
            Memory(
                id="rev1",
                type="review",
                review_of="tgt1",
                scopes=["generic"],
                content="4★ Helpful",
                tags=["review:tgt1", "Agent", "stars:4"],
            )
        )

        store2 = MemoryStore(path=tmp_path_file)
        review = store2.get_review_for("tgt1")
        assert review is not None
        assert review.id == "rev1"
        assert review.review_of == "tgt1"
        assert review.content == "4★ Helpful"

    def test_v6_to_v7_migration(self, tmp_path_file):
        """Loading a v6 file converts inline review dict to standalone review Memory."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        v6_data = {
            "schema_version": 6,
            "global_enabled": True,
            "memories": [
                {
                    "id": "m1",
                    "type": "pitfall",
                    "scopes": ["generic"],
                    "content": "Test pitfall",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "sess1",
                    "enabled": True,
                    "source": "extracted",
                    "tags": ["test"],
                    "supersedes": "",
                    "access_count": 0,
                    "last_accessed": "",
                    "version": 1,
                    "archived": False,
                    "review": {
                        "stars": 4,
                        "comment": "Helpful",
                        "agent": "VizAgent",
                        "model": "gemini-2.5-flash",
                        "session_id": "s1",
                        "created_at": "2026-02-20T14:00:00",
                    },
                },
                {
                    "id": "m2",
                    "type": "preference",
                    "scopes": ["generic"],
                    "content": "No review",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "sess1",
                    "enabled": True,
                    "source": "extracted",
                    "tags": [],
                    "supersedes": "",
                    "access_count": 0,
                    "last_accessed": "",
                    "version": 1,
                    "archived": False,
                    "review": None,
                },
            ],
        }
        tmp_path_file.write_text(json.dumps(v6_data))

        store = MemoryStore(path=tmp_path_file)
        all_mems = store.get_all()

        # m1 should have review_of="" (it's the target, not the review)
        m1 = next(m for m in all_mems if m.id == "m1")
        assert m1.review_of == ""

        # m2 should have review_of=""
        m2 = next(m for m in all_mems if m.id == "m2")
        assert m2.review_of == ""

        # A standalone review memory should have been created for m1
        review = store.get_review_for("m1")
        assert review is not None
        assert review.type == "review"
        assert review.review_of == "m1"
        assert "4★" in review.content
        assert "Helpful" in review.content
        assert review.scopes == ["generic"]  # inherited from target

        # No review for m2
        assert store.get_review_for("m2") is None

        # File re-saved with current schema
        data = json.loads(tmp_path_file.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        for mem_data in data["memories"]:
            assert "review" not in mem_data
            assert "review_of" in mem_data

    def test_v5_to_v7_migration(self, tmp_path_file):
        """Loading a v5 file migrates through v6 then v7: last review becomes standalone."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        v5_data = {
            "schema_version": 5,
            "global_enabled": True,
            "memories": [
                {
                    "id": "m1",
                    "type": "pitfall",
                    "scopes": ["generic"],
                    "content": "Test pitfall",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "sess1",
                    "enabled": True,
                    "confidence": 0.85,
                    "source": "extracted",
                    "tags": ["test"],
                    "supersedes": "",
                    "access_count": 0,
                    "last_accessed": "",
                    "version": 1,
                    "archived": False,
                    "reviews": [
                        {
                            "stars": 3,
                            "comment": "OK",
                            "agent": "A",
                            "model": "M",
                            "session_id": "s1",
                            "created_at": "2026-01-01",
                        },
                        {
                            "stars": 5,
                            "comment": "Great",
                            "agent": "B",
                            "model": "M2",
                            "session_id": "s2",
                            "created_at": "2026-01-15",
                        },
                    ],
                },
            ],
        }
        tmp_path_file.write_text(json.dumps(v5_data))

        store = MemoryStore(path=tmp_path_file)
        # m1 should exist without inline review
        m1 = next(m for m in store.get_all() if m.id == "m1")
        assert m1.review_of == ""

        # v5→v6 keeps last review (5★), then v6→v7 converts to standalone
        review = store.get_review_for("m1")
        assert review is not None
        assert "5★" in review.content
        assert "Great" in review.content

    def test_v4_to_v7_migration(self, tmp_path_file):
        """Loading a v4 file migrates through all versions; no review for memory without one."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        v4_data = {
            "schema_version": 4,
            "global_enabled": True,
            "memories": [
                {
                    "id": "m1",
                    "type": "pitfall",
                    "scopes": ["generic"],
                    "content": "Test pitfall",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "sess1",
                    "enabled": True,
                    "confidence": 0.8,
                    "source": "extracted",
                    "tags": ["test"],
                    "supersedes": "",
                    "access_count": 0,
                    "last_accessed": "",
                    "version": 1,
                    "archived": False,
                }
            ],
        }
        tmp_path_file.write_text(json.dumps(v4_data))

        store = MemoryStore(path=tmp_path_file)
        m = store.get_all()[0]
        assert m.review_of == ""
        assert store.get_review_for("m1") is None

        # File re-saved with current schema
        data = json.loads(tmp_path_file.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert "confidence" not in data["memories"][0]
        assert "reviews" not in data["memories"][0]
        assert "review" not in data["memories"][0]
        assert "review_of" in data["memories"][0]


# ---- Injection tracking ----


class TestInjectionTracking:
    def test_build_prompt_section_tracks_ids(self, store):
        """build_prompt_section() records injected memory IDs with OrchestratorAgent."""
        store.add(Memory(id="inj1", type="preference", content="Generic pref"))
        store.add(Memory(id="inj2", type="pitfall", content="Generic pitfall"))
        store._last_injected_ids.clear()
        store.build_prompt_section()
        assert "inj1" in store._last_injected_ids
        assert "inj2" in store._last_injected_ids
        assert store._last_injected_ids["inj1"] == "OrchestratorAgent"
        assert store._last_injected_ids["inj2"] == "OrchestratorAgent"

    def test_format_for_injection_tracks_ids(self, store):
        """format_for_injection() records injected memory IDs with agent name."""
        store.add(
            Memory(id="scoped1", type="pitfall", scopes=["data_ops"], content="Test")
        )
        store._last_injected_ids.clear()
        store.format_for_injection(scope="data_ops")
        assert "scoped1" in store._last_injected_ids
        assert store._last_injected_ids["scoped1"] == "DataOpsAgent"

    def test_injected_ids_cleared_per_message(self, store):
        """_last_injected_ids can be cleared between messages."""
        store.add(Memory(id="x1", type="preference", content="Test"))
        store.build_prompt_section()
        assert "x1" in store._last_injected_ids
        store._last_injected_ids.clear()
        assert len(store._last_injected_ids) == 0


# ---- Git auto-commit ----


class TestGitAutoCommit:
    def test_git_init_on_first_save(self, tmp_path_file):
        """git init is called when .git/ doesn't exist."""
        calls = []
        original_run = subprocess.run

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            # Simulate "git diff --cached --quiet" returning 1 (changes staged)
            if cmd[:3] == ["git", "diff", "--cached"]:
                result = MagicMock()
                result.returncode = 1
                return result
            return MagicMock(returncode=0)

        with patch("agent.memory.subprocess.run", side_effect=mock_run):
            _git_commit_data(tmp_path_file.parent)

        cmd_strs = [" ".join(c) for c in calls]
        assert any("git init" in s for s in cmd_strs)
        assert any("git config user.email" in s for s in cmd_strs)
        assert any("git config user.name" in s for s in cmd_strs)
        assert any("git add" in s for s in cmd_strs)
        assert any("git commit" in s for s in cmd_strs)

    def test_git_init_skipped_when_exists(self, tmp_path_file):
        """git init is NOT called when .git/ already exists."""
        # Create .git dir to simulate existing repo
        git_dir = tmp_path_file.parent / ".git"
        git_dir.mkdir(parents=True, exist_ok=True)

        calls = []

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:3] == ["git", "diff", "--cached"]:
                result = MagicMock()
                result.returncode = 1
                return result
            return MagicMock(returncode=0)

        with patch("agent.memory.subprocess.run", side_effect=mock_run):
            _git_commit_data(tmp_path_file.parent)

        cmd_strs = [" ".join(c) for c in calls]
        assert not any("git init" in s for s in cmd_strs)
        # But add and commit should still happen
        assert any("git add" in s for s in cmd_strs)
        assert any("git commit" in s for s in cmd_strs)

    def test_no_commit_when_nothing_staged(self, tmp_path_file):
        """git commit is NOT called when git diff --cached --quiet returns 0."""
        git_dir = tmp_path_file.parent / ".git"
        git_dir.mkdir(parents=True, exist_ok=True)

        calls = []

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            # returncode=0 means no staged changes
            return MagicMock(returncode=0)

        with patch("agent.memory.subprocess.run", side_effect=mock_run):
            _git_commit_data(tmp_path_file.parent)

        cmd_strs = [" ".join(c) for c in calls]
        assert any("git add" in s for s in cmd_strs)
        assert not any("git commit" in s for s in cmd_strs)

    def test_git_failure_does_not_propagate(self, tmp_path_file):
        """Git failures are swallowed — save() still succeeds."""

        def mock_run(cmd, **kwargs):
            raise subprocess.SubprocessError("git not found")

        with patch("agent.memory.subprocess.run", side_effect=mock_run):
            # Should not raise
            _git_commit_data(tmp_path_file.parent)

    def test_save_calls_git_commit(self, tmp_path_file):
        """MemoryStore.save() triggers _git_commit_data."""
        with patch("agent.memory._git_commit_data") as mock_git:
            store = MemoryStore(path=tmp_path_file)
            store.add(Memory(type="preference", content="Test"))
            assert mock_git.called
            mock_git.assert_called_with(tmp_path_file.parent)

    def test_git_timeout_does_not_propagate(self, tmp_path_file):
        """subprocess.TimeoutExpired is caught and logged."""

        def mock_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 10)

        with patch("agent.memory.subprocess.run", side_effect=mock_run):
            # Should not raise
            _git_commit_data(tmp_path_file.parent)

    def test_gitignore_created_on_init(self, tmp_path_file):
        """Lazy init creates .gitignore that tracks memory.json and pipelines/."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)

        def mock_run(cmd, **kwargs):
            if cmd[:3] == ["git", "diff", "--cached"]:
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        with patch("agent.memory.subprocess.run", side_effect=mock_run):
            _git_commit_data(tmp_path_file.parent)

        gitignore = tmp_path_file.parent / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text()
        assert "*" in content
        assert "!memory.json" in content
        assert "!.gitignore" in content
        assert "!pipelines/" in content
        assert "!pipelines/*.json" in content
