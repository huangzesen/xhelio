"""Tests for data_ops.ops_library — persistent custom operations library."""

import json
import os
import pytest
from pathlib import Path

from data_ops.ops_library import OpsLibrary, reset_ops_library


@pytest.fixture
def tmp_library(tmp_path):
    """Create an OpsLibrary backed by a temp file."""
    path = tmp_path / "test_ops_library.json"
    return OpsLibrary(path=path, max_entries=50)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module singleton before each test."""
    reset_ops_library()
    yield
    reset_ops_library()


SAMPLE_CODE = (
    "merged = pd.concat([df_BR, df_BT, df_BN], axis=1)\n"
    "sq = merged.pow(2)\n"
    "summed = sq.sum(axis=1, skipna=False)\n"
    "mag = summed.pow(0.5)\n"
    "result = mag.to_frame('magnitude')"
)


class TestAddNewEntry:
    def test_add_and_verify(self, tmp_library):
        entry = tmp_library.add_or_update(
            description="Compute vector magnitude from 3 components",
            code=SAMPLE_CODE,
            source_labels=["DS.BR", "DS.BT", "DS.BN"],
            units="nT",
            session_id="sess_001",
        )
        assert entry["id"]
        assert len(entry["id"]) == 8
        assert entry["use_count"] == 1
        assert entry["num_sources"] == 3
        assert entry["source_type"] == "df"
        assert entry["units"] == "nT"
        assert entry["description"] == "Compute vector magnitude from 3 components"
        assert entry["code"] == SAMPLE_CODE

    def test_entry_persisted_to_disk(self, tmp_library):
        tmp_library.add_or_update(
            description="Test op",
            code=SAMPLE_CODE,
            source_labels=["A.x"],
        )
        data = json.loads(tmp_library._path.read_text())
        assert len(data["entries"]) == 1
        assert data["version"] == 1


class TestDedupByDescription:
    def test_same_description_different_case(self, tmp_library):
        tmp_library.add_or_update(
            description="Compute Magnitude",
            code="a\nb\nc\nd\ne",
            source_labels=["A.x"],
        )
        tmp_library.add_or_update(
            description="compute magnitude",
            code="f\ng\nh\ni\nj",
            source_labels=["B.x"],
        )
        entries = tmp_library.get_top_entries()
        assert len(entries) == 1
        assert entries[0]["use_count"] == 2
        assert entries[0]["code"] == "f\ng\nh\ni\nj"  # updated

    def test_description_with_extra_whitespace(self, tmp_library):
        tmp_library.add_or_update(
            description="  compute   magnitude  ",
            code="a\nb\nc\nd\ne",
            source_labels=["A.x"],
        )
        tmp_library.add_or_update(
            description="compute magnitude",
            code="f\ng\nh\ni\nj",
            source_labels=["B.x"],
        )
        entries = tmp_library.get_top_entries()
        assert len(entries) == 1
        assert entries[0]["use_count"] == 2

    def test_description_strips_from_ref(self, tmp_library):
        tmp_library.add_or_update(
            description="Compute magnitude",
            code="a\nb\nc\nd\ne",
            source_labels=["A.x"],
        )
        tmp_library.add_or_update(
            description="Compute magnitude [from ab12cd34]",
            code="f\ng\nh\ni\nj",
            source_labels=["B.x"],
        )
        entries = tmp_library.get_top_entries()
        assert len(entries) == 1
        assert entries[0]["use_count"] == 2


class TestEviction:
    def test_evicts_when_full(self, tmp_path):
        lib = OpsLibrary(path=tmp_path / "lib.json", max_entries=3)
        for i in range(3):
            lib.add_or_update(
                description=f"Op {i}",
                code=f"line{i}",
                source_labels=["A.x"],
            )
        assert len(lib.get_top_entries(limit=100)) == 3

        # Adding a 4th should evict one
        lib.add_or_update(
            description="Op 3",
            code="line3",
            source_labels=["A.x"],
        )
        entries = lib.get_top_entries(limit=100)
        assert len(entries) == 3
        descs = {e["description"] for e in entries}
        assert "Op 3" in descs

    def test_eviction_tiebreak_oldest(self, tmp_path):
        """Same use_count → oldest last_used_at is evicted."""
        lib = OpsLibrary(path=tmp_path / "lib.json", max_entries=3)

        # Add 3 entries; all use_count=1. Manually set timestamps.
        for i in range(3):
            lib.add_or_update(
                description=f"Op {i}",
                code=f"line{i}",
                source_labels=["A.x"],
            )
        # Adjust timestamps so Op 0 is oldest
        lib._entries[0]["last_used_at"] = "2025-01-01T00:00:00+00:00"
        lib._entries[1]["last_used_at"] = "2025-06-01T00:00:00+00:00"
        lib._entries[2]["last_used_at"] = "2025-12-01T00:00:00+00:00"

        # Add 4th → Op 0 (oldest, same use_count) should be evicted
        lib.add_or_update(
            description="Op 3",
            code="line3",
            source_labels=["A.x"],
        )
        entries = lib.get_top_entries(limit=100)
        descs = {e["description"] for e in entries}
        assert "Op 0" not in descs
        assert "Op 3" in descs

    def test_evicts_lowest_use_count(self, tmp_path):
        """Lowest use_count is evicted, even if not oldest."""
        lib = OpsLibrary(path=tmp_path / "lib.json", max_entries=3)
        for i in range(3):
            lib.add_or_update(
                description=f"Op {i}",
                code=f"line{i}",
                source_labels=["A.x"],
            )
        # Bump use_count for Op 0 and Op 2
        lib._entries[0]["use_count"] = 5
        lib._entries[2]["use_count"] = 3

        # Op 1 has use_count=1 → should be evicted
        lib.add_or_update(
            description="Op 3",
            code="line3",
            source_labels=["A.x"],
        )
        entries = lib.get_top_entries(limit=100)
        descs = {e["description"] for e in entries}
        assert "Op 1" not in descs


class TestRecordReuse:
    def test_bumps_use_count(self, tmp_library):
        entry = tmp_library.add_or_update(
            description="Some op",
            code="a\nb\nc\nd\ne",
            source_labels=["A.x"],
        )
        assert entry["use_count"] == 1

        result = tmp_library.record_reuse(entry["id"])
        assert result is True

        entries = tmp_library.get_top_entries()
        assert entries[0]["use_count"] == 2

    def test_nonexistent_id_returns_false(self, tmp_library):
        result = tmp_library.record_reuse("deadbeef")
        assert result is False


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "lib.json"
        lib1 = OpsLibrary(path=path, max_entries=50)
        lib1.add_or_update(
            description="Persistent op",
            code=SAMPLE_CODE,
            source_labels=["DS.BR", "DS.BT", "DS.BN"],
            units="nT",
        )

        # Create a new instance from the same file
        lib2 = OpsLibrary(path=path, max_entries=50)
        entries = lib2.get_top_entries()
        assert len(entries) == 1
        assert entries[0]["description"] == "Persistent op"
        assert entries[0]["code"] == SAMPLE_CODE
        assert entries[0]["units"] == "nT"


class TestBuildPromptSection:
    def test_returns_empty_when_no_entries(self, tmp_library):
        result = tmp_library.build_prompt_section()
        assert result == ""

    def test_returns_markdown(self, tmp_library):
        tmp_library.add_or_update(
            description="Butterworth low-pass filter",
            code=SAMPLE_CODE,
            source_labels=["DS.Bmag"],
            units="nT",
        )
        section = tmp_library.build_prompt_section()
        assert "## Saved Operations Library" in section
        assert "Butterworth low-pass filter" in section
        assert "```python" in section
        assert SAMPLE_CODE in section
        assert "Sources: 1 df" in section
        assert "Units: nT" in section
        assert "Used 1 time" in section

    def test_plural_use_count(self, tmp_library):
        entry = tmp_library.add_or_update(
            description="Some op",
            code="a\nb\nc\nd\ne",
            source_labels=["A.x"],
        )
        tmp_library.record_reuse(entry["id"])
        section = tmp_library.build_prompt_section()
        assert "Used 2 times" in section


class TestInferSourceType:
    def test_df_only(self, tmp_library):
        entry = tmp_library.add_or_update(
            description="DF op",
            code="result = df_Bmag.pow(2).sum(axis=1).pow(0.5).to_frame('mag')\nline2\nline3\nline4\nline5",
            source_labels=["DS.Bmag"],
        )
        assert entry["source_type"] == "df"

    def test_da_only(self, tmp_library):
        entry = tmp_library.add_or_update(
            description="DA op",
            code="result = da_spec.mean(dim='frequency')\nline2\nline3\nline4\nline5",
            source_labels=["DS.spec"],
        )
        assert entry["source_type"] == "da"

    def test_mixed(self, tmp_library):
        entry = tmp_library.add_or_update(
            description="Mixed op",
            code="merged = pd.concat([df_Bmag], axis=1)\nresult = da_spec.mean()\nline3\nline4\nline5",
            source_labels=["DS.Bmag", "DS.spec"],
        )
        assert entry["source_type"] == "mixed"

    def test_bare_df_alias(self, tmp_library):
        entry = tmp_library.add_or_update(
            description="Bare df op",
            code="result = df.pow(2).sum(axis=1).pow(0.5)\nline2\nline3\nline4\nline5",
            source_labels=["DS.Bmag"],
        )
        assert entry["source_type"] == "df"


class TestNormalizeDescription:
    def test_basic(self):
        assert OpsLibrary._normalize_description("Hello  World") == "hello world"

    def test_strips_from_ref(self):
        assert OpsLibrary._normalize_description(
            "Compute magnitude [from ab12cd34]"
        ) == "compute magnitude"

    def test_whitespace_collapse(self):
        assert OpsLibrary._normalize_description("  a   b   c  ") == "a b c"
