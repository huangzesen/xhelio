"""
Tests for mission override system.

The override system allows learned knowledge to be persisted as sparse
JSON patch files in ``{data_dir}/mission_overrides/``.  At load time,
these are deep-merged on top of the auto-generated mission JSON.

Two levels:
- Mission-level: ``{overrides_dir}/{stem}.json``
- Dataset-level: ``{overrides_dir}/{stem}/{dataset_id}.json``

Run with: python -m pytest tests/test_mission_overrides.py -v
"""

import json
import pytest
from pathlib import Path

from knowledge.mission_loader import (
    _deep_merge,
    _load_override,
    _save_override,
    _get_overrides_dir,
    update_mission_override,
    load_mission,
    clear_cache,
    _CDAWEB_DIR,
    _SOURCE_DIRS,
)
from knowledge.metadata_client import (
    _load_dataset_override,
    update_dataset_override,
    get_dataset_info,
    clear_cache as clear_metadata_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_mission() -> dict:
    """Return a minimal base mission dict for testing."""
    return {
        "id": "TEST",
        "name": "Test Mission",
        "keywords": ["test", "original"],
        "observatory_group": "TestGroup",
        "profile": {
            "description": "Original description.",
            "coordinate_systems": ["GSE"],
            "typical_cadence": "1s",
            "data_caveats": ["caveat1"],
            "analysis_patterns": [],
        },
        "instruments": {
            "MAG": {
                "name": "Magnetometer",
                "keywords": ["magnetic", "field"],
                "datasets": {
                    "TEST_MAG_L2": {
                        "description": "Level 2 MAG data",
                        "start_date": "2020-01-01",
                        "stop_date": "2024-01-01",
                    }
                },
            },
            "SWE": {
                "name": "Solar Wind Experiment",
                "keywords": ["plasma", "solar wind"],
                "datasets": {
                    "TEST_SWE_L2": {
                        "description": "Level 2 SWE data",
                        "start_date": "2020-01-01",
                        "stop_date": "2024-01-01",
                    }
                },
            },
        },
        "_meta": {"generated": "2024-01-01"},
    }


# ---------------------------------------------------------------------------
# TestDeepMerge — unit tests for the generic merge logic
# ---------------------------------------------------------------------------

class TestDeepMerge:

    def test_scalar_replacement(self):
        base = {"name": "Old"}
        _deep_merge(base, {"name": "New"})
        assert base["name"] == "New"

    def test_list_replacement(self):
        base = {"keywords": ["a", "b"]}
        _deep_merge(base, {"keywords": ["x", "y", "z"]})
        assert base["keywords"] == ["x", "y", "z"]

    def test_dict_recursive_merge(self):
        """Nested dicts are merged recursively, not replaced."""
        base = {"profile": {"description": "Old", "cadence": "1s"}}
        _deep_merge(base, {"profile": {"description": "New"}})
        assert base["profile"]["description"] == "New"
        assert base["profile"]["cadence"] == "1s"

    def test_new_key_added(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert base == {"a": 1, "b": 2}

    def test_nested_new_key_added(self):
        base = {"profile": {"description": "Old"}}
        _deep_merge(base, {"profile": {"special_notes": "Important"}})
        assert base["profile"]["special_notes"] == "Important"
        assert base["profile"]["description"] == "Old"

    def test_deeply_nested_merge(self):
        """Three levels of nesting."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        _deep_merge(base, {"a": {"b": {"c": 99}}})
        assert base["a"]["b"]["c"] == 99
        assert base["a"]["b"]["d"] == 2

    def test_dict_replaced_by_scalar(self):
        """If patch has a scalar where base has a dict, scalar wins."""
        base = {"profile": {"description": "Old"}}
        _deep_merge(base, {"profile": "just a string"})
        assert base["profile"] == "just a string"

    def test_scalar_replaced_by_dict(self):
        """If patch has a dict where base has a scalar, dict wins."""
        base = {"profile": "just a string"}
        _deep_merge(base, {"profile": {"description": "New"}})
        assert base["profile"] == {"description": "New"}

    def test_empty_patch_is_noop(self):
        import copy
        base = _make_base_mission()
        original = copy.deepcopy(base)
        _deep_merge(base, {})
        assert base == original

    def test_returns_base(self):
        base = {"a": 1}
        result = _deep_merge(base, {"b": 2})
        assert result is base

    def test_full_mission_override(self):
        """Realistic mission override scenario."""
        base = _make_base_mission()
        _deep_merge(base, {
            "name": "Updated Name",
            "keywords": ["new1", "new2"],
            "profile": {"description": "Better description."},
            "instruments": {
                "MAG": {"name": "Fluxgate Magnetometer"},
            },
        })
        assert base["name"] == "Updated Name"
        assert base["keywords"] == ["new1", "new2"]
        assert base["profile"]["description"] == "Better description."
        assert base["profile"]["coordinate_systems"] == ["GSE"]  # untouched
        assert base["instruments"]["MAG"]["name"] == "Fluxgate Magnetometer"
        assert base["instruments"]["MAG"]["keywords"] == ["magnetic", "field"]  # untouched

    def test_can_override_any_field(self):
        """Generic merge has no allowlist — any field can be patched."""
        base = _make_base_mission()
        _deep_merge(base, {
            "id": "CHANGED",
            "observatory_group": "NewGroup",
            "_meta": {"generated": "2025-01-01"},
        })
        # All fields are updated — no restrictions
        assert base["id"] == "CHANGED"
        assert base["observatory_group"] == "NewGroup"
        assert base["_meta"]["generated"] == "2025-01-01"

    def test_instrument_datasets_can_be_patched(self):
        """Datasets within instruments can be patched (generic merge)."""
        base = _make_base_mission()
        _deep_merge(base, {
            "instruments": {
                "MAG": {
                    "datasets": {
                        "TEST_MAG_L2": {"description": "Updated desc"}
                    }
                }
            }
        })
        assert base["instruments"]["MAG"]["datasets"]["TEST_MAG_L2"]["description"] == "Updated desc"
        # start_date still present (deep merge)
        assert base["instruments"]["MAG"]["datasets"]["TEST_MAG_L2"]["start_date"] == "2020-01-01"

    def test_new_instrument_added(self):
        """A new instrument can be added via override."""
        base = _make_base_mission()
        _deep_merge(base, {
            "instruments": {
                "EPHEM": {"name": "Ephemeris", "keywords": ["position"]}
            }
        })
        assert "EPHEM" in base["instruments"]
        assert base["instruments"]["EPHEM"]["name"] == "Ephemeris"
        # Existing instruments untouched
        assert base["instruments"]["MAG"]["name"] == "Magnetometer"


# ---------------------------------------------------------------------------
# TestLoadOverride — file loading
# ---------------------------------------------------------------------------

class TestLoadOverride:

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        assert _load_override("nonexistent") is None

    def test_valid_file_loads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        override = {"name": "Custom Name"}
        (tmp_path / "test.json").write_text(json.dumps(override))
        result = _load_override("test")
        assert result == override

    def test_malformed_json_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        (tmp_path / "bad.json").write_text("{not valid json")
        assert _load_override("bad") is None

    def test_non_dict_json_returns_none(self, tmp_path, monkeypatch):
        """A JSON array is not a valid override."""
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        (tmp_path / "arr.json").write_text('["not", "a", "dict"]')
        assert _load_override("arr") is None


# ---------------------------------------------------------------------------
# TestSaveOverride and TestUpdateMissionOverride
# ---------------------------------------------------------------------------

class TestSaveOverride:

    def test_save_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        _save_override("test", {"name": "Saved"})
        path = tmp_path / "test.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "Saved"

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        nested = tmp_path / "sub" / "dir"
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: nested,
        )
        _save_override("test", {"a": 1})
        assert (nested / "test.json").exists()


class TestUpdateMissionOverride:

    def test_creates_new_override(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        result = update_mission_override("test", {"name": "New"})
        assert result == {"name": "New"}
        assert (tmp_path / "test.json").exists()

    def test_merges_into_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        # First write
        update_mission_override("test", {"name": "First", "a": 1})
        # Second write — merges
        result = update_mission_override("test", {"b": 2})
        assert result == {"name": "First", "a": 1, "b": 2}

    def test_invalidates_cache(self, tmp_path, monkeypatch):
        from knowledge.mission_loader import _mission_cache
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        _mission_cache["test"] = {"cached": True}
        update_mission_override("test", {"name": "New"})
        assert "test" not in _mission_cache


# ---------------------------------------------------------------------------
# TestLoadMissionWithOverrides — integration
# ---------------------------------------------------------------------------

class TestLoadMissionWithOverrides:

    @pytest.fixture(autouse=True)
    def fresh(self):
        clear_cache()
        yield
        clear_cache()

    def _find_any_mission_stem(self) -> str | None:
        """Return the stem of any mission JSON on disk, or None."""
        files = sorted(_CDAWEB_DIR.glob("*.json"))
        return files[0].stem if files else None

    def test_load_mission_applies_override(self, tmp_path, monkeypatch):
        stem = self._find_any_mission_stem()
        if stem is None:
            pytest.skip("No mission JSON files on disk")

        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        override = {"profile": {"description": "OVERRIDDEN DESC"}}
        (tmp_path / f"{stem}.json").write_text(json.dumps(override))

        mission = load_mission(stem)
        assert mission["profile"]["description"] == "OVERRIDDEN DESC"

    def test_cache_contains_merged_data(self, tmp_path, monkeypatch):
        stem = self._find_any_mission_stem()
        if stem is None:
            pytest.skip("No mission JSON files on disk")

        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        override = {"name": "CACHED NAME"}
        (tmp_path / f"{stem}.json").write_text(json.dumps(override))

        m1 = load_mission(stem)
        m2 = load_mission(stem)
        assert m1 is m2  # same cached object
        assert m1["name"] == "CACHED NAME"

    def test_no_override_unchanged(self, tmp_path, monkeypatch):
        stem = self._find_any_mission_stem()
        if stem is None:
            pytest.skip("No mission JSON files on disk")

        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        # No override file created — tmp_path is empty

        # Load the raw JSON for comparison
        with open(_CDAWEB_DIR / f"{stem}.json", "r") as f:
            raw = json.load(f)

        mission = load_mission(stem)
        assert mission["name"] == raw["name"]
        assert mission["profile"] == raw["profile"]


# ---------------------------------------------------------------------------
# TestDatasetOverrides — dataset-level overrides in metadata_client
# ---------------------------------------------------------------------------

class TestDatasetOverrides:

    @pytest.fixture(autouse=True)
    def fresh(self):
        clear_metadata_cache()
        yield
        clear_metadata_cache()

    def test_load_dataset_override_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        assert _load_dataset_override("FAKE_DATASET") is None

    def test_load_dataset_override_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        # Create dataset override at {overrides_dir}/ace/AC_H2_MFI.json
        ds_dir = tmp_path / "ace"
        ds_dir.mkdir()
        override = {"_note": "Test note"}
        (ds_dir / "AC_H2_MFI.json").write_text(json.dumps(override))

        result = _load_dataset_override("AC_H2_MFI")
        assert result == {"_note": "Test note"}

    def test_load_dataset_override_malformed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        ds_dir = tmp_path / "ace"
        ds_dir.mkdir()
        (ds_dir / "AC_H2_MFI.json").write_text("{bad json")
        assert _load_dataset_override("AC_H2_MFI") is None

    def test_update_dataset_override_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        result = update_dataset_override(
            "TEST_DS", {"note": "hello"}, mission_stem="test"
        )
        assert result == {"note": "hello"}
        path = tmp_path / "test" / "TEST_DS.json"
        assert path.exists()

    def test_update_dataset_override_merges(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )
        update_dataset_override("TEST_DS", {"a": 1}, mission_stem="test")
        result = update_dataset_override("TEST_DS", {"b": 2}, mission_stem="test")
        assert result == {"a": 1, "b": 2}

    def test_get_dataset_info_merges_override(self, tmp_path, monkeypatch):
        """Dataset override is merged into get_dataset_info() result."""
        monkeypatch.setattr(
            "knowledge.mission_loader._get_overrides_dir",
            lambda: tmp_path,
        )

        # Find a real dataset to test with
        for source_dir in _SOURCE_DIRS:
            found = False
            if not source_dir.exists():
                continue
            for mission_dir in source_dir.iterdir():
                if not mission_dir.is_dir():
                    continue
                metadata_dir = mission_dir / "metadata"
                if not metadata_dir.exists():
                    continue
                ds_files = [f for f in metadata_dir.glob("*.json") if not f.name.startswith("_")]
                if ds_files:
                    dataset_id = ds_files[0].stem
                    stem = mission_dir.name
                    found = True
                    break
            if found:
                break
        else:
            pytest.skip("No cached dataset metadata on disk")

        # Create a dataset override
        ds_dir = tmp_path / stem
        ds_dir.mkdir()
        (ds_dir / f"{dataset_id}.json").write_text(
            json.dumps({"_custom_note": "Injected by override"})
        )

        info = get_dataset_info(dataset_id)
        assert info.get("_custom_note") == "Injected by override"
        # Original fields still present
        assert "parameters" in info
