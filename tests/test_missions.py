"""
Generic mission test bed.

Tests iterate over all populated missions on disk — no hardcoded mission IDs.
When the primary mission list changes, zero tests need updating.

Run with: python -m pytest tests/test_missions.py -v
"""

import json
import pytest
from datetime import datetime
from pathlib import Path

from knowledge.mission_loader import (
    load_mission,
    load_all_missions,
    get_routing_table,
    get_mission_datasets,
    get_mission_ids,
    clear_cache,
    _ENVOYS_DIR,
)
from knowledge.catalog import (
    SPACECRAFT,
    list_spacecraft,
    match_spacecraft,
    match_instrument,
    search_by_keywords,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_cache():
    """Clear the module cache before each test."""
    clear_cache()
    yield
    clear_cache()


def _all_mission_jsons() -> list[Path]:
    """Discover all envoy JSON files on disk."""
    jsons = []
    for kind_dir in sorted(_ENVOYS_DIR.iterdir()):
        if kind_dir.is_dir():
            jsons.extend(kind_dir.glob("*.json"))
    return sorted(jsons)


def _all_mission_ids() -> list[str]:
    """Load all missions and return their canonical IDs."""
    return sorted(load_all_missions().keys())


def _missions_with_keywords() -> list[str]:
    """Return mission IDs that have at least one instrument with keywords."""
    result = []
    for sc_id, sc_info in SPACECRAFT.items():
        for inst in sc_info["instruments"].values():
            if inst.get("keywords"):
                result.append(sc_id)
                break
    return result


# ---------------------------------------------------------------------------
# JSON file integrity — every file on disk is well-formed
# ---------------------------------------------------------------------------


class TestJsonIntegrity:
    """Verify all mission JSON files have valid structure."""

    def test_all_json_files_parse(self):
        for filepath in _all_mission_jsons():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "id" in data, f"{filepath.name} missing 'id'"
            assert "name" in data, f"{filepath.name} missing 'name'"
            assert "keywords" in data, f"{filepath.name} missing 'keywords'"
            assert "instruments" in data, f"{filepath.name} missing 'instruments'"

    def test_all_datasets_have_date_ranges(self):
        """Every dataset should have start_date and stop_date."""
        for filepath in _all_mission_jsons():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for inst_id, inst in data["instruments"].items():
                for ds_id, ds_info in inst["datasets"].items():
                    assert "start_date" in ds_info, (
                        f"{data['id']}/{inst_id}/{ds_id} missing start_date"
                    )
                    assert "stop_date" in ds_info, (
                        f"{data['id']}/{inst_id}/{ds_id} missing stop_date"
                    )

    def test_date_ranges_are_valid(self):
        """start_date should be before or equal to stop_date for every dataset."""
        for filepath in _all_mission_jsons():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for inst_id, inst in data["instruments"].items():
                for ds_id, ds_info in inst["datasets"].items():
                    start = ds_info.get("start_date", "")
                    stop = ds_info.get("stop_date", "")
                    if start and stop:
                        assert start <= stop, (
                            f"{data['id']}/{ds_id}: start_date {start} > stop_date {stop}"
                        )

    def test_no_datasets_have_tier(self):
        """Tier field has been removed from all dataset entries."""
        for filepath in _all_mission_jsons():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for inst in data["instruments"].values():
                for ds_id, ds_info in inst["datasets"].items():
                    assert "tier" not in ds_info, (
                        f"{data['id']}/{ds_id} still has tier field"
                    )

    def test_datasets_are_dicts(self):
        """Datasets should be dicts keyed by dataset ID, not lists."""
        for filepath in _all_mission_jsons():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for inst in data["instruments"].values():
                assert isinstance(inst["datasets"], dict)

    def test_at_least_one_mission_exists(self):
        assert len(_all_mission_jsons()) > 0, "No mission JSON files found"


# ---------------------------------------------------------------------------
# Mission loader — caching, loading, routing table
# ---------------------------------------------------------------------------


class TestMissionLoader:
    def test_load_is_case_insensitive(self):
        """Loading by lowercase stem should work."""
        ids = _all_mission_ids()
        if ids:
            mission = load_mission(ids[0].lower())
            assert mission["id"] == ids[0]

    def test_caching(self):
        ids = _all_mission_ids()
        if ids:
            m1 = load_mission(ids[0])
            m2 = load_mission(ids[0])
            assert m1 is m2  # Same object = cached

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_mission("NONEXISTENT_MISSION_XYZ")

    def test_all_missions_keyed_by_id(self):
        missions = load_all_missions()
        for mission_id, mission in missions.items():
            assert mission["id"] == mission_id

    def test_mission_ids_are_sorted(self):
        ids = get_mission_ids()
        assert ids == sorted(ids)

    def test_routing_table_covers_all_missions(self):
        table = get_routing_table()
        table_ids = {e["id"] for e in table}
        file_ids = set(_all_mission_ids())
        assert table_ids == file_ids

    def test_routing_table_entry_structure(self):
        for entry in get_routing_table():
            assert "id" in entry
            assert "name" in entry
            assert "capabilities" in entry
            assert isinstance(entry["capabilities"], list)

    def test_get_mission_datasets_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            get_mission_datasets("NONEXISTENT_MISSION_XYZ")


# ---------------------------------------------------------------------------
# Catalog — every mission is discoverable and has datasets
# ---------------------------------------------------------------------------


class TestCatalogDiscovery:
    """Every populated mission should be findable through the catalog."""

    def test_all_missions_in_catalog(self):
        """Every JSON on disk appears in the SPACECRAFT catalog."""
        catalog_ids = {s["id"] for s in list_spacecraft()}
        for mission_id in _all_mission_ids():
            assert mission_id in catalog_ids, f"{mission_id} not found in catalog"

    def test_every_mission_has_datasets(self):
        """Every mission should have at least one dataset."""
        for sc_id, sc_info in SPACECRAFT.items():
            total = sum(
                len(inst["datasets"]) for inst in sc_info["instruments"].values()
            )
            assert total > 0, f"{sc_id} has no datasets"

    def test_every_mission_findable_by_own_id(self):
        """match_spacecraft(id.lower()) should return the mission."""
        for sc_id in _all_mission_ids():
            result = match_spacecraft(sc_id.lower())
            assert result == sc_id, (
                f"match_spacecraft({sc_id.lower()!r}) returned {result!r}, expected {sc_id!r}"
            )

    def test_every_mission_findable_by_keyword(self):
        """Each mission should be findable by at least one of its keywords."""
        missions = load_all_missions()
        for sc_id, mission in missions.items():
            keywords = mission.get("keywords", [])
            found = False
            for kw in keywords:
                result = match_spacecraft(kw)
                if result == sc_id:
                    found = True
                    break
            assert found, f"{sc_id} not findable by any of its keywords: {keywords}"

    def test_dataset_ids_are_nonempty_strings(self):
        for sc_id, sc_info in SPACECRAFT.items():
            for inst_info in sc_info["instruments"].values():
                for ds in inst_info["datasets"]:
                    assert isinstance(ds, str) and len(ds) > 0

    def test_missions_have_profiles(self):
        """Every mission should have a profile with description."""
        for mission_id in _all_mission_ids():
            mission = load_mission(mission_id)
            assert "profile" in mission, f"{mission_id} missing profile"
            assert "description" in mission["profile"]


# ---------------------------------------------------------------------------
# Keyword routing — missions with keywords support instrument matching
# ---------------------------------------------------------------------------


class TestKeywordRouting:
    """Missions with instrument keywords should support keyword search."""

    def test_missions_with_keywords_have_capabilities(self):
        """Missions whose instruments have keywords should appear with
        capabilities in the routing table."""
        table = {e["id"]: e for e in get_routing_table()}
        for sc_id in _missions_with_keywords():
            entry = table.get(sc_id)
            assert entry is not None, f"{sc_id} missing from routing table"
            assert len(entry["capabilities"]) > 0, (
                f"{sc_id} has instrument keywords but no routing capabilities"
            )

    def test_keyword_search_returns_a_spacecraft(self):
        """search_by_keywords('<mission_keyword> <instrument_keyword>')
        should return a valid spacecraft (may differ if keywords overlap)."""
        missions = load_all_missions()
        all_ids = set(missions.keys())
        for sc_id in _missions_with_keywords():
            mission = missions[sc_id]
            # Pick first mission keyword and first instrument keyword
            sc_keyword = mission["keywords"][0] if mission["keywords"] else None
            inst_keyword = None
            for inst in mission["instruments"].values():
                if inst.get("keywords"):
                    inst_keyword = inst["keywords"][0]
                    break
            if sc_keyword and inst_keyword:
                result = search_by_keywords(f"{sc_keyword} {inst_keyword}")
                if result is not None:
                    assert result["mission"] in all_ids, (
                        f"search '{sc_keyword} {inst_keyword}' returned "
                        f"unknown mission {result['mission']}"
                    )

    def test_bare_mission_search_lists_instruments(self):
        """Searching just a mission keyword should list available instruments."""
        for sc_id in _all_mission_ids():
            mission = load_mission(sc_id)
            if not mission["keywords"]:
                continue
            kw = mission["keywords"][0]
            result = search_by_keywords(kw)
            if result is not None and result["mission"] == sc_id:
                assert "available_instruments" in result

    def test_match_instrument_with_keywords(self):
        """match_instrument should return something for missions with
        instrument keywords."""
        for sc_id in _missions_with_keywords():
            sc = SPACECRAFT[sc_id]
            for inst_id, inst in sc["instruments"].items():
                if inst.get("keywords"):
                    # Try the first keyword
                    kw = inst["keywords"][0]
                    result = match_instrument(sc_id, kw)
                    # Should match *some* instrument (maybe not this exact one
                    # if keywords overlap)
                    assert result is not None, (
                        f"match_instrument({sc_id!r}, {kw!r}) returned None "
                        f"but {inst_id} has keyword {kw!r}"
                    )
                    break  # One check per mission is enough
