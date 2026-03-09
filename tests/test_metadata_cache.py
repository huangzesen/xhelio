"""
Tests for local metadata cache functionality.

Tests the local file cache in metadata_client.py: _find_local_cache(),
get_dataset_info() with local files, list_parameters from cache,
and list_cached_datasets().

Run with: python -m pytest tests/test_metadata_cache.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from knowledge.metadata_client import (
    _find_local_cache,
    get_dataset_info,
    list_parameters,
    list_cached_datasets,
    browse_datasets,
    list_missions,
    validate_dataset_id,
    validate_parameter_id,
    clear_cache,
)


# Sample metadata response for testing
SAMPLE_METADATA_INFO = {
    "startDate": "2018-10-06T00:00:00.000Z",
    "stopDate": "2025-12-31T00:00:00.000Z",
    "parameters": [
        {"name": "Time", "type": "isotime", "units": "UTC", "length": 24},
        {
            "name": "psp_fld_l2_mag_RTN_1min",
            "type": "double",
            "units": "nT",
            "size": [3],
            "description": "Magnetic field in RTN coordinates",
        },
        {
            "name": "psp_fld_l2_quality_flags",
            "type": "integer",
            "units": None,
            "size": [1],
            "description": "Quality flags",
        },
    ],
}

SAMPLE_INDEX = {
    "mission_id": "PSP",
    "dataset_count": 2,
    "generated_at": "2026-02-07T00:00:00Z",
    "datasets": [
        {
            "id": "PSP_FLD_L2_MAG_RTN_1MIN",
            "description": "PSP FIELDS Magnetometer 1-min RTN",
            "start_date": "2018-10-06",
            "stop_date": "2025-12-31",
            "parameter_count": 2,
            "instrument": "FIELDS/MAG",
        },
        {
            "id": "PSP_SWP_SPC_L3I",
            "description": "PSP SWEAP SPC Level 3i",
            "start_date": "2018-10-06",
            "stop_date": "2025-12-31",
            "parameter_count": 5,
            "instrument": "SWEAP",
        },
    ],
}


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear in-memory cache before each test."""
    clear_cache()


@pytest.fixture
def fake_missions_dir(tmp_path):
    """Create a temporary missions directory with cache files.

    Structure mirrors the real layout:
      envoys/cdaweb/psp/metadata/...
      envoys/cdaweb/ace/metadata/...
    """
    missions_dir = tmp_path / "envoys"
    cdaweb_dir = missions_dir / "cdaweb"

    # Create PSP cache under cdaweb/
    psp_meta = cdaweb_dir / "psp" / "metadata"
    psp_meta.mkdir(parents=True)

    # Write sample metadata file
    cache_file = psp_meta / "PSP_FLD_L2_MAG_RTN_1MIN.json"
    cache_file.write_text(json.dumps(SAMPLE_METADATA_INFO), encoding="utf-8")

    # Write _index.json
    index_file = psp_meta / "_index.json"
    index_file.write_text(json.dumps(SAMPLE_INDEX), encoding="utf-8")

    # Create ACE cache (second mission for list_missions tests)
    ace_meta = cdaweb_dir / "ace" / "metadata"
    ace_meta.mkdir(parents=True)
    ace_index = {
        "mission_id": "ACE",
        "dataset_count": 1,
        "generated_at": "2026-02-07T00:00:00Z",
        "datasets": [
            {
                "id": "AC_H2_MFI",
                "description": "ACE MFI Level 2 Data",
                "start_date": "1998-01-01",
                "stop_date": "2025-12-31",
                "parameter_count": 2,
                "instrument": "MFI",
            },
        ],
    }
    ace_index_file = ace_meta / "_index.json"
    ace_index_file.write_text(json.dumps(ace_index), encoding="utf-8")
    ace_cache = ace_meta / "AC_H2_MFI.json"
    ace_meta_info = {
        "startDate": "1998-01-01T00:00:00.000Z",
        "stopDate": "2025-12-31T00:00:00.000Z",
        "parameters": [
            {"name": "Time", "type": "isotime", "units": "UTC", "length": 24},
            {"name": "BGSEc", "type": "double", "units": "nT", "size": [3],
             "description": "Magnetic field in GSE coordinates"},
            {"name": "Magnitude", "type": "double", "units": "nT", "size": [1],
             "description": "Total magnetic field"},
        ],
    }
    ace_cache.write_text(json.dumps(ace_meta_info), encoding="utf-8")

    # Create a dir without _index.json (should be skipped by list_missions)
    empty_mission = cdaweb_dir / "empty_mission" / "metadata"
    empty_mission.mkdir(parents=True)

    # Also create a non-directory file (the psp.json mission file)
    mission_file = cdaweb_dir / "psp.json"
    mission_file.write_text("{}", encoding="utf-8")

    # Return the cdaweb dir as the source dir list item
    return cdaweb_dir


class TestFindLocalCache:
    def test_returns_path_when_exists(self, fake_missions_dir):
        """_find_local_cache returns the path when a cache file exists."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = _find_local_cache("PSP_FLD_L2_MAG_RTN_1MIN")
            assert result is not None
            assert result.name == "PSP_FLD_L2_MAG_RTN_1MIN.json"
            assert result.exists()

    def test_returns_none_when_missing(self, fake_missions_dir):
        """_find_local_cache returns None for a dataset not in cache."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = _find_local_cache("NONEXISTENT_DATASET")
            assert result is None

    def test_skips_non_directory_entries(self, fake_missions_dir):
        """_find_local_cache skips files like psp.json (not directories)."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            # Should not crash even though psp.json exists at missions_dir level
            result = _find_local_cache("PSP_FLD_L2_MAG_RTN_1MIN")
            assert result is not None


class TestGetDatasetInfoLocalCache:
    def test_uses_local_cache(self, fake_missions_dir):
        """get_dataset_info loads from local file, no network needed."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            info = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN")
            assert info is not None
            assert info["startDate"] == "2018-10-06T00:00:00.000Z"
            assert len(info["parameters"]) == 3

    def test_local_cache_populates_memory_cache(self, fake_missions_dir):
        """Reading from local file also populates the in-memory cache."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            info1 = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN")
            # Delete the file to prove second call uses memory
            cache_file = fake_missions_dir / "psp" / "metadata" / "PSP_FLD_L2_MAG_RTN_1MIN.json"
            cache_file.unlink()
            info2 = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN")
            assert info1 == info2

    def test_use_cache_false_skips_local_uses_master_cdf(self, fake_missions_dir):
        """use_cache=False skips both memory and local file, hits Master CDF."""
        master_info = {"startDate": "2018-01-01", "parameters": [{"name": "Time", "type": "isotime"}]}
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]), \
             patch("knowledge.master_cdf.fetch_dataset_metadata_from_master", return_value=master_info):
            info = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN", use_cache=False)
            assert info["startDate"] == "2018-01-01"

    def test_use_cache_false_raises_when_master_cdf_fails(self, fake_missions_dir):
        """use_cache=False raises ValueError when Master CDF returns None."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]), \
             patch("knowledge.master_cdf.fetch_dataset_metadata_from_master", return_value=None):
            with pytest.raises(ValueError, match="No metadata available"):
                get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN", use_cache=False)


class TestListParametersFromCache:
    def test_lists_parameters_from_local_cache(self, fake_missions_dir):
        """list_parameters works with locally cached metadata."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            params = list_parameters("PSP_FLD_L2_MAG_RTN_1MIN")
            assert len(params) == 2
            names = [p["name"] for p in params]
            assert "psp_fld_l2_mag_RTN_1min" in names
            assert "psp_fld_l2_quality_flags" in names
            # Time should be excluded
            assert "Time" not in names

    def test_parameter_structure(self, fake_missions_dir):
        """Parameters from cache have correct structure."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            params = list_parameters("PSP_FLD_L2_MAG_RTN_1MIN")
            mag_param = next(p for p in params if p["name"] == "psp_fld_l2_mag_RTN_1min")
            assert mag_param["units"] == "nT"
            assert mag_param["size"] == [3]
            assert mag_param["dataset_id"] == "PSP_FLD_L2_MAG_RTN_1MIN"

    def test_returns_empty_for_uncached_invalid_dataset(self, fake_missions_dir):
        """list_parameters returns empty list for dataset not in cache or network."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]), \
             patch("knowledge.metadata_client.requests.get", return_value=mock_resp):
            params = list_parameters("NONEXISTENT_DATASET_XYZ")
            assert params == []


class TestListCachedDatasets:
    def test_loads_index(self, fake_missions_dir):
        """list_cached_datasets loads the _index.json correctly."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            index = list_cached_datasets("PSP")
            assert index is not None
            assert index["mission_id"] == "PSP"
            assert index["dataset_count"] == 2
            assert len(index["datasets"]) == 2

    def test_case_insensitive(self, fake_missions_dir):
        """Mission ID lookup is case-insensitive."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            index = list_cached_datasets("psp")
            assert index is not None
            assert index["mission_id"] == "PSP"

    def test_returns_none_when_no_index(self, fake_missions_dir):
        """Returns None when no _index.json exists for the mission."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = list_cached_datasets("NONEXISTENT")
            assert result is None

    def test_index_dataset_entries(self, fake_missions_dir):
        """Index entries have expected fields."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            index = list_cached_datasets("PSP")
            ds = index["datasets"][0]
            assert "id" in ds
            assert "description" in ds
            assert "start_date" in ds
            assert "stop_date" in ds
            assert "parameter_count" in ds
            assert "instrument" in ds


class TestBrowseDatasets:
    def test_browse_datasets_no_exclusion_file_returns_all(self, fake_missions_dir):
        """Without an exclusion file, browse_datasets returns all datasets."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            datasets = browse_datasets("PSP")
            assert datasets is not None
            assert len(datasets) == 2  # All datasets from SAMPLE_INDEX

    def test_browse_datasets_filters_calibration(self, fake_missions_dir):
        """browse_datasets filters out datasets matching exclusion IDs."""
        # Create exclusion file
        exclude_path = fake_missions_dir / "psp" / "metadata" / "_calibration_exclude.json"
        exclude_data = {
            "description": "Test exclusions",
            "patterns": [],
            "ids": ["PSP_FLD_L2_MAG_RTN_1MIN"],
        }
        exclude_path.write_text(json.dumps(exclude_data), encoding="utf-8")

        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            datasets = browse_datasets("PSP")
            assert datasets is not None
            assert len(datasets) == 1
            assert datasets[0]["id"] == "PSP_SWP_SPC_L3I"

    def test_browse_datasets_pattern_matching(self, fake_missions_dir):
        """browse_datasets filters out datasets matching glob patterns."""
        exclude_path = fake_missions_dir / "psp" / "metadata" / "_calibration_exclude.json"
        exclude_data = {
            "description": "Test pattern exclusions",
            "patterns": ["PSP_FLD_*"],
            "ids": [],
        }
        exclude_path.write_text(json.dumps(exclude_data), encoding="utf-8")

        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            datasets = browse_datasets("PSP")
            assert datasets is not None
            assert len(datasets) == 1
            assert datasets[0]["id"] == "PSP_SWP_SPC_L3I"

    def test_browse_datasets_returns_none_for_missing_mission(self, fake_missions_dir):
        """browse_datasets returns None when no _index.json exists."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = browse_datasets("NONEXISTENT")
            assert result is None


class TestMultiSourceMerge:
    """Tests for merging _index.json from multiple source directories (CDAWeb + PPI)."""

    @pytest.fixture
    def multi_source_dirs(self, tmp_path):
        """Create two source dirs with overlapping missions (like CDAWeb + PPI)."""
        cdaweb_dir = tmp_path / "cdaweb"
        ppi_dir = tmp_path / "ppi"

        # CDAWeb: juno with 2 ephemeris datasets
        juno_cdaweb = cdaweb_dir / "juno" / "metadata"
        juno_cdaweb.mkdir(parents=True)
        cdaweb_index = {
            "mission_id": "JUNO",
            "dataset_count": 2,
            "generated_at": "2026-02-15T00:00:00Z",
            "datasets": [
                {"id": "JUNO_HELIO1DAY_POSITION", "description": "Position daily",
                 "start_date": "2011-08-06", "stop_date": "2025-10-20",
                 "parameter_count": 7, "instrument": ""},
                {"id": "JUNO_HELIO1HR_POSITION", "description": "Position hourly",
                 "start_date": "2011-08-06", "stop_date": "2028-09-30",
                 "parameter_count": 7, "instrument": ""},
            ],
        }
        (juno_cdaweb / "_index.json").write_text(
            json.dumps(cdaweb_index), encoding="utf-8"
        )

        # PPI: juno with 2 plasma waves datasets
        juno_ppi = ppi_dir / "juno" / "metadata"
        juno_ppi.mkdir(parents=True)
        ppi_index = {
            "mission_id": "JUNO",
            "dataset_count": 2,
            "generated_at": "2026-02-15T00:00:00Z",
            "datasets": [
                {"id": "urn:nasa:pds:juno_waves:data_jupiter", "description": "Waves Jupiter",
                 "start_date": "2016-08-27", "stop_date": "2024-03-08",
                 "parameter_count": 26, "instrument": ""},
                {"id": "urn:nasa:pds:juno_waves:data_io", "description": "Waves Io",
                 "start_date": "2023-12-30", "stop_date": "2024-02-04",
                 "parameter_count": 31, "instrument": ""},
            ],
        }
        (juno_ppi / "_index.json").write_text(
            json.dumps(ppi_index), encoding="utf-8"
        )

        # PPI-only mission: galileo (no CDAWeb counterpart)
        galileo_ppi = ppi_dir / "galileo" / "metadata"
        galileo_ppi.mkdir(parents=True)
        galileo_index = {
            "mission_id": "GALILEO",
            "dataset_count": 1,
            "generated_at": "2026-02-15T00:00:00Z",
            "datasets": [
                {"id": "urn:nasa:pds:galileo_mag:data", "description": "Galileo MAG",
                 "start_date": "1995-01-01", "stop_date": "2003-09-21",
                 "parameter_count": 5, "instrument": ""},
            ],
        }
        (galileo_ppi / "_index.json").write_text(
            json.dumps(galileo_index), encoding="utf-8"
        )

        return cdaweb_dir, ppi_dir

    def test_list_cached_datasets_separate_sources(self, multi_source_dirs):
        """list_cached_datasets returns only CDAWeb data for non-PPI mission IDs."""
        cdaweb_dir, ppi_dir = multi_source_dirs
        with patch("knowledge.metadata_client._SOURCE_DIRS", [cdaweb_dir, ppi_dir]):
            # "JUNO" (no _PPI suffix) returns only CDAWeb datasets
            index = list_cached_datasets("JUNO")
            assert index is not None
            assert index["dataset_count"] == 2
            ds_ids = [ds["id"] for ds in index["datasets"]]
            assert "JUNO_HELIO1DAY_POSITION" in ds_ids
            assert "JUNO_HELIO1HR_POSITION" in ds_ids

    def test_list_cached_datasets_ppi_suffix(self, multi_source_dirs):
        """list_cached_datasets with _PPI suffix returns only PPI data."""
        cdaweb_dir, ppi_dir = multi_source_dirs
        with patch("knowledge.metadata_client._SOURCE_DIRS", [cdaweb_dir, ppi_dir]):
            index = list_cached_datasets("JUNO_PPI")
            assert index is not None
            assert index["dataset_count"] == 2
            ds_ids = [ds["id"] for ds in index["datasets"]]
            assert "urn:nasa:pds:juno_waves:data_jupiter" in ds_ids
            assert "urn:nasa:pds:juno_waves:data_io" in ds_ids

    def test_list_cached_datasets_ppi_only_mission(self, multi_source_dirs):
        """PPI-only missions (no CDAWeb counterpart) are found via fallback."""
        cdaweb_dir, ppi_dir = multi_source_dirs
        with patch("knowledge.metadata_client._SOURCE_DIRS", [cdaweb_dir, ppi_dir]):
            index = list_cached_datasets("GALILEO")
            assert index is not None
            assert index["dataset_count"] == 1
            assert index["datasets"][0]["id"] == "urn:nasa:pds:galileo_mag:data"

    def test_browse_datasets_separate_sources(self, multi_source_dirs):
        """browse_datasets returns only CDAWeb data for non-PPI mission IDs."""
        cdaweb_dir, ppi_dir = multi_source_dirs
        with patch("knowledge.metadata_client._SOURCE_DIRS", [cdaweb_dir, ppi_dir]):
            datasets = browse_datasets("JUNO")
            assert datasets is not None
            assert len(datasets) == 2
            ds_ids = [ds["id"] for ds in datasets]
            assert "JUNO_HELIO1DAY_POSITION" in ds_ids
            assert "JUNO_HELIO1HR_POSITION" in ds_ids


class TestListMissions:
    def test_returns_correct_structure(self, fake_missions_dir):
        """list_missions returns list of dicts with mission_id and dataset_count."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            missions = list_missions()
            assert isinstance(missions, list)
            assert len(missions) >= 2  # PSP and ACE
            for m in missions:
                assert "mission_id" in m
                assert "dataset_count" in m

    def test_includes_missions_with_index(self, fake_missions_dir):
        """list_missions includes missions that have _index.json."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            missions = list_missions()
            ids = [m["mission_id"] for m in missions]
            assert "PSP" in ids
            assert "ACE" in ids

    def test_skips_dirs_without_index(self, fake_missions_dir):
        """list_missions skips mission dirs that have no _index.json."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            missions = list_missions()
            ids = [m["mission_id"] for m in missions]
            assert "EMPTY_MISSION" not in ids

    def test_skips_non_directories(self, fake_missions_dir):
        """list_missions skips non-directory entries (like psp.json)."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            # Should not crash even with psp.json file at top level
            missions = list_missions()
            assert len(missions) >= 2


class TestValidateDatasetId:
    def test_valid_cached_dataset(self, fake_missions_dir):
        """validate_dataset_id returns valid=True for cached dataset with mission_id."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = validate_dataset_id("PSP_FLD_L2_MAG_RTN_1MIN")
            assert result["valid"] is True
            assert result["mission_id"] == "PSP"
            assert "found" in result["message"].lower()

    def test_valid_ace_dataset(self, fake_missions_dir):
        """validate_dataset_id works for ACE dataset too."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = validate_dataset_id("AC_H2_MFI")
            assert result["valid"] is True
            assert result["mission_id"] == "ACE"

    def test_invalid_dataset(self, fake_missions_dir):
        """validate_dataset_id returns valid=False for unknown dataset."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = validate_dataset_id("FAKE_DATASET_XYZ")
            assert result["valid"] is False
            assert result["mission_id"] is None
            assert "not found" in result["message"].lower()
            assert "browse_datasets" in result["message"]


class TestValidateParameterId:
    def test_valid_parameter(self, fake_missions_dir):
        """validate_parameter_id returns valid=True for known parameter."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = validate_parameter_id("AC_H2_MFI", "BGSEc")
            assert result["valid"] is True
            assert "BGSEc" in result["available_parameters"]
            assert "Magnitude" in result["available_parameters"]

    def test_invalid_parameter(self, fake_missions_dir):
        """validate_parameter_id returns valid=False with available parameter list."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = validate_parameter_id("AC_H2_MFI", "NonexistentParam")
            assert result["valid"] is False
            assert "BGSEc" in result["available_parameters"]
            assert "Magnitude" in result["available_parameters"]
            assert "NonexistentParam" in result["message"]
            assert "Available parameters" in result["message"]

    def test_invalid_dataset(self, fake_missions_dir):
        """validate_parameter_id returns valid=False for unknown dataset."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = validate_parameter_id("FAKE_DATASET", "BGSEc")
            assert result["valid"] is False
            assert result["available_parameters"] == []
            assert "not found" in result["message"].lower()

    def test_excludes_time_parameter(self, fake_missions_dir):
        """validate_parameter_id excludes Time from available parameters."""
        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_missions_dir]):
            result = validate_parameter_id("AC_H2_MFI", "BGSEc")
            assert "Time" not in result["available_parameters"]
