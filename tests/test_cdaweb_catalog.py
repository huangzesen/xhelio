"""Tests for knowledge.catalog_search — full dataset catalog search."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from knowledge.catalog_search import (
    get_full_catalog,
    search_catalog,
    get_catalog_stats,
    _substring_search,
    CATALOG_CACHE,
    CACHE_TTL_SECONDS,
)


# Sample catalog data for testing
SAMPLE_CATALOG = [
    {"id": "AC_H2_MFI", "title": "ACE Magnetometer 16-Second Data in GSE coordinates"},
    {"id": "AC_H0_SWE", "title": "ACE Solar Wind Electron, Proton, and Alpha Monitor"},
    {"id": "C1_CP_FGM_SPIN", "title": "Cluster C1 FGM Spin Resolution Magnetic Field"},
    {"id": "C2_CP_FGM_SPIN", "title": "Cluster C2 FGM Spin Resolution Magnetic Field"},
    {"id": "THA_L2_FGM", "title": "THEMIS-A Level 2 FGM Magnetic Field Data"},
    {"id": "VG2_48S_MAG", "title": "Voyager 2 48-Second Magnetic Field Data"},
    {"id": "GOES16_EXIS_L1B", "title": "GOES 16 EXIS L1b Extreme Ultraviolet"},
    {"id": "PSP_FLD_L2_MAG_RTN_1MIN", "title": "PSP FIELDS Fluxgate Magnetometer"},
    {"id": "OMNI_HRO_1MIN", "title": "OMNI High Resolution 1-Minute Data"},
    {"id": "RBSPA_REL04_ECT-HOPE", "title": "Van Allen Probe A ECT-HOPE"},
]


# ── search_catalog (substring mode) ─────────────────────────────────

@patch("config.CATALOG_SEARCH_METHOD", "substring")
class TestSearchCatalog:
    """Test catalog search in substring mode with mock data."""

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    def test_search_by_spacecraft(self, mock_catalog):
        results = search_catalog("cluster")
        assert len(results) == 2
        assert all("cluster" in r["title"].lower() or "C" in r["id"] for r in results)

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    def test_search_by_instrument(self, mock_catalog):
        results = search_catalog("fgm")
        assert len(results) >= 2
        assert all("fgm" in r["id"].lower() or "fgm" in r["title"].lower() for r in results)

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    def test_search_multi_word(self, mock_catalog):
        results = search_catalog("cluster magnetic")
        assert len(results) == 2

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    def test_search_case_insensitive(self, mock_catalog):
        results = search_catalog("VOYAGER")
        assert len(results) == 1
        assert results[0]["id"] == "VG2_48S_MAG"

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    def test_search_no_match(self, mock_catalog):
        results = search_catalog("nonexistent_spacecraft_xyzzy")
        assert len(results) == 0

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    def test_search_max_results(self, mock_catalog):
        results = search_catalog("magnetic", max_results=2)
        assert len(results) <= 2

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    def test_search_empty_query(self, mock_catalog):
        results = search_catalog("")
        assert len(results) == 0

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    def test_search_result_format(self, mock_catalog):
        results = search_catalog("ace")
        assert len(results) >= 1
        for r in results:
            assert "id" in r
            assert "title" in r

    @patch("knowledge.catalog_search.get_full_catalog", return_value=[])
    def test_search_empty_catalog(self, mock_catalog):
        results = search_catalog("anything")
        assert len(results) == 0

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    def test_search_by_dataset_id(self, mock_catalog):
        results = search_catalog("AC_H2_MFI")
        assert len(results) == 1
        assert results[0]["id"] == "AC_H2_MFI"


# ── search_catalog (semantic dispatch) ───────────────────────────────

class TestSemanticSearch:
    """Test semantic search dispatch and fallback logic."""

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    @patch("config.CATALOG_SEARCH_METHOD", "substring")
    def test_uses_substring_when_configured(self, mock_catalog):
        """Explicit substring config bypasses fastembed entirely."""
        with patch("knowledge.catalog_search._ensure_fastembed") as mock_fe:
            results = search_catalog("ace")
            mock_fe.assert_not_called()
        assert len(results) >= 1

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    @patch("config.CATALOG_SEARCH_METHOD", "semantic")
    @patch("knowledge.catalog_search._ensure_fastembed", return_value=False)
    def test_falls_back_when_fastembed_unavailable(self, mock_fe, mock_catalog):
        """When fastembed is not installed, falls back to substring."""
        results = search_catalog("ace")
        assert len(results) >= 1
        assert all("ac" in r["id"].lower() for r in results)

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    @patch("config.CATALOG_SEARCH_METHOD", "semantic")
    @patch("knowledge.catalog_search._ensure_fastembed", return_value=True)
    @patch("knowledge.catalog_search._semantic_search")
    def test_dispatches_to_semantic_when_available(self, mock_sem, mock_fe, mock_catalog):
        """When fastembed is available and config=semantic, uses _semantic_search."""
        mock_sem.return_value = [{"id": "FAKE_DS", "title": "Fake"}]
        results = search_catalog("solar wind")
        mock_sem.assert_called_once_with("solar wind", SAMPLE_CATALOG, 20)
        assert results == [{"id": "FAKE_DS", "title": "Fake"}]

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    @patch("config.CATALOG_SEARCH_METHOD", "semantic")
    def test_empty_query_returns_empty(self, mock_catalog):
        """Empty query returns [] regardless of search method."""
        results = search_catalog("")
        assert results == []

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    @patch("config.CATALOG_SEARCH_METHOD", "semantic")
    def test_whitespace_query_returns_empty(self, mock_catalog):
        """Whitespace-only query returns [] regardless of search method."""
        results = search_catalog("   ")
        assert results == []


# ── get_full_catalog ────────────────────────────────────────────────

class TestGetFullCatalog:
    """Test catalog fetching and caching."""

    @patch("knowledge.catalog_search._get_ppi_entries", return_value=[])
    @patch("knowledge.catalog_search.CATALOG_CACHE")
    def test_fetches_when_no_cache(self, mock_cache_path, mock_ppi):
        mock_cache_path.exists.return_value = False
        mock_cache_path.parent.mkdir = MagicMock()

        # Mock _fetch_from_cdas_rest to return sample catalog
        with patch("knowledge.catalog_search._fetch_from_cdas_rest",
                   return_value=SAMPLE_CATALOG):
            mock_file = MagicMock()
            with patch("builtins.open", return_value=mock_file):
                result = get_full_catalog()

        assert len(result) == len(SAMPLE_CATALOG)

    @patch("knowledge.catalog_search._get_ppi_entries", return_value=[])
    @patch("knowledge.catalog_search.CATALOG_CACHE")
    def test_uses_fresh_cache(self, mock_cache_path, mock_ppi):
        mock_cache_path.exists.return_value = True
        mock_cache_path.stat.return_value = MagicMock(
            st_mtime=time.time() - 100  # 100 seconds old = fresh
        )

        cache_content = json.dumps({"catalog": SAMPLE_CATALOG})
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = cache_content

        with patch("builtins.open", return_value=mock_file):
            with patch("json.load", return_value={"catalog": SAMPLE_CATALOG}):
                result = get_full_catalog()

        assert len(result) == len(SAMPLE_CATALOG)


# ── get_catalog_stats ───────────────────────────────────────────────

class TestGetCatalogStats:
    """Test catalog statistics."""

    @patch("knowledge.catalog_search.get_full_catalog", return_value=SAMPLE_CATALOG)
    @patch("knowledge.catalog_search.CATALOG_CACHE")
    def test_stats_with_cache(self, mock_cache_path, mock_catalog):
        mock_cache_path.exists.return_value = True
        mock_cache_path.stat.return_value = MagicMock(
            st_mtime=time.time() - 3600  # 1 hour old
        )

        stats = get_catalog_stats()
        assert stats["total_datasets"] == len(SAMPLE_CATALOG)
        assert stats["cache_exists"] is True
        assert stats["cache_age_hours"] is not None
        assert 0.9 <= stats["cache_age_hours"] <= 1.1

    @patch("knowledge.catalog_search.get_full_catalog", return_value=[])
    @patch("knowledge.catalog_search.CATALOG_CACHE")
    def test_stats_without_cache(self, mock_cache_path, mock_catalog):
        mock_cache_path.exists.return_value = False

        stats = get_catalog_stats()
        assert stats["total_datasets"] == 0
        assert stats["cache_exists"] is False
        assert stats["cache_age_hours"] is None
