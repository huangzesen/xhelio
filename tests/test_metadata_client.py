"""
Tests for the metadata client.

Run with: python -m pytest tests/test_metadata_client.py

Note: These tests require network access to CDAWeb.
Some tests are marked slow and can be skipped with: pytest -m "not slow"
"""

import json
import pytest
import requests
from unittest.mock import patch, MagicMock

from knowledge.metadata_client import (
    get_dataset_info,
    list_parameters,
    get_dataset_time_range,
    clear_cache,
)


def _mock_404(*args, **kwargs):
    """Return a 404 response to simulate server rejecting invalid dataset."""
    resp = MagicMock()
    resp.status_code = 404
    resp.raise_for_status.side_effect = requests.HTTPError(
        "404 Not Found", response=resp
    )
    return resp


@pytest.fixture(autouse=True)
def clear_metadata_cache():
    """Clear cache before each test."""
    clear_cache()


class TestGetDatasetInfo:
    @pytest.mark.slow
    def test_fetch_psp_mag_info(self):
        """Test fetching PSP MAG dataset info."""
        info = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN")
        assert info is not None
        assert "startDate" in info
        assert "stopDate" in info
        assert "parameters" in info
        assert len(info["parameters"]) > 0

    @pytest.mark.slow
    def test_fetch_ace_mag_info(self):
        """Test fetching ACE MAG dataset info."""
        info = get_dataset_info("AC_H2_MFI")
        assert info is not None
        assert "parameters" in info

    def test_caching_works(self):
        """Test that caching prevents duplicate requests."""
        # First call populates cache
        info1 = get_dataset_info("AC_H2_MFI")
        # Second call should use cache
        info2 = get_dataset_info("AC_H2_MFI", use_cache=True)
        assert info1 == info2

    def test_invalid_dataset_raises(self):
        """Test that invalid dataset ID raises an error."""
        with patch("knowledge.metadata_client.requests.get", side_effect=_mock_404):
            with pytest.raises(Exception):
                get_dataset_info("INVALID_DATASET_XYZ_123")


class TestListParameters:
    @pytest.mark.slow
    def test_list_psp_mag_parameters(self):
        """Test listing parameters for PSP MAG."""
        params = list_parameters("PSP_FLD_L2_MAG_RTN_1MIN")
        assert len(params) > 0

        # Check parameter structure
        for p in params:
            assert "name" in p
            assert "dataset_id" in p
            assert p["dataset_id"] == "PSP_FLD_L2_MAG_RTN_1MIN"

    @pytest.mark.slow
    def test_parameters_are_1d(self):
        """Test that returned parameters are 1D with size <= 3."""
        params = list_parameters("AC_H2_MFI")
        for p in params:
            assert len(p["size"]) == 1
            assert p["size"][0] <= 3

    @pytest.mark.slow
    def test_excludes_time_parameter(self):
        """Test that Time parameter is excluded."""
        params = list_parameters("AC_H2_MFI")
        names = [p["name"].lower() for p in params]
        assert "time" not in names

    def test_invalid_dataset_returns_empty(self):
        """Test that invalid dataset returns empty list."""
        with patch("knowledge.metadata_client.requests.get", side_effect=_mock_404):
            params = list_parameters("INVALID_DATASET_XYZ")
            assert params == []


class TestGetDatasetTimeRange:
    @pytest.mark.slow
    def test_get_time_range(self):
        """Test getting dataset time range."""
        time_range = get_dataset_time_range("AC_H2_MFI")
        assert time_range is not None
        assert "start" in time_range
        assert "stop" in time_range
        assert time_range["start"] is not None

    def test_invalid_dataset_returns_none(self):
        """Test that invalid dataset returns None."""
        with patch("knowledge.metadata_client.requests.get", side_effect=_mock_404):
            time_range = get_dataset_time_range("INVALID_XYZ")
            assert time_range is None


class TestLocalFirstCacheBehavior:
    """Test that get_dataset_info prefers local cache over network."""

    def test_local_cache_avoids_network(self, tmp_path):
        """When local cache exists, no network request is made."""
        fake_cdaweb = tmp_path / "envoys" / "cdaweb"
        psp_metadata = fake_cdaweb / "psp" / "metadata"
        psp_metadata.mkdir(parents=True)

        sample_info = {
            "startDate": "2018-10-06",
            "stopDate": "2025-12-31",
            "parameters": [
                {"name": "Time", "type": "isotime"},
                {"name": "test_param", "type": "double", "units": "nT"},
            ],
        }
        (psp_metadata / "PSP_FLD_L2_MAG_RTN_1MIN.json").write_text(
            json.dumps(sample_info), encoding="utf-8"
        )

        with patch("knowledge.metadata_client._SOURCE_DIRS", [fake_cdaweb]), \
             patch("knowledge.metadata_client.requests.get") as mock_get:
            info = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN")
            assert info["startDate"] == "2018-10-06"
            # Network should NOT have been called
            mock_get.assert_not_called()
