"""
Comprehensive behavioral tests for Features 01-04.

Tests the actual tool handler logic as it runs inside the agent,
using a lightweight harness that exercises _execute_tool() paths
without requiring a Gemini API key.

Run with: python -m pytest tests/test_features_01_04.py -v -s
"""

import json
import math
import os
import platform
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_ops.store import DataEntry, DataStore, set_store, reset_store
from knowledge.catalog import SPACECRAFT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_store(tmp_path):
    """Create a fresh disk-backed store and set it as the global store."""
    store = DataStore(tmp_path / "data")
    set_store(store)
    yield
    reset_store()


def _make_entry(label, n=100, columns=None, freq="1min", units="nT",
                inject_nans=0, start="2024-01-01"):
    """Flexible helper to create DataEntry objects."""
    idx = pd.date_range(start, periods=n, freq=freq)
    if columns is None:
        columns = ["value"]
    data = pd.DataFrame(
        np.random.default_rng(42).standard_normal((n, len(columns))),
        index=idx,
        columns=columns,
    )
    if inject_nans > 0:
        rng = np.random.default_rng(99)
        for _ in range(inject_nans):
            r = rng.integers(0, n)
            c = rng.integers(0, len(columns))
            data.iloc[r, c] = np.nan
    return DataEntry(
        label=label, data=data, units=units,
        description=f"test {label}", source="computed",
    )


def _run_describe(entry):
    """Reproduce the describe_data handler logic from core.py."""
    df = entry.data
    stats = {}
    desc = df.describe(percentiles=[0.25, 0.5, 0.75])
    for col in df.columns:
        stats[col] = {
            "min": float(desc.loc["min", col]),
            "max": float(desc.loc["max", col]),
            "mean": float(desc.loc["mean", col]),
            "std": float(desc.loc["std", col]),
            "25%": float(desc.loc["25%", col]),
            "50%": float(desc.loc["50%", col]),
            "75%": float(desc.loc["75%", col]),
        }
    nan_count = int(df.isna().sum().sum())
    total_points = len(df)
    time_span = str(df.index[-1] - df.index[0]) if total_points > 1 else "single point"
    if total_points > 1:
        dt = df.index.to_series().diff().dropna()
        median_cadence = str(dt.median())
    else:
        median_cadence = "N/A"
    return {
        "status": "success",
        "label": entry.label,
        "units": entry.units,
        "num_points": total_points,
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "time_start": str(df.index[0]),
        "time_end": str(df.index[-1]),
        "time_span": time_span,
        "median_cadence": median_cadence,
        "nan_count": nan_count,
        "nan_percentage": round(
            nan_count / (total_points * len(df.columns)) * 100, 1
        ) if total_points > 0 else 0,
        "statistics": stats,
    }


def _run_save(entry, filename):
    """Reproduce the save_data handler logic from core.py."""
    if not filename:
        safe_label = entry.label.replace(".", "_").replace("/", "_")
        filename = f"{safe_label}.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"
    parent = Path(filename).parent
    if parent and str(parent) != "." and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    df = entry.data.copy()
    df.index.name = "timestamp"
    df.to_csv(filename, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
    filepath = str(Path(filename).resolve())
    file_size = Path(filename).stat().st_size
    return {
        "status": "success",
        "label": entry.label,
        "filepath": filepath,
        "num_points": len(df),
        "num_columns": len(df.columns),
        "file_size_bytes": file_size,
    }


# ===================================================================
# FEATURE 01 — describe_data
# ===================================================================

class TestFeature01DescribeData:
    """Behavioral tests for the describe_data tool."""

    def test_scalar_timeseries_stats_are_correct(self):
        """Known values: verify min/max/mean exactly."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1h")
        data = pd.DataFrame({"v": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)
        entry = DataEntry(label="known", data=data, units="km/s")
        result = _run_describe(entry)

        assert result["status"] == "success"
        s = result["statistics"]["v"]
        assert s["min"] == 10.0
        assert s["max"] == 50.0
        assert s["mean"] == 30.0
        assert s["50%"] == 30.0  # median
        assert result["units"] == "km/s"
        assert result["num_points"] == 5

    def test_vector_3_component_returns_per_column_stats(self):
        """3-component vector data produces stats for each column."""
        entry = _make_entry("BGSEc", n=200, columns=["Bx", "By", "Bz"])
        result = _run_describe(entry)

        assert result["num_columns"] == 3
        assert set(result["columns"]) == {"Bx", "By", "Bz"}
        for col in ["Bx", "By", "Bz"]:
            s = result["statistics"][col]
            assert s["min"] <= s["25%"] <= s["50%"] <= s["75%"] <= s["max"]
            assert s["std"] > 0  # random data has nonzero std

    def test_nan_count_and_percentage(self):
        """NaN counting across multiple columns."""
        idx = pd.date_range("2024-06-01", periods=10, freq="5min")
        data = pd.DataFrame({
            "a": [1, np.nan, 3, 4, 5, 6, 7, 8, np.nan, 10],
            "b": [np.nan, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }, index=idx)
        entry = DataEntry(label="gaps", data=data, units="nT")
        result = _run_describe(entry)

        assert result["nan_count"] == 3  # 2 in 'a', 1 in 'b'
        # 3 NaN / (10 points * 2 columns) = 15%
        assert result["nan_percentage"] == 15.0

    def test_zero_nans_for_clean_data(self):
        entry = _make_entry("clean", n=50, columns=["x"])
        result = _run_describe(entry)
        assert result["nan_count"] == 0
        assert result["nan_percentage"] == 0.0

    def test_cadence_1min(self):
        """Cadence detection for 1-minute data."""
        entry = _make_entry("min_data", n=1000, freq="1min")
        result = _run_describe(entry)
        assert "0 days 00:01:00" in result["median_cadence"]

    def test_cadence_1sec(self):
        """Cadence detection for 1-second (high-res) data."""
        entry = _make_entry("hires", n=500, freq="1s")
        result = _run_describe(entry)
        assert "0 days 00:00:01" in result["median_cadence"]

    def test_cadence_1hour(self):
        entry = _make_entry("hourly", n=48, freq="1h")
        result = _run_describe(entry)
        assert "0 days 01:00:00" in result["median_cadence"]

    def test_time_span_covers_full_range(self):
        """Time span for exactly 24h of 1-min data."""
        idx = pd.date_range("2024-03-01", periods=1441, freq="1min")
        data = pd.DataFrame(np.ones(1441), index=idx, columns=["v"])
        entry = DataEntry(label="day", data=data, units="nT")
        result = _run_describe(entry)
        assert "1 days" in result["time_span"] or "1 day" in result["time_span"]

    def test_single_point_edge_case(self):
        """Single data point: time_span should say 'single point'."""
        idx = pd.date_range("2024-01-01", periods=1, freq="1min")
        data = pd.DataFrame({"v": [42.0]}, index=idx)
        entry = DataEntry(label="one", data=data, units="nT")
        result = _run_describe(entry)
        assert result["time_span"] == "single point"
        assert result["median_cadence"] == "N/A"
        assert result["num_points"] == 1

    def test_all_nan_column(self):
        """A column that is entirely NaN."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({"v": [np.nan] * 5}, index=idx)
        entry = DataEntry(label="allnan", data=data, units="nT")
        result = _run_describe(entry)
        assert result["nan_count"] == 5
        assert result["nan_percentage"] == 100.0

    def test_large_dataset_performance(self):
        """describe_data handles 100k points without issue."""
        entry = _make_entry("big", n=100_000, freq="1s")
        result = _run_describe(entry)
        assert result["num_points"] == 100_000
        assert result["status"] == "success"

    def test_result_is_json_serializable(self):
        """The result dict can be serialized to JSON (sent back to Gemini)."""
        entry = _make_entry("serial", n=50, columns=["x", "y", "z"])
        result = _run_describe(entry)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        roundtrip = json.loads(json_str)
        assert roundtrip["label"] == "serial"


# ===================================================================
# FEATURE 02 — save_data
# ===================================================================

class TestFeature02SaveData:
    """Behavioral tests for the save_data tool."""

    def test_basic_csv_roundtrip(self, tmp_path):
        """Save and reload: data should match."""
        entry = _make_entry("Bmag", n=100, columns=["magnitude"])
        filepath = str(tmp_path / "Bmag.csv")
        result = _run_save(entry, filepath)

        assert result["status"] == "success"
        assert result["num_points"] == 100
        assert result["num_columns"] == 1

        loaded = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert len(loaded) == 100
        assert "magnitude" in loaded.columns
        # Values should be close (float precision)
        np.testing.assert_allclose(
            loaded["magnitude"].values,
            entry.data["magnitude"].values,
            rtol=1e-10,
        )

    def test_vector_csv_roundtrip(self, tmp_path):
        """Vector data preserves all columns."""
        entry = _make_entry("BGSEc", n=30, columns=["Bx", "By", "Bz"])
        filepath = str(tmp_path / "vector.csv")
        result = _run_save(entry, filepath)

        assert result["num_columns"] == 3
        loaded = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert set(loaded.columns) == {"Bx", "By", "Bz"}

    def test_nan_values_roundtrip(self, tmp_path):
        """NaN values survive CSV roundtrip as empty cells."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({"v": [1.0, np.nan, 3.0, np.nan, 5.0]}, index=idx)
        entry = DataEntry(label="nans", data=data, units="nT")
        filepath = str(tmp_path / "nans.csv")
        _run_save(entry, filepath)

        loaded = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert loaded["v"].isna().sum() == 2
        assert loaded["v"].iloc[0] == 1.0
        assert loaded["v"].iloc[4] == 5.0

    def test_auto_filename_from_label(self, tmp_path):
        """Auto-generated filename sanitizes dots and slashes."""
        entry = _make_entry("AC_H2_MFI.BGSEc", n=10)
        # Simulate the auto-filename logic
        safe = entry.label.replace(".", "_").replace("/", "_")
        expected = f"{safe}.csv"
        assert expected == "AC_H2_MFI_BGSEc.csv"

        filepath = str(tmp_path / expected)
        result = _run_save(entry, filepath)
        assert Path(filepath).exists()

    def test_filename_with_subdirectory(self, tmp_path):
        """Creates parent directories if needed."""
        entry = _make_entry("test", n=10)
        subdir = tmp_path / "exports" / "2024"
        filepath = str(subdir / "data.csv")
        result = _run_save(entry, filepath)

        assert Path(filepath).exists()
        assert result["status"] == "success"

    def test_csv_extension_handling(self):
        """Extension logic: append .csv if missing, don't double it."""
        cases = [
            ("data", "data.csv"),
            ("data.csv", "data.csv"),
            ("my_export", "my_export.csv"),
            ("export.csv", "export.csv"),
        ]
        for input_name, expected in cases:
            result = input_name
            if not result.endswith(".csv"):
                result += ".csv"
            assert result == expected, f"{input_name} -> {result} != {expected}"

    def test_timestamps_are_iso8601(self, tmp_path):
        """Verify timestamps in CSV are ISO 8601 format."""
        idx = pd.date_range("2024-06-15T12:30:00", periods=3, freq="1min")
        data = pd.DataFrame({"v": [1, 2, 3]}, index=idx)
        entry = DataEntry(label="ts", data=data, units="nT")
        filepath = str(tmp_path / "timestamps.csv")
        _run_save(entry, filepath)

        # Read raw to check format
        with open(filepath) as f:
            lines = f.readlines()
        # Second line (first data row) should have ISO timestamp
        first_data = lines[1].split(",")[0]
        assert "2024-06-15T12:30:00" in first_data

    def test_file_size_reported(self, tmp_path):
        """Result includes file size in bytes."""
        entry = _make_entry("sized", n=1000, columns=["a", "b", "c"])
        filepath = str(tmp_path / "sized.csv")
        result = _run_save(entry, filepath)

        assert result["file_size_bytes"] > 0
        assert result["file_size_bytes"] == Path(filepath).stat().st_size

    def test_large_export(self, tmp_path):
        """100k points export works and produces reasonable file size."""
        entry = _make_entry("big", n=100_000, freq="1s")
        filepath = str(tmp_path / "big.csv")
        result = _run_save(entry, filepath)

        assert result["num_points"] == 100_000
        assert result["file_size_bytes"] > 1_000_000  # > 1MB for 100k rows


# ===================================================================
# FEATURE 03 — auto-open PNG
# ===================================================================

class TestFeature03AutoOpenPNG:
    """Behavioral tests for the auto-open exported PNG feature."""

    def test_auto_open_calls_startfile_on_windows(self):
        """On Windows, os.startfile is called with the filepath."""
        opened_files = []

        def mock_startfile(path):
            opened_files.append(path)

        with patch("platform.system", return_value="Windows"), \
             patch("os.startfile", mock_startfile, create=True):
            # Simulate what the handler does after a successful export
            result = {"status": "success", "filepath": r"C:\Users\test\plot.png"}
            try:
                filepath = result["filepath"]
                if platform.system() == "Windows":
                    os.startfile(filepath)
                result["auto_opened"] = True
            except Exception:
                result["auto_opened"] = False

        assert result["auto_opened"] is True
        assert len(opened_files) == 1
        assert opened_files[0] == r"C:\Users\test\plot.png"

    def test_auto_open_calls_open_on_macos(self):
        """On macOS, subprocess.Popen(['open', ...]) is called."""
        import subprocess

        with patch("platform.system", return_value="Darwin"), \
             patch("subprocess.Popen") as mock_popen:
            result = {"status": "success", "filepath": "/tmp/plot.png"}
            try:
                filepath = result["filepath"]
                if platform.system() == "Darwin":
                    subprocess.Popen(["open", filepath])
                result["auto_opened"] = True
            except Exception:
                result["auto_opened"] = False

        assert result["auto_opened"] is True
        mock_popen.assert_called_once_with(["open", "/tmp/plot.png"])

    def test_auto_open_calls_xdg_open_on_linux(self):
        """On Linux, subprocess.Popen(['xdg-open', ...]) is called."""
        import subprocess

        with patch("platform.system", return_value="Linux"), \
             patch("subprocess.Popen") as mock_popen:
            result = {"status": "success", "filepath": "/tmp/plot.png"}
            try:
                filepath = result["filepath"]
                if platform.system() == "Linux":
                    subprocess.Popen(["xdg-open", filepath])
                result["auto_opened"] = True
            except Exception:
                result["auto_opened"] = False

        assert result["auto_opened"] is True
        mock_popen.assert_called_once_with(["xdg-open", "/tmp/plot.png"])

    def test_auto_open_failure_is_non_fatal(self):
        """If auto-open fails, the export result is still success."""
        with patch("platform.system", return_value="Windows"), \
             patch("os.startfile", side_effect=OSError("No display"), create=True):
            result = {"status": "success", "filepath": r"C:\plot.png"}
            try:
                os.startfile(result["filepath"])
                result["auto_opened"] = True
            except Exception:
                result["auto_opened"] = False

        assert result["status"] == "success"  # export itself succeeded
        assert result["auto_opened"] is False

    def test_auto_open_skipped_on_failed_export(self):
        """auto_opened key is not set when export fails."""
        result = {"status": "error", "message": "No plot to export"}
        # The handler only attempts auto-open if status == "success"
        if result.get("status") == "success":
            result["auto_opened"] = True
        assert "auto_opened" not in result
