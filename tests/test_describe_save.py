"""
Tests for describe_data and save_data tool handlers.

Run with: python -m pytest tests/test_describe_save.py
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.store import DataEntry, DataStore, get_store, set_store, reset_store


@pytest.fixture(autouse=True)
def clean_store(tmp_path):
    """Reset the global store before each test and provide a disk-backed store."""
    reset_store()
    store = DataStore(tmp_path / "data")
    set_store(store)
    yield
    reset_store()


def _make_entry(label="test", n=100, vector=False, nan_count=0):
    """Helper to create a DataEntry for testing."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    if vector:
        data = pd.DataFrame(
            np.random.randn(n, 3), index=idx, columns=["x", "y", "z"]
        )
    else:
        data = pd.DataFrame(np.random.randn(n), index=idx, columns=["value"])
    # Inject NaNs if requested
    if nan_count > 0:
        flat = data.values.ravel()
        positions = np.random.choice(len(flat), size=min(nan_count, len(flat)), replace=False)
        flat[positions] = np.nan
    return DataEntry(
        label=label, data=data, units="nT",
        description="test entry", source="computed",
    )


def _describe(entry):
    """Simulate the describe_data handler logic on a DataEntry."""
    df = entry.data
    stats = {}
    desc = df.describe(percentiles=[0.25, 0.5, 0.75], include="all")
    for col in df.columns:
        if df[col].dtype.kind in ("f", "i", "u"):  # numeric
            col_stats = {
                "min": float(desc.loc["min", col]),
                "max": float(desc.loc["max", col]),
                "mean": float(desc.loc["mean", col]),
                "std": float(desc.loc["std", col]),
                "25%": float(desc.loc["25%", col]),
                "50%": float(desc.loc["50%", col]),
                "75%": float(desc.loc["75%", col]),
            }
        else:  # string/object/categorical columns
            col_stats = {
                "type": str(df[col].dtype),
                "count": int(desc.loc["count", col]),
                "unique": int(desc.loc["unique", col]) if "unique" in desc.index else None,
                "top": str(desc.loc["top", col]) if "top" in desc.index else None,
            }
        stats[col] = col_stats

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
        "nan_percentage": round(nan_count / (total_points * len(df.columns)) * 100, 1) if total_points > 0 else 0,
        "statistics": stats,
    }


class TestDescribeData:
    def test_scalar_stats(self):
        entry = _make_entry("Bmag", n=100, vector=False)
        result = _describe(entry)
        assert result["status"] == "success"
        assert result["label"] == "Bmag"
        assert result["num_points"] == 100
        assert result["num_columns"] == 1
        assert "value" in result["columns"]
        assert "value" in result["statistics"]
        stats = result["statistics"]["value"]
        assert stats["min"] <= stats["25%"] <= stats["50%"] <= stats["75%"] <= stats["max"]

    def test_vector_stats(self):
        entry = _make_entry("BGSEc", n=50, vector=True)
        result = _describe(entry)
        assert result["num_columns"] == 3
        assert set(result["columns"]) == {"x", "y", "z"}
        for col in ["x", "y", "z"]:
            assert col in result["statistics"]

    def test_nan_counting(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="1min")
        data = pd.DataFrame({"v": [1, 2, np.nan, 4, np.nan, 6, 7, 8, np.nan, 10]}, index=idx)
        entry = DataEntry(label="nans", data=data, units="nT")
        result = _describe(entry)
        assert result["nan_count"] == 3
        assert result["nan_percentage"] == 30.0

    def test_cadence_estimation(self):
        idx = pd.date_range("2024-01-01", periods=60, freq="1min")
        data = pd.DataFrame(np.ones(60), index=idx, columns=["v"])
        entry = DataEntry(label="cadence_test", data=data, units="nT")
        result = _describe(entry)
        # Median cadence should be 1 minute
        assert "0 days 00:01:00" in result["median_cadence"]

    def test_time_span(self):
        idx = pd.date_range("2024-01-01", periods=1440, freq="1min")
        data = pd.DataFrame(np.ones(1440), index=idx, columns=["v"])
        entry = DataEntry(label="span_test", data=data, units="nT")
        result = _describe(entry)
        assert "23:59:00" in result["time_span"]

    def test_string_columns(self):
        """describe_data should handle non-numeric columns without crashing."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D")
        data = pd.DataFrame({
            "class": ["X9.0", "X8.7", "X7.1", "X6.3", "X5.8"],
            "region": ["AR 3842", "AR 3664", "AR 3842", "AR 3590", "AR 3664"],
        }, index=idx)
        entry = DataEntry(label="flares", data=data, units="class")
        result = _describe(entry)
        assert result["status"] == "success"
        assert result["num_columns"] == 2
        # String columns should have type/count/unique/top instead of min/max/mean
        for col in ["class", "region"]:
            assert "type" in result["statistics"][col]
            assert "count" in result["statistics"][col]
            assert "unique" in result["statistics"][col]
            assert "min" not in result["statistics"][col]

    def test_mixed_columns(self):
        """describe_data should handle mix of numeric and string columns."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D")
        data = pd.DataFrame({
            "rank": [1, 2, 3, 4, 5],
            "class": ["X9.0", "X8.7", "X7.1", "X6.3", "X5.8"],
        }, index=idx)
        entry = DataEntry(label="mixed", data=data, units="")
        result = _describe(entry)
        assert result["status"] == "success"
        # rank is numeric
        assert "min" in result["statistics"]["rank"]
        assert result["statistics"]["rank"]["min"] == 1.0
        # class is string
        assert "type" in result["statistics"]["class"]
        assert "min" not in result["statistics"]["class"]

    def test_missing_label(self):
        store = get_store()
        entry = store.get("nonexistent")
        assert entry is None


class TestSaveData:
    def test_save_creates_file(self, tmp_path):
        entry = _make_entry("Bmag", n=50, vector=False)
        filepath = tmp_path / "test_output.csv"

        df = entry.data.copy()
        df.index.name = "timestamp"
        df.to_csv(filepath, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

        assert filepath.exists()
        loaded = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert len(loaded) == 50
        assert "value" in loaded.columns

    def test_save_vector_data(self, tmp_path):
        entry = _make_entry("BGSEc", n=30, vector=True)
        filepath = tmp_path / "vector.csv"

        df = entry.data.copy()
        df.index.name = "timestamp"
        df.to_csv(filepath, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

        loaded = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert len(loaded) == 30
        assert set(loaded.columns) == {"x", "y", "z"}

    def test_auto_filename_generation(self):
        label = "AC_H2_MFI.BGSEc"
        safe_label = label.replace(".", "_").replace("/", "_")
        filename = f"{safe_label}.csv"
        assert filename == "AC_H2_MFI_BGSEc.csv"

    def test_csv_extension_appended(self):
        filename = "mydata"
        if not filename.endswith(".csv"):
            filename += ".csv"
        assert filename == "mydata.csv"

    def test_csv_extension_not_doubled(self):
        filename = "mydata.csv"
        if not filename.endswith(".csv"):
            filename += ".csv"
        assert filename == "mydata.csv"

    def test_save_preserves_nans(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=5, freq="1min")
        data = pd.DataFrame({"v": [1.0, np.nan, 3.0, np.nan, 5.0]}, index=idx)
        entry = DataEntry(label="nans", data=data, units="nT")

        filepath = tmp_path / "nans.csv"
        df = entry.data.copy()
        df.index.name = "timestamp"
        df.to_csv(filepath, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

        loaded = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert loaded["v"].isna().sum() == 2
