"""Tests for the load_file tool handler (agent/tool_handlers/file_io.py)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from agent.tool_handlers.file_io import (
    _get_allowed_dirs,
    _load_file_to_dataframe,
    _try_parse_datetime_index,
    _validate_file_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_csv_with_datetime(tmp_path: Path) -> Path:
    """CSV with a 'timestamp' column that should be auto-detected."""
    p = tmp_path / "data.csv"
    p.write_text(
        textwrap.dedent("""\
        timestamp,value_a,value_b
        2024-01-01 00:00:00,1.0,10.0
        2024-01-01 01:00:00,2.0,20.0
        2024-01-01 02:00:00,3.0,30.0
        """)
    )
    return p


@pytest.fixture
def tmp_csv_custom_time(tmp_path: Path) -> Path:
    """CSV with a non-standard time column name."""
    p = tmp_path / "custom.csv"
    p.write_text(
        textwrap.dedent("""\
        obs_time,flux,energy
        2024-06-01 12:00:00,100.5,1.2
        2024-06-01 13:00:00,200.3,2.4
        """)
    )
    return p


@pytest.fixture
def tmp_csv_no_datetime(tmp_path: Path) -> Path:
    """CSV with no datetime columns at all."""
    p = tmp_path / "no_time.csv"
    p.write_text(
        textwrap.dedent("""\
        x,y,z
        1,2,3
        4,5,6
        7,8,9
        """)
    )
    return p


@pytest.fixture
def tmp_json_file(tmp_path: Path) -> Path:
    """JSON file in records orientation."""
    p = tmp_path / "data.json"
    records = [
        {"time": "2024-03-01", "speed": 400.0},
        {"time": "2024-03-02", "speed": 450.0},
    ]
    p.write_text(json.dumps(records))
    return p


@pytest.fixture
def tmp_tsv_file(tmp_path: Path) -> Path:
    """TSV file."""
    p = tmp_path / "data.tsv"
    p.write_text(
        textwrap.dedent("""\
        date\tvalue
        2024-01-01\t10
        2024-01-02\t20
        """)
    )
    return p


def _can_read_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TestLoadFileParsing
# ---------------------------------------------------------------------------


class TestLoadFileParsing:
    """Test file loading and datetime index parsing."""

    def test_csv_with_datetime_index(self, tmp_csv_with_datetime: Path):
        df = _load_file_to_dataframe(str(tmp_csv_with_datetime))
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"
        assert len(df) == 3
        assert list(df.columns) == ["value_a", "value_b"]

    def test_csv_with_explicit_time_column(self, tmp_csv_custom_time: Path):
        df = _load_file_to_dataframe(
            str(tmp_csv_custom_time), time_column="obs_time"
        )
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "obs_time"
        assert len(df) == 2

    def test_csv_without_datetime(self, tmp_csv_no_datetime: Path):
        df = _load_file_to_dataframe(str(tmp_csv_no_datetime))
        # Should NOT have a DatetimeIndex — values are plain integers
        assert not isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 3

    def test_json_loading(self, tmp_json_file: Path):
        df = _load_file_to_dataframe(str(tmp_json_file))
        assert len(df) == 2
        # 'time' column should be auto-detected as datetime index
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "speed" in df.columns

    def test_tsv_loading(self, tmp_tsv_file: Path):
        df = _load_file_to_dataframe(str(tmp_tsv_file))
        assert len(df) == 2
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "date"

    def test_nonexistent_file(self, tmp_path: Path):
        fake = str(tmp_path / "nope.csv")
        with pytest.raises(FileNotFoundError):
            _validate_file_path(fake)

    def test_unsupported_format(self, tmp_path: Path):
        p = tmp_path / "data.hdf5"
        p.write_text("not real")
        with pytest.raises(ValueError, match="Unsupported file format"):
            _load_file_to_dataframe(str(p))

    @pytest.mark.skipif(
        not _can_read_parquet(),
        reason="Neither pyarrow nor fastparquet is installed",
    )
    def test_parquet_loading(self, tmp_path: Path):
        df_orig = pd.DataFrame(
            {"epoch": pd.date_range("2024-01-01", periods=5, freq="h"), "val": range(5)}
        )
        p = tmp_path / "data.parquet"
        df_orig.to_parquet(p, index=False)

        df = _load_file_to_dataframe(str(p))
        assert len(df) == 5
        # 'epoch' is a known datetime name, should become the index
        assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# TestLoadFilePathSecurity
# ---------------------------------------------------------------------------


class TestLoadFilePathSecurity:
    """Test path validation and directory restrictions."""

    def test_reject_path_traversal(self, tmp_path: Path):
        """Paths like /etc/passwd should be rejected."""
        with pytest.raises(ValueError, match="outside allowed directories"):
            _validate_file_path("/etc/passwd")

    def test_accept_path_under_allowed_dir(self, tmp_path: Path, monkeypatch):
        """A file under an allowed directory should be accepted."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        f = allowed / "data.csv"
        f.write_text("a,b\n1,2\n")

        monkeypatch.setattr(
            "agent.tool_handlers.file_io._get_allowed_dirs",
            lambda: [allowed],
        )
        result = _validate_file_path(str(f))
        assert result == str(f.resolve())

    def test_reject_outside_mocked_allowed(self, tmp_path: Path, monkeypatch):
        """A file outside all mocked allowed dirs should be rejected."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        f = outside / "data.csv"
        f.write_text("a,b\n1,2\n")

        monkeypatch.setattr(
            "agent.tool_handlers.file_io._get_allowed_dirs",
            lambda: [allowed],
        )
        with pytest.raises(ValueError, match="outside allowed directories"):
            _validate_file_path(str(f))


# ---------------------------------------------------------------------------
# TestTryParseDatetimeIndex
# ---------------------------------------------------------------------------


class TestTryParseDatetimeIndex:
    """Test the datetime index auto-detection logic."""

    def test_explicit_time_column_in_columns(self):
        df = pd.DataFrame({
            "my_time": ["2024-01-01", "2024-01-02"],
            "val": [1, 2],
        })
        result = _try_parse_datetime_index(df, time_column="my_time")
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "my_time"

    def test_known_index_name(self):
        df = pd.DataFrame({"val": [1, 2]}, index=["2024-01-01", "2024-01-02"])
        df.index.name = "datetime"
        result = _try_parse_datetime_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_no_datetime_returns_unchanged(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = _try_parse_datetime_index(df)
        assert not isinstance(result.index, pd.DatetimeIndex)
