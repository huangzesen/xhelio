"""Unit tests for data_ops.fetch_ppi_archive — all mocked (no network)."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_ops.fetch_ppi_archive import (
    _discover_data_files,
    _discover_sol_organized,
    _discover_freq_organized,
    _filter_pairs_by_filename_time,
    _match_collection,
    _pair_data_and_labels,
    _parse_html_listing,
    _parse_sol_dir_dates,
    _parse_xml_label,
    _read_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_listing_html(names: list[str]) -> str:
    """Build a minimal Apache-style directory listing HTML."""
    links = "\n".join(f'<a href="{n}">{n}</a>' for n in names)
    return f"<html><body><pre>{links}</pre></body></html>"


def _make_entries(names: list[str]) -> list[dict]:
    """Simulate _list_directory output."""
    return [
        {"name": n, "is_dir": n.endswith("/"), "size": 0}
        for n in names
    ]


# ---------------------------------------------------------------------------
# TestParseHtmlListing
# ---------------------------------------------------------------------------

class TestParseHtmlListing:
    def test_basic_files(self):
        html = _make_listing_html(["foo.TAB", "foo.xml", "bar.csv", "bar.xml"])
        entries = _parse_html_listing(html)
        names = [e["name"] for e in entries]
        assert "foo.TAB" in names
        assert "foo.xml" in names

    def test_directories(self):
        html = _make_listing_html(["2005/", "2006/", "readme.txt"])
        entries = _parse_html_listing(html)
        dirs = [e for e in entries if e["is_dir"]]
        files = [e for e in entries if not e["is_dir"]]
        assert len(dirs) == 2
        assert len(files) == 1

    def test_skips_parent_and_sort_links(self):
        html = _make_listing_html(["../", "?C=N;O=D", "/", "data.tab"])
        entries = _parse_html_listing(html)
        assert len(entries) == 1
        assert entries[0]["name"] == "data.tab"

    def test_empty_listing(self):
        entries = _parse_html_listing("<html><body></body></html>")
        assert entries == []


# ---------------------------------------------------------------------------
# TestMatchCollection
# ---------------------------------------------------------------------------

class TestMatchCollection:
    def test_exact_match(self):
        assert _match_collection("data-1sec-krtp", ["data-1sec-krtp", "other"]) == "data-1sec-krtp"

    def test_hyphen_underscore_swap(self):
        assert _match_collection("data-1sec-krtp", ["data_1sec_krtp"]) == "data_1sec_krtp"

    def test_underscore_to_hyphen(self):
        assert _match_collection("data_1sec_krtp", ["data-1sec-krtp"]) == "data-1sec-krtp"

    def test_normalized_comparison(self):
        assert _match_collection("data1seckrtp", ["data-1sec-krtp"]) == "data-1sec-krtp"

    def test_no_match(self):
        assert _match_collection("nonexistent", ["a", "b", "c"]) is None


# ---------------------------------------------------------------------------
# TestPairDataAndLabels
# ---------------------------------------------------------------------------

class TestPairDataAndLabels:
    def test_tab_xml_pairing(self):
        files = ["data01.TAB", "data01.xml", "data02.TAB", "data02.xml"]
        pairs = _pair_data_and_labels("http://x/", files)
        assert len(pairs) == 2
        assert pairs[0] == ("http://x/data01.TAB", "http://x/data01.xml")

    def test_csv_xml_pairing(self):
        files = ["data.csv", "data.xml"]
        pairs = _pair_data_and_labels("http://x/", files)
        assert len(pairs) == 1

    def test_lblx_label(self):
        files = ["data.tab", "data.lblx"]
        pairs = _pair_data_and_labels("http://x/", files)
        assert len(pairs) == 1
        assert pairs[0][1] == "http://x/data.lblx"

    def test_unpaired_data_skipped(self):
        files = ["data.TAB", "other.xml"]  # no matching label
        pairs = _pair_data_and_labels("http://x/", files)
        assert len(pairs) == 0

    def test_collection_files_skipped(self):
        files = ["collection_inventory.csv", "collection_inventory.xml",
                 "data.tab", "data.xml"]
        pairs = _pair_data_and_labels("http://x/", files)
        assert len(pairs) == 1
        assert "data.tab" in pairs[0][0]


# ---------------------------------------------------------------------------
# TestDiscoverDataFiles — routing to correct discovery method
# ---------------------------------------------------------------------------

class TestDiscoverDataFiles:
    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_routes_to_year_organized(self, mock_list):
        mock_list.return_value = _make_entries(["2005/", "2006/", "readme.txt"])
        with patch("data_ops.fetch_ppi_archive._discover_year_organized") as mock_yr:
            mock_yr.return_value = [("a", "b")]
            result = _discover_data_files("http://x/", "2005-06-01", "2005-07-01")
            mock_yr.assert_called_once()
            assert result == [("a", "b")]

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_routes_to_orbit_organized(self, mock_list):
        mock_list.return_value = _make_entries(["2024017_orbit_58/", "2024080_orbit_59/"])
        with patch("data_ops.fetch_ppi_archive._discover_orbit_organized") as mock_orb:
            mock_orb.return_value = [("c", "d")]
            result = _discover_data_files("http://x/", "2024-01-01", "2024-02-01")
            mock_orb.assert_called_once()
            assert result == [("c", "d")]

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_routes_to_sol_organized(self, mock_list):
        mock_list.return_value = _make_entries([
            "SOL0004_SOL0029_20181130_20181226/",
            "SOL0030_SOL0060_20181227_20190126/",
        ])
        with patch("data_ops.fetch_ppi_archive._discover_sol_organized") as mock_sol:
            mock_sol.return_value = [("e", "f")]
            result = _discover_data_files("http://x/", "2019-01-01", "2019-01-15")
            mock_sol.assert_called_once()
            assert result == [("e", "f")]

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_routes_to_freq_organized(self, mock_list):
        mock_list.return_value = _make_entries(["20Hz/", "2Hz/"])
        with patch("data_ops.fetch_ppi_archive._discover_freq_organized") as mock_freq:
            mock_freq.return_value = [("g", "h")]
            result = _discover_data_files("http://x/", "2019-01-01", "2019-01-15")
            mock_freq.assert_called_once()
            assert result == [("g", "h")]

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_routes_to_flat(self, mock_list):
        mock_list.return_value = _make_entries(["data.tab", "data.xml"])
        result = _discover_data_files("http://x/", "2019-01-01", "2019-01-15")
        assert len(result) == 1

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_year_takes_priority_over_sol(self, mock_list):
        """When both YYYY/ and SOL*/ dirs exist, year takes priority."""
        mock_list.return_value = _make_entries([
            "2019/", "SOL0030_SOL0060_20181227_20190126/",
        ])
        with patch("data_ops.fetch_ppi_archive._discover_year_organized") as mock_yr:
            mock_yr.return_value = []
            _discover_data_files("http://x/", "2019-01-01", "2019-01-15")
            mock_yr.assert_called_once()


# ---------------------------------------------------------------------------
# TestParseSolDirDates
# ---------------------------------------------------------------------------

class TestParseSolDirDates:
    def test_standard_sol_dir(self):
        start, end = _parse_sol_dir_dates("SOL0004_SOL0029_20181130_20181226")
        assert start == pd.Timestamp("2018-11-30")
        assert end == pd.Timestamp("2018-12-26")

    def test_release_sol_dir(self):
        start, end = _parse_sol_dir_dates("release02_SOL0120_SOL0209_20190329_20190629")
        assert start == pd.Timestamp("2019-03-29")
        assert end == pd.Timestamp("2019-06-29")

    def test_single_date(self):
        start, end = _parse_sol_dir_dates("SOL0100_20190301")
        assert start == pd.Timestamp("2019-03-01")
        assert end == pd.Timestamp("2019-03-01")

    def test_no_date(self):
        start, end = _parse_sol_dir_dates("SOL0100_no_dates")
        assert start is None
        assert end is None

    def test_case_insensitive(self):
        start, end = _parse_sol_dir_dates("sol0004_SOL0029_20181130_20181226")
        assert start == pd.Timestamp("2018-11-30")


# ---------------------------------------------------------------------------
# TestDiscoverSolOrganized
# ---------------------------------------------------------------------------

class TestDiscoverSolOrganized:
    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_includes_overlapping_dir(self, mock_list):
        """Dir whose date range overlaps the request is included."""
        def side_effect(url):
            if url.endswith("SOL0030_SOL0060_20181227_20190126/"):
                return _make_entries(["data.tab", "data.xml"])
            return _make_entries([])

        mock_list.side_effect = side_effect
        dir_names = [
            "SOL0004_SOL0029_20181130_20181226",
            "SOL0030_SOL0060_20181227_20190126",
        ]
        t_min = pd.Timestamp("2019-01-01")
        t_max = pd.Timestamp("2019-01-15")
        pairs = _discover_sol_organized("http://x/", dir_names, t_min, t_max)
        assert len(pairs) == 1

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_excludes_non_overlapping_dir(self, mock_list):
        """Dir outside the request range is excluded."""
        mock_list.return_value = _make_entries([])
        dir_names = ["SOL0004_SOL0029_20181130_20181226"]
        t_min = pd.Timestamp("2020-01-01")
        t_max = pd.Timestamp("2020-02-01")
        pairs = _discover_sol_organized("http://x/", dir_names, t_min, t_max)
        assert len(pairs) == 0

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_includes_unparseable_dir(self, mock_list):
        """Dirs without parseable dates are included as a safety fallback."""
        mock_list.return_value = _make_entries(["data.tab", "data.xml"])
        dir_names = ["SOL0100_mystery"]
        t_min = pd.Timestamp("2020-01-01")
        t_max = pd.Timestamp("2020-02-01")
        pairs = _discover_sol_organized("http://x/", dir_names, t_min, t_max)
        assert len(pairs) == 1

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_skips_non_sol_dirs(self, mock_list):
        """Non-SOL directories are ignored."""
        mock_list.return_value = _make_entries([])
        dir_names = ["SOL0004_SOL0029_20181130_20181226", "readme", "browse"]
        t_min = pd.Timestamp("2018-12-01")
        t_max = pd.Timestamp("2018-12-31")
        pairs = _discover_sol_organized("http://x/", dir_names, t_min, t_max)
        # Only the SOL dir should trigger _list_directory
        assert mock_list.call_count == 1


# ---------------------------------------------------------------------------
# TestDiscoverFreqOrganized
# ---------------------------------------------------------------------------

class TestDiscoverFreqOrganized:
    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_freq_with_release_subdirs(self, mock_list):
        """Frequency dir with release-sol subdirs."""
        call_count = [0]

        def side_effect(url):
            call_count[0] += 1
            if url.endswith("2Hz/"):
                return _make_entries([
                    "release01_SOL0004_SOL0029_20181130_20181226/",
                    "release02_SOL0030_SOL0060_20181227_20190126/",
                ])
            elif "release02" in url:
                return _make_entries(["data.tab", "data.xml"])
            else:
                return _make_entries([])

        mock_list.side_effect = side_effect
        dir_names = ["2Hz", "20Hz"]
        t_min = pd.Timestamp("2019-01-01")
        t_max = pd.Timestamp("2019-01-15")
        pairs = _discover_freq_organized("http://x/", dir_names, t_min, t_max)
        assert len(pairs) == 1

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_freq_with_flat_files(self, mock_list):
        """Frequency dir with files directly (no subdirs)."""
        def side_effect(url):
            if url.endswith("1Hz/"):
                return _make_entries(["data.tab", "data.xml"])
            return _make_entries([])

        mock_list.side_effect = side_effect
        dir_names = ["1Hz"]
        t_min = pd.Timestamp("2019-01-01")
        t_max = pd.Timestamp("2019-12-31")
        pairs = _discover_freq_organized("http://x/", dir_names, t_min, t_max)
        assert len(pairs) == 1

    @patch("data_ops.fetch_ppi_archive._list_directory")
    def test_skips_non_freq_dirs(self, mock_list):
        """Non-frequency directories are ignored."""
        mock_list.return_value = _make_entries([])
        dir_names = ["2Hz", "browse", "readme"]
        t_min = pd.Timestamp("2019-01-01")
        t_max = pd.Timestamp("2019-12-31")
        _discover_freq_organized("http://x/", dir_names, t_min, t_max)
        # Only "2Hz" should trigger _list_directory
        assert mock_list.call_count == 1


# ---------------------------------------------------------------------------
# TestFilterPairsByFilenameTime
# ---------------------------------------------------------------------------

class TestFilterPairsByFilenameTime:
    def test_includes_overlapping_file(self):
        pairs = [
            ("http://x/ifg_20190101T000000_20190102T000000.tab",
             "http://x/ifg_20190101T000000_20190102T000000.xml"),
        ]
        t_min = pd.Timestamp("2019-01-01")
        t_max = pd.Timestamp("2019-01-03")
        result = _filter_pairs_by_filename_time(pairs, t_min, t_max)
        assert len(result) == 1

    def test_excludes_non_overlapping_file(self):
        pairs = [
            ("http://x/ifg_20180101T000000_20180102T000000.tab",
             "http://x/ifg_20180101T000000_20180102T000000.xml"),
        ]
        t_min = pd.Timestamp("2019-06-01")
        t_max = pd.Timestamp("2019-06-30")
        result = _filter_pairs_by_filename_time(pairs, t_min, t_max)
        assert len(result) == 0

    def test_date_only_timestamps(self):
        """Filenames with YYYYMMDD_YYYYMMDD (no time component)."""
        pairs = [
            ("http://x/data_20190101_20190115.tab",
             "http://x/data_20190101_20190115.xml"),
        ]
        t_min = pd.Timestamp("2019-01-10")
        t_max = pd.Timestamp("2019-01-20")
        result = _filter_pairs_by_filename_time(pairs, t_min, t_max)
        assert len(result) == 1

    def test_no_timestamps_returns_all(self):
        """If no filenames have timestamps, return all pairs."""
        pairs = [
            ("http://x/data_001.tab", "http://x/data_001.xml"),
            ("http://x/data_002.tab", "http://x/data_002.xml"),
        ]
        t_min = pd.Timestamp("2019-01-01")
        t_max = pd.Timestamp("2019-01-31")
        result = _filter_pairs_by_filename_time(pairs, t_min, t_max)
        assert len(result) == 2

    def test_mixed_with_and_without_timestamps(self):
        """Files without timestamps are included; files with timestamps are filtered."""
        pairs = [
            ("http://x/data_no_ts.tab", "http://x/data_no_ts.xml"),
            ("http://x/data_20190101_20190102.tab", "http://x/data_20190101_20190102.xml"),
            ("http://x/data_20200101_20200102.tab", "http://x/data_20200101_20200102.xml"),
        ]
        t_min = pd.Timestamp("2019-01-01")
        t_max = pd.Timestamp("2019-01-31")
        result = _filter_pairs_by_filename_time(pairs, t_min, t_max)
        # data_no_ts included (no timestamp), 2019 included, 2020 excluded
        assert len(result) == 2

    def test_empty_pairs(self):
        result = _filter_pairs_by_filename_time([], pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-31"))
        assert result == []


# ---------------------------------------------------------------------------
# TestParseXmlLabel
# ---------------------------------------------------------------------------

class TestParseXmlLabel:
    def test_fixed_width_label(self):
        xml = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
          <File_Area_Observational>
            <Table_Character>
              <records>100</records>
              <Record_Character>
                <Field_Character>
                  <name>SCET</name>
                  <field_location>1</field_location>
                  <field_length>24</field_length>
                </Field_Character>
                <Field_Character>
                  <name>BR</name>
                  <field_location>26</field_location>
                  <field_length>14</field_length>
                </Field_Character>
              </Record_Character>
            </Table_Character>
          </File_Area_Observational>
        </Product_Observational>
        """)
        label = _parse_xml_label(xml)
        assert label["table_type"] == "fixed_width"
        assert len(label["fields"]) == 2
        assert label["fields"][0]["name"] == "SCET"
        assert label["fields"][0]["offset"] == 1
        assert label["fields"][0]["length"] == 24
        assert label["records"] == 100

    def test_delimited_label(self):
        xml = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
          <File_Area_Observational>
            <Table_Delimited>
              <records>50</records>
              <field_delimiter>Semicolon</field_delimiter>
              <Record_Delimited>
                <Field_Delimited>
                  <name>Time</name>
                  <field_number>1</field_number>
                </Field_Delimited>
                <Field_Delimited>
                  <name>Bx_SC</name>
                  <field_number>2</field_number>
                </Field_Delimited>
              </Record_Delimited>
            </Table_Delimited>
          </File_Area_Observational>
        </Product_Observational>
        """)
        label = _parse_xml_label(xml)
        assert label["table_type"] == "delimited"
        assert label["delimiter"] == ";"
        assert len(label["fields"]) == 2
        assert label["fields"][1]["name"] == "Bx_SC"
        assert label["records"] == 50

    def test_comma_delimiter(self):
        xml = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
          <File_Area_Observational>
            <Table_Delimited>
              <field_delimiter>Comma</field_delimiter>
              <Record_Delimited></Record_Delimited>
            </Table_Delimited>
          </File_Area_Observational>
        </Product_Observational>
        """)
        label = _parse_xml_label(xml)
        assert label["delimiter"] == ","

    def test_no_table_raises(self):
        xml = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
          <File_Area_Observational>
          </File_Area_Observational>
        </Product_Observational>
        """)
        with pytest.raises(ValueError, match="No Table_Delimited"):
            _parse_xml_label(xml)


# ---------------------------------------------------------------------------
# TestReadTable
# ---------------------------------------------------------------------------

class TestReadTable:
    def test_read_fixed_width(self, tmp_path):
        # Create a minimal fixed-width data file
        data_file = tmp_path / "test.TAB"
        # Fields: SCET (cols 1-24), BR (cols 26-39)
        lines = [
            "2005-01-15T00:00:00.000  1.23456789012 \n",
            "2005-01-15T00:01:00.000  2.34567890123 \n",
        ]
        data_file.write_text("".join(lines))

        label = {
            "table_type": "fixed_width",
            "fields": [
                {"name": "SCET", "offset": 1, "length": 24, "type": "fixed_width"},
                {"name": "BR", "offset": 26, "length": 14, "type": "fixed_width"},
            ],
            "delimiter": None,
            "records": 2,
        }

        df = _read_table(data_file, label, "BR")
        assert df is not None
        assert len(df) == 2
        assert 1 in df.columns

    def test_read_delimited(self, tmp_path):
        data_file = tmp_path / "test.csv"
        data_file.write_text(
            "2019-01-01T00:00:00;1.5;2.5\n"
            "2019-01-01T00:01:00;3.5;4.5\n"
        )

        label = {
            "table_type": "delimited",
            "fields": [
                {"name": "Time", "field_number": 1, "type": "delimited"},
                {"name": "Bx_SC", "field_number": 2, "type": "delimited"},
                {"name": "By_SC", "field_number": 3, "type": "delimited"},
            ],
            "delimiter": ";",
            "records": 2,
        }

        df = _read_table(data_file, label, "Bx_SC")
        assert df is not None
        assert len(df) == 2
        assert df.iloc[0, 0] == pytest.approx(1.5)

    def test_missing_parameter_returns_none(self, tmp_path):
        data_file = tmp_path / "test.csv"
        data_file.write_text("2019-01-01T00:00:00;1.5\n")

        label = {
            "table_type": "delimited",
            "fields": [
                {"name": "Time", "field_number": 1, "type": "delimited"},
                {"name": "Bx_SC", "field_number": 2, "type": "delimited"},
            ],
            "delimiter": ";",
            "records": 1,
        }

        df = _read_table(data_file, label, "NONEXISTENT")
        assert df is None
