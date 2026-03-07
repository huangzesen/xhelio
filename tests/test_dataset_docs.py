"""
Tests for get_dataset_docs and supporting functions in metadata_client.py.

Tests HTML-to-text conversion, section extraction, documentation lookup,
truncation, fallback URL construction, and graceful degradation.

Run with: python -m pytest tests/test_dataset_docs.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

from knowledge.metadata_client import (
    _HTMLToText,
    _extract_dataset_section,
    _fallback_resource_url,
    _fetch_notes_section,
    get_dataset_docs,
    clear_cache,
)


# ---------------------------------------------------------------------------
# _HTMLToText tests
# ---------------------------------------------------------------------------

class TestHTMLToText:
    """Test the HTML-to-text converter."""

    def test_strips_tags(self):
        parser = _HTMLToText()
        parser.feed("<p>Hello <b>world</b></p>")
        assert parser.get_text() == "Hello world"

    def test_preserves_block_structure(self):
        parser = _HTMLToText()
        parser.feed("<p>First</p><p>Second</p>")
        text = parser.get_text()
        assert "First" in text
        assert "Second" in text
        assert "\n" in text

    def test_br_adds_newline(self):
        parser = _HTMLToText()
        parser.feed("Line 1<br>Line 2")
        assert "Line 1\nLine 2" in parser.get_text()

    def test_skips_script_content(self):
        parser = _HTMLToText()
        parser.feed("<p>Visible</p><script>var x = 1;</script><p>Also visible</p>")
        text = parser.get_text()
        assert "Visible" in text
        assert "Also visible" in text
        assert "var x" not in text

    def test_skips_style_content(self):
        parser = _HTMLToText()
        parser.feed("<style>.foo { color: red; }</style><p>Content</p>")
        text = parser.get_text()
        assert "Content" in text
        assert "color" not in text

    def test_collapses_blank_lines(self):
        parser = _HTMLToText()
        parser.feed("<p>A</p><p></p><p></p><p></p><p>B</p>")
        text = parser.get_text()
        assert "\n\n\n" not in text

    def test_empty_input(self):
        parser = _HTMLToText()
        parser.feed("")
        assert parser.get_text() == ""

    def test_plain_text_passes_through(self):
        parser = _HTMLToText()
        parser.feed("Just plain text")
        assert parser.get_text() == "Just plain text"

    def test_nested_skip_tags(self):
        parser = _HTMLToText()
        parser.feed("<script>outer<script>inner</script>still script</script>After")
        text = parser.get_text()
        assert "outer" not in text
        assert "inner" not in text
        assert "After" in text


# ---------------------------------------------------------------------------
# _extract_dataset_section tests
# ---------------------------------------------------------------------------

class TestExtractDatasetSection:
    """Test section extraction from CDAWeb Notes HTML."""

    SAMPLE_HTML = """
    <html><body>
    <h2>Datasets</h2>
    <hr>
    <a name="AC_H0_MFI"></a>
    <strong>AC_H0_MFI</strong>
    <p>Description of AC_H0_MFI dataset.</p>
    <p>Coordinate system: GSE</p>
    <hr>
    <a name="AC_H2_MFI"></a>
    <strong>AC_H2_MFI</strong>
    <p>Description of AC_H2_MFI dataset.</p>
    <p>16-second averages in GSE coordinates.</p>
    <p>Contact: N. Ness</p>
    <hr>
    <a name="AC_H3_MFI"></a>
    <strong>AC_H3_MFI</strong>
    <p>Another dataset.</p>
    <hr>
    </body></html>
    """

    def test_finds_section_by_anchor_name(self):
        section = _extract_dataset_section(self.SAMPLE_HTML, "AC_H2_MFI")
        assert section is not None
        assert "AC_H2_MFI" in section
        assert "16-second averages" in section

    def test_section_does_not_include_next_dataset(self):
        section = _extract_dataset_section(self.SAMPLE_HTML, "AC_H2_MFI")
        assert section is not None
        assert "AC_H3_MFI" not in section

    def test_returns_none_for_missing_dataset(self):
        section = _extract_dataset_section(self.SAMPLE_HTML, "NONEXISTENT_DS")
        assert section is None

    def test_finds_section_by_strong_tag(self):
        html = """
        <strong>MY_DATASET</strong>
        <p>Some documentation.</p>
        <hr>
        """
        section = _extract_dataset_section(html, "MY_DATASET")
        assert section is not None
        assert "Some documentation" in section

    def test_finds_section_by_id_attribute(self):
        html = """
        <a id="TEST_DS_01"></a>
        <p>Dataset info here.</p>
        <hr>
        """
        section = _extract_dataset_section(html, "TEST_DS_01")
        assert section is not None
        assert "Dataset info here" in section

    def test_back_to_top_boundary(self):
        html = """
        <a name="BACK_TOP_DS"></a>
        <p>Dataset description.</p>
        <a href="#">Back to top</a>
        <a name="NEXT_DS"></a>
        """
        # Pad the content so section is > 100 chars from start
        html = html.replace("<p>Dataset description.</p>",
                           "<p>Dataset description. " + "x" * 150 + "</p>")
        section = _extract_dataset_section(html, "BACK_TOP_DS")
        assert section is not None
        assert "NEXT_DS" not in section


# ---------------------------------------------------------------------------
# _fallback_resource_url tests
# ---------------------------------------------------------------------------

class TestFallbackResourceUrl:
    """Test fallback URL construction."""

    def test_ace_dataset(self):
        url = _fallback_resource_url("AC_H2_MFI")
        assert url == "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI"

    def test_psp_dataset(self):
        url = _fallback_resource_url("PSP_FLD_L2_MAG_RTN_1MIN")
        assert url == "https://cdaweb.gsfc.nasa.gov/misc/NotesP.html#PSP_FLD_L2_MAG_RTN_1MIN"

    def test_wind_dataset(self):
        url = _fallback_resource_url("WI_H2_MFI")
        assert url == "https://cdaweb.gsfc.nasa.gov/misc/NotesW.html#WI_H2_MFI"

    def test_lowercase_handled(self):
        url = _fallback_resource_url("ac_h2_mfi")
        assert "NotesA.html" in url


# ---------------------------------------------------------------------------
# get_dataset_docs tests (with mocks)
# ---------------------------------------------------------------------------

class TestGetDatasetDocs:
    """Test the main get_dataset_docs function with mocked dependencies."""

    def setup_method(self):
        clear_cache()

    def test_success_with_metadata_and_notes(self):
        fake_info = {
            "contact": "N. Ness @ Bartol Research Institute",
            "resourceURL": "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI",
            "parameters": [],
        }
        fake_html = """
        <a name="AC_H2_MFI"></a>
        <strong>AC_H2_MFI</strong>
        <p>ACE Magnetic Field 16-Second Level 2 Data in GSE Coordinates.</p>
        <p>doi:10.48322/fake-doi</p>
        <hr>
        """

        with patch("knowledge.metadata_client.get_dataset_info", return_value=fake_info), \
             patch("knowledge.metadata_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = fake_html
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_dataset_docs("AC_H2_MFI")

        assert result["dataset_id"] == "AC_H2_MFI"
        assert result["contact"] == "N. Ness @ Bartol Research Institute"
        assert result["resource_url"] == "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI"
        assert result["documentation"] is not None
        assert "16-Second Level 2" in result["documentation"]

    def test_truncation_at_max_chars(self):
        fake_info = {
            "contact": "Test Contact",
            "resourceURL": "https://cdaweb.gsfc.nasa.gov/misc/NotesT.html#TEST_DS",
            "parameters": [],
        }
        # Create a long documentation section
        long_content = "x" * 5000
        fake_html = f"""
        <a name="TEST_DS"></a>
        <p>{long_content}</p>
        <hr>
        """

        with patch("knowledge.metadata_client.get_dataset_info", return_value=fake_info), \
             patch("knowledge.metadata_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = fake_html
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_dataset_docs("TEST_DS", max_chars=100)

        assert result["documentation"] is not None
        assert len(result["documentation"]) <= 100 + len("\n[truncated]")
        assert result["documentation"].endswith("[truncated]")

    def test_fallback_url_when_metadata_fails(self):
        """When metadata fetch fails, construct URL from dataset ID first letter."""
        with patch("knowledge.metadata_client.get_dataset_info", side_effect=Exception("network error")), \
             patch("knowledge.metadata_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = '<a name="AC_H2_MFI"></a><p>Doc content</p><hr>'
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_dataset_docs("AC_H2_MFI")

        assert result["resource_url"] == "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI"
        assert result["documentation"] is not None

    def test_graceful_when_notes_page_unavailable(self):
        """When Notes page fetch fails, return partial result with contact info."""
        fake_info = {
            "contact": "Test PI",
            "resourceURL": "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI",
            "parameters": [],
        }

        with patch("knowledge.metadata_client.get_dataset_info", return_value=fake_info), \
             patch("knowledge.metadata_client.requests.get", side_effect=Exception("timeout")):
            result = get_dataset_docs("AC_H2_MFI")

        assert result["dataset_id"] == "AC_H2_MFI"
        assert result["contact"] == "Test PI"
        assert result["resource_url"] is not None
        assert result["documentation"] is None

    def test_graceful_when_section_not_found(self):
        """When section not found in page, documentation is None."""
        fake_info = {
            "contact": "Someone",
            "resourceURL": "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI",
            "parameters": [],
        }
        fake_html = "<html><body><p>No matching sections here.</p></body></html>"

        with patch("knowledge.metadata_client.get_dataset_info", return_value=fake_info), \
             patch("knowledge.metadata_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = fake_html
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_dataset_docs("AC_H2_MFI")

        assert result["documentation"] is None
        assert result["contact"] == "Someone"

    def test_metadata_without_resource_url(self):
        """When metadata has no resourceURL, fallback URL is constructed."""
        fake_info = {
            "contact": "Test Contact",
            "parameters": [],
        }

        with patch("knowledge.metadata_client.get_dataset_info", return_value=fake_info), \
             patch("knowledge.metadata_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = '<a name="WI_H2_MFI"></a><p>Wind data</p><hr>'
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_dataset_docs("WI_H2_MFI")

        assert "NotesW.html" in result["resource_url"]

    def test_notes_cache_is_used(self):
        """Second call for same base URL should use cache, not make another request."""
        fake_info = {
            "contact": "Test",
            "resourceURL": "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI",
            "parameters": [],
        }
        fake_html = '<a name="AC_H2_MFI"></a><p>Cached content</p><hr>'

        with patch("knowledge.metadata_client.get_dataset_info", return_value=fake_info), \
             patch("knowledge.metadata_client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = fake_html
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            # First call — makes HTTP request
            get_dataset_docs("AC_H2_MFI")
            assert mock_get.call_count == 1

            # Second call — should use cache
            result = get_dataset_docs("AC_H2_MFI")
            assert mock_get.call_count == 1  # No additional request
            assert result["documentation"] is not None


# ---------------------------------------------------------------------------
# _fetch_notes_section tests
# ---------------------------------------------------------------------------

class TestFetchNotesSection:
    """Test _fetch_notes_section with mocked HTTP."""

    def setup_method(self):
        clear_cache()

    def test_strips_fragment_from_url(self):
        """Base URL should not include the fragment."""
        fake_html = '<a name="DS_01"></a><p>Section content</p><hr>'

        with patch("data_ops.http_utils.request_with_retry") as mock_req:
            mock_resp = MagicMock()
            mock_resp.text = fake_html
            mock_req.return_value = mock_resp

            result = _fetch_notes_section(
                "https://cdaweb.gsfc.nasa.gov/misc/NotesD.html#DS_01",
                "DS_01",
            )

        mock_req.assert_called_once_with(
            "https://cdaweb.gsfc.nasa.gov/misc/NotesD.html",
        )
        assert result is not None
        assert "Section content" in result

    def test_returns_none_on_network_error(self):
        with patch("data_ops.http_utils.request_with_retry", side_effect=Exception("err")):
            result = _fetch_notes_section("https://example.com/Notes.html#X", "X")
        assert result is None
