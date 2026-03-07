"""
Tests for knowledge.function_catalog — function doc catalog, search, and docstring retrieval.

Run with: python -m pytest tests/test_function_catalog.py
"""

import pytest

from knowledge.function_catalog import (
    FUNCTION_CATALOG,
    search_functions,
    get_function_docstring,
    get_function_index_summary,
)


class TestFunctionCatalogBuild:
    def test_catalog_has_scipy_signal(self):
        assert "scipy.signal" in FUNCTION_CATALOG
        assert len(FUNCTION_CATALOG["scipy.signal"]) > 0

    def test_catalog_has_scipy_fft(self):
        assert "scipy.fft" in FUNCTION_CATALOG
        assert len(FUNCTION_CATALOG["scipy.fft"]) > 0

    def test_catalog_has_scipy_interpolate(self):
        assert "scipy.interpolate" in FUNCTION_CATALOG
        assert len(FUNCTION_CATALOG["scipy.interpolate"]) > 0

    def test_catalog_has_scipy_stats(self):
        assert "scipy.stats" in FUNCTION_CATALOG
        assert len(FUNCTION_CATALOG["scipy.stats"]) > 0

    def test_catalog_has_scipy_integrate(self):
        assert "scipy.integrate" in FUNCTION_CATALOG
        assert len(FUNCTION_CATALOG["scipy.integrate"]) > 0

    def test_catalog_has_pywt(self):
        assert "pywt" in FUNCTION_CATALOG
        assert len(FUNCTION_CATALOG["pywt"]) > 0

    def test_entry_structure(self):
        """Each entry has name, sandbox_call, and summary."""
        for pkg, entries in FUNCTION_CATALOG.items():
            for entry in entries[:3]:
                assert "name" in entry
                assert "sandbox_call" in entry
                assert "summary" in entry
                assert entry["sandbox_call"].startswith(pkg + ".")

    def test_no_private_functions(self):
        """No functions starting with underscore."""
        for entries in FUNCTION_CATALOG.values():
            for entry in entries:
                assert not entry["name"].startswith("_")


class TestSearchFunctions:
    def test_search_spectrogram(self):
        results = search_functions("spectrogram")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "spectrogram" in names

    def test_search_wavelet(self):
        results = search_functions("wavelet")
        assert len(results) > 0
        # Should find pywt functions
        packages = {r["package"] for r in results}
        assert "pywt" in packages

    def test_search_butter(self):
        results = search_functions("butter")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "butter" in names

    def test_search_with_package_filter(self):
        results = search_functions("filter", package="scipy.signal")
        assert len(results) > 0
        # All results should be from scipy.signal
        for r in results:
            assert r["package"] == "scipy.signal"

    def test_search_max_results(self):
        results = search_functions("signal", max_results=3)
        assert len(results) <= 3

    def test_search_empty_query(self):
        results = search_functions("")
        assert results == []

    def test_search_result_structure(self):
        results = search_functions("fft")
        assert len(results) > 0
        r = results[0]
        assert "package" in r
        assert "name" in r
        assert "sandbox_call" in r
        assert "summary" in r

    def test_search_interpolate(self):
        results = search_functions("interpolate")
        assert len(results) > 0

    def test_search_fallback_to_substring(self):
        """Substring match when word-boundary fails."""
        results = search_functions("filtfilt")
        assert len(results) > 0
        assert any(r["name"] == "filtfilt" for r in results)


class TestGetFunctionDocstring:
    def test_get_butter_docstring(self):
        result = get_function_docstring("scipy.signal", "butter")
        assert "error" not in result
        assert result["name"] == "butter"
        assert result["package"] == "scipy.signal"
        assert "docstring" in result
        assert len(result["docstring"]) > 0
        assert "signature" in result
        assert result["sandbox_call"] == "scipy.signal.butter"

    def test_get_cwt_docstring(self):
        result = get_function_docstring("pywt", "cwt")
        assert "error" not in result
        assert result["name"] == "cwt"
        assert "docstring" in result

    def test_invalid_function(self):
        result = get_function_docstring("scipy.signal", "nonexistent_function_xyz")
        assert "error" in result

    def test_invalid_package(self):
        result = get_function_docstring("nonexistent.package", "butter")
        assert "error" in result

    def test_docstring_truncation(self):
        """Long docstrings should be truncated."""
        result = get_function_docstring("scipy.signal", "butter")
        assert len(result.get("docstring", "")) <= 3100  # 3000 + truncation notice


class TestGetFunctionIndexSummary:
    def test_produces_text(self):
        summary = get_function_index_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_contains_package_names(self):
        summary = get_function_index_summary()
        assert "scipy.signal" in summary
        assert "pywt" in summary

    def test_contains_function_names(self):
        summary = get_function_index_summary()
        # Should contain at least some common function names
        assert "butter" in summary or "spectrogram" in summary or "fft" in summary
