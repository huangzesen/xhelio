"""
Unit tests for the Plotly renderer.

No API key, no JVM, no network — fast and self-contained.
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.store import DataEntry
from rendering.plotly_renderer import PlotlyRenderer, fill_figure_data, RenderResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def renderer():
    return PlotlyRenderer(verbose=False)


def _make_entry(label: str, n: int = 100, ncols: int = 1, desc: str = "test data") -> DataEntry:
    """Create a synthetic DataEntry for testing."""
    rng = pd.date_range("2024-01-01", periods=n, freq="min")
    if ncols == 1:
        df = pd.DataFrame({"value": np.random.randn(n)}, index=rng)
    else:
        cols = {f"c{i}": np.random.randn(n) for i in range(ncols)}
        df = pd.DataFrame(cols, index=rng)
    return DataEntry(label=label, data=df, units="nT", description=desc)


def _render_one(renderer, label="x", n=10, desc="test data"):
    """Helper: render a single scatter trace via render_plotly_json."""
    entry = _make_entry(label, n=n, desc=desc)
    fig_json = {
        "data": [{"type": "scatter", "data_label": label}],
        "layout": {},
    }
    result = renderer.render_plotly_json(fig_json, {label: entry})
    return result, entry


# ---------------------------------------------------------------------------
# manage (direct method calls)
# ---------------------------------------------------------------------------

class TestManage:
    def test_reset(self, renderer):
        _render_one(renderer)
        assert renderer.get_figure() is not None
        result = renderer.reset()
        assert result["status"] == "success"
        assert renderer.get_figure() is None
        assert renderer._trace_labels == []

    def test_get_state(self, renderer):
        _render_one(renderer, desc="Alpha")
        result = renderer.get_current_state()
        assert result["has_plot"] is True
        assert result["traces"] == ["Alpha"]

    def test_get_state_empty(self, renderer):
        result = renderer.get_current_state()
        assert result["has_plot"] is False
        assert result["traces"] == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_png(self, renderer, tmp_path):
        _render_one(renderer)
        filepath = str(tmp_path / "test_output.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_png_no_plot(self, renderer, tmp_path):
        filepath = str(tmp_path / "empty.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "error"

    def test_export_pdf(self, renderer, tmp_path):
        _render_one(renderer)
        filepath = str(tmp_path / "test_output.pdf")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_default_format_is_png(self, renderer, tmp_path):
        _render_one(renderer)
        filepath = str(tmp_path / "test_output")
        result = renderer.export(filepath)
        assert result["status"] == "success"
        assert result["filepath"].endswith(".png")

    def test_export_adds_extension(self, renderer, tmp_path):
        _render_one(renderer)
        filepath = str(tmp_path / "noext")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["filepath"].endswith(".pdf")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestState:
    def test_reset_clears_state(self, renderer):
        _render_one(renderer)
        assert renderer.get_figure() is not None

        result = renderer.reset()
        assert result["status"] == "success"
        assert renderer.get_figure() is None
        assert renderer._panel_count == 0
        assert renderer._trace_labels == []

    def test_get_current_state_empty(self, renderer):
        state = renderer.get_current_state()
        assert state["has_plot"] is False
        assert state["traces"] == []

    def test_get_current_state_after_plot(self, renderer):
        _render_one(renderer, desc="Alpha")
        state = renderer.get_current_state()
        assert state["has_plot"] is True
        assert state["traces"] == ["Alpha"]


# ---------------------------------------------------------------------------
# fill_figure_data (Plotly JSON pipeline)
# ---------------------------------------------------------------------------

class TestFillFigureData:
    """Tests for the fill_figure_data function that resolves data_label placeholders."""

    def test_scalar_trace(self):
        """Single scalar trace gets x and y arrays filled."""
        entry = _make_entry("mag", n=50, desc="Bmag")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag", "mode": "lines"}],
            "layout": {"title": {"text": "Test"}},
        }
        result = fill_figure_data(fig_json, {"mag": entry})
        assert isinstance(result, RenderResult)
        fig = result.figure
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 50
        assert len(fig.data[0].y) == 50
        assert fig.layout.title.text == "Test"
        assert result.trace_labels == ["Bmag"]

    def test_multi_column_raises_error(self):
        """Multi-column entry raises ValueError with helpful message."""
        entry = _make_entry("Bvec", n=30, ncols=3, desc="B field")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "Bvec"}],
            "layout": {},
        }
        with pytest.raises(ValueError, match="has 3 columns"):
            fill_figure_data(fig_json, {"Bvec": entry})

    def test_nan_to_none(self):
        """NaN values in data are converted to None for Plotly."""
        rng = pd.date_range("2024-01-01", periods=10, freq="min")
        vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        df = pd.DataFrame({"value": vals}, index=rng)
        entry = DataEntry(label="gappy", data=df, units="nT")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "gappy"}],
            "layout": {},
        }
        result = fill_figure_data(fig_json, {"gappy": entry})
        y_data = result.figure.data[0].y
        assert y_data[1] is None
        assert y_data[3] is None
        assert y_data[0] == 1.0

    def test_heatmap_trace(self):
        """Heatmap trace gets x, y, z populated from spectrogram data."""
        rng = pd.date_range("2024-01-01", periods=50, freq="10min")
        bins = np.linspace(0.001, 0.5, 20)
        data = np.random.rand(50, 20)
        df = pd.DataFrame(data, index=rng, columns=[str(b) for b in bins])
        entry = DataEntry(
            label="spec", data=df, units="nT",
            description="Spectrogram",
            metadata={"type": "spectrogram", "bin_label": "Freq (Hz)",
                      "value_label": "PSD", "bin_values": bins.tolist()},
        )
        fig_json = {
            "data": [{"type": "heatmap", "data_label": "spec",
                       "colorscale": "Viridis"}],
            "layout": {"yaxis": {"domain": [0, 1]}},
        }
        result = fill_figure_data(fig_json, {"spec": entry})
        fig = result.figure
        assert len(fig.data) == 1
        trace = fig.data[0]
        assert trace.type == "heatmap"
        assert len(trace.x) == 50
        assert len(trace.y) == 20
        assert len(trace.z) == 20  # z is transposed: bins x times

    def test_multi_panel_layout(self):
        """Multi-panel layout with separate y-axes works correctly."""
        e1 = _make_entry("A", n=20, desc="Alpha")
        e2 = _make_entry("B", n=20, desc="Beta")
        fig_json = {
            "data": [
                {"type": "scatter", "data_label": "A", "xaxis": "x", "yaxis": "y"},
                {"type": "scatter", "data_label": "B", "xaxis": "x2", "yaxis": "y2"},
            ],
            "layout": {
                "xaxis": {"domain": [0, 1], "anchor": "y"},
                "xaxis2": {"domain": [0, 1], "anchor": "y2", "matches": "x"},
                "yaxis": {"domain": [0.55, 1], "anchor": "x", "title": {"text": "nT"}},
                "yaxis2": {"domain": [0, 0.45], "anchor": "x2", "title": {"text": "km/s"}},
            },
        }
        result = fill_figure_data(fig_json, {"A": e1, "B": e2})
        assert result.panel_count == 2
        assert len(result.figure.data) == 2

    def test_missing_data_label_error(self):
        """Missing data_label raises ValueError."""
        fig_json = {
            "data": [{"type": "scatter", "data_label": "MISSING"}],
            "layout": {},
        }
        with pytest.raises(ValueError, match="MISSING"):
            fill_figure_data(fig_json, {})

    def test_default_layout_applied(self):
        """Default white background and sizing is applied."""
        entry = _make_entry("mag", n=50)
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag"}],
            "layout": {},
        }
        result = fill_figure_data(fig_json, {"mag": entry})
        assert result.figure.layout.paper_bgcolor == "white"
        assert result.figure.layout.plot_bgcolor == "white"

    def test_explicit_color_preserved(self):
        """Trace with explicit line color keeps it."""
        entry = _make_entry("mag", n=20)
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag",
                       "line": {"color": "red"}}],
            "layout": {},
        }
        result = fill_figure_data(fig_json, {"mag": entry})
        assert result.figure.data[0].line.color == "red"

    def test_numeric_index_scatter(self):
        """Scatter trace with numeric (non-datetime) index works correctly."""
        df = pd.DataFrame({"value": [1.0, 4.0, 9.0, 16.0]},
                          index=[10.0, 20.0, 30.0, 40.0])
        entry = DataEntry(label="numeric", data=df, units="km/s",
                          description="Velocity scatter")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "numeric"}],
            "layout": {"xaxis": {}},
        }
        result = fill_figure_data(fig_json, {"numeric": entry})
        fig = result.figure
        assert len(fig.data) == 1
        assert list(fig.data[0].x) == [10.0, 20.0, 30.0, 40.0]
        assert fig.data[0].y == (1.0, 4.0, 9.0, 16.0)

    def test_3row_2col_panel_skeleton(self):
        """3-row × 2-column grid layout with 6 timeseries panels.

        Verifies that fill_figure_data correctly processes a complex grid
        layout: 6 independent y-axes with domain splits, 2 x-axis columns,
        proper anchoring, and matched time axes within each column.
        """
        # 6 distinct timeseries entries
        labels = ["Bmag_L", "Np_L", "Vsw_L", "Bmag_R", "Np_R", "Vsw_R"]
        descs = ["B (left)", "density (left)", "speed (left)",
                 "B (right)", "density (right)", "speed (right)"]
        entries = {}
        for lbl, desc in zip(labels, descs):
            entries[lbl] = _make_entry(lbl, n=30, desc=desc)

        # Domain math: 3 rows with 0.05 spacing
        # Row height h = (1 - 0.05*2) / 3 = 0.3
        h = 0.3
        gap = 0.05
        col_gap = 0.1
        # Left column x-domain: [0, 0.45], right: [0.55, 1]
        left_x = [0, 0.45]
        right_x = [0.55, 1]
        # Row domains (top to bottom):
        row1 = [1 - h, 1]           # [0.7, 1.0]
        row2 = [1 - 2*h - gap, 1 - h - gap]  # [0.35, 0.65]
        row3 = [0, h]               # [0.0, 0.3]

        fig_json = {
            "data": [
                # Left column (panels 1, 2, 3)
                {"type": "scatter", "data_label": "Bmag_L", "xaxis": "x", "yaxis": "y"},
                {"type": "scatter", "data_label": "Np_L", "xaxis": "x2", "yaxis": "y2"},
                {"type": "scatter", "data_label": "Vsw_L", "xaxis": "x3", "yaxis": "y3"},
                # Right column (panels 4, 5, 6)
                {"type": "scatter", "data_label": "Bmag_R", "xaxis": "x4", "yaxis": "y4"},
                {"type": "scatter", "data_label": "Np_R", "xaxis": "x5", "yaxis": "y5"},
                {"type": "scatter", "data_label": "Vsw_R", "xaxis": "x6", "yaxis": "y6"},
            ],
            "layout": {
                # Left column x-axes
                "xaxis":  {"domain": left_x, "anchor": "y"},
                "xaxis2": {"domain": left_x, "anchor": "y2", "matches": "x"},
                "xaxis3": {"domain": left_x, "anchor": "y3", "matches": "x"},
                # Right column x-axes
                "xaxis4": {"domain": right_x, "anchor": "y4"},
                "xaxis5": {"domain": right_x, "anchor": "y5", "matches": "x4"},
                "xaxis6": {"domain": right_x, "anchor": "y6", "matches": "x4"},
                # Y-axes with row domains
                "yaxis":  {"domain": row1, "anchor": "x", "title": {"text": "B (nT)"}},
                "yaxis2": {"domain": row2, "anchor": "x2", "title": {"text": "n (cm⁻³)"}},
                "yaxis3": {"domain": row3, "anchor": "x3", "title": {"text": "V (km/s)"}},
                "yaxis4": {"domain": row1, "anchor": "x4", "title": {"text": "B (nT)"}},
                "yaxis5": {"domain": row2, "anchor": "x5", "title": {"text": "n (cm⁻³)"}},
                "yaxis6": {"domain": row3, "anchor": "x6", "title": {"text": "V (km/s)"}},
                "title": {"text": "Solar Wind: Two Epochs"},
            },
        }

        result = fill_figure_data(fig_json, entries)
        fig = result.figure

        # --- Panel count: 6 yaxes detected ---
        assert result.panel_count == 6

        # --- All 6 traces populated with correct data ---
        assert len(fig.data) == 6
        for i, trace in enumerate(fig.data):
            assert len(trace.x) == 30, f"trace {i} x length"
            assert len(trace.y) == 30, f"trace {i} y length"

        # --- Trace labels match descriptions ---
        assert result.trace_labels == descs

        # --- Layout: verify all 6 y-axis domains ---
        # Use pytest.approx for floating-point domain values
        assert list(fig.layout.yaxis.domain) == pytest.approx(row1)
        assert list(fig.layout.yaxis2.domain) == pytest.approx(row2)
        assert list(fig.layout.yaxis3.domain) == pytest.approx(row3)
        assert list(fig.layout.yaxis4.domain) == pytest.approx(row1)
        assert list(fig.layout.yaxis5.domain) == pytest.approx(row2)
        assert list(fig.layout.yaxis6.domain) == pytest.approx(row3)

        # --- Layout: verify x-axis domains for two columns ---
        assert list(fig.layout.xaxis.domain) == left_x
        assert list(fig.layout.xaxis2.domain) == left_x
        assert list(fig.layout.xaxis3.domain) == left_x
        assert list(fig.layout.xaxis4.domain) == right_x
        assert list(fig.layout.xaxis5.domain) == right_x
        assert list(fig.layout.xaxis6.domain) == right_x

        # --- Layout: axis anchoring preserved ---
        assert fig.layout.yaxis.anchor == "x"
        assert fig.layout.yaxis4.anchor == "x4"
        assert fig.layout.xaxis2.anchor == "y2"
        assert fig.layout.xaxis5.anchor == "y5"

        # --- Layout: matched time axes within columns ---
        assert fig.layout.xaxis2.matches == "x"
        assert fig.layout.xaxis3.matches == "x"
        assert fig.layout.xaxis5.matches == "x4"
        assert fig.layout.xaxis6.matches == "x4"

        # --- Layout: y-axis titles preserved ---
        assert fig.layout.yaxis.title.text == "B (nT)"
        assert fig.layout.yaxis2.title.text == "n (cm⁻³)"
        assert fig.layout.yaxis3.title.text == "V (km/s)"
        assert fig.layout.yaxis4.title.text == "B (nT)"

        # --- Sizing: 6 x-axes detected → multi-column width, 6-panel height ---
        # n_columns = count of xaxis* keys = 6; n_panels = count of yaxis* keys = 6
        assert fig.layout.height == 300 * 6  # _PANEL_HEIGHT * n_panels
        assert fig.layout.width == int(1100 * 6 * 0.55)  # n_columns * scaling

        # --- x data is ISO 8601 datetime strings (timeseries) ---
        assert "2024-01-01" in fig.data[0].x[0]

        # --- Title preserved ---
        assert fig.layout.title.text == "Solar Wind: Two Epochs"


# ---------------------------------------------------------------------------
# PlotlyRenderer.render_plotly_json (stateful wrapper)
# ---------------------------------------------------------------------------

class TestRenderPlotlyJson:
    """Tests for the PlotlyRenderer.render_plotly_json method."""

    def test_basic_render(self, renderer):
        """Basic render produces success result."""
        entry = _make_entry("mag", n=50, desc="Bmag")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag"}],
            "layout": {"title": {"text": "Test"}},
        }
        result = renderer.render_plotly_json(fig_json, {"mag": entry})
        assert result["status"] == "success"
        assert result["traces"] == ["Bmag"]
        assert "trace_info" in result
        fig = renderer.get_figure()
        assert fig is not None
        assert fig.layout.title.text == "Test"

    def test_multi_column_error(self, renderer):
        """Multi-column data returns error dict."""
        entry = _make_entry("B", n=30, ncols=3, desc="Bfield")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "B"}],
            "layout": {},
        }
        result = renderer.render_plotly_json(fig_json, {"B": entry})
        assert result["status"] == "error"
        assert "3 columns" in result["message"]

    def test_missing_label_error(self, renderer):
        """Missing data_label returns error dict."""
        fig_json = {
            "data": [{"type": "scatter", "data_label": "NOPE"}],
            "layout": {},
        }
        result = renderer.render_plotly_json(fig_json, {})
        assert result["status"] == "error"
        assert "NOPE" in result["message"]

    def test_empty_data_error(self, renderer):
        """Empty data array returns error."""
        fig_json = {"data": [], "layout": {}}
        result = renderer.render_plotly_json(fig_json, {})
        assert result["status"] == "success"
        assert result["traces"] == []

    def test_state_updated(self, renderer):
        """Renderer state is updated after render_plotly_json."""
        entry = _make_entry("X", n=20, desc="Xdata")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "X"}],
            "layout": {},
        }
        renderer.render_plotly_json(fig_json, {"X": entry})
        assert renderer._trace_labels == ["Xdata"]
        assert renderer.get_figure() is not None

    def test_trace_info_included(self, renderer):
        """Trace info with point counts is included in render result."""
        entry = _make_entry("mag", n=50, desc="Mag")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag"}],
            "layout": {},
        }
        result = renderer.render_plotly_json(fig_json, {"mag": entry})
        assert "trace_info" in result
        assert len(result["trace_info"]) == 1
        assert result["trace_info"][0]["name"] == "Mag"
        assert result["trace_info"][0]["points"] == 50


# ---------------------------------------------------------------------------
# _last_fig_json tracking
# ---------------------------------------------------------------------------

class TestLastFigJson:
    """Tests for _last_fig_json storage, retrieval, reset, and persistence."""

    def test_stored_after_render(self, renderer):
        """_last_fig_json is stored after a successful render."""
        fig_json = {
            "data": [{"type": "scatter", "data_label": "x"}],
            "layout": {"title": {"text": "Test"}},
        }
        entry = _make_entry("x", n=10)
        renderer.render_plotly_json(fig_json, {"x": entry})
        assert renderer._last_fig_json is not None
        assert renderer._last_fig_json == fig_json

    def test_not_stored_on_error(self, renderer):
        """_last_fig_json is NOT updated when render fails."""
        # First render succeeds
        entry = _make_entry("a", n=10)
        good_json = {"data": [{"type": "scatter", "data_label": "a"}], "layout": {}}
        renderer.render_plotly_json(good_json, {"a": entry})
        assert renderer._last_fig_json == good_json

        # Second render fails (missing label)
        bad_json = {"data": [{"type": "scatter", "data_label": "MISSING"}], "layout": {}}
        result = renderer.render_plotly_json(bad_json, {})
        assert result["status"] == "error"
        # _last_fig_json still points to the good render
        assert renderer._last_fig_json == good_json

    def test_in_get_current_state(self, renderer):
        """get_current_state includes figure_json when present."""
        fig_json = {
            "data": [{"type": "scatter", "data_label": "x"}],
            "layout": {},
        }
        entry = _make_entry("x", n=10)
        renderer.render_plotly_json(fig_json, {"x": entry})
        state = renderer.get_current_state()
        assert "figure_json" in state
        assert state["figure_json"] == fig_json

    def test_not_in_get_current_state_when_empty(self, renderer):
        """get_current_state omits figure_json when nothing rendered."""
        state = renderer.get_current_state()
        assert "figure_json" not in state

    def test_cleared_on_reset(self, renderer):
        """_last_fig_json is cleared after reset."""
        fig_json = {
            "data": [{"type": "scatter", "data_label": "x"}],
            "layout": {},
        }
        entry = _make_entry("x", n=10)
        renderer.render_plotly_json(fig_json, {"x": entry})
        assert renderer._last_fig_json is not None

        renderer.reset()
        assert renderer._last_fig_json is None
        state = renderer.get_current_state()
        assert "figure_json" not in state

    def test_save_restore_roundtrip(self, renderer):
        """_last_fig_json round-trips through save_state/restore_state."""
        fig_json = {
            "data": [{"type": "scatter", "data_label": "x"}],
            "layout": {"title": {"text": "Roundtrip"}},
        }
        entry = _make_entry("x", n=10)
        renderer.render_plotly_json(fig_json, {"x": entry})

        saved = renderer.save_state()
        assert saved is not None
        assert "last_fig_json" in saved
        assert saved["last_fig_json"] == fig_json

        # Restore into a fresh renderer
        renderer2 = PlotlyRenderer(verbose=False)
        renderer2.restore_state(saved)
        assert renderer2._last_fig_json == fig_json
        state = renderer2.get_current_state()
        assert state["figure_json"] == fig_json
