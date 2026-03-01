"""
Plotly-based renderer for visualization.

The ``fill_figure_data()`` function resolves ``data_label`` placeholders
in LLM-generated Plotly figure JSON, populating real data arrays.

The ``PlotlyRenderer`` class is a thin stateful wrapper providing:
- ``render_plotly_json()`` — fill data → copy into state
- ``export()``, ``reset()`` — structural operations
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_ops.store import DataEntry

logger = logging.getLogger("xhelio")

# Explicit layout defaults — prevent dark theme CSS from overriding
_DEFAULT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_color="#2a3f5f",
    autosize=False,
)

_PANEL_HEIGHT = 300  # px per subplot panel
_DEFAULT_WIDTH = 1100  # px figure width

# Stride thresholds for large traces
_MAX_PLOT_POINTS_DEFAULT = 10_000  # fallback if config unavailable
_WEBGL_THRESHOLD = 100_000  # switch to scattergl above this


def _get_max_plot_points() -> int:
    """Read max_plot_points from config (lazy import to avoid circular deps)."""
    try:
        from config import MAX_PLOT_POINTS
        return int(MAX_PLOT_POINTS)
    except Exception:
        return _MAX_PLOT_POINTS_DEFAULT


def _wrap_axis_title(text: str, max_len: int = 20) -> str:
    """Insert <br> into a long axis title at word boundaries."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = f"{current} {w}".strip() if current else w
        if len(candidate) > max_len and current:
            lines.append(current)
            current = w
        else:
            current = candidate
    if current:
        lines.append(current)
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# RenderResult — return type of fill_figure_data
# ---------------------------------------------------------------------------

class RenderResult:
    """Result of a stateless fill_figure_data() call."""

    __slots__ = ("figure", "trace_labels", "panel_count", "stride_info")

    def __init__(
        self,
        figure: go.Figure,
        trace_labels: list[str],
        panel_count: int,
        stride_info: list[dict] | None = None,
    ):
        self.figure = figure
        self.trace_labels = trace_labels
        self.panel_count = panel_count
        self.stride_info = stride_info or []


# ---------------------------------------------------------------------------
# Stateless figure builder
# ---------------------------------------------------------------------------

def _extract_index_data(entry: DataEntry) -> tuple[list, bool]:
    """Extract index values from a DataEntry for use as x-axis data.

    DatetimeIndex / xarray time coords → ISO 8601 strings (is_datetime=True).
    Numeric or other indices → raw values as list (is_datetime=False).

    Returns:
        Tuple of (values list, is_datetime flag).
    """
    if entry.is_xarray:
        import pandas as pd
        return [pd.Timestamp(t).isoformat() for t in entry.data.coords["time"].values], True
    idx = entry.data.index
    if hasattr(idx, 'tz') or str(idx.dtype).startswith('datetime'):
        return [t.isoformat() for t in idx], True
    return list(idx), False


def _update_axis_range(
    ranges: dict[str, list[str]], ax_key: str, x_min: str, x_max: str,
) -> None:
    """Expand the tracked [min, max] range for a given axis key."""
    if ax_key not in ranges:
        ranges[ax_key] = [x_min, x_max]
    else:
        if x_min < ranges[ax_key][0]:
            ranges[ax_key][0] = x_min
        if x_max > ranges[ax_key][1]:
            ranges[ax_key][1] = x_max


# ---------------------------------------------------------------------------
# Fill function — resolves data_label placeholders in Plotly JSON
# ---------------------------------------------------------------------------

def fill_figure_data(
    fig_json: dict,
    entries: dict[str, DataEntry],
) -> RenderResult:
    """Fill data_label placeholders in a Plotly figure JSON with actual data.

    The LLM generates a Plotly figure JSON where each trace has a ``data_label``
    field instead of actual x/y/z arrays.  This function resolves those
    placeholders by looking up the corresponding DataEntry objects, extracting
    time and value arrays, and populating the trace dicts.

    Handles:
    - Time extraction (DatetimeIndex/xarray → ISO 8601 strings)
    - NaN → None conversion (Plotly requirement)
    - Spectrogram/heatmap data population (x=times, y=bins, z=values)

    For multi-column (vector) data, the LLM must emit one trace per column.
    If a single trace references multi-column data, an error is raised.

    Args:
        fig_json: Plotly figure dict with ``data`` and ``layout`` keys.
            Each trace in ``data`` must have a ``data_label`` field.
        entries: Mapping of label → DataEntry for all referenced labels.

    Returns:
        RenderResult with the constructed go.Figure and metadata.
    """
    layout = fig_json.get("layout", {})
    raw_traces = fig_json.get("data", [])

    filled_traces: list[dict] = []
    trace_labels: list[str] = []
    stride_info: list[dict] = []
    # Track which x-axes carry datetime data (for setting type: "date")
    datetime_xaxes: set[str] = set()
    # Track x-data range per datetime axis (for explicit range on auto-ranged axes)
    # Maps layout key ("xaxis", "xaxis2") → [min_iso, max_iso]
    datetime_xaxis_ranges: dict[str, list[str]] = {}

    for trace_dict in raw_traces:
        trace = dict(trace_dict)  # shallow copy
        label = trace.pop("data_label", None)
        if label is None:
            # Trace without data_label — pass through as-is (e.g., shapes)
            filled_traces.append(trace)
            continue

        # Resolve entry
        entry = entries.get(label)
        if entry is None:
            raise ValueError(f"data_label '{label}' not found in provided entries")

        trace_type = trace.get("type", "scatter").lower()
        is_heatmap = trace_type in ("heatmap", "heatmapgl")

        display_name = trace.get("name") or entry.description or entry.label

        if is_heatmap:
            _fill_heatmap_trace(trace, entry)
            # spectrograms are always time-based
            # Map trace xaxis ref ("x", "x2") → layout key ("xaxis", "xaxis2")
            xref = trace.get("xaxis", "x")
            ax_key = "xaxis" + xref[1:]  # "x" → "xaxis", "x2" → "xaxis2"
            datetime_xaxes.add(ax_key)
            # Track heatmap x range
            x_vals = trace.get("x", [])
            if x_vals:
                _update_axis_range(datetime_xaxis_ranges, ax_key, x_vals[0], x_vals[-1])
        else:
            # Reject multi-column data — LLM must emit one trace per column
            if entry.values.ndim == 2 and entry.values.shape[1] > 1:
                ncols = entry.values.shape[1]
                col_names = list(entry.data.columns) if hasattr(entry.data, 'columns') else [str(i) for i in range(ncols)]
                sub_labels = [f"'{entry.label}.{c}'" for c in col_names]
                raise ValueError(
                    f"Entry '{entry.label}' has {ncols} columns. "
                    f"Use one trace per column with data_labels: {', '.join(sub_labels)}"
                )
            vals = entry.values.ravel() if entry.values.ndim > 1 else entry.values
            x_data, is_dt = _extract_index_data(entry)
            if is_dt:
                xref = trace.get("xaxis", "x")
                ax_key = "xaxis" + xref[1:]
                datetime_xaxes.add(ax_key)
                # Track data range for this axis
                if x_data:
                    _update_axis_range(datetime_xaxis_ranges, ax_key, x_data[0], x_data[-1])

            # Stride large traces to keep figure JSON small and rendering fast
            n_original = len(vals)
            max_pts = _get_max_plot_points()
            if n_original > max_pts:
                stride = n_original // max_pts
                x_data = x_data[::stride]
                vals = vals[::stride]
                n_after = len(vals)
                logger.info(
                    "[Render] Strided '%s': %s → %s points (every %sth point)",
                    display_name, f"{n_original:,}", f"{n_after:,}", f"{stride:,}",
                )
                stride_info.append({
                    "trace": display_name,
                    "original": n_original,
                    "plotted": n_after,
                    "stride": stride,
                })
                # Switch to WebGL for very large original datasets
                if n_original > _WEBGL_THRESHOLD and trace_type == "scatter":
                    trace["type"] = "scattergl"
                    logger.debug("[Render] Switched '%s' to scattergl (WebGL)", display_name)

            trace["x"] = x_data
            # Vectorized conversion: mask non-finite values as None for Plotly.
            # Uses numpy masked array → filled with NaN → object array with
            # None, which is faster than a Python-level per-element loop.
            mask = ~np.isfinite(vals)
            if mask.any():
                result = vals.astype(object)
                result[mask] = None
                trace["y"] = result.tolist()
            else:
                trace["y"] = vals.tolist()
            trace.setdefault("mode", "lines")

        trace["name"] = display_name
        filled_traces.append(trace)
        trace_labels.append(display_name)

    # Count panels from layout
    n_panels = sum(1 for key in layout if key.startswith("yaxis"))
    n_panels = max(n_panels, 1)
    n_columns = sum(1 for key in layout if key.startswith("xaxis"))
    n_columns = max(n_columns, 1)

    # Apply default layout settings
    width = (_DEFAULT_WIDTH if n_columns == 1
             else int(_DEFAULT_WIDTH * n_columns * 0.55))
    height = _PANEL_HEIGHT * n_panels

    defaults = dict(
        **_DEFAULT_LAYOUT,
        width=width,
        height=height,
        legend=dict(font=dict(size=11), tracegroupgap=2),
    )
    # Merge defaults under layout (user layout wins)
    merged_layout = {**defaults, **layout}

    # Ensure x-axes with datetime traces have type: "date" set explicitly.
    # Without this, Plotly can misinterpret ISO 8601 strings in static export
    # when no range is set (e.g., per-column zoom where one column has a range
    # and the other relies on auto-range).
    #
    # Also set explicit range on datetime axes that don't already have one.
    # This works around a kaleido bug where `matches` + auto-range + datetime
    # produces wrong date scales in static export (e.g., "Jan 2000" instead of
    # "Jan 2024").
    for key in datetime_xaxes:
        if key in merged_layout and isinstance(merged_layout[key], dict):
            merged_layout[key].setdefault("type", "date")
            if key in datetime_xaxis_ranges and "range" not in merged_layout[key]:
                merged_layout[key]["range"] = datetime_xaxis_ranges[key]

    # Auto-fill "matches" on secondary x-axes for zoom synchronization.
    # Group x-axes by column (same domain), then for each column with 2+
    # axes, set "matches" on secondary axes pointing to the primary.
    from collections import defaultdict
    xaxis_columns: dict[tuple, list[str]] = defaultdict(list)
    for key, val in merged_layout.items():
        if key.startswith("xaxis") and isinstance(val, dict):
            domain = tuple(round(d, 2) for d in val.get("domain", [0, 1]))
            xaxis_columns[domain].append(key)
    for _domain, axes in xaxis_columns.items():
        if len(axes) <= 1:
            continue
        # Sort so "xaxis" < "xaxis2" < "xaxis3" — primary is first
        axes.sort(key=lambda k: int(k[5:] or "1"))
        primary = axes[0]
        # Plotly "matches" uses the trace-axis ref: "xaxis" → "x", "xaxis2" → "x2"
        primary_ref = "x" + primary[5:]  # "xaxis" → "x", "xaxis2" → "x2"
        for secondary in axes[1:]:
            merged_layout[secondary].setdefault("matches", primary_ref)

    # Auto-wrap long y-axis titles with <br> so they don't get clipped
    for key, val in merged_layout.items():
        if key.startswith("yaxis") and isinstance(val, dict):
            title = val.get("title")
            if isinstance(title, str) and len(title) > 20 and "<br>" not in title:
                val["title"] = _wrap_axis_title(title, max_len=20)
            elif isinstance(title, dict):
                text = title.get("text", "")
                if len(text) > 20 and "<br>" not in text:
                    title["text"] = _wrap_axis_title(text, max_len=20)

    fig = go.Figure({"data": filled_traces, "layout": merged_layout})

    return RenderResult(
        figure=fig,
        trace_labels=trace_labels,
        panel_count=n_panels,
        stride_info=stride_info,
    )


def _fill_heatmap_trace(
    trace: dict,
    entry: DataEntry,
) -> None:
    """Populate a heatmap trace dict with data from a DataEntry."""
    meta = entry.metadata or {}
    times, _ = _extract_index_data(entry)

    if entry.is_xarray:
        da = entry.data
        non_time_dims = [d for d in da.dims if d != "time"]
        if non_time_dims:
            last_dim = non_time_dims[-1]
            if last_dim in da.coords:
                bin_values = [float(v) for v in da.coords[last_dim].values]
            else:
                bin_values = list(range(da.sizes[last_dim]))
        else:
            bin_values = [0]
        z_values = da.values.astype(float)
        if z_values.ndim > 2:
            middle_axes = tuple(range(1, z_values.ndim - 1))
            z_values = np.nanmean(z_values, axis=middle_axes)
    else:
        bin_values = meta.get("bin_values")
        if bin_values is None:
            try:
                bin_values = [float(c) for c in entry.data.columns]
            except (ValueError, TypeError):
                bin_values = list(range(len(entry.data.columns)))
        z_values = entry.data.values.astype(float)

    trace["x"] = times
    trace["y"] = [float(b) for b in bin_values]
    trace["z"] = z_values.T.tolist()


class PlotlyRenderer:
    """Stateful Plotly renderer for heliophysics data visualization."""

    def __init__(self, verbose: bool = False, gui_mode: bool = False):
        self.verbose = verbose
        self.gui_mode = gui_mode
        self._figure: Optional[go.Figure] = None
        self._panel_count: int = 0
        self._trace_labels: list[str] = []
        self._last_fig_json: dict | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [PlotlyRenderer] {msg}")
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, filepath: str, format: str = "png") -> dict:
        """Export the current plot to a file (PNG or PDF).

        Args:
            filepath: Output file path.
            format: 'png' (default) or 'pdf'.

        Returns:
            Result dict with status, filepath, and size_bytes.
        """
        # Ensure correct extension
        if format == "pdf" and not filepath.endswith(".pdf"):
            filepath += ".pdf"
        elif format == "png" and not filepath.endswith(".png"):
            filepath += ".png"

        filepath = str(Path(filepath).resolve())
        parent = Path(filepath).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        if self._figure is None or len(self._figure.data) == 0:
            return {"status": "error",
                    "message": "No plot to export. Plot data first before exporting."}

        self._log(f"Exporting {format.upper()} to {filepath}...")
        try:
            self._figure.write_image(filepath, format=format)
        except Exception as e:
            return {"status": "error", "message": f"{format.upper()} export failed: {e}"}

        path_obj = Path(filepath)
        if path_obj.exists() and path_obj.stat().st_size > 0:
            return {
                "status": "success",
                "filepath": str(path_obj.resolve()),
                "size_bytes": path_obj.stat().st_size,
            }
        return {"status": "error", "message": f"{format.upper()} file not created or is empty: {filepath}"}

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        self._log("Resetting canvas...")
        self._figure = None
        self._panel_count = 0
        self._trace_labels.clear()
        self._last_fig_json = None
        return {"status": "success", "message": "Canvas reset."}

    def get_current_state(self) -> dict:
        state = {
            "uri": None,
            "panel_count": self._panel_count,
            "has_plot": self._figure is not None and len(self._figure.data) > 0,
            "traces": list(self._trace_labels),
        }
        if self._last_fig_json is not None:
            state["figure_json"] = self._last_fig_json
        return state

    # ------------------------------------------------------------------
    # Public API: render_plotly_json
    # ------------------------------------------------------------------

    def render_plotly_json(
        self,
        fig_json: dict,
        entries: dict[str, DataEntry],
    ) -> dict:
        """Create a plot from LLM-generated Plotly figure JSON.

        The LLM produces a Plotly figure dict with ``data_label`` placeholders
        in each trace.  This method resolves them via ``fill_figure_data()``,
        copies the result into renderer state, and returns basic metadata.

        Args:
            fig_json: Plotly figure dict (``data`` + ``layout``).
            entries: Mapping of label → DataEntry for all referenced labels.

        Returns:
            Result dict with status, panels, traces, display.
        """
        try:
            build_result = fill_figure_data(fig_json, entries)
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": str(e)}

        # Copy into stateful fields
        self._figure = build_result.figure
        self._panel_count = build_result.panel_count
        self._trace_labels = build_result.trace_labels
        self._last_fig_json = fig_json

        # Build basic trace info for the LLM
        trace_info = []
        for i, trace in enumerate(self._figure.data):
            name = self._trace_labels[i] if i < len(self._trace_labels) else (trace.name or f"trace_{i}")
            y = trace.y
            if y is not None:
                points = len(y)
            elif hasattr(trace, 'z') and trace.z is not None:
                z_arr = np.asarray(trace.z)
                points = f"{z_arr.shape[0]}x{z_arr.shape[1]}" if z_arr.ndim == 2 else len(trace.z)
            else:
                points = 0
            trace_info.append({"name": name, "points": points})

        result = {
            "status": "success",
            "panels": build_result.panel_count,
            "traces": list(build_result.trace_labels),
            "trace_info": trace_info,
            "display": "plotly",
        }
        if build_result.stride_info:
            result["stride_applied"] = build_result.stride_info
        return result

    # ------------------------------------------------------------------
    # Accessor for web UI / external use
    # ------------------------------------------------------------------

    def get_figure(self) -> Optional[go.Figure]:
        """Return the current Plotly figure (or None if nothing plotted)."""
        return self._figure

    # ------------------------------------------------------------------
    # Serialization for session persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict | None:
        """Serialize the renderer state (figure + metadata) to a dict.

        Saves only the compact LLM template (``last_fig_json`` with
        ``data_label`` placeholders) instead of the full Plotly figure.
        The full figure is reconstructed from the DataStore on restore.

        Falls back to serializing the full figure only when
        ``_last_fig_json`` is unavailable (e.g. figures built outside
        ``render_plotly_json``).

        Returns None if there is no figure to save.
        """
        if self._figure is None:
            return None
        state: dict = {
            "panel_count": self._panel_count,
            "trace_labels": list(self._trace_labels),
        }
        if self._last_fig_json is not None:
            # Compact path: save only the LLM template (~5-10 KB)
            state["last_fig_json"] = self._last_fig_json
        else:
            # Fallback: save the full figure (legacy / non-render_plotly_json path)
            state["figure_json"] = self._figure.to_json()
        return state

    def restore_state(
        self,
        state: dict,
        entries: dict[str, "DataEntry"] | None = None,
    ) -> None:
        """Restore renderer state from a dict produced by save_state().

        Args:
            state: Dict from ``save_state()``.
            entries: Optional mapping of data_label → DataEntry.  When
                provided alongside ``last_fig_json``, the full figure is
                reconstructed via ``fill_figure_data()`` (fast path).
                Otherwise falls back to ``pio.from_json()`` for legacy
                sessions that saved the full ``figure_json``.
        """
        last_fig_json = state.get("last_fig_json")

        # Always restore metadata
        self._panel_count = state.get("panel_count", 0)
        self._trace_labels = state.get("trace_labels", [])
        self._last_fig_json = last_fig_json

        # New fast path: replay from compact LLM template + DataStore entries
        if last_fig_json and entries:
            try:
                result = fill_figure_data(last_fig_json, entries)
                self._figure = result.figure
                self._panel_count = result.panel_count
                self._trace_labels = result.trace_labels
                return
            except Exception:
                # If replay fails (e.g. missing entry), fall through to legacy
                pass

        # Legacy path: full figure JSON stored in state
        fig_json = state.get("figure_json")
        if not fig_json:
            return

        import plotly.io as pio

        self._figure = pio.from_json(fig_json)
        self._panel_count = state.get("panel_count", 0)
        self._trace_labels = state.get("trace_labels", [])
