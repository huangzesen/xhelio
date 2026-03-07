"""
Test that the LLM (Gemini) generates correct Plotly panel skeleton JSON.

Sends the actual visualization agent prompt to Gemini with a
render_plotly_json tool, asks for a 3-row × 2-column layout with
timeseries data, and validates the returned figure_json structure.

Requires GOOGLE_API_KEY in .env.  Skipped automatically if missing.

Run with: venv/bin/python -m pytest tests/test_viz_llm_skeleton.py -xvs
"""

import json
import os

import pytest

# Skip unless --slow AND API key are both present
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    ),
]


@pytest.fixture(scope="module")
def adapter():
    """Create a GeminiAdapter with the real API key."""
    from agent.llm.gemini_adapter import GeminiAdapter

    return GeminiAdapter(api_key=os.environ["GOOGLE_API_KEY"])


@pytest.fixture(scope="module")
def viz_system_prompt():
    """The actual visualization agent system prompt."""
    from knowledge.prompt_builder import build_viz_plotly_prompt

    return build_viz_plotly_prompt(gui_mode=False)


@pytest.fixture(scope="module")
def render_tool_schema():
    """The actual render_plotly_json tool schema as FunctionSchema."""
    from agent.llm.base import FunctionSchema
    from agent.tools import get_tool_schemas

    schemas = get_tool_schemas(names=["render_plotly_json"])
    for s in schemas:
        if s["name"] == "render_plotly_json":
            return FunctionSchema(
                name=s["name"],
                description=s["description"],
                parameters=s["parameters"],
            )
    pytest.fail("render_plotly_json schema not found in tool registry")


@pytest.fixture(scope="module")
def fake_data_listing():
    """Simulate the output of list_fetched_data for 6 timeseries entries."""
    labels = {
        "AC_H2_MFI.Bmag": {"units": "nT", "desc": "ACE B magnitude"},
        "AC_H2_SWE.Np": {"units": "cm⁻³", "desc": "ACE proton density"},
        "AC_H2_SWE.Vp": {"units": "km/s", "desc": "ACE solar wind speed"},
        "WI_H2_MFI.Bmag": {"units": "nT", "desc": "Wind B magnitude"},
        "WI_H2_SWE.Np": {"units": "cm⁻³", "desc": "Wind proton density"},
        "WI_H2_SWE.Vp": {"units": "km/s", "desc": "Wind solar wind speed"},
    }
    entries = []
    for label, info in labels.items():
        entries.append({
            "label": label,
            "columns": ["value"],
            "num_points": 5000,
            "shape": "scalar",
            "units": info["units"],
            "description": info["desc"],
            "source": "cdf",
            "is_timeseries": True,
            "time_min": "2024-01-01 00:00:00",
            "time_max": "2024-01-07 23:59:00",
        })
    return entries


def _extract_figure_json(response):
    """Extract figure_json from a tool call response.

    Returns (figure_json_dict, error_message).
    """
    if not response.tool_calls:
        return None, f"No tool calls in response. Text: {response.text[:500]}"
    for tc in response.tool_calls:
        if tc.name == "render_plotly_json":
            fig = tc.args.get("figure_json")
            if fig is None:
                return None, f"render_plotly_json called but no figure_json arg: {tc.args}"
            return fig, None
    names = [tc.name for tc in response.tool_calls]
    return None, f"No render_plotly_json call found. Tool calls: {names}"


class TestVizLLMSkeleton:
    """Test LLM generates correct panel skeleton for 3×2 grid layout."""

    @pytest.fixture(scope="class")
    def llm_figure_json(
        self, adapter, viz_system_prompt, render_tool_schema, fake_data_listing
    ):
        """Call Gemini once and cache the result for all tests in this class."""
        # Build the user request — include the fake data listing as context
        data_summary = json.dumps(fake_data_listing, indent=2)
        user_request = (
            f"Create a 3-row × 2-column comparison plot.\n\n"
            f"Left column: ACE data (AC_H2_MFI.Bmag, AC_H2_SWE.Np, AC_H2_SWE.Vp)\n"
            f"Right column: Wind data (WI_H2_MFI.Bmag, WI_H2_SWE.Np, WI_H2_SWE.Vp)\n\n"
            f"Row 1: magnetic field magnitude\n"
            f"Row 2: proton density\n"
            f"Row 3: solar wind speed\n\n"
            f"Use matching y-axis labels for each row across columns.\n\n"
            f"Data currently in memory:\n{data_summary}"
        )

        chat = adapter.create_chat(
            model="gemini-2.5-flash",
            system_prompt=viz_system_prompt,
            tools=[render_tool_schema],
            thinking="low",
        )
        response = chat.send(user_request)

        fig_json, err = _extract_figure_json(response)
        if err:
            pytest.fail(err)

        return fig_json

    def test_has_6_traces(self, llm_figure_json):
        """LLM should generate exactly 6 traces (one per panel)."""
        traces = llm_figure_json.get("data", [])
        assert len(traces) == 6, f"Expected 6 traces, got {len(traces)}: {traces}"

    def test_all_traces_have_data_label(self, llm_figure_json):
        """Every trace must have a data_label field."""
        for i, trace in enumerate(llm_figure_json["data"]):
            assert "data_label" in trace, f"trace {i} missing data_label: {trace}"

    def test_correct_labels_used(self, llm_figure_json):
        """Traces should reference exactly the 6 labels from the listing."""
        expected = {
            "AC_H2_MFI.Bmag", "AC_H2_SWE.Np", "AC_H2_SWE.Vp",
            "WI_H2_MFI.Bmag", "WI_H2_SWE.Np", "WI_H2_SWE.Vp",
        }
        actual = {t["data_label"] for t in llm_figure_json["data"]}
        assert actual == expected, f"Label mismatch: expected {expected}, got {actual}"

    def test_6_yaxes_defined(self, llm_figure_json):
        """Layout should define exactly 6 y-axes (yaxis through yaxis6)."""
        layout = llm_figure_json.get("layout", {})
        yaxes = [k for k in layout if k.startswith("yaxis")]
        assert len(yaxes) == 6, f"Expected 6 yaxes, got {len(yaxes)}: {yaxes}"

    def test_6_xaxes_defined(self, llm_figure_json):
        """Layout should define exactly 6 x-axes (xaxis through xaxis6)."""
        layout = llm_figure_json.get("layout", {})
        xaxes = [k for k in layout if k.startswith("xaxis")]
        assert len(xaxes) == 6, f"Expected 6 xaxes, got {len(xaxes)}: {xaxes}"

    def test_two_column_x_domains(self, llm_figure_json):
        """X-axes should form two distinct domain groups (left and right columns)."""
        layout = llm_figure_json.get("layout", {})
        xaxes = {k: v for k, v in layout.items() if k.startswith("xaxis")}

        # Extract unique x-domain start values to identify columns
        domain_starts = set()
        for ax_name, ax_def in xaxes.items():
            domain = ax_def.get("domain", [0, 1])
            domain_starts.add(round(domain[0], 2))

        assert len(domain_starts) == 2, (
            f"Expected 2 column domain starts, got {domain_starts}. "
            f"X-axes: {json.dumps(xaxes, indent=2)}"
        )

    def test_three_row_y_domains(self, llm_figure_json):
        """Y-axes should form 3 distinct row domain groups."""
        layout = llm_figure_json.get("layout", {})
        yaxes = {k: v for k, v in layout.items() if k.startswith("yaxis")}

        # Each row pair (left+right) should share the same y-domain
        domain_groups = set()
        for ax_name, ax_def in yaxes.items():
            domain = ax_def.get("domain", [0, 1])
            # Round to avoid float precision issues
            domain_groups.add((round(domain[0], 2), round(domain[1], 2)))

        assert len(domain_groups) == 3, (
            f"Expected 3 row domain groups, got {len(domain_groups)}: {domain_groups}. "
            f"Y-axes: {json.dumps(yaxes, indent=2)}"
        )

    def test_y_domains_non_overlapping(self, llm_figure_json):
        """Y-axis domains should not overlap (rows should be stacked)."""
        layout = llm_figure_json.get("layout", {})
        yaxes = {k: v for k, v in layout.items() if k.startswith("yaxis")}

        # Get unique domain ranges
        domains = set()
        for ax_def in yaxes.values():
            d = ax_def.get("domain", [0, 1])
            domains.add((round(d[0], 2), round(d[1], 2)))

        domains_sorted = sorted(domains, key=lambda d: d[0])
        for i in range(len(domains_sorted) - 1):
            _, top = domains_sorted[i]
            bottom_next, _ = domains_sorted[i + 1]
            assert top <= bottom_next + 0.01, (
                f"Y domains overlap: {domains_sorted[i]} and {domains_sorted[i+1]}"
            )

    def test_traces_reference_valid_axes(self, llm_figure_json):
        """Every trace's xaxis/yaxis reference should correspond to a layout axis."""
        layout = llm_figure_json.get("layout", {})
        traces = llm_figure_json.get("data", [])

        # Build set of defined axis references
        # xaxis → "x", xaxis2 → "x2", etc.
        defined_x = set()
        defined_y = set()
        for key in layout:
            if key.startswith("xaxis"):
                suffix = key[5:]  # "" for xaxis, "2" for xaxis2, etc.
                defined_x.add(f"x{suffix}")
            elif key.startswith("yaxis"):
                suffix = key[5:]
                defined_y.add(f"y{suffix}")

        for i, trace in enumerate(traces):
            xref = trace.get("xaxis", "x")
            yref = trace.get("yaxis", "y")
            assert xref in defined_x, (
                f"trace {i} references xaxis='{xref}' but layout has {defined_x}"
            )
            assert yref in defined_y, (
                f"trace {i} references yaxis='{yref}' but layout has {defined_y}"
            )

    def test_x_axes_have_matches(self, llm_figure_json):
        """Within each column, secondary x-axes should use 'matches' for zoom sync."""
        layout = llm_figure_json.get("layout", {})
        xaxes = {k: v for k, v in layout.items() if k.startswith("xaxis")}

        # Group axes by column (same domain)
        from collections import defaultdict
        columns = defaultdict(list)
        for name, ax_def in xaxes.items():
            domain = tuple(round(d, 2) for d in ax_def.get("domain", [0, 1]))
            columns[domain].append((name, ax_def))

        for domain, axes in columns.items():
            if len(axes) <= 1:
                continue
            # At least one secondary axis should have "matches"
            has_matches = any(
                ax_def.get("matches") is not None
                for _, ax_def in axes
            )
            assert has_matches, (
                f"Column domain={domain}: no 'matches' for zoom sync among {[n for n, _ in axes]}"
            )

    def test_figure_json_renders_successfully(self, llm_figure_json):
        """The generated JSON should pass through fill_figure_data without error."""
        import numpy as np
        import pandas as pd
        from data_ops.store import DataEntry
        from rendering.plotly_renderer import fill_figure_data

        # Create fake entries matching the labels
        entries = {}
        for trace in llm_figure_json["data"]:
            label = trace["data_label"]
            if label not in entries:
                rng = pd.date_range("2024-01-01", periods=100, freq="min")
                df = pd.DataFrame({"value": np.random.randn(100)}, index=rng)
                entries[label] = DataEntry(
                    label=label, data=df, units="test", description=label,
                )

        result = fill_figure_data(llm_figure_json, entries)
        assert result.panel_count == 6
        assert len(result.figure.data) == 6
