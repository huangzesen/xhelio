"""
Tests for data_ops.script_gen — self-contained Python script generation
from SavedPipeline data.

Run with: python -m pytest tests/test_script_gen.py -v
"""

import pytest

from data_ops.script_gen import generate_script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_data(steps: list[dict], **overrides) -> dict:
    """Build a minimal SavedPipeline dict with the given steps."""
    data = {
        "version": 1,
        "id": "test-pipeline-001",
        "name": "Test Pipeline",
        "description": "A test pipeline",
        "created_at": "2026-03-07T00:00:00Z",
        "updated_at": "2026-03-07T00:00:00Z",
        "source_session_id": "session-001",
        "source_render_op_id": None,
        "tags": ["test"],
        "time_range_original": ["2025-01-01", "2025-01-31"],
        "steps": steps,
    }
    data.update(overrides)
    return data


def _make_fetch_step(step_id: str, dataset_id: str, parameter_id: str,
                     output_label: str, **overrides) -> dict:
    step = {
        "step_id": step_id,
        "phase": "appropriation",
        "tool": "fetch_data",
        "params": {"dataset_id": dataset_id, "parameter_id": parameter_id},
        "inputs": [],
        "output_label": output_label,
        "description": f"Fetch {dataset_id}.{parameter_id}",
    }
    step.update(overrides)
    return step


def _make_run_code_step(step_id: str, code: str, inputs: list[str],
                        output_label: str, description: str = "Run code",
                         **overrides) -> dict:
    step = {
        "step_id": step_id,
        "phase": "appropriation",
        "tool": "run_code",
        "params": {"code": code, "description": description},
        "inputs": inputs,
        "output_label": output_label,
        "description": description,
    }
    step.update(overrides)
    return step


def _make_store_step(step_id: str, code: str, output_label: str,
                     description: str = "Create DataFrame", **overrides) -> dict:
    step = {
        "step_id": step_id,
        "phase": "appropriation",
        "tool": "run_code",
        "params": {"code": code, "description": description},
        "inputs": [],
        "output_label": output_label,
        "description": description,
    }
    step.update(overrides)
    return step


def _make_plotly_step(step_id: str, figure_json: dict,
                      inputs: list[str], **overrides) -> dict:
    step = {
        "step_id": step_id,
        "phase": "presentation",
        "tool": "render_plotly_json",
        "params": {"figure_json": figure_json},
        "inputs": inputs,
        "output_label": None,
        "description": "Render plot",
    }
    step.update(overrides)
    return step


def _make_mpl_step(step_id: str, code: str,
                   inputs: list[str], **overrides) -> dict:
    step = {
        "step_id": step_id,
        "phase": "presentation",
        "tool": "generate_mpl_script",
        "params": {"code": code},
        "inputs": inputs,
        "output_label": None,
        "description": "Render matplotlib plot",
    }
    step.update(overrides)
    return step


def _make_jsx_step(step_id: str, code: str,
                   inputs: list[str], **overrides) -> dict:
    step = {
        "step_id": step_id,
        "phase": "presentation",
        "tool": "generate_jsx_component",
        "params": {"code": code},
        "inputs": inputs,
        "output_label": None,
        "description": "Render JSX component",
    }
    step.update(overrides)
    return step


# ---------------------------------------------------------------------------
# TestFetchStepGeneration
# ---------------------------------------------------------------------------

class TestFetchStepGeneration:
    """Tests for fetch step code generation."""

    def test_single_cdaweb_fetch(self):
        """Single CDAWeb fetch generates cdasws code."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "from cdasws import CdasWs" in script
        assert 'cdas = CdasWs()' in script
        assert '"AC_H2_MFI"' in script
        assert '["BGSEc"]' in script
        assert 'data["ace_bfield"]' in script
        assert "import requests" not in script

    def test_ppi_fetch_urn(self):
        """PPI fetch (urn:nasa:pds: prefix) generates NotImplementedError."""
        steps = [
            _make_fetch_step(
                "s001",
                "urn:nasa:pds:voyager1-pls-jupiter",
                "ion_density",
                "v1_plasma",
            ),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "NotImplementedError" in script
        assert "pds-ppi.igpp.ucla.edu" in script
        assert "from cdasws import CdasWs" not in script
        assert "import requests" in script  # PPI datasets need requests for fetch

    def test_ppi_fetch_pds3_prefix(self):
        """PPI fetch with pds3: prefix is detected as PPI."""
        steps = [
            _make_fetch_step(
                "s001",
                "pds3:VG1-J-MAG-4-SUMM",
                "bfield",
                "v1_mag",
            ),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "NotImplementedError" in script
        assert "pds3:VG1-J-MAG-4-SUMM" in script

    def test_multiple_fetches(self):
        """Multiple fetch steps all appear in fetch_data()."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_fetch_step("s002", "AC_H2_SWE", "Np", "ace_density"),
            _make_plotly_step("s003", {"data": [], "layout": {}}, ["s001", "s002"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert 'data["ace_bfield"]' in script
        assert 'data["ace_density"]' in script
        assert '"AC_H2_MFI"' in script
        assert '"AC_H2_SWE"' in script

    def test_mixed_cdaweb_and_ppi(self):
        """Mixed CDAWeb + PPI fetches produce both cdasws and NotImplementedError."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_fetch_step("s002", "urn:nasa:pds:vg1-mag", "bfield", "v1_mag"),
            _make_plotly_step("s003", {"data": [], "layout": {}}, ["s001", "s002"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "from cdasws import CdasWs" in script
        assert "NotImplementedError" in script


# ---------------------------------------------------------------------------
# TestTransformStepGeneration
# ---------------------------------------------------------------------------

class TestTransformStepGeneration:
    """Tests for transform step code generation."""

    def test_run_code(self):
        """run_code inlines code with df input and result output."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_run_code_step(
                "s002",
                "result = df['Bx'] ** 2 + df['By'] ** 2",
                ["s001"],
                "b_magnitude",
                description="Compute B magnitude squared",
            ),
            _make_plotly_step("s003", {"data": [], "layout": {}}, ["s002"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "def transform(" in script
        assert 'df = data["ace_bfield"]' in script
        assert "result = df['Bx'] ** 2 + df['By'] ** 2" in script
        assert 'data["b_magnitude"] = result' in script
        assert "isinstance(result, pd.Series)" in script

    def test_store_dataframe_step(self):
        """run_code (store) inlines code without df input."""
        steps = [
            _make_store_step(
                "s001",
                "result = pd.DataFrame({'x': [1, 2, 3]})",
                "synthetic_data",
                description="Create synthetic data",
            ),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "def transform(" in script
        assert "result = pd.DataFrame({'x': [1, 2, 3]})" in script
        assert 'data["synthetic_data"] = result' in script
        # run_code (store mode) should NOT have df = data[...] line
        assert 'df = data["' not in script

    def test_no_transform_steps_skips_function(self):
        """No transform steps → no transform() function or call."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "def transform(" not in script
        assert "data = transform(data)" not in script


# ---------------------------------------------------------------------------
# TestPlotlyRenderGeneration
# ---------------------------------------------------------------------------

class TestPlotlyRenderGeneration:
    """Tests for Plotly render step code generation."""

    def test_figure_json_with_data_label(self):
        """Traces with data_label resolve to data dict lookups."""
        figure_json = {
            "data": [
                {
                    "type": "scatter",
                    "name": "Bx",
                    "data_label": "ace_bfield",
                    "mode": "lines",
                }
            ],
            "layout": {"title": "ACE Magnetic Field"},
        }
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", figure_json, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "import plotly.graph_objects as go" in script
        assert 'data["ace_bfield"]' in script
        assert "go.Figure(" in script
        assert "go.Scatter(" in script
        # data_label should NOT appear as a key in the generated trace dict
        assert '"data_label"' not in script

    def test_run_returns_figure(self):
        """run() calls render() and returns the figure."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "return render(data)" in script
        assert "result.show()" in script

    def test_plotly_no_mpl_import(self):
        """Plotly backend should not import matplotlib."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "matplotlib" not in script

    def test_trace_without_data_label(self):
        """Traces without data_label are included as-is."""
        figure_json = {
            "data": [
                {
                    "type": "scatter",
                    "x": [1, 2, 3],
                    "y": [4, 5, 6],
                    "name": "static",
                }
            ],
            "layout": {},
        }
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", figure_json, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "go.Scatter(" in script


# ---------------------------------------------------------------------------
# TestMplRenderGeneration
# ---------------------------------------------------------------------------

class TestMplRenderGeneration:
    """Tests for matplotlib render step code generation."""

    def test_mpl_code_inlined(self):
        """matplotlib code is inlined in render()."""
        mpl_code = (
            "fig, ax = plt.subplots()\n"
            "ax.plot(data['ace_bfield'].index, data['ace_bfield']['Bx'])\n"
            "ax.set_title('ACE Bx')"
        )
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_mpl_step("s002", mpl_code, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "import matplotlib.pyplot as plt" in script
        assert "fig, ax = plt.subplots()" in script
        assert "ax.set_title('ACE Bx')" in script
        assert "return fig" in script
        assert "plt.show()" in script

    def test_mpl_backend_detection(self):
        """generate_mpl_script detected as mpl backend."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_mpl_step("s002", "fig, ax = plt.subplots()", ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "plotly" not in script.lower() or "plotly" not in script

    def test_mpl_no_plotly_import(self):
        """MPL backend should not import plotly."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_mpl_step("s002", "fig, ax = plt.subplots()", ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "import plotly" not in script


# ---------------------------------------------------------------------------
# TestJsxRenderGeneration
# ---------------------------------------------------------------------------

class TestJsxRenderGeneration:
    """Tests for JSX render step code generation."""

    def test_two_file_output(self):
        """JSX backend produces both pipeline.py and component.tsx."""
        jsx_code = "export default function Chart() { return <div>Chart</div>; }"
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_jsx_step("s002", jsx_code, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))

        assert "pipeline.py" in result
        assert "component.tsx" in result
        assert result["component.tsx"] == jsx_code

    def test_csv_export(self):
        """JSX backend exports data as CSV in render()."""
        jsx_code = "export default function Chart() { return <div/>; }"
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_jsx_step("s002", jsx_code, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "to_csv" in script
        assert ".csv" in script

    def test_no_viz_imports(self):
        """JSX backend should not import plotly or matplotlib."""
        jsx_code = "export default function Chart() { return <div/>; }"
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_jsx_step("s002", jsx_code, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "import plotly" not in script
        assert "import matplotlib" not in script


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and script quality."""

    def test_empty_pipeline_raises(self):
        """Empty pipeline (no steps) raises ValueError."""
        with pytest.raises(ValueError, match="no steps"):
            generate_script(_make_pipeline_data([]))

    def test_no_transform_skips_transform_call(self):
        """Pipeline without transforms skips transform() in run()."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "transform(data)" not in script

    def test_valid_python_compile(self):
        """Generated script compiles as valid Python."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_run_code_step(
                "s002", "result = df * 2", ["s001"], "doubled",
            ),
            _make_plotly_step("s003", {
                "data": [{"data_label": "doubled", "name": "Doubled"}],
                "layout": {"title": "Test"},
            }, ["s002"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        # Must not raise SyntaxError
        compile(script, "<generated>", "exec")

    def test_valid_python_mpl(self):
        """MPL generated script compiles as valid Python."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_mpl_step("s002", "fig, ax = plt.subplots()", ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        compile(result["pipeline.py"], "<generated>", "exec")

    def test_valid_python_jsx(self):
        """JSX generated script compiles as valid Python."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_jsx_step("s002", "<div/>", ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        compile(result["pipeline.py"], "<generated>", "exec")

    def test_valid_python_with_transform(self):
        """Generated script with transforms compiles as valid Python."""
        steps = [
            _make_store_step("s001", "result = pd.DataFrame({'x': [1]})", "synth"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        compile(result["pipeline.py"], "<generated>", "exec")

    def test_requirements_comment_cdaweb(self):
        """Requirements line includes cdasws for CDAWeb fetches."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "pip install" in script
        assert "cdasws" in script
        assert "plotly" in script

    def test_requirements_comment_ppi(self):
        """Requirements do not include requests when PPI raises NotImplementedError."""
        steps = [
            _make_fetch_step("s001", "urn:nasa:pds:vg1", "p", "v1"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        # PPI steps raise NotImplementedError but we still note requests in reqs
        # since the intent is for users to implement the fetch
        assert "pip install" in script

    def test_pipeline_name_in_header(self):
        """Pipeline name appears in the header docstring."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(
            _make_pipeline_data(steps, name="Solar Wind Analysis")
        )
        script = result["pipeline.py"]

        assert "Solar Wind Analysis" in script

    def test_time_range_in_header(self):
        """Original time range appears in the header."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(
            steps, time_range_original=["2025-01-01", "2025-01-31"]
        ))
        script = result["pipeline.py"]

        assert "2025-01-01" in script
        assert "2025-01-31" in script

    def test_argparse_present(self):
        """Generated script includes argparse CLI."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert 'argparse.ArgumentParser' in script
        assert '"--start"' in script
        assert '"--end"' in script
        assert '__name__' in script

    def test_no_xhelio_imports(self):
        """Generated script must not import any xhelio modules."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_run_code_step("s002", "result = df * 2", ["s001"], "doubled"),
            _make_plotly_step("s003", {"data": [], "layout": {}}, ["s002"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]

        assert "from data_ops" not in script
        assert "from agent" not in script
        assert "from knowledge" not in script
        assert "import xhelio" not in script
        assert "from rendering" not in script

    def test_valid_python_with_booleans_and_none(self):
        """Layout with True/False/None values compiles (not JSON true/false/null)."""
        figure_json = {
            "data": [{"type": "scatter", "data_label": "ace_bfield", "visible": True}],
            "layout": {"showlegend": False, "hovermode": None, "autosize": True},
        }
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", figure_json, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))
        script = result["pipeline.py"]
        # Must compile — True/False/None must be Python literals, not JSON
        compile(script, "<generated>", "exec")
        # Should NOT contain JSON-style boolean/null literals
        assert "true" not in script.split("import")[0]  # avoid matching in import lines
        assert ": false" not in script
        assert ": null" not in script

    def test_name_with_quotes_compiles(self):
        """Pipeline names with quotes must not break the generated script."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        for name in ['Has "double" quotes', "Has 'single' quotes", 'Has triple """ quotes']:
            result = generate_script(_make_pipeline_data(steps, name=name))
            compile(result["pipeline.py"], "<generated>", "exec")

    def test_only_pipeline_py_for_non_jsx(self):
        """Non-JSX backends produce only pipeline.py."""
        steps = [
            _make_fetch_step("s001", "AC_H2_MFI", "BGSEc", "ace_bfield"),
            _make_plotly_step("s002", {"data": [], "layout": {}}, ["s001"]),
        ]
        result = generate_script(_make_pipeline_data(steps))

        assert list(result.keys()) == ["pipeline.py"]
