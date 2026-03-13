"""Tests for rendering/mpl_sandbox.py — data label extraction and script execution."""

import json
import textwrap

import pandas as pd
import pytest

from rendering.mpl_sandbox import extract_data_labels


# ---- Data Label Extraction Tests ----


class TestExtractDataLabels:
    def test_single_label(self):
        code = 'df = load_data("AC_H2_MFI.Magnitude")'
        assert extract_data_labels(code) == ["AC_H2_MFI.Magnitude"]

    def test_multiple_labels(self):
        code = 'a = load_data("X")\nb = load_data("Y")'
        assert extract_data_labels(code) == ["X", "Y"]

    def test_deduplicates(self):
        code = 'a = load_data("X")\nb = load_data("X")'
        assert extract_data_labels(code) == ["X"]

    def test_no_labels(self):
        code = "fig, ax = plt.subplots()"
        assert extract_data_labels(code) == []


# ---- Script Execution Tests (using data_ops/sandbox.py) ----


class TestMplScriptExecution:
    """Test mpl script execution using the shared sandbox."""

    def _build_preamble(self, staged_labels, output_path):
        """Build the matplotlib preamble (mirrors handle_generate_mpl_script)."""
        labels_json = json.dumps(staged_labels)
        return f'''import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

_OUTPUT_PATH = {repr(str(output_path))}
_STAGED_LABELS = json.loads({repr(labels_json)})

def load_data(label):
    from pathlib import Path
    p = Path(f"{{label}}.parquet")
    if not p.exists():
        raise KeyError(f"Label '{{label}}' not found. Available labels: {{available_labels()}}")
    return pd.read_parquet(p)

def load_meta(label):
    from pathlib import Path
    p = Path(f"{{label}}.meta.json")
    if not p.exists():
        return {{}}
    with open(p) as f:
        return json.load(f)

def available_labels():
    return _STAGED_LABELS

# === User script starts below ===
'''

    def _build_epilogue(self):
        return '''

# === Auto-generated epilogue ===
plt.savefig(_OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close("all")
'''

    def test_simple_plot_produces_png(self, tmp_path):
        from data_ops.sandbox import execute_sandboxed

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        output_path = tmp_path / "output.png"

        user_code = textwrap.dedent("""\
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
            ax.set_title("Simple Test")
        """)

        wrapped = self._build_preamble([], output_path) + user_code + self._build_epilogue()
        output, _ = execute_sandboxed(wrapped, work_dir=sandbox_dir)

        assert output_path.exists()
        with open(output_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"\x89PNG"

    @staticmethod
    def _stage_df(label: str, df: "pd.DataFrame", sandbox_dir, **meta_fields):
        df.to_parquet(sandbox_dir / f"{label}.parquet")
        meta = {"label": label}
        meta.update(meta_fields)
        (sandbox_dir / f"{label}.meta.json").write_text(json.dumps(meta, default=str))

    def test_load_data_reads_staged_parquet(self, tmp_path):
        from data_ops.sandbox import execute_sandboxed

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        output_path = tmp_path / "output.png"

        df = pd.DataFrame(
            {"value": [10.0, 20.0, 30.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )
        self._stage_df("TEST.Value", df, sandbox_dir, units="nT")

        user_code = textwrap.dedent("""\
            df = load_data("TEST.Value")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            fig, ax = plt.subplots()
            ax.plot(df.index, df["value"])
        """)

        wrapped = self._build_preamble(["TEST.Value"], output_path) + user_code + self._build_epilogue()
        output, _ = execute_sandboxed(wrapped, work_dir=sandbox_dir)

        assert output_path.exists()
        assert "Shape: (3, 1)" in output
        assert "value" in output

    def test_load_meta_reads_staged_metadata(self, tmp_path):
        from data_ops.sandbox import execute_sandboxed

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        output_path = tmp_path / "output.png"

        df = pd.DataFrame({"x": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
        self._stage_df("TEST.X", df, sandbox_dir, units="km/s", description="test data")

        user_code = textwrap.dedent("""\
            meta = load_meta("TEST.X")
            print(f"Units: {meta.get('units', 'unknown')}")
            print(f"Desc: {meta.get('description', 'none')}")
            fig, ax = plt.subplots()
            ax.plot([1], [1])
        """)

        wrapped = self._build_preamble(["TEST.X"], output_path) + user_code + self._build_epilogue()
        output, _ = execute_sandboxed(wrapped, work_dir=sandbox_dir)

        assert "Units: km/s" in output
        assert "Desc: test data" in output

    def test_available_labels_returns_staged(self, tmp_path):
        from data_ops.sandbox import execute_sandboxed

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        output_path = tmp_path / "output.png"

        user_code = textwrap.dedent("""\
            labels = available_labels()
            print(f"Labels: {labels}")
            fig, ax = plt.subplots()
            ax.plot([1], [1])
        """)

        wrapped = self._build_preamble(["A.x", "B.y"], output_path) + user_code + self._build_epilogue()
        output, _ = execute_sandboxed(wrapped, work_dir=sandbox_dir)

        assert "A.x" in output
        assert "B.y" in output

    def test_load_data_missing_label_error(self, tmp_path):
        from data_ops.sandbox import execute_sandboxed

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        output_path = tmp_path / "output.png"

        user_code = 'df = load_data("NONEXISTENT")'

        wrapped = self._build_preamble([], output_path) + user_code + self._build_epilogue()
        output, _ = execute_sandboxed(wrapped, work_dir=sandbox_dir)

        assert "NONEXISTENT" in output
        assert "not found" in output.lower() or "KeyError" in output

    def test_script_with_error_returns_traceback(self, tmp_path):
        from data_ops.sandbox import execute_sandboxed

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        output_path = tmp_path / "output.png"

        user_code = "raise ValueError('intentional error')"
        wrapped = self._build_preamble([], output_path) + user_code + self._build_epilogue()
        output, _ = execute_sandboxed(wrapped, work_dir=sandbox_dir)

        assert "ValueError" in output
        assert "intentional error" in output
