"""Tests for rendering/mpl_sandbox.py — AST validation and subprocess execution."""

import json
import textwrap

import pandas as pd
import pytest

from rendering.mpl_sandbox import validate_mpl_script


# ---- AST Validation Tests ----


class TestValidateMplScript:
    """Test the AST validation for matplotlib scripts."""

    def test_valid_simple_script(self):
        code = textwrap.dedent("""\
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [4, 5, 6])
            ax.set_title("Test")
        """)
        violations = validate_mpl_script(code)
        assert violations == []

    def test_valid_numpy_import(self):
        code = "import numpy as np\nx = np.array([1, 2, 3])"
        violations = validate_mpl_script(code)
        assert violations == []

    def test_valid_scipy_import(self):
        code = "from scipy.signal import butter"
        violations = validate_mpl_script(code)
        assert violations == []

    def test_valid_matplotlib_submodule(self):
        code = "from mpl_toolkits.mplot3d import Axes3D"
        violations = validate_mpl_script(code)
        assert violations == []

    def test_valid_datetime_import(self):
        code = "import datetime\nfrom datetime import timedelta"
        violations = validate_mpl_script(code)
        assert violations == []

    def test_valid_collections_import(self):
        code = "from collections import defaultdict"
        violations = validate_mpl_script(code)
        assert violations == []

    def test_blocked_socket_import(self):
        code = "import socket"
        violations = validate_mpl_script(code)
        assert len(violations) == 1
        assert "socket" in violations[0]

    def test_blocked_subprocess_import(self):
        code = "import subprocess"
        violations = validate_mpl_script(code)
        assert len(violations) == 1
        assert "subprocess" in violations[0]

    def test_blocked_requests_import(self):
        code = "import requests"
        violations = validate_mpl_script(code)
        assert len(violations) == 1
        assert "requests" in violations[0]

    def test_blocked_from_import(self):
        code = "from http.client import HTTPConnection"
        violations = validate_mpl_script(code)
        assert len(violations) == 1
        assert "http" in violations[0]

    def test_blocked_ctypes(self):
        code = "import ctypes"
        violations = validate_mpl_script(code)
        assert len(violations) == 1
        assert "ctypes" in violations[0]

    def test_blocked_pickle(self):
        code = "import pickle"
        violations = validate_mpl_script(code)
        assert len(violations) == 1
        assert "pickle" in violations[0]

    def test_blocked_importlib(self):
        code = "import importlib"
        violations = validate_mpl_script(code)
        assert len(violations) == 1
        assert "importlib" in violations[0]

    def test_unknown_import_rejected(self):
        code = "import some_random_package"
        violations = validate_mpl_script(code)
        assert len(violations) == 1
        assert "Unknown import" in violations[0]

    def test_blocked_exec_builtin(self):
        code = "exec('print(1)')"
        violations = validate_mpl_script(code)
        assert any("exec" in v for v in violations)

    def test_blocked_eval_builtin(self):
        code = "eval('1+1')"
        violations = validate_mpl_script(code)
        assert any("eval" in v for v in violations)

    def test_blocked_compile_builtin(self):
        code = "compile('x=1', '<string>', 'exec')"
        violations = validate_mpl_script(code)
        assert any("compile" in v for v in violations)

    def test_blocked___import__(self):
        code = "__import__('os')"
        violations = validate_mpl_script(code)
        assert any("__import__" in v for v in violations)

    def test_blocked_open_builtin(self):
        code = "f = open('/etc/passwd')"
        violations = validate_mpl_script(code)
        assert any("open" in v for v in violations)

    def test_blocked_input_builtin(self):
        code = "x = input('Enter: ')"
        violations = validate_mpl_script(code)
        assert any("input" in v for v in violations)

    def test_blocked_dunder_attr(self):
        code = "x.__class__.__bases__"
        violations = validate_mpl_script(code)
        assert any("__class__" in v for v in violations)

    def test_blocked_system_attr(self):
        code = "import os\nos.system('ls')"
        violations = validate_mpl_script(code)
        assert any("system" in v for v in violations)

    def test_blocked_popen_attr(self):
        code = "import os\nos.popen('ls')"
        violations = validate_mpl_script(code)
        assert any("popen" in v for v in violations)

    def test_blocked_global_nonlocal(self):
        code = "def f():\n    global x\n    x = 1"
        violations = validate_mpl_script(code)
        assert any("global" in v for v in violations)

    def test_blocked_async(self):
        code = "async def f():\n    pass"
        violations = validate_mpl_script(code)
        assert any("Async" in v for v in violations)

    def test_syntax_error(self):
        code = "def f(\n"
        violations = validate_mpl_script(code)
        assert len(violations) == 1
        assert "Syntax error" in violations[0]

    def test_multiple_violations(self):
        code = "import socket\nimport subprocess\nexec('bad')"
        violations = validate_mpl_script(code)
        assert len(violations) >= 3

    def test_allowed_pandas(self):
        code = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2]})"
        violations = validate_mpl_script(code)
        assert violations == []

    def test_allowed_xarray(self):
        code = "import xarray as xr"
        violations = validate_mpl_script(code)
        assert violations == []


# ---- Script Execution Tests (new flow: _stage_entry + execute_sandboxed) ----


class TestMplScriptExecution:
    """Test the new mpl script execution flow using run_code sandbox."""

    def _build_preamble(self, staged_labels, output_path):
        """Build the matplotlib preamble (same as _handle_generate_mpl_script)."""
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
    path = f"{{label}}.parquet"
    import os
    if not os.path.exists(path):
        raise KeyError(f"Label '{{label}}' not found. Available labels: {{available_labels()}}")
    return pd.read_parquet(path)

def load_meta(label):
    path = f"{{label}}.meta.json"
    import os
    if not os.path.exists(path):
        return {{}}
    with open(path) as f:
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
        """A simple plot should produce a PNG file."""
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
        """Stage a DataFrame as parquet + meta.json (mirrors _stage_entry/_stage_meta)."""
        df.to_parquet(sandbox_dir / f"{label}.parquet")
        meta = {"label": label}
        meta.update(meta_fields)
        (sandbox_dir / f"{label}.meta.json").write_text(json.dumps(meta, default=str))

    def test_load_data_reads_staged_parquet(self, tmp_path):
        """load_data() should read parquet files staged as .parquet."""
        from data_ops.sandbox import execute_sandboxed

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        output_path = tmp_path / "output.png"

        # Create and stage data
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
        """load_meta() should read .meta.json files staged as .meta.json."""
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
        """available_labels() should return the list of staged labels."""
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
        """load_data() should raise KeyError for missing labels."""
        from data_ops.sandbox import execute_sandboxed

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        output_path = tmp_path / "output.png"

        user_code = 'df = load_data("NONEXISTENT")'

        wrapped = self._build_preamble([], output_path) + user_code + self._build_epilogue()
        output, _ = execute_sandboxed(wrapped, work_dir=sandbox_dir)

        # execute_sandboxed captures exceptions in stdout
        assert "NONEXISTENT" in output
        assert "not found" in output.lower() or "KeyError" in output

    def test_script_with_error_returns_traceback(self, tmp_path):
        """Script errors should appear in output."""
        from data_ops.sandbox import execute_sandboxed

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        output_path = tmp_path / "output.png"

        user_code = "raise ValueError('intentional error')"
        wrapped = self._build_preamble([], output_path) + user_code + self._build_epilogue()
        output, _ = execute_sandboxed(wrapped, work_dir=sandbox_dir)

        assert "ValueError" in output
        assert "intentional error" in output
