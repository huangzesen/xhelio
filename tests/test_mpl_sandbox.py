"""Tests for rendering/mpl_sandbox.py — AST validation and subprocess execution."""

import json
import pickle
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from rendering.mpl_sandbox import (
    validate_mpl_script,
    build_script_wrapper,
    run_mpl_script,
    MplSandboxResult,
)


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


# ---- Script Wrapper Tests ----


class TestBuildScriptWrapper:
    """Test the script wrapper builder."""

    def test_wrapper_includes_preamble(self, tmp_path):
        code = "fig, ax = plt.subplots()"
        data_dir = tmp_path / "data"
        output_path = tmp_path / "output.png"
        labels_index = {"label1": "abc123"}

        wrapped = build_script_wrapper(code, data_dir, output_path, labels_index)

        assert "matplotlib.use(\"Agg\")" in wrapped
        assert "import matplotlib.pyplot as plt" in wrapped
        assert "import numpy as np" in wrapped
        assert "import pandas as pd" in wrapped
        assert "def load_data" in wrapped
        assert "def load_meta" in wrapped
        assert "def available_labels" in wrapped

    def test_wrapper_includes_user_code(self, tmp_path):
        code = "fig, ax = plt.subplots()\nax.plot([1, 2, 3])"
        wrapped = build_script_wrapper(
            code, tmp_path, tmp_path / "out.png", {}
        )
        assert "fig, ax = plt.subplots()" in wrapped
        assert "ax.plot([1, 2, 3])" in wrapped

    def test_wrapper_includes_epilogue(self, tmp_path):
        code = "plt.plot([1, 2, 3])"
        wrapped = build_script_wrapper(
            code, tmp_path, tmp_path / "out.png", {}
        )
        assert "plt.savefig" in wrapped
        assert 'plt.close("all")' in wrapped

    def test_wrapper_data_dir_path(self, tmp_path):
        code = "pass"
        data_dir = tmp_path / "my_data"
        wrapped = build_script_wrapper(
            code, data_dir, tmp_path / "out.png", {}
        )
        assert str(data_dir) in wrapped

    def test_wrapper_labels_index(self, tmp_path):
        code = "pass"
        labels = {"AC_H2_MFI.Magnitude": "abc12345", "PSP_Bmag": "def67890"}
        wrapped = build_script_wrapper(
            code, tmp_path, tmp_path / "out.png", labels
        )
        assert "AC_H2_MFI.Magnitude" in wrapped
        assert "abc12345" in wrapped


# ---- Script Execution Tests ----


class TestRunMplScript:
    """Test full script execution in subprocess."""

    def test_simple_plot_produces_png(self, tmp_path):
        """A simple matplotlib script should produce a PNG file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "mpl_outputs"
        labels_index = {}
        (data_dir / "_labels.json").write_text("{}")

        code = textwrap.dedent("""\
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
            ax.set_title("Simple Test")
        """)

        result = run_mpl_script(
            code=code,
            data_dir=data_dir,
            output_dir=output_dir,
            labels_index=labels_index,
            script_id="test_001",
        )

        assert result.success, f"Script failed: {result.stderr}"
        assert result.output_path is not None
        assert Path(result.output_path).exists()
        assert result.exit_code == 0
        # Verify it's a PNG (starts with PNG magic bytes)
        with open(result.output_path, "rb") as f:
            magic = f.read(8)
        assert magic[:4] == b"\x89PNG"

    def test_script_with_print_captures_stdout(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "mpl_outputs"
        (data_dir / "_labels.json").write_text("{}")

        code = textwrap.dedent("""\
            print("Hello from script")
            fig, ax = plt.subplots()
            ax.plot([1, 2], [3, 4])
        """)

        result = run_mpl_script(
            code=code,
            data_dir=data_dir,
            output_dir=output_dir,
            labels_index={},
            script_id="test_print",
        )

        assert result.success
        assert "Hello from script" in result.stdout

    def test_script_with_error_returns_stderr(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "mpl_outputs"
        (data_dir / "_labels.json").write_text("{}")

        code = "raise ValueError('intentional error')"

        result = run_mpl_script(
            code=code,
            data_dir=data_dir,
            output_dir=output_dir,
            labels_index={},
            script_id="test_error",
        )

        assert not result.success
        assert "ValueError" in result.stderr
        assert "intentional error" in result.stderr
        assert result.exit_code != 0

    def test_validation_failure_returns_violations(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "mpl_outputs"

        code = "import socket\nimport subprocess"

        result = run_mpl_script(
            code=code,
            data_dir=data_dir,
            output_dir=output_dir,
            labels_index={},
            script_id="test_blocked",
        )

        assert not result.success
        assert "validation failed" in result.stderr.lower()
        assert "socket" in result.stderr

    def test_timeout_enforcement(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "mpl_outputs"
        (data_dir / "_labels.json").write_text("{}")

        code = textwrap.dedent("""\
            import time
            time.sleep(10)
            fig, ax = plt.subplots()
            ax.plot([1], [1])
        """)

        result = run_mpl_script(
            code=code,
            data_dir=data_dir,
            output_dir=output_dir,
            labels_index={},
            script_id="test_timeout",
            timeout=2.0,
        )

        assert not result.success
        assert "timed out" in result.stderr.lower()

    def test_script_saves_to_mpl_scripts_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "mpl_outputs"
        (data_dir / "_labels.json").write_text("{}")

        code = "fig, ax = plt.subplots()\nax.plot([1], [1])"

        result = run_mpl_script(
            code=code,
            data_dir=data_dir,
            output_dir=output_dir,
            labels_index={},
            script_id="test_save",
        )

        assert result.script_path is not None
        script_path = Path(result.script_path)
        assert script_path.exists()
        assert script_path.name == "test_save.py"
        assert script_path.parent.name == "mpl_scripts"

    def test_load_data_helper(self, tmp_path):
        """Test that the load_data() helper can read pickled DataFrames."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "mpl_outputs"

        # Create a mock data entry
        hash_dir = data_dir / "abcd1234"
        hash_dir.mkdir()
        df = pd.DataFrame(
            {"value": [10.0, 20.0, 30.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )
        df.to_pickle(hash_dir / "data.pkl")
        meta = {"label": "TEST.Value", "units": "nT"}
        (hash_dir / "meta.json").write_text(json.dumps(meta))

        labels_index = {"TEST.Value": "abcd1234"}
        (data_dir / "_labels.json").write_text(json.dumps(labels_index))

        code = textwrap.dedent("""\
            labels = available_labels()
            print(f"Labels: {labels}")
            df = load_data("TEST.Value")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            meta = load_meta("TEST.Value")
            print(f"Units: {meta.get('units', 'unknown')}")
            fig, ax = plt.subplots()
            ax.plot(df.index, df["value"])
        """)

        result = run_mpl_script(
            code=code,
            data_dir=data_dir,
            output_dir=output_dir,
            labels_index=labels_index,
            script_id="test_load_data",
        )

        assert result.success, f"Script failed: {result.stderr}"
        assert "TEST.Value" in result.stdout
        assert "Shape: (3, 1)" in result.stdout
        assert "Units: nT" in result.stdout

    def test_load_data_missing_label_error(self, tmp_path):
        """Test that load_data() raises a clear error for missing labels."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "mpl_outputs"
        (data_dir / "_labels.json").write_text("{}")

        code = textwrap.dedent("""\
            df = load_data("NONEXISTENT")
        """)

        result = run_mpl_script(
            code=code,
            data_dir=data_dir,
            output_dir=output_dir,
            labels_index={},
            script_id="test_missing",
        )

        assert not result.success
        assert "NONEXISTENT" in result.stderr
        assert "not found" in result.stderr.lower()


class TestMplSandboxResult:
    """Test the MplSandboxResult dataclass."""

    def test_default_values(self):
        result = MplSandboxResult(success=False)
        assert result.success is False
        assert result.output_path is None
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.script_path is None
        assert result.exit_code == -1

    def test_success_result(self):
        result = MplSandboxResult(
            success=True,
            output_path="/tmp/out.png",
            stdout="OK",
            stderr="",
            script_path="/tmp/script.py",
            exit_code=0,
        )
        assert result.success is True
        assert result.output_path == "/tmp/out.png"
        assert result.exit_code == 0
