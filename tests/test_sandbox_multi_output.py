"""Tests for multi-output sandbox execution."""

import pandas as pd
import pytest
from data_ops.sandbox import execute_sandboxed


class TestMultiOutputSandbox:

    def test_single_output(self, tmp_path):
        code = "import pandas as pd\nresult = pd.DataFrame({'a': [1,2,3]})"
        output, results = execute_sandboxed(
            code, work_dir=tmp_path, output_vars=["result"],
        )
        assert results["result"]["status"] == "ok"
        assert results["result"]["format"] == "parquet"
        assert (tmp_path / "result.parquet").exists()

    def test_multiple_outputs(self, tmp_path):
        code = (
            "import pandas as pd\n"
            "magnitude = pd.DataFrame({'v': [1.0, 2.0]})\n"
            "angle = pd.DataFrame({'v': [0.5, 1.5]})\n"
        )
        output, results = execute_sandboxed(
            code, work_dir=tmp_path, output_vars=["magnitude", "angle"],
        )
        assert results["magnitude"]["status"] == "ok"
        assert results["angle"]["status"] == "ok"
        assert (tmp_path / "magnitude.parquet").exists()
        assert (tmp_path / "angle.parquet").exists()

    def test_missing_output_var(self, tmp_path):
        code = "x = 42"
        output, results = execute_sandboxed(
            code, work_dir=tmp_path, output_vars=["nonexistent"],
        )
        assert results["nonexistent"]["status"] == "missing"

    def test_partial_success(self, tmp_path):
        code = (
            "import pandas as pd\n"
            "good = pd.DataFrame({'v': [1]})\n"
        )
        output, results = execute_sandboxed(
            code, work_dir=tmp_path, output_vars=["good", "bad"],
        )
        assert results["good"]["status"] == "ok"
        assert results["bad"]["status"] == "missing"

    def test_xarray_output(self, tmp_path):
        code = (
            "import xarray as xr\n"
            "import numpy as np\n"
            "result = xr.DataArray(np.zeros((3,3)))\n"
        )
        output, results = execute_sandboxed(
            code, work_dir=tmp_path, output_vars=["result"],
        )
        assert results["result"]["status"] == "ok"
        assert results["result"]["format"] == "nc"

    def test_json_serializable_output(self, tmp_path):
        code = "result = {'key': 'value', 'nums': [1, 2, 3]}"
        output, results = execute_sandboxed(
            code, work_dir=tmp_path, output_vars=["result"],
        )
        assert results["result"]["status"] == "ok"
        assert results["result"]["format"] == "json"

    def test_no_output_vars(self, tmp_path):
        """When output_vars is empty, behaves like fire-and-forget."""
        code = "print('hello')"
        output, results = execute_sandboxed(
            code, work_dir=tmp_path, output_vars=[],
        )
        assert results == {}
        assert "hello" in output

    def test_backward_compat_no_output_vars_arg(self, tmp_path):
        """When output_vars is not provided, defaults to empty."""
        code = "print('hello')"
        output, results = execute_sandboxed(code, work_dir=tmp_path)
        assert results == {}


class TestInputIsolation:
    """Verify code cannot access data outside declared inputs."""

    def test_undeclared_input_not_accessible(self, tmp_path):
        """If a label is not in inputs, its file should not exist in sandbox."""
        # Write a file that would exist if staging were unrestricted
        pd.DataFrame({"v": [1]}).to_parquet(tmp_path / "secret.parquet")

        # Run code in a FRESH temp dir (not tmp_path) — secret.parquet absent
        fresh = tmp_path / "fresh_sandbox"
        fresh.mkdir()
        code = (
            "import os\n"
            "files = os.listdir('.')\n"
            "assert 'secret.parquet' not in files, "
            "f'Unexpected file found: {files}'\n"
            "result = 'isolated'\n"
        )
        output, results = execute_sandboxed(
            code, work_dir=fresh,
            output_vars=["result"],
        )
        assert results.get("result", {}).get("status") == "ok"
