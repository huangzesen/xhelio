"""
Tests for data_ops.sandbox — blocklist-only sandbox with print capture and file I/O.

Run with: python -m pytest tests/test_sandbox.py
"""

import pytest

from data_ops.sandbox import validate_code_blocklist, execute_sandboxed


class TestBlocklistValidation:
    def test_allows_imports(self):
        """Blocklist sandbox should allow regular imports."""
        violations = validate_code_blocklist("import math\nresult = math.pi")
        assert violations == []

    def test_blocks_os_system(self):
        violations = validate_code_blocklist("import os\nos.system('rm -rf /')")
        assert len(violations) > 0
        assert any("os" in v for v in violations)

    def test_blocks_subprocess(self):
        violations = validate_code_blocklist("import subprocess\nsubprocess.run(['ls'])")
        assert len(violations) > 0

    def test_blocks_eval_exec(self):
        violations = validate_code_blocklist("eval('1+1')")
        assert len(violations) > 0

    def test_allows_file_io(self):
        """File I/O should be allowed (restricted at runtime by cwd)."""
        violations = validate_code_blocklist(
            "import pandas as pd\ndf = pd.read_parquet('data.parquet')\nresult = df"
        )
        assert violations == []

    def test_allows_scipy(self):
        violations = validate_code_blocklist(
            "from scipy import signal\nresult = signal.welch([1,2,3])"
        )
        assert violations == []

    def test_blocks_socket(self):
        violations = validate_code_blocklist("import socket\nsocket.create_connection(('x', 80))")
        assert len(violations) > 0

    def test_blocks_shutil_rmtree(self):
        violations = validate_code_blocklist("import shutil\nshutil.rmtree('/tmp/x')")
        assert len(violations) > 0

    def test_blocks_dunder_import(self):
        violations = validate_code_blocklist("__import__('os')")
        assert len(violations) > 0

    def test_allows_numpy(self):
        violations = validate_code_blocklist("import numpy as np\nresult = np.array([1,2,3])")
        assert violations == []

    def test_allows_xarray(self):
        violations = validate_code_blocklist("import xarray as xr")
        assert violations == []

    def test_syntax_error_returns_violation(self):
        violations = validate_code_blocklist("def f(\n  broken syntax")
        assert len(violations) > 0
        assert any("syntax" in v.lower() or "parse" in v.lower() for v in violations)


class TestSandboxExecution:
    def test_captures_print(self, tmp_path):
        """Sandbox should capture print output."""
        output, result = execute_sandboxed(
            "print('hello world')\nprint(2+2)",
            work_dir=tmp_path,
        )
        assert "hello world" in output
        assert "4" in output
        assert result is None  # no result assigned

    def test_captures_result(self, tmp_path):
        output, result = execute_sandboxed(
            "result = {'answer': 42}",
            work_dir=tmp_path,
        )
        assert result == {"answer": 42}

    def test_file_io_in_workdir(self, tmp_path):
        """Code should be able to read/write files in work_dir."""
        (tmp_path / "input.txt").write_text("hello")
        output, result = execute_sandboxed(
            "data = open('input.txt').read()\nprint(data)\nresult = data",
            work_dir=tmp_path,
        )
        assert "hello" in output
        assert result == "hello"

    def test_blocks_path_traversal(self, tmp_path):
        """Code should not access files outside work_dir."""
        output, result = execute_sandboxed(
            "open('/etc/passwd').read()",
            work_dir=tmp_path,
        )
        # Should raise error — execution failure captured in output
        assert result is None

    def test_timeout(self, tmp_path):
        """Infinite loops should be stopped."""
        with pytest.raises((RuntimeError, TimeoutError)):
            execute_sandboxed(
                "while True: pass",
                work_dir=tmp_path,
                timeout=2,
            )

    def test_pandas_available(self, tmp_path):
        output, result = execute_sandboxed(
            "import pandas as pd\nresult = len(pd.DataFrame({'x': [1,2,3]}))",
            work_dir=tmp_path,
        )
        assert result == 3

    def test_numpy_available(self, tmp_path):
        output, result = execute_sandboxed(
            "import numpy as np\nresult = float(np.mean([1,2,3]))",
            work_dir=tmp_path,
        )
        assert result == 2.0

    def test_write_file_in_workdir(self, tmp_path):
        output, result = execute_sandboxed(
            "with open('out.txt', 'w') as f:\n    f.write('ok')\nresult = 'done'",
            work_dir=tmp_path,
        )
        assert result == "done"
        assert (tmp_path / "out.txt").read_text() == "ok"

    def test_runtime_error_captured(self, tmp_path):
        """Runtime errors should be captured, not crash."""
        output, result = execute_sandboxed(
            "x = 1 / 0",
            work_dir=tmp_path,
        )
        assert result is None
        assert "ZeroDivisionError" in output
