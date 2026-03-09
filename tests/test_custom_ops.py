"""
Tests for data_ops.custom_ops — AST validator (allowlist sandbox).

The execution functions (execute_custom_operation, run_multi_source_operation, etc.)
have been removed in favor of data_ops.sandbox + agent.tool_handlers.sandbox (run_code).
Only validate_code and the sandbox registry are tested here.

Run with: python -m pytest tests/test_custom_ops.py
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.custom_ops import validate_code


class TestReloadSandboxRegistry:
    def test_reload_sandbox_registry(self):
        """reload_sandbox_registry() rebuilds derived constants."""
        from data_ops import custom_ops

        old_module_names = frozenset(custom_ops._MODULE_NAMES)
        custom_ops.reload_sandbox_registry()
        assert custom_ops._MODULE_NAMES == old_module_names


# ── Validator Tests ──────────────────────────────────────────────────────────


class TestValidatePandasCode:
    def test_valid_simple_operation(self):
        assert validate_code("result = df * 2") == []

    def test_valid_multiline(self):
        code = "mean = df.mean()\nresult = df - mean"
        assert validate_code(code) == []

    def test_valid_numpy_operation(self):
        assert validate_code("result = np.log10(df.abs())") == []

    def test_valid_rolling(self):
        assert validate_code("result = df.rolling(10, center=True, min_periods=1).mean()") == []

    def test_valid_interpolate(self):
        assert validate_code("result = df.interpolate(method='linear')") == []

    def test_valid_clip(self):
        assert validate_code("result = df.clip(lower=-50, upper=50)") == []

    def test_valid_complex_multiline(self):
        code = "z = (df - df.mean()) / df.std()\nmask = z.abs() < 3\nresult = df[mask].reindex(df.index)"
        assert validate_code(code) == []

    def test_reject_no_result_assignment(self):
        violations = validate_code("x = df * 2")
        assert any("result" in v for v in violations)

    def test_reject_import(self):
        violations = validate_code("import os\nresult = df")
        assert any("Import" in v for v in violations)

    def test_reject_from_import(self):
        violations = validate_code("from os import path\nresult = df")
        assert any("Import" in v for v in violations)

    def test_reject_exec(self):
        violations = validate_code("exec('x=1')\nresult = df")
        assert any("exec" in v for v in violations)

    def test_reject_eval(self):
        violations = validate_code("result = eval('df * 2')")
        assert any("eval" in v for v in violations)

    def test_reject_open(self):
        violations = validate_code("open('test.txt')\nresult = df")
        assert any("open" in v for v in violations)

    def test_reject_dunder_access(self):
        violations = validate_code("result = df.__class__")
        assert any("__class__" in v for v in violations)

    def test_reject_global(self):
        violations = validate_code("global x\nresult = df")
        assert any("global" in v for v in violations)

    def test_reject_nonlocal(self):
        violations = validate_code("nonlocal x\nresult = df")
        assert any("global/nonlocal" in v.lower() or "nonlocal" in v.lower() for v in violations)

    def test_reject_syntax_error(self):
        violations = validate_code("result = df +")
        assert any("Syntax" in v for v in violations)

    def test_reject_async(self):
        violations = validate_code("async def f(): pass\nresult = df")
        assert any("Async" in v or "async" in v for v in violations)

    def test_require_result_false_allows_no_assignment(self):
        violations = validate_code("x = 42", require_result=False)
        assert violations == []

    def test_require_result_false_still_blocks_imports(self):
        violations = validate_code("import os", require_result=False)
        assert any("Import" in v for v in violations)

    def test_require_result_false_still_blocks_exec(self):
        violations = validate_code("exec('x=1')", require_result=False)
        assert any("exec" in v for v in violations)

    def test_require_result_false_still_blocks_dunder(self):
        violations = validate_code("x = obj.__class__", require_result=False)
        assert any("__class__" in v for v in violations)

    def test_require_result_default_is_true(self):
        violations = validate_code("x = 42")
        assert any("result" in v for v in violations)
