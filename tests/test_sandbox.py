"""
Tests for data_ops.sandbox — threat-category blocklist sandbox.

Run with: python -m pytest tests/test_sandbox.py
"""

import pytest

from data_ops.sandbox import (
    BLOCKED_ATTRS,
    BLOCKED_BUILTINS,
    BLOCKED_IMPORTS,
    THREAT_CATEGORIES,
    build_sandbox_rules_prompt,
    execute_sandboxed,
    validate_code_blocklist,
)


# ---- Threat Category Coverage ----


class TestThreatCategories:
    """Verify that THREAT_CATEGORIES correctly flatten into runtime sets."""

    def test_all_categories_have_entries(self):
        """Each category should have at least one blocked item."""
        for cat_name, cat in THREAT_CATEGORIES.items():
            total = sum(len(cat.get(k, [])) for k in ("imports", "builtins", "attrs"))
            assert total > 0, f"Category '{cat_name}' has no blocked items"

    def test_imports_flattened(self):
        """All category imports should appear in BLOCKED_IMPORTS."""
        for cat in THREAT_CATEGORIES.values():
            for imp in cat.get("imports", []):
                assert imp in BLOCKED_IMPORTS, f"{imp} not in BLOCKED_IMPORTS"

    def test_builtins_flattened(self):
        """All category builtins should appear in BLOCKED_BUILTINS."""
        for cat in THREAT_CATEGORIES.values():
            for b in cat.get("builtins", []):
                assert b in BLOCKED_BUILTINS, f"{b} not in BLOCKED_BUILTINS"

    def test_attrs_flattened(self):
        """All category attrs should appear in BLOCKED_ATTRS."""
        for cat in THREAT_CATEGORIES.values():
            for a in cat.get("attrs", []):
                assert a in BLOCKED_ATTRS, f"{a} not in BLOCKED_ATTRS"

    def test_network_category_has_socket(self):
        assert "socket" in BLOCKED_IMPORTS

    def test_process_spawn_has_subprocess(self):
        assert "subprocess" in BLOCKED_IMPORTS

    def test_process_spawn_has_multiprocessing(self):
        assert "multiprocessing" in BLOCKED_IMPORTS

    def test_code_injection_has_eval(self):
        assert "eval" in BLOCKED_BUILTINS

    def test_code_injection_has_pickle(self):
        assert "pickle" in BLOCKED_IMPORTS

    def test_ffi_has_ctypes(self):
        assert "ctypes" in BLOCKED_IMPORTS

    def test_filesystem_damage_has_rmtree(self):
        assert "rmtree" in BLOCKED_ATTRS

    def test_information_leak_has_environ(self):
        assert "environ" in BLOCKED_ATTRS

    def test_reflection_has_getattr(self):
        assert "getattr" in BLOCKED_BUILTINS

    def test_subprocess_hang_has_breakpoint(self):
        assert "breakpoint" in BLOCKED_BUILTINS


# ---- AST Validation ----


class TestBlocklistValidation:
    """Test validate_code_blocklist against the threat categories."""

    # --- Allowed imports (previously blocked, now unblocked) ---

    def test_allows_os_import(self):
        """os is allowed — dangerous attrs are blocked separately."""
        violations = validate_code_blocklist("import os\nprint(os.path.exists('/tmp'))")
        assert violations == []

    def test_allows_sys_import(self):
        violations = validate_code_blocklist("import sys\nprint(sys.version)")
        assert violations == []

    def test_allows_shutil_import(self):
        """shutil is allowed — rmtree is blocked as an attr."""
        violations = validate_code_blocklist("import shutil\nshutil.copy('a', 'b')")
        assert violations == []

    def test_allows_threading_import(self):
        violations = validate_code_blocklist("import threading")
        assert violations == []

    def test_allows_signal_import(self):
        violations = validate_code_blocklist("import signal")
        assert violations == []

    # --- Still-allowed imports ---

    def test_allows_math(self):
        violations = validate_code_blocklist("import math\nresult = math.pi")
        assert violations == []

    def test_allows_numpy(self):
        violations = validate_code_blocklist("import numpy as np\nresult = np.array([1,2,3])")
        assert violations == []

    def test_allows_pandas(self):
        violations = validate_code_blocklist(
            "import pandas as pd\ndf = pd.read_parquet('data.parquet')"
        )
        assert violations == []

    def test_allows_scipy(self):
        violations = validate_code_blocklist(
            "from scipy import signal\nresult = signal.welch([1,2,3])"
        )
        assert violations == []

    def test_allows_xarray(self):
        violations = validate_code_blocklist("import xarray as xr")
        assert violations == []

    # --- Blocked imports ---

    def test_blocks_subprocess(self):
        violations = validate_code_blocklist("import subprocess\nsubprocess.run(['ls'])")
        assert any("subprocess" in v for v in violations)

    def test_blocks_socket(self):
        violations = validate_code_blocklist("import socket")
        assert any("socket" in v for v in violations)

    def test_blocks_ctypes(self):
        violations = validate_code_blocklist("import ctypes")
        assert any("ctypes" in v for v in violations)

    def test_blocks_importlib(self):
        violations = validate_code_blocklist("import importlib")
        assert any("importlib" in v for v in violations)

    def test_blocks_multiprocessing(self):
        violations = validate_code_blocklist("import multiprocessing")
        assert any("multiprocessing" in v for v in violations)

    def test_blocks_pickle(self):
        violations = validate_code_blocklist("import pickle")
        assert any("pickle" in v for v in violations)

    def test_blocks_marshal(self):
        violations = validate_code_blocklist("import marshal")
        assert any("marshal" in v for v in violations)

    def test_blocks_http(self):
        violations = validate_code_blocklist("from http.client import HTTPConnection")
        assert any("http" in v for v in violations)

    # --- Blocked builtins ---

    def test_blocks_eval(self):
        violations = validate_code_blocklist("eval('1+1')")
        assert any("eval" in v for v in violations)

    def test_blocks_exec(self):
        violations = validate_code_blocklist("exec('x=1')")
        assert any("exec" in v for v in violations)

    def test_blocks_dunder_import(self):
        violations = validate_code_blocklist("__import__('os')")
        assert any("__import__" in v for v in violations)

    # --- Dangerous attrs still blocked even with os unblocked ---

    def test_blocks_os_system(self):
        violations = validate_code_blocklist("import os\nos.system('ls')")
        assert any("system" in v for v in violations)

    def test_blocks_os_popen(self):
        violations = validate_code_blocklist("import os\nos.popen('ls')")
        assert any("popen" in v for v in violations)

    def test_blocks_os_environ(self):
        violations = validate_code_blocklist("import os\nprint(os.environ)")
        assert any("environ" in v for v in violations)

    def test_blocks_os_kill(self):
        violations = validate_code_blocklist("import os\nos.kill(1, 9)")
        assert any("kill" in v for v in violations)

    def test_blocks_shutil_rmtree(self):
        violations = validate_code_blocklist("import shutil\nshutil.rmtree('/tmp/x')")
        assert any("rmtree" in v for v in violations)

    # --- Dunder attribute access ---

    def test_blocks_dunder_class(self):
        violations = validate_code_blocklist("x = ''.__class__")
        assert any("__class__" in v for v in violations)

    def test_blocks_dunder_bases(self):
        violations = validate_code_blocklist("x.__bases__")
        assert any("__bases__" in v for v in violations)

    def test_blocks_dunder_subclasses(self):
        violations = validate_code_blocklist("x.__subclasses__()")
        assert any("__subclasses__" in v for v in violations)

    # --- Syntax errors ---

    def test_syntax_error(self):
        violations = validate_code_blocklist("def f(\n  broken syntax")
        assert any("syntax" in v.lower() or "parse" in v.lower() for v in violations)


# ---- Dynamic Prompt Generation ----


class TestBuildSandboxRulesPrompt:
    def test_contains_all_categories(self):
        prompt = build_sandbox_rules_prompt()
        for cat_name in THREAT_CATEGORIES:
            label = cat_name.replace("_", " ").title()
            assert label in prompt, f"Category '{label}' not in prompt"

    def test_contains_blocked_items(self):
        prompt = build_sandbox_rules_prompt()
        assert "socket" in prompt
        assert "subprocess" in prompt
        assert "eval" in prompt

    def test_mentions_dunder(self):
        prompt = build_sandbox_rules_prompt()
        assert "dunder" in prompt.lower() or "__class__" in prompt


# ---- Sandbox Execution ----


class TestSandboxExecution:
    def test_captures_print(self, tmp_path):
        output, outputs = execute_sandboxed(
            "print('hello world')\nprint(2+2)",
            work_dir=tmp_path,
        )
        assert "hello world" in output
        assert "4" in output
        assert outputs == {}

    def test_captures_result(self, tmp_path):
        output, outputs = execute_sandboxed(
            "import json\nresult = {'answer': 42}",
            work_dir=tmp_path,
            output_vars=["result"],
        )
        assert "result" in outputs
        assert outputs["result"]["status"] == "ok"
        assert outputs["result"]["format"] == "json"

    def test_file_io_in_workdir(self, tmp_path):
        (tmp_path / "input.txt").write_text("hello")
        output, outputs = execute_sandboxed(
            "data = open('input.txt').read()\nprint(data)",
            work_dir=tmp_path,
        )
        assert "hello" in output

    def test_blocks_path_traversal(self, tmp_path):
        output, outputs = execute_sandboxed(
            "open('/etc/passwd').read()",
            work_dir=tmp_path,
        )
        assert outputs == {}

    def test_timeout(self, tmp_path):
        with pytest.raises((RuntimeError, TimeoutError)):
            execute_sandboxed(
                "while True: pass",
                work_dir=tmp_path,
                timeout=2,
            )

    def test_pandas_available(self, tmp_path):
        output, outputs = execute_sandboxed(
            "import pandas as pd\nresult = pd.DataFrame({'x': [1,2,3]})",
            work_dir=tmp_path,
            output_vars=["result"],
        )
        assert outputs["result"]["status"] == "ok"
        assert outputs["result"]["format"] == "parquet"

    def test_numpy_available(self, tmp_path):
        output, outputs = execute_sandboxed(
            "import numpy as np\nresult = float(np.mean([1,2,3]))\nprint(result)",
            work_dir=tmp_path,
        )
        assert "2.0" in output

    def test_write_file_in_workdir(self, tmp_path):
        output, outputs = execute_sandboxed(
            "with open('out.txt', 'w') as f:\n    f.write('ok')\nresult = 'done'",
            work_dir=tmp_path,
            output_vars=["result"],
        )
        assert outputs["result"]["status"] == "ok"
        assert (tmp_path / "out.txt").read_text() == "ok"

    def test_runtime_error_captured(self, tmp_path):
        output, outputs = execute_sandboxed(
            "x = 1 / 0",
            work_dir=tmp_path,
        )
        assert outputs == {}
        assert "ZeroDivisionError" in output

    # --- Integration: os allowed but dangerous ops blocked at runtime ---

    def test_os_path_works_at_runtime(self, tmp_path):
        """import os + os.path.exists should work in the sandbox."""
        (tmp_path / "test.txt").write_text("hi")
        output, _ = execute_sandboxed(
            "import os\nprint(os.path.exists('test.txt'))",
            work_dir=tmp_path,
        )
        assert "True" in output

    def test_os_listdir_works_at_runtime(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        output, _ = execute_sandboxed(
            "import os\nprint(os.listdir('.'))",
            work_dir=tmp_path,
        )
        assert "a.txt" in output

    def test_shutil_copy_works_at_runtime(self, tmp_path):
        (tmp_path / "src.txt").write_text("data")
        output, _ = execute_sandboxed(
            "import shutil\nshutil.copy('src.txt', 'dst.txt')\nprint('ok')",
            work_dir=tmp_path,
        )
        assert "ok" in output
        assert (tmp_path / "dst.txt").read_text() == "data"
