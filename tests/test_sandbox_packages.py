"""Tests for the merged manage_sandbox_packages handler."""

from unittest.mock import patch, MagicMock

import agent.tool_handlers.sandbox_packages as _mod
from agent.tool_handlers.sandbox_packages import handle_manage_sandbox_packages

# All auto_install / subprocess / importlib patches target the module object
# via patch.object to guarantee the right namespace is patched.


class TestManageSandboxPackages:
    def _make_ctx(self, perm_approved=True):
        ctx = MagicMock()
        ctx.request_permission = MagicMock(return_value={
            "approved": perm_approved,
            "reason": "ok" if perm_approved else "denied by user",
        })
        return ctx

    def test_list_action(self):
        ctx = self._make_ctx()
        result = handle_manage_sandbox_packages(ctx, {"action": "list"})
        assert result["status"] == "ok"
        assert "packages" in result

    def test_install_missing_pip_name(self):
        ctx = self._make_ctx()
        result = handle_manage_sandbox_packages(ctx, {
            "action": "install",
            "import_path": "foo",
            "sandbox_alias": "foo",
            "description": "test",
        })
        assert result["status"] == "error"
        assert "pip_name" in result["message"]

    def test_install_invalid_pip_name(self):
        ctx = self._make_ctx()
        result = handle_manage_sandbox_packages(ctx, {
            "action": "install",
            "pip_name": "--index-url http://evil.com",
            "import_path": "foo",
            "sandbox_alias": "foo",
            "description": "test",
        })
        assert result["status"] == "error"
        assert "Invalid" in result["message"]

    @patch.object(_mod, "_get_auto_install", return_value=False)
    def test_install_denied_by_user(self, _mock):
        ctx = self._make_ctx(perm_approved=False)
        result = handle_manage_sandbox_packages(ctx, {
            "action": "install",
            "pip_name": "requests",
            "import_path": "requests",
            "sandbox_alias": "requests",
            "description": "HTTP library",
        })
        assert result["status"] == "denied"

    @patch.object(_mod, "_get_auto_install", return_value=True)
    @patch.object(_mod, "subprocess")
    @patch.object(_mod, "importlib")
    @patch.object(_mod, "_register_package")
    def test_install_auto_approve_skips_permission(self, mock_reg, mock_imp, mock_sub, _mock_auto):
        mock_sub.run.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx()
        result = handle_manage_sandbox_packages(ctx, {
            "action": "install",
            "pip_name": "requests",
            "import_path": "requests",
            "sandbox_alias": "requests",
            "description": "HTTP library",
        })
        assert result["status"] == "installed"
        ctx.request_permission.assert_not_called()

    def test_add_missing_import_path(self):
        ctx = self._make_ctx()
        result = handle_manage_sandbox_packages(ctx, {
            "action": "add",
            "sandbox_alias": "foo",
            "description": "test",
        })
        assert result["status"] == "error"

    @patch.object(_mod, "_get_auto_install", return_value=True)
    @patch.object(_mod, "importlib")
    @patch.object(_mod, "_register_package")
    def test_add_auto_approve_skips_permission(self, mock_reg, mock_imp, _mock_auto):
        ctx = self._make_ctx()
        result = handle_manage_sandbox_packages(ctx, {
            "action": "add",
            "import_path": "requests",
            "sandbox_alias": "requests",
            "description": "HTTP library",
        })
        assert result["status"] == "added"
        ctx.request_permission.assert_not_called()

    @patch.object(_mod, "_get_auto_install", return_value=False)
    @patch.object(_mod, "subprocess")
    @patch.object(_mod, "importlib")
    @patch.object(_mod, "_register_package")
    def test_install_approved_by_user(self, mock_reg, mock_imp, mock_sub, _mock_auto):
        mock_sub.run.return_value = MagicMock(returncode=0)
        ctx = self._make_ctx(perm_approved=True)
        result = handle_manage_sandbox_packages(ctx, {
            "action": "install",
            "pip_name": "requests",
            "import_path": "requests",
            "sandbox_alias": "requests",
            "description": "HTTP library",
        })
        assert result["status"] == "installed"
        ctx.request_permission.assert_called_once()
        mock_reg.assert_called_once()

    @patch.object(_mod, "_get_auto_install", return_value=True)
    @patch.object(_mod, "subprocess")
    def test_install_pip_failure(self, mock_sub, _mock_auto):
        mock_sub.run.return_value = MagicMock(returncode=1, stderr="No matching distribution")
        ctx = self._make_ctx()
        result = handle_manage_sandbox_packages(ctx, {
            "action": "install",
            "pip_name": "nonexistent-pkg-xyz",
            "import_path": "nonexistent",
            "sandbox_alias": "nonexistent",
            "description": "test",
        })
        assert result["status"] == "error"
        assert "failed" in result["message"].lower()

    @patch.object(_mod, "_get_auto_install", return_value=True)
    @patch.object(_mod, "subprocess")
    def test_install_pip_timeout(self, mock_sub, _mock_auto):
        import subprocess as real_subprocess
        mock_sub.run.side_effect = real_subprocess.TimeoutExpired(cmd="pip", timeout=120)
        mock_sub.TimeoutExpired = real_subprocess.TimeoutExpired
        ctx = self._make_ctx()
        result = handle_manage_sandbox_packages(ctx, {
            "action": "install",
            "pip_name": "slow-pkg",
            "import_path": "slow",
            "sandbox_alias": "slow",
            "description": "test",
        })
        assert result["status"] == "error"
        assert "timed out" in result["message"]

    def test_unknown_action(self):
        ctx = self._make_ctx()
        result = handle_manage_sandbox_packages(ctx, {"action": "unknown"})
        assert result["status"] == "error"
