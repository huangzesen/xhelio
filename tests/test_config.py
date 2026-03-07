"""Tests for config.get_data_dir() — configurable data directory."""

import os
from pathlib import Path
from unittest import mock

import config


class TestGetDataDir:
    """Test get_data_dir() resolution priority."""

    def setup_method(self):
        """Reset cached value before each test."""
        config._reset_data_dir()

    def teardown_method(self):
        """Reset cached value after each test."""
        config._reset_data_dir()

    def test_default_is_home_xhelio(self):
        """With no env var or config key, returns ~/.xhelio."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XHELIO_DIR", None)
            with mock.patch("config.get", return_value=None):
                result = config.get_data_dir()
        assert result == Path.home() / ".xhelio"

    def test_env_var_overrides_default(self, tmp_path):
        """XHELIO_DIR env var takes precedence over default."""
        target = tmp_path / "custom-dir"
        with mock.patch.dict(os.environ, {"XHELIO_DIR": str(target)}):
            result = config.get_data_dir()
        assert result == target.resolve()

    def test_config_key_overrides_default(self, tmp_path):
        """data_dir config key overrides the default."""
        target = tmp_path / "config-dir"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XHELIO_DIR", None)
            with mock.patch("config.get", side_effect=lambda k, d=None: str(target) if k == "data_dir" else d):
                result = config.get_data_dir()
        assert result == target.resolve()

    def test_env_var_beats_config_key(self, tmp_path):
        """XHELIO_DIR env var has higher priority than config key."""
        env_dir = tmp_path / "env-dir"
        cfg_dir = tmp_path / "cfg-dir"
        with mock.patch.dict(os.environ, {"XHELIO_DIR": str(env_dir)}):
            with mock.patch("config.get", side_effect=lambda k, d=None: str(cfg_dir) if k == "data_dir" else d):
                result = config.get_data_dir()
        assert result == env_dir.resolve()

    def test_result_is_cached(self, tmp_path):
        """Second call returns cached value without re-reading env/config."""
        target = tmp_path / "cached-dir"
        with mock.patch.dict(os.environ, {"XHELIO_DIR": str(target)}):
            first = config.get_data_dir()
        # Even after removing the env var, cached value persists
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XHELIO_DIR", None)
            second = config.get_data_dir()
        assert first == second == target.resolve()

    def test_tilde_expansion(self):
        """Tilde paths in env var are expanded."""
        with mock.patch.dict(os.environ, {"XHELIO_DIR": "~/my-helio-data"}):
            result = config.get_data_dir()
        assert "~" not in str(result)
        assert result == (Path.home() / "my-helio-data").resolve()

    def test_reset_clears_cache(self, tmp_path):
        """_reset_data_dir() allows re-resolution."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        with mock.patch.dict(os.environ, {"XHELIO_DIR": str(dir1)}):
            first = config.get_data_dir()
        assert first == dir1.resolve()

        config._reset_data_dir()
        with mock.patch.dict(os.environ, {"XHELIO_DIR": str(dir2)}):
            second = config.get_data_dir()
        assert second == dir2.resolve()
