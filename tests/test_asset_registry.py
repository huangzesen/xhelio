"""
Tests for data_ops.asset_registry — AssetMeta and AssetRegistry.

Run with: python -m pytest tests/test_asset_registry.py
"""

import json

import numpy as np
import pandas as pd
import pytest

from data_ops.store import DataEntry, DataStore
from data_ops.asset_registry import AssetMeta, AssetRegistry


@pytest.fixture
def session_dir(tmp_path):
    """Create a temporary session directory."""
    d = tmp_path / "session_abc"
    d.mkdir()
    return d


@pytest.fixture
def data_store(session_dir):
    """Create a DataStore in the session directory."""
    return DataStore(session_dir / "data")


@pytest.fixture
def registry(session_dir, data_store):
    """Create an AssetRegistry."""
    return AssetRegistry(session_dir, data_store)


# ------------------------------------------------------------------
# File registration
# ------------------------------------------------------------------


class TestRegisterFile:
    def test_register_file_returns_asset_meta(self, registry, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b\n1,2\n")
        asset = registry.register_file(
            filename="test.csv",
            path=f,
            size_bytes=12,
            mime_type="text/csv",
        )
        assert isinstance(asset, AssetMeta)
        assert asset.kind == "file"
        assert asset.name == "test.csv"
        assert asset.asset_id.startswith("file_")
        assert asset.metadata["size_bytes"] == 12
        assert asset.metadata["mime_type"] == "text/csv"
        assert asset.metadata["extension"] == ".csv"
        assert asset.metadata["path"] == str(f)

    def test_register_file_appears_in_list(self, registry, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        registry.register_file(filename="data.json", path=f, size_bytes=2)
        assets = registry.list_assets(kind="file")
        assert len(assets) == 1
        assert assets[0]["kind"] == "file"
        assert assets[0]["name"] == "data.json"


# ------------------------------------------------------------------
# Figure registration
# ------------------------------------------------------------------


class TestRegisterFigure:
    def test_register_figure_returns_asset_meta(self, registry):
        fig_json = {"layout": {"title": {"text": "B-field"}}, "data": []}
        asset = registry.register_figure(
            fig_json=fig_json,
            trace_labels=["AC_H2_MFI.BGSEc"],
            panel_count=1,
            op_id="op_001",
        )
        assert isinstance(asset, AssetMeta)
        assert asset.kind == "figure"
        assert asset.asset_id == "fig_001"
        assert asset.name == "B-field"
        assert asset.metadata["op_id"] == "op_001"
        assert asset.metadata["trace_labels"] == ["AC_H2_MFI.BGSEc"]
        assert asset.metadata["panel_count"] == 1

    def test_figure_counter_increments(self, registry):
        for i in range(3):
            asset = registry.register_figure(
                fig_json={}, trace_labels=[], panel_count=1, op_id=f"op_{i}",
            )
        assert asset.asset_id == "fig_003"

    def test_figure_title_fallback(self, registry):
        asset = registry.register_figure(
            fig_json={}, trace_labels=[], panel_count=1, op_id="op_x",
        )
        assert asset.name == "Figure 1"

    def test_figure_title_from_string(self, registry):
        asset = registry.register_figure(
            fig_json={"layout": {"title": "My Plot"}},
            trace_labels=[], panel_count=1, op_id="op_x",
        )
        assert asset.name == "My Plot"


# ------------------------------------------------------------------
# list_assets — unified listing
# ------------------------------------------------------------------


class TestListAssets:
    def test_list_all_kinds(self, registry, data_store, tmp_path):
        # Add a data entry
        idx = pd.date_range("2024-01-01", periods=5, freq="1s")
        entry = DataEntry(
            label="test_data", data=pd.DataFrame({"v": np.ones(5)}, index=idx),
            units="nT", source="computed",
        )
        entry.id = "d001"
        data_store.put(entry)

        # Add a file
        f = tmp_path / "upload.csv"
        f.write_text("x\n1\n")
        registry.register_file(filename="upload.csv", path=f, size_bytes=4)

        # Add a figure
        registry.register_figure(
            fig_json={}, trace_labels=["test_data"], panel_count=1, op_id="op_1",
        )

        all_assets = registry.list_assets()
        kinds = {a["kind"] for a in all_assets}
        assert kinds == {"data", "file", "figure"}
        assert len(all_assets) == 3

    def test_filter_by_kind(self, registry, tmp_path):
        f = tmp_path / "a.csv"
        f.write_text("")
        registry.register_file(filename="a.csv", path=f, size_bytes=0)
        registry.register_figure(
            fig_json={}, trace_labels=[], panel_count=1, op_id="op_1",
        )

        assert len(registry.list_assets(kind="file")) == 1
        assert len(registry.list_assets(kind="figure")) == 1
        assert len(registry.list_assets(kind="data")) == 0

    def test_figure_listing_omits_fig_json(self, registry):
        registry.register_figure(
            fig_json={"data": [{"x": [1, 2, 3]}]},
            trace_labels=["a"], panel_count=1, op_id="op_1",
        )
        figures = registry.list_assets(kind="figure")
        assert "fig_json" not in figures[0]["metadata"]
        assert figures[0]["metadata"]["op_id"] == "op_1"


# ------------------------------------------------------------------
# get_asset
# ------------------------------------------------------------------


class TestGetAsset:
    def test_get_file(self, registry, tmp_path):
        f = tmp_path / "x.csv"
        f.write_text("")
        asset = registry.register_file(filename="x.csv", path=f, size_bytes=0)
        found = registry.get_asset(asset.asset_id)
        assert found is not None
        assert found.asset_id == asset.asset_id

    def test_get_figure(self, registry):
        asset = registry.register_figure(
            fig_json={}, trace_labels=[], panel_count=1, op_id="op_1",
        )
        found = registry.get_asset(asset.asset_id)
        assert found is not None
        assert found.kind == "figure"

    def test_get_missing_returns_none(self, registry):
        assert registry.get_asset("nonexistent_id") is None


# ------------------------------------------------------------------
# Persistence (save / load round-trip)
# ------------------------------------------------------------------


class TestPersistence:
    def test_save_load_round_trip(self, session_dir, data_store, tmp_path):
        reg1 = AssetRegistry(session_dir, data_store)

        # Register assets
        f = tmp_path / "file.csv"
        f.write_text("data")
        reg1.register_file(filename="file.csv", path=f, size_bytes=4, mime_type="text/csv")
        reg1.register_figure(
            fig_json={"layout": {"title": "Test"}},
            trace_labels=["a", "b"], panel_count=2, op_id="op_42",
        )
        reg1.save()

        # Verify assets.json was written
        assets_path = session_dir / "assets.json"
        assert assets_path.exists()

        # Load into a new registry
        reg2 = AssetRegistry(session_dir, data_store)
        files = reg2.list_assets(kind="file")
        figures = reg2.list_assets(kind="figure")

        assert len(files) == 1
        assert files[0]["name"] == "file.csv"
        assert files[0]["metadata"]["mime_type"] == "text/csv"

        assert len(figures) == 1
        assert figures[0]["name"] == "Test"
        assert figures[0]["metadata"]["op_id"] == "op_42"

    def test_figure_counter_persists(self, session_dir, data_store):
        reg1 = AssetRegistry(session_dir, data_store)
        reg1.register_figure(fig_json={}, trace_labels=[], panel_count=1, op_id="a")
        reg1.register_figure(fig_json={}, trace_labels=[], panel_count=1, op_id="b")
        reg1.save()

        reg2 = AssetRegistry(session_dir, data_store)
        asset = reg2.register_figure(fig_json={}, trace_labels=[], panel_count=1, op_id="c")
        assert asset.asset_id == "fig_003"

    def test_missing_assets_json_handled_gracefully(self, session_dir, data_store):
        """New sessions have no assets.json — should load without error."""
        reg = AssetRegistry(session_dir, data_store)
        assert reg.list_assets() == []

    def test_corrupt_assets_json_handled_gracefully(self, session_dir, data_store):
        """Corrupt file should not crash the registry."""
        (session_dir / "assets.json").write_text("NOT VALID JSON {{{")
        reg = AssetRegistry(session_dir, data_store)
        assert reg.list_assets() == []


# ------------------------------------------------------------------
# DataStore integration
# ------------------------------------------------------------------


class TestDataStoreIntegration:
    def test_data_entries_appear_in_list_assets(self, registry, data_store):
        idx = pd.date_range("2024-01-01", periods=10, freq="1s")
        entry = DataEntry(
            label="ACE.Bmag",
            data=pd.DataFrame({"value": np.ones(10)}, index=idx),
            units="nT",
            source="cdf",
        )
        entry.id = "ace_001"
        data_store.put(entry)

        assets = registry.list_assets(kind="data")
        assert len(assets) == 1
        assert assets[0]["asset_id"] == "ace_001"
        assert assets[0]["kind"] == "data"
        assert assets[0]["name"] == "ACE.Bmag"

    def test_data_not_duplicated_in_all_listing(self, registry, data_store):
        idx = pd.date_range("2024-01-01", periods=5, freq="1s")
        entry = DataEntry(
            label="test", data=pd.DataFrame({"v": np.ones(5)}, index=idx),
            units="nT", source="computed",
        )
        entry.id = "t001"
        data_store.put(entry)

        all_assets = registry.list_assets()
        data_assets = [a for a in all_assets if a["kind"] == "data"]
        assert len(data_assets) == 1
