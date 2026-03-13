"""
Tests for data_ops.asset_registry — AssetMeta and AssetRegistry.

Run with: python -m pytest tests/test_asset_registry.py
"""

import json
from pathlib import Path

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

    def test_register_file_with_custom_asset_id(self, registry, tmp_path):
        f = tmp_path / "test.h5"
        f.write_bytes(b"fake hdf5")
        asset = registry.register_file(
            filename="Test File",
            path=f,
            size_bytes=9,
            asset_id="file_custom01",
        )
        assert asset.asset_id == "file_custom01"
        assert registry.get_asset("file_custom01") is not None

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


# ------------------------------------------------------------------
# register_image
# ------------------------------------------------------------------


def test_register_image(tmp_path):
    """register_image creates a figure asset with image_path and source_url."""
    from data_ops.store import DataStore
    from data_ops.asset_registry import AssetRegistry

    session_dir = tmp_path / "session_img"
    session_dir.mkdir()
    store = DataStore(session_dir / "data")
    ar = AssetRegistry(session_dir, store)

    meta = ar.register_image(
        name="APOD 2026-03-12",
        image_path="/tmp/session/mpl_outputs/20260312_120000_abc123.png",
        source_url="https://apod.nasa.gov/apod/image/2603/example.jpg",
    )

    assert meta.kind == "figure"
    assert meta.asset_id == "fig_001"
    assert meta.name == "APOD 2026-03-12"
    assert meta.metadata["image_path"] == "/tmp/session/mpl_outputs/20260312_120000_abc123.png"
    assert meta.metadata["source_url"] == "https://apod.nasa.gov/apod/image/2603/example.jpg"
    assert "fig_json" not in meta.metadata


def test_register_image_persists(tmp_path):
    """register_image entries survive save/load cycle."""
    from data_ops.store import DataStore
    from data_ops.asset_registry import AssetRegistry

    session_dir = tmp_path / "session_persist"
    session_dir.mkdir()
    store = DataStore(session_dir / "data")
    ar = AssetRegistry(session_dir, store)
    ar.register_image(name="Test Image", image_path="/tmp/test.png")
    ar.save()

    ar2 = AssetRegistry(session_dir, store)
    assets = ar2.list_assets(kind="figure")
    assert len(assets) == 1
    assert assets[0]["name"] == "Test Image"


def test_register_image_appears_in_list_assets(tmp_path):
    """register_image entries show up in list_assets with kind=figure and kind=None."""
    from data_ops.store import DataStore
    from data_ops.asset_registry import AssetRegistry

    session_dir = tmp_path / "session_list"
    session_dir.mkdir()
    store = DataStore(session_dir / "data")
    ar = AssetRegistry(session_dir, store)
    ar.register_image(name="Web Figure", image_path="/tmp/fig.png")

    all_assets = ar.list_assets()
    figure_assets = ar.list_assets(kind="figure")
    assert any(a["name"] == "Web Figure" for a in all_assets)
    assert any(a["name"] == "Web Figure" for a in figure_assets)


# ------------------------------------------------------------------
# New fields (source_path, session_path, lineage, figure_kind)
# ------------------------------------------------------------------


class TestAssetMetaNewFields:
    def test_default_new_fields_are_none(self):
        meta = AssetMeta(
            asset_id="file_abc", kind="file", name="test.csv",
            created_at="2026-01-01T00:00:00Z", metadata={},
        )
        assert meta.source_path is None
        assert meta.session_path is None
        assert meta.lineage is None
        assert meta.figure_kind is None

    def test_new_fields_set_explicitly(self):
        meta = AssetMeta(
            asset_id="fig_001", kind="figure", name="Plot 1",
            created_at="2026-01-01T00:00:00Z", metadata={},
            source_path=None, session_path=None,
            lineage={"data_sources": ["d001"]}, figure_kind="plotly",
        )
        assert meta.figure_kind == "plotly"
        assert meta.lineage == {"data_sources": ["d001"]}


class TestBackwardCompatLoad:
    def test_old_assets_json_loads_with_none_fields(self, session_dir, data_store):
        """Old assets.json without new fields should load with None defaults."""
        old_data = {
            "figure_counter": 1,
            "files": {
                "file_abc": {
                    "asset_id": "file_abc", "kind": "file",
                    "name": "old.csv", "created_at": "2026-01-01T00:00:00Z",
                    "metadata": {"path": "/tmp/old.csv", "size_bytes": 100},
                }
            },
            "figures": {
                "fig_001": {
                    "asset_id": "fig_001", "kind": "figure",
                    "name": "Old Plot", "created_at": "2026-01-01T00:00:00Z",
                    "metadata": {"op_id": "op_1"},
                }
            },
        }
        (session_dir / "assets.json").write_text(json.dumps(old_data))
        reg = AssetRegistry(session_dir, data_store)

        files = reg.list_assets(kind="file")
        assert len(files) == 1
        file_asset = reg.get_asset("file_abc")
        assert file_asset.source_path is None
        assert file_asset.session_path is None
        assert file_asset.lineage is None
        assert file_asset.figure_kind is None

        fig_asset = reg.get_asset("fig_001")
        assert fig_asset.figure_kind is None


# ------------------------------------------------------------------
# Remove assets
# ------------------------------------------------------------------


class TestRemoveAssets:
    def test_remove_file(self, registry, tmp_path):
        f = tmp_path / "removeme.csv"
        f.write_text("data")
        asset = registry.register_file(filename="removeme.csv", path=f, size_bytes=4)
        assert registry.get_asset(asset.asset_id) is not None

        result = registry.remove_file(asset.asset_id)
        assert result is True
        assert registry.get_asset(asset.asset_id) is None
        assert len(registry.list_assets(kind="file")) == 0

    def test_remove_file_with_session_copy(self, registry, tmp_path, session_dir):
        """remove_file deletes the session copy if it exists."""
        uploads = session_dir / "uploads"
        uploads.mkdir()
        copy = uploads / "data.csv"
        copy.write_text("data")

        f = tmp_path / "data.csv"
        f.write_text("data")
        asset = registry.register_file(filename="data.csv", path=f, size_bytes=4)
        # Simulate prepare having set session_path
        asset.session_path = str(copy)

        registry.remove_file(asset.asset_id)
        assert not copy.exists()

    def test_remove_file_missing_returns_false(self, registry):
        assert registry.remove_file("nonexistent") is False

    def test_remove_figure(self, registry, session_dir):
        asset = registry.register_figure(
            fig_json={}, trace_labels=[], panel_count=1, op_id="op_1",
        )
        result = registry.remove_figure(asset.asset_id)
        assert result is True
        assert registry.get_asset(asset.asset_id) is None

    def test_remove_figure_deletes_files(self, registry, session_dir):
        """remove_figure deletes thumbnail and JSON files."""
        plotly_dir = session_dir / "plotly_outputs"
        plotly_dir.mkdir()
        thumb = plotly_dir / "op_1.png"
        thumb.write_bytes(b"fake png")
        json_file = plotly_dir / "op_1.json"
        json_file.write_text("{}")

        asset = registry.register_figure(
            fig_json={}, trace_labels=[], panel_count=1, op_id="op_1",
            thumbnail_path=str(thumb),
        )
        registry.remove_figure(asset.asset_id)
        assert not thumb.exists()
        assert not json_file.exists()

    def test_remove_figure_missing_returns_false(self, registry):
        assert registry.remove_figure("fig_999") is False


# ------------------------------------------------------------------
# list_assets_enriched
# ------------------------------------------------------------------


class TestListAssetsEnriched:
    def test_summary_counts(self, registry, data_store, tmp_path):
        # Add 1 data, 1 file, 2 figures
        idx = pd.date_range("2024-01-01", periods=5, freq="1s")
        entry = DataEntry(
            label="test", data=pd.DataFrame({"v": np.ones(5)}, index=idx),
            units="nT", source="file",
        )
        entry.id = "d001"
        data_store.put(entry)

        f = tmp_path / "a.csv"
        f.write_text("")
        registry.register_file(filename="a.csv", path=f, size_bytes=0)
        registry.register_figure(fig_json={}, trace_labels=["test"], panel_count=1, op_id="op_1")
        registry.register_figure(fig_json={}, trace_labels=[], panel_count=1, op_id="op_2")

        result = registry.list_assets_enriched()
        assert result["summary"] == {"data": 1, "files": 1, "figures": 2}
        assert len(result["data"]) == 1
        assert len(result["files"]) == 1
        assert len(result["figures"]) == 2

    def test_filter_by_kind(self, registry, tmp_path):
        f = tmp_path / "b.csv"
        f.write_text("")
        registry.register_file(filename="b.csv", path=f, size_bytes=0)
        registry.register_figure(fig_json={}, trace_labels=[], panel_count=1, op_id="op_1")

        result = registry.list_assets_enriched(kind="file")
        assert result["summary"]["files"] == 1
        assert "data" not in result
        assert "figures" not in result

    def test_figure_kind_in_output(self, registry):
        asset = registry.register_figure(
            fig_json={}, trace_labels=[], panel_count=1, op_id="op_1",
        )
        asset.figure_kind = "plotly"

        result = registry.list_assets_enriched(kind="figure")
        assert result["figures"][0]["figure_kind"] == "plotly"

    def test_file_has_local_copy_field(self, registry, tmp_path):
        f = tmp_path / "c.csv"
        f.write_text("")
        asset = registry.register_file(filename="c.csv", path=f, size_bytes=0)

        result = registry.list_assets_enriched(kind="file")
        assert result["files"][0]["has_local_copy"] is False

        asset.session_path = "/tmp/copy.csv"
        result = registry.list_assets_enriched(kind="file")
        assert result["files"][0]["has_local_copy"] is True

    def test_figure_data_sources(self, registry):
        registry.register_figure(
            fig_json={}, trace_labels=["ACE.Bmag", "ACE.Np"],
            panel_count=2, op_id="op_1",
        )
        result = registry.list_assets_enriched(kind="figure")
        assert result["figures"][0]["data_sources"] == ["ACE.Bmag", "ACE.Np"]
