"""
Tests for data_ops.store — DataEntry and DataStore (disk-backed).

Run with: python -m pytest tests/test_store.py
"""

import threading

import numpy as np
import pandas as pd
import pytest

from data_ops.store import (
    DataEntry,
    DataStore,
    generate_id,
    _compute_product_hash,
    get_store,
    set_store,
    reset_store,
    build_source_map,
    describe_sources,
)


@pytest.fixture(autouse=True)
def clean_store():
    """Reset the global store before each test."""
    reset_store()
    yield
    reset_store()


def _make_entry(label="test", n=10, vector=False):
    """Helper to create a DataEntry for testing."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1s")
    if vector:
        data = pd.DataFrame(np.random.randn(n, 3), index=idx, columns=["x", "y", "z"])
    else:
        data = pd.DataFrame(np.random.randn(n), index=idx, columns=["value"])
    return DataEntry(
        label=label,
        data=data,
        units="nT",
        description="test entry",
        source="computed",
    )


class TestDataEntry:
    def test_summary_scalar(self):
        entry = _make_entry("Bmag", n=100, vector=False)
        s = entry.summary()
        assert s["label"] == "Bmag"
        assert s["num_points"] == 100
        assert s["shape"] == "scalar"
        assert s["units"] == "nT"
        assert s["source"] == "computed"
        assert s["time_min"] is not None
        assert s["time_max"] is not None

    def test_new_fields_defaults(self):
        """New fields have sensible defaults."""
        entry = _make_entry("A")
        assert entry.id == ""
        assert entry.time_range is None
        assert entry.physical_quantity == ""
        assert entry.array_shape == ""
        assert entry.comment == ""

    def test_new_fields_set(self):
        """New fields can be set on construction."""
        entry = DataEntry(
            label="test",
            data=pd.DataFrame(
                {"v": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], name="time")
            ),
            id="abcd1234_1",
            time_range=("2024-01-01T00:00:00", "2024-01-07T23:59:59"),
            physical_quantity="magnetic_field",
            array_shape="vector[3]",
            comment="ACE MFI for solar wind study",
        )
        assert entry.id == "abcd1234_1"
        assert entry.time_range[0] == "2024-01-01T00:00:00"
        assert entry.physical_quantity == "magnetic_field"
        assert entry.array_shape == "vector[3]"
        assert entry.comment == "ACE MFI for solar wind study"

    def test_summary_includes_id(self):
        """summary() must include the id field."""
        entry = DataEntry(
            label="test",
            data=pd.DataFrame(
                {"v": [1.0, 2.0]},
                index=pd.date_range("2024-01-01", periods=2, freq="1h"),
            ),
            id="abcd1234_1",
            time_range=("2024-01-01", "2024-01-07"),
            physical_quantity="magnetic_field",
            array_shape="scalar",
            comment="test note",
        )
        s = entry.summary()
        assert s["id"] == "abcd1234_1"
        assert s["time_range"] == ("2024-01-01", "2024-01-07")
        assert s["physical_quantity"] == "magnetic_field"
        assert s["array_shape"] == "scalar"
        assert s["comment"] == "test note"

    def test_summary_vector(self):
        entry = _make_entry("B", n=50, vector=True)
        s = entry.summary()
        assert s["shape"] == "vector[3]"
        assert s["num_points"] == 50

    def test_summary_empty(self):
        empty_df = pd.DataFrame(dtype=np.float64)
        empty_df.index = pd.DatetimeIndex([], name="time")
        entry = DataEntry(label="empty", data=empty_df)
        s = entry.summary()
        assert s["num_points"] == 0
        assert s["time_min"] is None
        assert s["time_max"] is None

    def test_backward_compat_time(self):
        entry = _make_entry("A", n=5)
        t = entry.time
        assert isinstance(t, np.ndarray)
        assert np.issubdtype(t.dtype, np.datetime64)
        assert len(t) == 5

    def test_backward_compat_values_scalar(self):
        entry = _make_entry("A", n=5, vector=False)
        v = entry.values
        assert isinstance(v, np.ndarray)
        assert v.ndim == 1
        assert len(v) == 5

    def test_backward_compat_values_vector(self):
        entry = _make_entry("A", n=5, vector=True)
        v = entry.values
        assert isinstance(v, np.ndarray)
        assert v.ndim == 2
        assert v.shape == (5, 3)

    def test_summary_non_timeseries(self):
        """Non-timeseries entry uses index_min/index_max instead of time_min/time_max."""
        df = pd.DataFrame({"flux": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
        entry = DataEntry(
            label="events",
            data=df,
            units="",
            description="Event catalog",
            source="created",
            is_timeseries=False,
        )
        s = entry.summary()
        assert s["label"] == "events"
        assert s["num_points"] == 3
        assert "time_min" not in s
        assert "time_max" not in s
        assert s["index_min"] == "10"
        assert s["index_max"] == "30"

    def test_summary_non_timeseries_empty(self):
        """Empty non-timeseries entry should have None index_min/index_max."""
        df = pd.DataFrame({"val": pd.Series(dtype=float)})
        entry = DataEntry(label="empty_nt", data=df, is_timeseries=False)
        s = entry.summary()
        assert s["index_min"] is None
        assert s["index_max"] is None
        assert "time_min" not in s

    def test_is_timeseries_default_true(self):
        """is_timeseries defaults to True."""
        entry = _make_entry("A")
        assert entry.is_timeseries is True

    def test_summary_includes_is_timeseries_true(self):
        """Timeseries summary dict includes is_timeseries: True."""
        entry = _make_entry("Bmag", n=10)
        s = entry.summary()
        assert s["is_timeseries"] is True

    def test_summary_includes_is_timeseries_false(self):
        """Non-timeseries summary dict includes is_timeseries: False."""
        df = pd.DataFrame({"flux": [1.0, 2.0]}, index=[10, 20])
        entry = DataEntry(label="events", data=df, is_timeseries=False)
        s = entry.summary()
        assert s["is_timeseries"] is False


class TestDataStore:
    def test_put_and_get(self, tmp_path):
        store = DataStore(tmp_path / "data")
        entry = _make_entry("A")
        store.put(entry)
        got = store.get("A")
        assert got is not None
        assert got.label == "A"
        pd.testing.assert_frame_equal(got.data, entry.data)
        assert store.get("B") is None

    def test_has(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("A"))
        assert store.has("A")
        assert not store.has("B")

    def test_overwrite(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("A", n=10))
        store.put(_make_entry("A", n=20))
        assert len(store.get("A").data) == 20

    def test_remove(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("A"))
        assert store.remove("A") is True
        assert store.get("A") is None
        assert store.remove("A") is False

    def test_list_entries(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("A"))
        store.put(_make_entry("B"))
        entries = store.list_entries()
        assert len(entries) == 2
        labels = {e["label"] for e in entries}
        assert labels == {"A", "B"}

    def test_clear(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("A"))
        store.put(_make_entry("B"))
        store.clear()
        assert len(store) == 0

    def test_len(self, tmp_path):
        store = DataStore(tmp_path / "data")
        assert len(store) == 0
        store.put(_make_entry("A"))
        assert len(store) == 1


class TestGetStore:
    def test_raises_without_set(self):
        """get_store() raises RuntimeError if no store has been set."""
        with pytest.raises(RuntimeError, match="No DataStore"):
            get_store()

    def test_set_and_get(self, tmp_path):
        store = DataStore(tmp_path / "data")
        set_store(store)
        assert get_store() is store

    def test_reset(self, tmp_path):
        store = DataStore(tmp_path / "data")
        set_store(store)
        reset_store()
        with pytest.raises(RuntimeError):
            get_store()


class TestBuildSourceMap:
    def test_single_label(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("DATASET.BR", n=10))
        sources, err = build_source_map(store, ["DATASET.BR"])
        assert err is None
        assert "df_BR" in sources
        assert len(sources["df_BR"]) == 10

    def test_three_scalars(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("DATASET.BR", n=10))
        store.put(_make_entry("DATASET.BT", n=10))
        store.put(_make_entry("DATASET.BN", n=10))
        sources, err = build_source_map(
            store, ["DATASET.BR", "DATASET.BT", "DATASET.BN"]
        )
        assert err is None
        assert set(sources.keys()) == {"df_BR", "df_BT", "df_BN"}

    def test_missing_label(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("DATASET.BR", n=10))
        sources, err = build_source_map(store, ["DATASET.BR", "DATASET.MISSING"])
        assert sources is None
        assert "DATASET.MISSING" in err

    def test_duplicate_suffix(self, tmp_path):
        """Duplicate suffixes are now allowed with ID-based disambiguation."""
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("A.val", n=10))
        store.put(_make_entry("B.val", n=10))
        sources, err = build_source_map(store, ["A.val", "B.val"])
        assert err is None
        # With ID-based disambiguation, both entries should be available
        assert len(sources) == 2

    def test_no_dot_label(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("Bmag", n=10))
        sources, err = build_source_map(store, ["Bmag"])
        assert err is None
        assert "df_Bmag" in sources


class TestDescribeSources:
    def test_basic_summary(self, tmp_path):
        store = DataStore(tmp_path / "data")
        idx = pd.date_range("2024-01-01", periods=100, freq="1min")
        data = pd.DataFrame({"BR": np.random.randn(100)}, index=idx)
        data.iloc[50:60] = np.nan  # 10% NaN
        store.put(DataEntry(label="DATASET.BR", data=data, units="nT"))

        info = describe_sources(store, ["DATASET.BR"])
        assert "df_BR" in info
        summary = info["df_BR"]
        assert summary["label"] == "DATASET.BR"
        assert summary["points"] == 100
        assert len(summary["columns"]) == 1
        assert summary["nan_pct"] == 10.0
        assert len(summary["time_range"]) == 2
        assert summary["cadence"] != ""

    def test_missing_label_skipped(self, tmp_path):
        store = DataStore(tmp_path / "data")
        info = describe_sources(store, ["NONEXISTENT"])
        assert info == {}

    def test_non_timeseries_summary(self, tmp_path):
        """Non-timeseries entries use index_range instead of time_range/cadence."""
        store = DataStore(tmp_path / "data")
        df = pd.DataFrame({"flux": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
        entry = DataEntry(label="EVENTS.catalog", data=df, is_timeseries=False)
        store.put(entry)
        info = describe_sources(store, ["EVENTS.catalog"])
        assert "df_catalog" in info
        summary = info["df_catalog"]
        assert "time_range" not in summary
        assert "cadence" not in summary
        assert summary["index_range"] == ["10", "30"]


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_is_timeseries_persists(self, tmp_path):
        """is_timeseries flag should survive save/load cycle."""
        store = DataStore(tmp_path / "data")
        # Timeseries entry
        ts_entry = _make_entry("ts_data", n=5)
        assert ts_entry.is_timeseries is True
        store.put(ts_entry)
        # Non-timeseries entry
        df = pd.DataFrame({"val": [1.0, 2.0]}, index=[0, 1])
        nt_entry = DataEntry(label="nt_data", data=df, is_timeseries=False)
        store.put(nt_entry)

        # Create a new store pointing at the same directory
        store2 = DataStore(tmp_path / "data")
        assert len(store2) == 2

        loaded_ts = store2.get("ts_data")
        assert loaded_ts.is_timeseries is True
        loaded_nt = store2.get("nt_data")
        assert loaded_nt.is_timeseries is False

    def test_persists_across_instances(self, tmp_path):
        """A new DataStore at the same dir sees previously stored data."""
        data_dir = tmp_path / "data"
        store1 = DataStore(data_dir)
        store1.put(_make_entry("X", n=20))
        store1.put(_make_entry("Y", n=30))

        store2 = DataStore(data_dir)
        assert len(store2) == 2
        assert store2.has("X")
        assert store2.has("Y")
        x = store2.get("X")
        assert len(x.data) == 20

    def test_hash_routing_pds_urn(self, tmp_path):
        """PDS URN labels with colons/slashes round-trip correctly."""
        store = DataStore(tmp_path / "data")
        label = "urn:nasa:pds:cassini-mag-cal:data-1sec-krtp"
        entry = _make_entry(label, n=5)
        store.put(entry)
        assert store.has(label)

        # Reload from disk
        store2 = DataStore(tmp_path / "data")
        assert store2.has(label)
        got = store2.get(label)
        assert got.label == label
        pd.testing.assert_frame_equal(got.data, entry.data)


# ---------------------------------------------------------------------------
# Cross-thread visibility tests
# ---------------------------------------------------------------------------


class TestCrossThreadVisibility:
    def test_worker_sees_main_put(self, tmp_path):
        """Worker thread sees data put by the main thread."""
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("main_data", n=10))

        result = {}

        def worker():
            entry = store.get("main_data")
            result["found"] = entry is not None
            result["label"] = entry.label if entry else None

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert result["found"] is True
        assert result["label"] == "main_data"

    def test_main_sees_worker_put(self, tmp_path):
        """Main thread sees data put by a worker thread."""
        store = DataStore(tmp_path / "data")

        def worker():
            store.put(_make_entry("worker_data", n=15))

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert store.has("worker_data")
        entry = store.get("worker_data")
        assert entry is not None
        assert len(entry.data) == 15


# ---------------------------------------------------------------------------
# Concurrent access tests
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    """Verify DataStore thread-safety under concurrent access."""

    def test_concurrent_put_get(self, tmp_path):
        """10 threads doing put/get simultaneously should not corrupt state."""
        store = DataStore(tmp_path / "data")
        errors = []

        def worker(thread_id):
            try:
                for i in range(20):
                    label = f"thread_{thread_id}_entry_{i}"
                    entry = _make_entry(label=label, n=5)
                    store.put(entry)
                    retrieved = store.get(label)
                    if retrieved is None:
                        errors.append(f"{label} was None after put")
                    elif retrieved.label != label:
                        errors.append(f"{label} had wrong label: {retrieved.label}")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent errors: {errors}"
        assert len(store) == 200  # 10 threads × 20 entries

    def test_concurrent_list_entries(self, tmp_path):
        """list_entries should not crash while other threads modify the store."""
        store = DataStore(tmp_path / "data")
        errors = []

        def writer():
            try:
                for i in range(50):
                    store.put(_make_entry(label=f"w_{i}", n=3))
            except Exception as e:
                errors.append(f"Writer error: {e}")

        def reader():
            try:
                for _ in range(50):
                    store.list_entries()
            except Exception as e:
                errors.append(f"Reader error: {e}")

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent errors: {errors}"


# ---------------------------------------------------------------------------
# Async stats tests
# ---------------------------------------------------------------------------


class TestAsyncStats:
    def test_computing_flag_removed(self, tmp_path):
        """After put(), the .computing flag is eventually removed."""
        import time

        store = DataStore(tmp_path / "data")
        entry = _make_entry("A", n=10)
        store.put(entry)

        # Wait for async stats to finish
        # The hash folder contains the ID plus a short label hash
        entry_id = entry.id
        ids = store._ids
        h = None
        for eid, folder in ids.items():
            if eid == entry_id:
                h = folder
                break
        assert h is not None, "Entry should be in store"
        flag = tmp_path / "data" / h / ".computing"
        deadline = time.monotonic() + 5.0
        while flag.exists() and time.monotonic() < deadline:
            time.sleep(0.05)

        assert not flag.exists(), ".computing flag was not removed"

    def test_list_entries_has_stats(self, tmp_path):
        """list_entries() returns full stats after async computation."""
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("A", n=100))

        # list_entries waits for .computing to clear
        entries = store.list_entries()
        assert len(entries) == 1
        s = entries[0]
        assert s["num_points"] == 100
        assert s["label"] == "A"


class TestIDGeneration:
    def test_product_hash_fetch(self):
        h = _compute_product_hash(dataset_id="AC_H2_MFI", parameter_id="BGSEc")
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_product_hash_deterministic(self):
        h1 = _compute_product_hash(dataset_id="AC_H2_MFI", parameter_id="BGSEc")
        h2 = _compute_product_hash(dataset_id="AC_H2_MFI", parameter_id="BGSEc")
        assert h1 == h2

    def test_product_hash_different_params(self):
        h1 = _compute_product_hash(dataset_id="AC_H2_MFI", parameter_id="BGSEc")
        h2 = _compute_product_hash(dataset_id="AC_H2_MFI", parameter_id="Magnitude")
        assert h1 != h2

    def test_product_hash_computed(self):
        h = _compute_product_hash(
            source_ids=["a1b2c3d4_1", "e5f6a7b8_1"], code="result = df_A + df_B"
        )
        assert len(h) == 8

    def test_product_hash_created(self):
        h = _compute_product_hash(output_label="my_catalog")
        assert len(h) == 8

    def test_generate_id_sequential(self, tmp_path):
        store = DataStore(tmp_path / "data")
        id1 = generate_id(store, output_label="A")
        id2 = generate_id(store, output_label="A")
        # Same hash prefix, sequential suffix
        prefix1, n1 = id1.rsplit("_", 1)
        prefix2, n2 = id2.rsplit("_", 1)
        assert prefix1 == prefix2
        assert int(n1) == 1
        assert int(n2) == 2

    def test_generate_id_different_products(self, tmp_path):
        store = DataStore(tmp_path / "data")
        id1 = generate_id(store, output_label="A")
        id2 = generate_id(store, output_label="B")
        prefix1 = id1.rsplit("_", 1)[0]
        prefix2 = id2.rsplit("_", 1)[0]
        assert prefix1 != prefix2


class TestStoreWithIDs:
    def test_put_assigns_id(self, tmp_path):
        """put() auto-assigns an ID if entry.id is empty."""
        store = DataStore(tmp_path / "data")
        entry = _make_entry("AC_H2_MFI.BGSEc")
        assert entry.id == ""
        store.put(entry)
        assert entry.id != ""
        assert "_" in entry.id

    def test_get_by_id(self, tmp_path):
        store = DataStore(tmp_path / "data")
        entry = _make_entry("A")
        store.put(entry)
        got = store.get(entry.id)
        assert got is not None
        assert got.label == "A"

    def test_get_by_label_returns_most_recent(self, tmp_path):
        store = DataStore(tmp_path / "data")
        e1 = _make_entry("A", n=10)
        store.put(e1)
        e2 = _make_entry("A", n=20)
        store.put(e2)
        got = store.get("A")
        assert len(got.data) == 20  # Most recent

    def test_same_label_different_ids(self, tmp_path):
        """Two puts with same label create two separate entries."""
        store = DataStore(tmp_path / "data")
        e1 = _make_entry("A", n=10)
        store.put(e1)
        e2 = _make_entry("A", n=20)
        store.put(e2)
        assert e1.id != e2.id
        assert len(store) == 2

    def test_get_by_label_returns_list(self, tmp_path):
        store = DataStore(tmp_path / "data")
        e1 = _make_entry("A", n=10)
        store.put(e1)
        e2 = _make_entry("A", n=20)
        store.put(e2)
        entries = store.get_by_label("A")
        assert len(entries) == 2

    def test_list_entries_includes_id(self, tmp_path):
        store = DataStore(tmp_path / "data")
        store.put(_make_entry("A"))
        entries = store.list_entries()
        assert len(entries) == 1
        assert "id" in entries[0]
        assert entries[0]["id"] != ""

    def test_remove_by_id(self, tmp_path):
        store = DataStore(tmp_path / "data")
        entry = _make_entry("A")
        store.put(entry)
        assert store.remove(entry.id)
        assert store.get(entry.id) is None

    def test_has_by_id(self, tmp_path):
        store = DataStore(tmp_path / "data")
        entry = _make_entry("A")
        store.put(entry)
        assert store.has(entry.id)
        assert store.has("A")  # Label still works


class TestMergeEntries:
    def test_merge_two_timeseries(self, tmp_path):
        store = DataStore(tmp_path / "data")
        idx1 = pd.date_range("2024-01-01", periods=10, freq="1h")
        e1 = DataEntry(
            label="A",
            data=pd.DataFrame({"v": range(10)}, index=idx1, dtype=float),
            units="nT",
        )
        e1.id = generate_id(store, output_label="A")
        store.put(e1)

        idx2 = pd.date_range("2024-01-05", periods=10, freq="1h")
        e2 = DataEntry(
            label="A",
            data=pd.DataFrame({"v": range(10, 20)}, index=idx2, dtype=float),
            units="nT",
        )
        e2.id = generate_id(store, output_label="A")
        store.put(e2)

        assert e1.id.rsplit("_", 1)[0] == e2.id.rsplit("_", 1)[0]  # Same hash

        merged = store.merge_entries([e1.id, e2.id])
        assert merged is not None
        assert len(merged.data) == 20  # Both ranges combined
        assert merged.label == "A"
        # Original entries removed
        assert store.get(e1.id) is None
        assert store.get(e2.id) is None
        # Merged entry exists
        assert store.get(merged.id) is not None

    def test_merge_different_products_fails(self, tmp_path):
        store = DataStore(tmp_path / "data")
        e1 = _make_entry("A")
        e1.id = generate_id(store, output_label="A")
        store.put(e1)
        e2 = _make_entry("B")
        e2.id = generate_id(store, output_label="B")
        store.put(e2)
        with pytest.raises(ValueError, match="same data product"):
            store.merge_entries([e1.id, e2.id])


class TestPersistence:
    def test_id_persists_across_instances(self, tmp_path):
        """IDs persist across DataStore instances."""
        data_dir = tmp_path / "data"
        store1 = DataStore(data_dir)
        entry = _make_entry("X", n=20)
        store1.put(entry)
        saved_id = entry.id

        store2 = DataStore(data_dir)
        loaded = store2.get(saved_id)
        assert loaded is not None
        assert loaded.id == saved_id
        assert loaded.label == "X"

    def test_ids_json_created(self, tmp_path):
        """_ids.json is created when entries are stored."""
        data_dir = tmp_path / "data"
        store = DataStore(data_dir)
        store.put(_make_entry("Y", n=10))

        ids_path = data_dir / "_ids.json"
        assert ids_path.exists()

    def test_label_index_json_created(self, tmp_path):
        """_label_index.json is created when entries are stored."""
        data_dir = tmp_path / "data"
        store = DataStore(data_dir)
        store.put(_make_entry("Z", n=10))

        label_index_path = data_dir / "_label_index.json"
        assert label_index_path.exists()
