"""Tests for data_ops.operations_log."""

import json
import threading

import pytest

from data_ops.operations_log import OperationsLog, get_operations_log, reset_operations_log


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global singleton before/after each test."""
    reset_operations_log()
    yield
    reset_operations_log()


class TestOperationsLog:
    """Unit tests for the OperationsLog class."""

    def test_record_creates_entry(self):
        log = OperationsLog()
        rec = log.record(
            tool="fetch_data",
            args={"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            outputs=["AC_H2_MFI.BGSEc"],
        )
        assert rec["id"] == "op_001"
        assert rec["tool"] == "fetch_data"
        assert rec["status"] == "success"
        assert rec["outputs"] == ["AC_H2_MFI.BGSEc"]
        assert rec["inputs"] == []
        assert rec["error"] is None
        assert "timestamp" in rec

    def test_id_auto_increments(self):
        log = OperationsLog()
        r1 = log.record(tool="fetch_data", args={}, outputs=["a"])
        r2 = log.record(tool="custom_operation", args={}, outputs=["b"])
        r3 = log.record(tool="store_dataframe", args={}, outputs=["c"])
        assert r1["id"] == "op_001"
        assert r2["id"] == "op_002"
        assert r3["id"] == "op_003"

    def test_record_with_inputs_and_error(self):
        log = OperationsLog()
        rec = log.record(
            tool="custom_operation",
            args={"code": "bad code"},
            inputs=["AC_H2_MFI.BGSEc"],
            outputs=[],
            status="error",
            error="Execution error: name 'bad' is not defined",
        )
        assert rec["status"] == "error"
        assert rec["inputs"] == ["AC_H2_MFI.BGSEc"]
        assert rec["outputs"] == []
        assert "not defined" in rec["error"]

    def test_get_records_returns_copy(self):
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["a"])
        records = log.get_records()
        assert len(records) == 1
        # Mutating the returned list shouldn't affect internal state
        records.clear()
        assert len(log.get_records()) == 1

    def test_len(self):
        log = OperationsLog()
        assert len(log) == 0
        log.record(tool="fetch_data", args={}, outputs=["a"])
        assert len(log) == 1
        log.record(tool="fetch_data", args={}, outputs=["b"])
        assert len(log) == 2

    def test_clear(self):
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["a"])
        log.record(tool="fetch_data", args={}, outputs=["b"])
        assert len(log) == 2
        log.clear()
        assert len(log) == 0
        # Counter should reset too
        rec = log.record(tool="fetch_data", args={}, outputs=["c"])
        assert rec["id"] == "op_001"

    def test_json_roundtrip(self, tmp_path):
        log = OperationsLog()
        log.record(
            tool="fetch_data",
            args={"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            outputs=["AC_H2_MFI.BGSEc"],
        )
        log.record(
            tool="custom_operation",
            args={"code": "result = df.mean()", "output_label": "mean"},
            inputs=["AC_H2_MFI.BGSEc"],
            outputs=["mean"],
        )

        path = tmp_path / "operations.json"
        log.save_to_file(path)

        # Verify the JSON file is valid
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["tool"] == "fetch_data"
        assert data[1]["tool"] == "custom_operation"

        # Load into a fresh log
        log2 = OperationsLog()
        count = log2.load_from_file(path)
        assert count == 2
        records = log2.get_records()
        assert records[0]["id"] == "op_001"
        assert records[1]["id"] == "op_002"
        assert records[1]["inputs"] == ["AC_H2_MFI.BGSEc"]

    def test_counter_resumes_after_load(self, tmp_path):
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["a"])
        log.record(tool="fetch_data", args={}, outputs=["b"])
        log.record(tool="fetch_data", args={}, outputs=["c"])

        path = tmp_path / "operations.json"
        log.save_to_file(path)

        log2 = OperationsLog()
        log2.load_from_file(path)
        # Next record should be op_004
        rec = log2.record(tool="fetch_data", args={}, outputs=["d"])
        assert rec["id"] == "op_004"

    def test_load_from_records(self):
        records = [
            {"id": "op_001", "tool": "fetch_data", "args": {}, "outputs": ["a"],
             "inputs": [], "status": "success", "error": None, "timestamp": "2026-01-01T00:00:00+00:00"},
            {"id": "op_002", "tool": "custom_operation", "args": {}, "outputs": ["b"],
             "inputs": ["a"], "status": "success", "error": None, "timestamp": "2026-01-01T00:01:00+00:00"},
        ]
        log = OperationsLog()
        count = log.load_from_records(records)
        assert count == 2
        assert len(log) == 2
        # Counter resumes
        rec = log.record(tool="fetch_data", args={}, outputs=["c"])
        assert rec["id"] == "op_003"

    def test_thread_safety(self):
        log = OperationsLog()
        n_threads = 10
        n_per_thread = 50
        errors = []

        def worker(thread_id):
            try:
                for i in range(n_per_thread):
                    log.record(
                        tool="fetch_data",
                        args={"thread": thread_id, "i": i},
                        outputs=[f"t{thread_id}_{i}"],
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(log) == n_threads * n_per_thread

        # All IDs should be unique
        records = log.get_records()
        ids = [r["id"] for r in records]
        assert len(set(ids)) == len(ids)


class TestGetPipeline:
    """Tests for OperationsLog.get_pipeline()."""

    def test_basic_chain(self):
        """fetch → compute → render produces correct pipeline."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={"dataset_id": "AC_H2_MFI"}, outputs=["Bx"])
        log.record(
            tool="custom_operation",
            args={"code": "mag = ..."},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        log.record(
            tool="render_plotly_json",
            args={"figure": {}},
            inputs=["Bmag"],
            outputs=[],
        )
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        tools = [r["tool"] for r in pipeline]
        assert tools == ["fetch_data", "custom_operation", "render_plotly_json"]

    def test_superseded_computation_keeps_last(self):
        """When a label is produced twice, only the last producer is kept."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="custom_operation",
            args={"code": "wrong"},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        log.record(
            tool="custom_operation",
            args={"code": "correct"},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        # Should have fetch + last custom_operation only
        assert len(pipeline) == 2
        assert pipeline[0]["tool"] == "fetch_data"
        assert pipeline[1]["args"]["code"] == "correct"

    def test_dedup_skips_excluded(self):
        """fetch_data with already_loaded=true is skipped; real fetch is used."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={"dataset_id": "AC_H2_MFI"}, outputs=["Bx"])
        log.record(
            tool="fetch_data",
            args={"dataset_id": "AC_H2_MFI", "already_loaded": True},
            outputs=["Bx"],
        )
        pipeline = log.get_pipeline({"Bx"})
        # The dedup record is ignored during producer selection, so the
        # real fetch (op_001) is the last producer and appears in the pipeline.
        assert len(pipeline) == 1
        assert pipeline[0]["tool"] == "fetch_data"
        assert pipeline[0]["args"].get("already_loaded") is None

    def test_error_records_excluded(self):
        """Error records are never included in the pipeline."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="custom_operation",
            args={"code": "bad"},
            inputs=["Bx"],
            outputs=[],
            status="error",
            error="Execution error",
        )
        log.record(
            tool="custom_operation",
            args={"code": "good"},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        assert len(pipeline) == 2
        assert all(r["status"] == "success" for r in pipeline)

    def test_transitive_input_resolution(self):
        """A → B → C: requesting C pulls in A and B."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(
            tool="custom_operation", args={"code": "B = f(A)"}, inputs=["A"], outputs=["B"]
        )
        log.record(
            tool="custom_operation", args={"code": "C = f(B)"}, inputs=["B"], outputs=["C"]
        )
        pipeline = log.get_pipeline({"C"})
        tools = [r["tool"] for r in pipeline]
        assert tools == ["fetch_data", "custom_operation", "custom_operation"]
        assert pipeline[0]["outputs"] == ["A"]
        assert pipeline[1]["outputs"] == ["B"]
        assert pipeline[2]["outputs"] == ["C"]

    def test_render_plotly_json_included(self):
        """The last successful render_plotly_json is included even if not a label producer."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json",
            args={"figure": {"data": []}},
            inputs=["Bx"],
            outputs=[],
        )
        pipeline = log.get_pipeline({"Bx"})
        tools = [r["tool"] for r in pipeline]
        assert "render_plotly_json" in tools

    def test_all_renders_included(self):
        """Multiple render_plotly_json calls — all successful ones are kept."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json",
            args={"figure": {"version": 1}},
            inputs=["Bx"],
            outputs=[],
        )
        log.record(
            tool="render_plotly_json",
            args={"figure": {"version": 2}},
            inputs=["Bx"],
            outputs=[],
        )
        pipeline = log.get_pipeline({"Bx"})
        renders = [r for r in pipeline if r["tool"] == "render_plotly_json"]
        assert len(renders) == 2
        assert renders[0]["args"]["figure"]["version"] == 1
        assert renders[1]["args"]["figure"]["version"] == 2

    def test_contributes_to_multiple_renders(self):
        """Shared data contributes to multiple render products."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(tool="fetch_data", args={}, outputs=["By"])
        log.record(
            tool="render_plotly_json",
            args={"figure": {"plot": "mag"}},
            inputs=["Bx"],
            outputs=[],
        )
        log.record(
            tool="render_plotly_json",
            args={"figure": {"plot": "both"}},
            inputs=["Bx", "By"],
            outputs=[],
        )
        pipeline = log.get_pipeline({"Bx", "By"})
        renders = [r for r in pipeline if r["tool"] == "render_plotly_json"]
        assert len(renders) == 2
        # Bx fetch contributes to both renders
        bx_fetch = [r for r in pipeline if r["tool"] == "fetch_data" and "Bx" in r["outputs"]][0]
        assert len(bx_fetch["contributes_to"]) == 2
        # By fetch contributes only to the second render
        by_fetch = [r for r in pipeline if r["tool"] == "fetch_data" and "By" in r["outputs"]][0]
        assert len(by_fetch["contributes_to"]) == 1

    def test_empty_labels_returns_empty(self):
        """Empty final_labels produces an empty pipeline."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        assert log.get_pipeline(set()) == []

    def test_empty_log_returns_empty(self):
        """Empty log produces an empty pipeline regardless of labels."""
        log = OperationsLog()
        assert log.get_pipeline({"Bx"}) == []

    def test_manage_plot_reset_excluded(self):
        """manage_plot with action=reset is excluded from pipeline."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="manage_plot",
            args={"action": "reset"},
            inputs=[],
            outputs=[],
        )
        log.record(
            tool="render_plotly_json",
            args={"figure": {}},
            inputs=["Bx"],
            outputs=[],
        )
        pipeline = log.get_pipeline({"Bx"})
        tools = [r["tool"] for r in pipeline]
        assert "manage_plot" not in tools

    def test_render_inputs_resolved_transitively(self):
        """render_plotly_json inputs trigger transitive resolution."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(
            tool="custom_operation", args={}, inputs=["A"], outputs=["B"]
        )
        # Final labels don't include A or B, but render needs B
        log.record(
            tool="render_plotly_json",
            args={"figure": {}},
            inputs=["B"],
            outputs=[],
        )
        # Only ask for labels that aren't produced by render
        pipeline = log.get_pipeline({"A"})
        # Should include fetch(A), compute(B), render — because render references B
        tools = [r["tool"] for r in pipeline]
        assert "fetch_data" in tools
        assert "custom_operation" in tools
        assert "render_plotly_json" in tools

    def test_dedup_does_not_shadow_real_fetch(self):
        """A dedup fetch after the real fetch must not drop the real one."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={"dataset_id": "X"}, outputs=["Bx"])
        log.record(
            tool="fetch_data",
            args={"dataset_id": "X", "already_loaded": True},
            outputs=["Bx"],
        )
        log.record(
            tool="custom_operation",
            args={"code": "mag"},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        tools = [r["tool"] for r in pipeline]
        assert tools == ["fetch_data", "custom_operation"]
        # The fetch in the pipeline is the real one, not the dedup
        assert pipeline[0]["args"].get("already_loaded") is None

    def test_chronological_order_preserved(self):
        """Pipeline records are in the same order as the original log."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(tool="fetch_data", args={}, outputs=["B"])
        log.record(
            tool="custom_operation", args={}, inputs=["A", "B"], outputs=["C"]
        )
        pipeline = log.get_pipeline({"A", "B", "C"})
        ids = [r["id"] for r in pipeline]
        assert ids == sorted(ids)

    def test_contributes_to_basic_chain(self):
        """All ops in fetch → compute → render contribute to the render product."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="custom_operation", args={}, inputs=["Bx"], outputs=["Bmag"]
        )
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bmag"], outputs=[]
        )
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        # All three should contribute to op_003 (the render)
        for rec in pipeline:
            assert rec["contributes_to"] == ["op_003"]

    def test_contributes_to_orphan(self):
        """Ops not feeding the render have empty contributes_to."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])  # orphan
        log.record(tool="fetch_data", args={}, outputs=["B"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["B"], outputs=[]
        )
        pipeline = log.get_pipeline({"A", "B"})
        by_id = {r["id"]: r for r in pipeline}
        assert by_id["op_001"]["contributes_to"] == []   # orphan
        assert by_id["op_002"]["contributes_to"] == ["op_003"]
        assert by_id["op_003"]["contributes_to"] == ["op_003"]

    def test_contributes_to_transitive(self):
        """Transitive deps of render are all marked as contributing."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(
            tool="custom_operation", args={}, inputs=["A"], outputs=["B"]
        )
        log.record(
            tool="custom_operation", args={}, inputs=["B"], outputs=["C"]
        )
        log.record(
            tool="render_plotly_json", args={}, inputs=["C"], outputs=[]
        )
        pipeline = log.get_pipeline({"A", "B", "C"})
        # All should contribute to op_004
        for rec in pipeline:
            assert rec["contributes_to"] == ["op_004"]

    def test_contributes_to_no_render(self):
        """Without render, all ops have empty contributes_to."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(
            tool="custom_operation", args={}, inputs=["A"], outputs=["B"]
        )
        pipeline = log.get_pipeline({"A", "B"})
        for rec in pipeline:
            assert rec["contributes_to"] == []

    def test_contributes_to_not_in_original_record(self):
        """contributes_to is added to pipeline output, not mutated into the log."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["A"], outputs=[]
        )
        pipeline = log.get_pipeline({"A"})
        assert "contributes_to" in pipeline[0]
        # Original records should NOT have it
        original = log.get_records()
        assert "contributes_to" not in original[0]


class TestGetPipelineMermaid:
    """Tests for OperationsLog.get_pipeline_mermaid()."""

    def test_basic_flowchart(self):
        """fetch → compute → render produces valid Mermaid with edges."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="custom_operation", args={}, inputs=["Bx"], outputs=["Bmag"]
        )
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bmag"], outputs=[]
        )
        mermaid = log.get_pipeline_mermaid({"Bx", "Bmag"})
        assert mermaid.startswith("graph TD")
        # Nodes present
        assert 'op_001["fetch\\nBx"]' in mermaid
        assert 'op_002["compute\\nBmag"]' in mermaid
        assert 'op_003["plot"]' in mermaid
        # Edges present
        assert "op_001 -->|Bx| op_002" in mermaid
        assert "op_002 -->|Bmag| op_003" in mermaid

    def test_empty_pipeline_returns_empty_string(self):
        log = OperationsLog()
        assert log.get_pipeline_mermaid(set()) == ""

    def test_multiple_outputs(self):
        """A record with multiple outputs shows them comma-separated."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx", "By", "Bz"])
        mermaid = log.get_pipeline_mermaid({"Bx", "By", "Bz"})
        assert "Bx, By, Bz" in mermaid

    def test_multiple_inputs(self):
        """A record with multiple inputs gets an edge from each producer."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(tool="fetch_data", args={}, outputs=["B"])
        log.record(
            tool="custom_operation", args={}, inputs=["A", "B"], outputs=["C"]
        )
        mermaid = log.get_pipeline_mermaid({"A", "B", "C"})
        assert "op_001 -->|A| op_003" in mermaid
        assert "op_002 -->|B| op_003" in mermaid


class TestInputProducers:
    """Tests for input_producers snapshot and product families."""

    def test_input_producers_snapshot(self):
        """input_producers is auto-populated on record()."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        rec = log.record(
            tool="custom_operation", args={}, inputs=["Bx"], outputs=["Bmag"]
        )
        assert rec["input_producers"] == {"Bx": "op_001"}

    def test_input_producers_tracks_overwrite(self):
        """Re-fetching a label updates the producer snapshot for later ops."""
        log = OperationsLog()
        # First fetch of Bx
        log.record(tool="fetch_data", args={"range": "jan"}, outputs=["Bx"])
        # First render sees op_001
        r1 = log.record(
            tool="render_plotly_json", args={"figure_json": {}},
            inputs=["Bx"], outputs=[],
        )
        assert r1["input_producers"] == {"Bx": "op_001"}

        # Re-fetch Bx (different time range)
        log.record(tool="fetch_data", args={"range": "feb"}, outputs=["Bx"])
        # Second render sees op_003 (the new fetch)
        r2 = log.record(
            tool="render_plotly_json", args={"figure_json": {}},
            inputs=["Bx"], outputs=[],
        )
        assert r2["input_producers"] == {"Bx": "op_003"}

    def test_product_family_same_inputs(self):
        """Two renders with same input set get the same product_family."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bx"], outputs=[]
        )
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bx"], outputs=[]
        )
        pipeline = log.get_pipeline({"Bx"})
        renders = [r for r in pipeline if r["tool"] == "render_plotly_json"]
        assert len(renders) == 2
        assert renders[0]["product_family"] == renders[1]["product_family"]

    def test_product_family_different_inputs(self):
        """Two renders with different input sets get different families."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(tool="fetch_data", args={}, outputs=["By"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bx"], outputs=[]
        )
        log.record(
            tool="render_plotly_json", args={}, inputs=["By"], outputs=[]
        )
        pipeline = log.get_pipeline({"Bx", "By"})
        renders = [r for r in pipeline if r["tool"] == "render_plotly_json"]
        assert len(renders) == 2
        assert renders[0]["product_family"] != renders[1]["product_family"]

    def test_product_family_state_index(self):
        """state_index and state_count annotations are correct."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bx"], outputs=[]
        )
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bx"], outputs=[]
        )
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bx"], outputs=[]
        )
        pipeline = log.get_pipeline({"Bx"})
        renders = [r for r in pipeline if r["tool"] == "render_plotly_json"]
        assert len(renders) == 3
        for i, r in enumerate(renders):
            assert r["state_index"] == i
            assert r["state_count"] == 3
            assert r["product_family"] == "op_002"  # first render's ID

    def test_load_from_records_rebuilds_producers(self):
        """load_from_records rebuilds _current_producers correctly."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(tool="fetch_data", args={}, outputs=["By"])
        records = log.get_records()

        log2 = OperationsLog()
        log2.load_from_records(records)
        assert log2._current_producers == {"Bx": "op_001", "By": "op_002"}

        # New record should see the rebuilt producers
        rec = log2.record(
            tool="render_plotly_json", args={}, inputs=["Bx", "By"], outputs=[]
        )
        assert rec["input_producers"] == {"Bx": "op_001", "By": "op_002"}

    def test_backward_compat_no_input_producers(self):
        """Records without input_producers still work via last_producer fallback."""
        log = OperationsLog()
        # Simulate old-format records loaded from file (no input_producers key)
        old_records = [
            {
                "id": "op_001", "tool": "fetch_data", "args": {},
                "outputs": ["Bx"], "inputs": [], "status": "success",
                "error": None, "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": "op_002", "tool": "render_plotly_json", "args": {},
                "outputs": [], "inputs": ["Bx"], "status": "success",
                "error": None, "timestamp": "2026-01-01T00:01:00+00:00",
            },
        ]
        log.load_from_records(old_records)
        pipeline = log.get_pipeline({"Bx"})
        # Should still resolve: fetch → render
        tools = [r["tool"] for r in pipeline]
        assert tools == ["fetch_data", "render_plotly_json"]
        # fetch should contribute to the render
        assert pipeline[0]["contributes_to"] == ["op_002"]

    def test_mermaid_collapsed_families(self):
        """Mermaid output collapses multi-state families into one node."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bx"], outputs=[]
        )
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bx"], outputs=[]
        )
        mermaid = log.get_pipeline_mermaid({"Bx"})
        # Should have a collapsed node with "2 states"
        assert "2 states" in mermaid
        # Should NOT have separate nodes for op_002 and op_004
        # The family representative is op_002 (first render)
        assert 'op_002["plot (2 states)"]' in mermaid
        # op_004 should not appear as a separate node
        lines = mermaid.split("\n")
        op_004_nodes = [l for l in lines if l.strip().startswith("op_004[")]
        assert len(op_004_nodes) == 0


class TestGetStatePipeline:
    """Tests for OperationsLog.get_state_pipeline()."""

    def test_single_render(self):
        """Basic chain: fetch → compute → render returns all three."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="custom_operation", args={}, inputs=["Bx"], outputs=["Bmag"]
        )
        log.record(
            tool="render_plotly_json", args={"figure_json": {}},
            inputs=["Bmag"], outputs=[],
        )
        pipeline = log.get_state_pipeline("op_003", {"Bx", "Bmag"})
        ids = [r["id"] for r in pipeline]
        assert ids == ["op_001", "op_002", "op_003"]
        # All should contribute to op_003
        for rec in pipeline:
            assert rec["contributes_to"] == ["op_003"]

    def test_isolates_overwrite(self):
        """Two renders of the same label each get their own upstream fetch."""
        log = OperationsLog()
        # State 1: fetch Bx (jan) → render
        log.record(tool="fetch_data", args={"range": "jan"}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json", args={"figure_json": {}},
            inputs=["Bx"], outputs=[],
        )
        # State 2: fetch Bx (feb) → render
        log.record(tool="fetch_data", args={"range": "feb"}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json", args={"figure_json": {}},
            inputs=["Bx"], outputs=[],
        )

        # State 1 pipeline: only op_001 + op_002
        state1 = log.get_state_pipeline("op_002", {"Bx"})
        assert [r["id"] for r in state1] == ["op_001", "op_002"]

        # State 2 pipeline: only op_003 + op_004
        state2 = log.get_state_pipeline("op_004", {"Bx"})
        assert [r["id"] for r in state2] == ["op_003", "op_004"]

    def test_unknown_op_id(self):
        """Returns [] for a non-existent op ID."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        assert log.get_state_pipeline("op_999", {"Bx"}) == []

    def test_non_render_op_id(self):
        """Returns [] when op_id refers to a non-render operation."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        assert log.get_state_pipeline("op_001", {"Bx"}) == []


class TestSingleton:
    """Tests for the module-level singleton helpers."""

    def test_get_operations_log_returns_same_instance(self):
        log1 = get_operations_log()
        log2 = get_operations_log()
        assert log1 is log2

    def test_reset_creates_new_instance(self):
        log1 = get_operations_log()
        log1.record(tool="fetch_data", args={}, outputs=["a"])
        reset_operations_log()
        log2 = get_operations_log()
        assert log2 is not log1
        assert len(log2) == 0


class TestScopedIds:
    """Tests for session-scoped operation IDs."""

    def test_scoped_id_generation(self):
        """OperationsLog(session_id='sess_abc') produces 'sess_abc:op_001'."""
        log = OperationsLog(session_id="sess_abc")
        rec = log.record(tool="fetch_data", args={}, outputs=["a"])
        assert rec["id"] == "sess_abc:op_001"

    def test_scoped_id_auto_increments(self):
        log = OperationsLog(session_id="s1")
        r1 = log.record(tool="fetch_data", args={}, outputs=["a"])
        r2 = log.record(tool="fetch_data", args={}, outputs=["b"])
        r3 = log.record(tool="fetch_data", args={}, outputs=["c"])
        assert r1["id"] == "s1:op_001"
        assert r2["id"] == "s1:op_002"
        assert r3["id"] == "s1:op_003"

    def test_no_session_id_backward_compat(self):
        """No session_id → still produces 'op_001' (backward compat)."""
        log = OperationsLog()
        rec = log.record(tool="fetch_data", args={}, outputs=["a"])
        assert rec["id"] == "op_001"

    def test_empty_session_id_backward_compat(self):
        """Empty string session_id → plain 'op_001'."""
        log = OperationsLog(session_id="")
        rec = log.record(tool="fetch_data", args={}, outputs=["a"])
        assert rec["id"] == "op_001"

    def test_max_counter_with_scoped_ids(self):
        """_max_counter_from_records handles scoped IDs."""
        records = [
            {"id": "sess_abc:op_005"},
            {"id": "sess_abc:op_010"},
            {"id": "op_003"},
        ]
        assert OperationsLog._max_counter_from_records(records) == 10

    def test_load_from_records_migrates_old_ids(self):
        """load_from_records with session_id migrates old-style IDs."""
        old_records = [
            {"id": "op_001", "tool": "fetch_data", "args": {}, "outputs": ["Bx"],
             "inputs": [], "status": "success", "error": None,
             "timestamp": "2026-01-01T00:00:00+00:00", "input_producers": {}},
            {"id": "op_002", "tool": "render_plotly_json", "args": {}, "outputs": [],
             "inputs": ["Bx"], "status": "success", "error": None,
             "timestamp": "2026-01-01T00:01:00+00:00",
             "input_producers": {"Bx": "op_001"}},
        ]
        log = OperationsLog(session_id="sess_xyz")
        log.load_from_records(old_records)
        records = log.get_records()

        # IDs should be migrated
        assert records[0]["id"] == "sess_xyz:op_001"
        assert records[1]["id"] == "sess_xyz:op_002"

        # input_producers should also be migrated
        assert records[1]["input_producers"] == {"Bx": "sess_xyz:op_001"}

    def test_load_from_records_no_double_migration(self):
        """Already-scoped IDs are not double-prefixed."""
        records = [
            {"id": "sess_xyz:op_001", "tool": "fetch_data", "args": {}, "outputs": ["a"],
             "inputs": [], "status": "success", "error": None,
             "timestamp": "2026-01-01T00:00:00+00:00", "input_producers": {}},
        ]
        log = OperationsLog(session_id="sess_xyz")
        log.load_from_records(records)
        assert log.get_records()[0]["id"] == "sess_xyz:op_001"

    def test_load_from_records_no_migration_without_session_id(self):
        """Without session_id, IDs are left as-is."""
        records = [
            {"id": "op_001", "tool": "fetch_data", "args": {}, "outputs": ["a"],
             "inputs": [], "status": "success", "error": None,
             "timestamp": "2026-01-01T00:00:00+00:00"},
        ]
        log = OperationsLog()
        log.load_from_records(records)
        assert log.get_records()[0]["id"] == "op_001"

    def test_counter_resumes_after_load_scoped(self):
        """Counter resumes correctly after loading scoped records."""
        records = [
            {"id": "sess_abc:op_003", "tool": "fetch_data", "args": {}, "outputs": ["a"],
             "inputs": [], "status": "success", "error": None,
             "timestamp": "2026-01-01T00:00:00+00:00"},
        ]
        log = OperationsLog(session_id="sess_abc")
        log.load_from_records(records)
        rec = log.record(tool="fetch_data", args={}, outputs=["b"])
        assert rec["id"] == "sess_abc:op_004"

    def test_current_producers_rebuilt_with_scoped_ids(self):
        """_current_producers uses scoped IDs after migration."""
        records = [
            {"id": "op_001", "tool": "fetch_data", "args": {}, "outputs": ["Bx"],
             "inputs": [], "status": "success", "error": None,
             "timestamp": "2026-01-01T00:00:00+00:00"},
        ]
        log = OperationsLog(session_id="s1")
        log.load_from_records(records)
        assert log._current_producers == {"Bx": "s1:op_001"}

    def test_json_roundtrip_scoped(self, tmp_path):
        """Scoped IDs survive save/load roundtrip."""
        log = OperationsLog(session_id="s1")
        log.record(tool="fetch_data", args={}, outputs=["a"])
        log.record(tool="custom_operation", args={}, inputs=["a"], outputs=["b"])

        path = tmp_path / "operations.json"
        log.save_to_file(path)

        log2 = OperationsLog(session_id="s1")
        log2.load_from_file(path)
        records = log2.get_records()
        assert records[0]["id"] == "s1:op_001"
        assert records[1]["id"] == "s1:op_002"

    def test_get_pipeline_with_scoped_ids(self):
        """get_pipeline works with scoped IDs."""
        log = OperationsLog(session_id="s1")
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(tool="custom_operation", args={}, inputs=["Bx"], outputs=["Bmag"])
        log.record(tool="render_plotly_json", args={}, inputs=["Bmag"], outputs=[])
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        tools = [r["tool"] for r in pipeline]
        assert tools == ["fetch_data", "custom_operation", "render_plotly_json"]
        assert pipeline[0]["id"] == "s1:op_001"
        assert pipeline[2]["contributes_to"] == ["s1:op_003"]

    def test_get_state_pipeline_with_scoped_ids(self):
        """get_state_pipeline works with scoped IDs."""
        log = OperationsLog(session_id="s1")
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(tool="render_plotly_json", args={}, inputs=["Bx"], outputs=[])
        pipeline = log.get_state_pipeline("s1:op_002", {"Bx"})
        assert len(pipeline) == 2
        assert pipeline[0]["id"] == "s1:op_001"
        assert pipeline[1]["id"] == "s1:op_002"


class TestPipelineStatus:
    """Tests for pipeline_status tracking on render ops."""

    def test_set_pipeline_status(self):
        """set_pipeline_status updates the record."""
        log = OperationsLog()
        log.record(tool="render_plotly_json", args={}, inputs=[], outputs=[])
        assert log.set_pipeline_status("op_001", "registered") is True
        rec = log.get_records()[0]
        assert rec["pipeline_status"] == "registered"

    def test_set_pipeline_status_not_found(self):
        """set_pipeline_status returns False for unknown op_id."""
        log = OperationsLog()
        assert log.set_pipeline_status("op_999", "registered") is False

    def test_set_pipeline_status_scoped(self):
        """set_pipeline_status works with scoped IDs."""
        log = OperationsLog(session_id="s1")
        log.record(tool="render_plotly_json", args={}, inputs=[], outputs=[])
        assert log.set_pipeline_status("s1:op_001", "discarded") is True
        rec = log.get_records()[0]
        assert rec["pipeline_status"] == "discarded"

    def test_get_render_ops_by_status_fresh(self):
        """Records without pipeline_status default to 'fresh'."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["a"])
        log.record(tool="render_plotly_json", args={}, inputs=["a"], outputs=[])
        log.record(tool="render_plotly_json", args={}, inputs=["a"], outputs=[],
                   status="error", error="bad")

        fresh = log.get_render_ops_by_status("fresh")
        assert len(fresh) == 1
        assert fresh[0]["id"] == "op_002"

    def test_get_render_ops_by_status_registered(self):
        """After marking registered, op appears in registered filter."""
        log = OperationsLog()
        log.record(tool="render_plotly_json", args={}, inputs=[], outputs=[])
        log.record(tool="render_plotly_json", args={}, inputs=[], outputs=[])
        log.set_pipeline_status("op_001", "registered")

        fresh = log.get_render_ops_by_status("fresh")
        registered = log.get_render_ops_by_status("registered")
        assert len(fresh) == 1
        assert fresh[0]["id"] == "op_002"
        assert len(registered) == 1
        assert registered[0]["id"] == "op_001"

    def test_get_render_ops_by_status_discarded(self):
        """Discarded ops appear in discarded filter, not fresh."""
        log = OperationsLog()
        log.record(tool="render_plotly_json", args={}, inputs=[], outputs=[])
        log.set_pipeline_status("op_001", "discarded")

        fresh = log.get_render_ops_by_status("fresh")
        discarded = log.get_render_ops_by_status("discarded")
        assert len(fresh) == 0
        assert len(discarded) == 1

    def test_pipeline_status_survives_roundtrip(self, tmp_path):
        """pipeline_status is persisted through save/load."""
        log = OperationsLog(session_id="s1")
        log.record(tool="render_plotly_json", args={}, inputs=[], outputs=[])
        log.set_pipeline_status("s1:op_001", "registered")

        path = tmp_path / "operations.json"
        log.save_to_file(path)

        log2 = OperationsLog(session_id="s1")
        log2.load_from_file(path)
        registered = log2.get_render_ops_by_status("registered")
        assert len(registered) == 1
        assert registered[0]["id"] == "s1:op_001"

    def test_pipeline_status_not_on_non_render_ops(self):
        """get_render_ops_by_status only returns render_plotly_json ops."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["a"])
        log.record(tool="render_plotly_json", args={}, inputs=["a"], outputs=[])

        fresh = log.get_render_ops_by_status("fresh")
        assert len(fresh) == 1
        assert fresh[0]["tool"] == "render_plotly_json"
