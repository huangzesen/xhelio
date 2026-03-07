"""Tests for data_ops.pipeline — Pipeline DAG with staleness and mutation."""

import pytest
import numpy as np
import pandas as pd

from data_ops.pipeline import (
    Pipeline,
    PipelineNode,
    PipelineEdge,
    NodeState,
    NodeType,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic operation records
# ---------------------------------------------------------------------------

def _fetch_record(op_id, dataset_id, parameter_id, time_range, output_label):
    """Create a synthetic fetch_data record."""
    return {
        "id": op_id,
        "tool": "fetch_data",
        "status": "success",
        "inputs": [],
        "outputs": [output_label],
        "args": {
            "dataset_id": dataset_id,
            "parameter_id": parameter_id,
            "time_range": time_range,
            "time_range_resolved": time_range,
        },
        "input_producers": {},
    }


def _compute_record(op_id, code, input_labels, output_label, input_producers=None):
    """Create a synthetic custom_operation record."""
    if input_producers is None:
        input_producers = {}
    return {
        "id": op_id,
        "tool": "custom_operation",
        "status": "success",
        "inputs": input_labels,
        "outputs": [output_label],
        "args": {
            "code": code,
            "output_label": output_label,
            "description": "test compute",
        },
        "input_producers": input_producers,
    }


def _render_record(op_id, input_labels, input_producers=None):
    """Create a synthetic render_plotly_json record."""
    if input_producers is None:
        input_producers = {}
    traces = [{"data_label": lbl, "type": "scatter"} for lbl in input_labels]
    return {
        "id": op_id,
        "tool": "render_plotly_json",
        "status": "success",
        "inputs": input_labels,
        "outputs": [],
        "args": {
            "figure_json": {"data": traces, "layout": {"title": {"text": "Test"}}},
        },
        "input_producers": input_producers,
    }


def _error_record(op_id, tool="fetch_data"):
    """Create a failed record."""
    return {
        "id": op_id,
        "tool": tool,
        "status": "error",
        "inputs": [],
        "outputs": ["bad_label"],
        "args": {},
        "error": "test error",
        "input_producers": {},
    }


def _simple_pipeline_records():
    """A linear pipeline: fetch → compute → render.

    fetch(op_001) → "A.field"
    compute(op_002) "A.field" → "A.magnitude"
    render(op_003) "A.magnitude" → (plot)
    """
    return [
        _fetch_record("op_001", "DS1", "P1", "2024-01-01 to 2024-01-31", "A.field"),
        _compute_record(
            "op_002", "result = df_field.apply(np.linalg.norm, axis=1)",
            ["A.field"], "A.magnitude",
            input_producers={"A.field": "op_001"},
        ),
        _render_record(
            "op_003", ["A.magnitude"],
            input_producers={"A.magnitude": "op_002"},
        ),
    ]


def _diamond_pipeline_records():
    """A diamond DAG: two fetches → compute (merging both) → render.

    fetch(op_001) → "A.bx"
    fetch(op_002) → "B.by"
    compute(op_003) "A.bx", "B.by" → "merged.ratio"
    render(op_004) "merged.ratio" → (plot)
    """
    return [
        _fetch_record("op_001", "DS1", "BX", "2024-01-01 to 2024-01-31", "A.bx"),
        _fetch_record("op_002", "DS2", "BY", "2024-01-01 to 2024-01-31", "B.by"),
        _compute_record(
            "op_003", "result = df_bx / df_by",
            ["A.bx", "B.by"], "merged.ratio",
            input_producers={"A.bx": "op_001", "B.by": "op_002"},
        ),
        _render_record(
            "op_004", ["merged.ratio"],
            input_producers={"merged.ratio": "op_003"},
        ),
    ]


def _multi_branch_records():
    """Branching DAG: one fetch feeds two computes, each rendered.

    fetch(op_001) → "A.field"
    compute(op_002) "A.field" → "A.magnitude"
    compute(op_003) "A.field" → "A.smoothed"
    render(op_004) "A.magnitude", "A.smoothed" → (plot)
    """
    return [
        _fetch_record("op_001", "DS1", "P1", "2024-01-01 to 2024-01-31", "A.field"),
        _compute_record(
            "op_002", "result = df_field.apply(np.linalg.norm, axis=1)",
            ["A.field"], "A.magnitude",
            input_producers={"A.field": "op_001"},
        ),
        _compute_record(
            "op_003", "result = df_field.rolling(10).mean()",
            ["A.field"], "A.smoothed",
            input_producers={"A.field": "op_001"},
        ),
        _render_record(
            "op_004", ["A.magnitude", "A.smoothed"],
            input_producers={"A.magnitude": "op_002", "A.smoothed": "op_003"},
        ),
    ]


# ===========================================================================
# Test group: Construction
# ===========================================================================

class TestConstruction:
    def test_from_records_simple(self):
        records = _simple_pipeline_records()
        pipe = Pipeline.from_records(records)
        assert len(pipe) == 3
        assert "op_001" in pipe
        assert "op_002" in pipe
        assert "op_003" in pipe

    def test_from_records_diamond(self):
        records = _diamond_pipeline_records()
        pipe = Pipeline.from_records(records)
        assert len(pipe) == 4

    def test_from_records_skips_errors(self):
        records = _simple_pipeline_records() + [_error_record("op_099")]
        pipe = Pipeline.from_records(records)
        assert "op_099" not in pipe
        assert len(pipe) == 3

    def test_from_records_skips_dedup_fetch(self):
        """Fetch records with already_loaded=True should be skipped."""
        records = _simple_pipeline_records()
        dedup = {
            "id": "op_098",
            "tool": "fetch_data",
            "status": "success",
            "inputs": [],
            "outputs": ["A.field"],
            "args": {"dataset_id": "DS1", "parameter_id": "P1", "already_loaded": True},
            "input_producers": {},
        }
        records.append(dedup)
        pipe = Pipeline.from_records(records)
        assert "op_098" not in pipe

    def test_from_records_skips_unknown_tools(self):
        """manage_plot and other unknown tools should be skipped."""
        records = _simple_pipeline_records()
        manage = {
            "id": "op_098",
            "tool": "manage_plot",
            "status": "success",
            "inputs": [],
            "outputs": [],
            "args": {"action": "reset"},
            "input_producers": {},
        }
        records.append(manage)
        pipe = Pipeline.from_records(records)
        assert "op_098" not in pipe

    def test_edge_inference(self):
        """Edges should be inferred from input_producers."""
        records = _simple_pipeline_records()
        pipe = Pipeline.from_records(records)
        edge_pairs = {(e.source_id, e.target_id) for e in pipe._edges}
        assert ("op_001", "op_002") in edge_pairs
        assert ("op_002", "op_003") in edge_pairs

    def test_edge_inference_diamond(self):
        records = _diamond_pipeline_records()
        pipe = Pipeline.from_records(records)
        edge_pairs = {(e.source_id, e.target_id) for e in pipe._edges}
        assert ("op_001", "op_003") in edge_pairs
        assert ("op_002", "op_003") in edge_pairs
        assert ("op_003", "op_004") in edge_pairs

    def test_node_types(self):
        records = _simple_pipeline_records()
        pipe = Pipeline.from_records(records)
        assert pipe.get_node("op_001").node_type == NodeType.FETCH
        assert pipe.get_node("op_002").node_type == NodeType.COMPUTE
        assert pipe.get_node("op_003").node_type == NodeType.RENDER

    def test_all_nodes_clean(self):
        records = _simple_pipeline_records()
        pipe = Pipeline.from_records(records)
        for nid in pipe._nodes:
            assert pipe._nodes[nid].state == NodeState.CLEAN

    def test_from_operations_log(self):
        """Test construction via from_operations_log."""
        from data_ops.operations_log import OperationsLog

        log = OperationsLog()
        log.record(
            tool="fetch_data",
            args={"dataset_id": "DS1", "parameter_id": "P1", "time_range": "2024-01-01 to 2024-01-31"},
            outputs=["A.field"],
        )
        log.record(
            tool="render_plotly_json",
            args={"figure_json": {"data": [{"data_label": "A.field"}], "layout": {}}},
            outputs=[],
            inputs=["A.field"],
        )

        pipe = Pipeline.from_operations_log(log, final_labels={"A.field"})
        assert len(pipe) == 2


# ===========================================================================
# Test group: Topology
# ===========================================================================

class TestTopology:
    def test_topological_order_simple(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        order = pipe.topological_order()
        assert order.index("op_001") < order.index("op_002")
        assert order.index("op_002") < order.index("op_003")

    def test_topological_order_diamond(self):
        pipe = Pipeline.from_records(_diamond_pipeline_records())
        order = pipe.topological_order()
        # Both fetches before compute, compute before render
        assert order.index("op_001") < order.index("op_003")
        assert order.index("op_002") < order.index("op_003")
        assert order.index("op_003") < order.index("op_004")

    def test_children(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        assert pipe.children("op_001") == {"op_002"}
        assert pipe.children("op_002") == {"op_003"}
        assert pipe.children("op_003") == set()

    def test_parents(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        assert pipe.parents("op_001") == set()
        assert pipe.parents("op_002") == {"op_001"}
        assert pipe.parents("op_003") == {"op_002"}

    def test_parents_diamond(self):
        pipe = Pipeline.from_records(_diamond_pipeline_records())
        assert pipe.parents("op_003") == {"op_001", "op_002"}

    def test_descendants(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        assert pipe.descendants("op_001") == {"op_002", "op_003"}
        assert pipe.descendants("op_002") == {"op_003"}
        assert pipe.descendants("op_003") == set()

    def test_ancestors(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        assert pipe.ancestors("op_001") == set()
        assert pipe.ancestors("op_002") == {"op_001"}
        assert pipe.ancestors("op_003") == {"op_001", "op_002"}

    def test_descendants_diamond(self):
        pipe = Pipeline.from_records(_diamond_pipeline_records())
        assert pipe.descendants("op_001") == {"op_003", "op_004"}

    def test_ancestors_diamond(self):
        pipe = Pipeline.from_records(_diamond_pipeline_records())
        assert pipe.ancestors("op_004") == {"op_001", "op_002", "op_003"}

    def test_children_multi_branch(self):
        pipe = Pipeline.from_records(_multi_branch_records())
        children = pipe.children("op_001")
        assert children == {"op_002", "op_003"}


# ===========================================================================
# Test group: Staleness
# ===========================================================================

class TestStaleness:
    def test_propagate_from_root(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        affected = pipe.propagate_staleness("op_001")
        assert affected == {"op_001", "op_002", "op_003"}
        assert pipe.get_node("op_001").state == NodeState.STALE
        assert pipe.get_node("op_002").state == NodeState.STALE
        assert pipe.get_node("op_003").state == NodeState.STALE

    def test_propagate_from_middle(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        affected = pipe.propagate_staleness("op_002")
        assert affected == {"op_002", "op_003"}
        assert pipe.get_node("op_001").state == NodeState.CLEAN

    def test_propagate_leaf(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        affected = pipe.propagate_staleness("op_003")
        assert affected == {"op_003"}

    def test_propagate_diamond(self):
        """In a diamond, staleness from one branch affects the merge point."""
        pipe = Pipeline.from_records(_diamond_pipeline_records())
        affected = pipe.propagate_staleness("op_001")
        assert "op_003" in affected  # merge node
        assert "op_004" in affected  # render
        assert "op_002" not in affected  # other branch

    def test_get_stale_nodes_order(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        pipe.propagate_staleness("op_001")
        stale = pipe.get_stale_nodes()
        assert stale == ["op_001", "op_002", "op_003"]

    def test_get_stale_nodes_empty(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        assert pipe.get_stale_nodes() == []

    def test_propagate_nonexistent(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        affected = pipe.propagate_staleness("op_999")
        assert affected == set()

    def test_pending_in_stale_nodes(self):
        """PENDING nodes should appear in get_stale_nodes()."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        pipe._nodes["op_001"].state = NodeState.PENDING
        stale = pipe.get_stale_nodes()
        assert "op_001" in stale


# ===========================================================================
# Test group: Mutation
# ===========================================================================

class TestMutation:
    def test_update_params(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        affected = pipe.update_node_params("op_001", {
            "time_range": "2024-02-01 to 2024-02-28",
            "time_range_resolved": "2024-02-01 to 2024-02-28",
        })
        node = pipe.get_node("op_001")
        assert node.params["time_range"] == "2024-02-01 to 2024-02-28"
        assert "op_001" in affected
        assert "op_002" in affected
        assert "op_003" in affected

    def test_update_params_nonexistent(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        with pytest.raises(KeyError):
            pipe.update_node_params("op_999", {"foo": "bar"})

    def test_remove_node(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        result = pipe.remove_node("op_002")
        assert result["removed"] == "op_002"
        assert "A.magnitude" in result["orphaned_labels"]
        assert "op_002" not in pipe
        assert len(pipe) == 2

    def test_remove_node_nonexistent(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        with pytest.raises(KeyError):
            pipe.remove_node("op_999")

    def test_remove_node_cleans_edges(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        pipe.remove_node("op_002")
        edge_ids = {(e.source_id, e.target_id) for e in pipe._edges}
        assert all("op_002" not in pair for pair in edge_ids)

    def test_insert_node(self):
        """Insert a smoothing step between compute and render."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        new_id = pipe.insert_node(
            after_id="op_002",
            tool="custom_operation",
            params={"code": "result = df_magnitude.rolling(10).mean()"},
            output_label="A.smoothed",
        )
        assert new_id in pipe
        new_node = pipe.get_node(new_id)
        assert new_node.state == NodeState.PENDING
        assert new_node.inputs == ["A.magnitude"]
        assert new_node.outputs == ["A.smoothed"]

        # Render should now consume A.smoothed instead of A.magnitude
        render = pipe.get_node("op_003")
        assert "A.smoothed" in render.inputs

        # Edges should be: op_002 → new → op_003
        edge_pairs = {(e.source_id, e.target_id) for e in pipe._edges}
        assert ("op_002", new_id) in edge_pairs
        assert (new_id, "op_003") in edge_pairs

    def test_insert_node_nonexistent(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        with pytest.raises(KeyError):
            pipe.insert_node("op_999", "custom_operation", {}, "X.out")


# ===========================================================================
# Test group: Execution
# ===========================================================================

class TestExecution:
    def _make_store_with_data(self, tmp_path=None):
        """Create a DataStore with test data matching simple pipeline."""
        import tempfile
        from pathlib import Path
        from data_ops.store import DataStore, DataEntry

        data_dir = tmp_path / "data" if tmp_path else Path(tempfile.mkdtemp()) / "data"
        store = DataStore(data_dir)
        df = pd.DataFrame(
            {"Bx": [1.0, 2.0, 3.0], "By": [4.0, 5.0, 6.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )
        store.put(DataEntry(label="A.field", data=df, source="cdf"))
        return store

    def test_no_stale_nodes(self):
        """When nothing is stale, execute_stale is a no-op."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        store = self._make_store_with_data()
        result = pipe.execute_stale(store)
        assert result["executed"] == 0
        assert result["skipped"] == 0

    def test_stale_compute_executes(self):
        """Mark compute stale, verify it re-executes."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        store = self._make_store_with_data()

        # Update compute code to something simple
        pipe._nodes["op_002"].params["code"] = "result = df_field.sum(axis=1)"
        pipe._nodes["op_002"].state = NodeState.STALE

        result = pipe.execute_stale(store)
        assert result["executed"] == 1
        assert pipe.get_node("op_002").state == NodeState.CLEAN

        # Verify data was written
        entry = store.get("A.magnitude")
        assert entry is not None

    def test_error_skips_descendants(self):
        """If a node errors, its descendants should be skipped."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        store = self._make_store_with_data()

        # Set bad code that will error
        pipe._nodes["op_002"].params["code"] = "result = undefined_variable"
        pipe.propagate_staleness("op_002")

        result = pipe.execute_stale(store)
        assert len(result["errors"]) >= 1
        assert pipe.get_node("op_002").state == NodeState.ERROR

    def test_backdating_skips_unchanged(self):
        """If output hash matches, descendants should be skipped."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        store = self._make_store_with_data()

        # Execute compute to get initial data + hash
        pipe._nodes["op_002"].params["code"] = "result = df_field.sum(axis=1)"
        pipe._nodes["op_002"].state = NodeState.STALE
        pipe.execute_stale(store)

        # Now mark it stale again but with same code → same output
        pipe.propagate_staleness("op_002")
        result = pipe.execute_stale(store)
        assert result["backdated"] >= 1

    def test_progress_callback(self):
        """Progress callback should be called for each stale node."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        store = self._make_store_with_data()

        pipe._nodes["op_002"].params["code"] = "result = df_field.sum(axis=1)"
        pipe._nodes["op_002"].state = NodeState.STALE

        calls = []
        def cb(step, total, node_id, tool):
            calls.append((step, total, node_id, tool))

        pipe.execute_stale(store, progress_cb=cb)
        assert len(calls) == 1
        assert calls[0][2] == "op_002"


# ===========================================================================
# Test group: Serialization
# ===========================================================================

class TestSerialization:
    def test_roundtrip(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        d = pipe.to_dict()
        pipe2 = Pipeline.from_dict(d)

        assert len(pipe2) == len(pipe)
        assert set(pipe2._nodes.keys()) == set(pipe._nodes.keys())
        assert len(pipe2._edges) == len(pipe._edges)

    def test_roundtrip_preserves_state(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        pipe.propagate_staleness("op_001")

        d = pipe.to_dict()
        pipe2 = Pipeline.from_dict(d)

        for nid in pipe._nodes:
            assert pipe2.get_node(nid).state == pipe.get_node(nid).state

    def test_to_summary_structure(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        s = pipe.to_summary()

        assert s["node_count"] == 3
        assert s["edge_count"] == 2
        assert s["stale_count"] == 0
        assert len(s["nodes"]) == 3
        assert len(s["edges"]) == 2

        # Check node summary fields
        node = s["nodes"][0]
        assert "id" in node
        assert "type" in node
        assert "state" in node
        assert "inputs" in node
        assert "outputs" in node

    def test_to_summary_with_stale(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        pipe.propagate_staleness("op_001")
        s = pipe.to_summary()
        assert s["stale_count"] == 3
        assert s["stale_nodes"] == ["op_001", "op_002", "op_003"]

    def test_to_summary_fetch_details(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        s = pipe.to_summary()
        fetch_node = [n for n in s["nodes"] if n["type"] == "fetch"][0]
        assert fetch_node["dataset"] == "DS1"
        assert fetch_node["parameter"] == "P1"

    def test_to_summary_includes_description(self):
        """Compute nodes with a description should include it in the summary."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        s = pipe.to_summary()
        compute_node = [n for n in s["nodes"] if n["type"] == "compute"][0]
        assert compute_node["description"] == "test compute"

    def test_to_summary_render_details(self):
        pipe = Pipeline.from_records(_simple_pipeline_records())
        s = pipe.to_summary()
        render_node = [n for n in s["nodes"] if n["type"] == "render"][0]
        assert "traces" in render_node
        assert render_node["traces"] == 1

    def test_node_to_dict_from_dict_roundtrip(self):
        node = PipelineNode(
            id="op_001", node_type=NodeType.FETCH, tool="fetch_data",
            params={"dataset_id": "DS1"}, inputs=[], outputs=["A.field"],
            state=NodeState.STALE, output_hash="abc123",
            input_producers={},
        )
        d = node.to_dict()
        node2 = PipelineNode.from_dict(d)
        assert node2.id == node.id
        assert node2.state == node.state
        assert node2.output_hash == node.output_hash

    def test_edge_to_dict_from_dict_roundtrip(self):
        edge = PipelineEdge("op_001", "op_002", "A.field")
        d = edge.to_dict()
        edge2 = PipelineEdge.from_dict(d)
        assert edge2.source_id == edge.source_id
        assert edge2.target_id == edge.target_id
        assert edge2.label == edge.label


# ===========================================================================
# Test group: Node Detail
# ===========================================================================

class TestNodeDetail:
    def test_node_detail_compute(self):
        """node_detail for a compute node returns full code, description, parents, children."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        detail = pipe.node_detail("op_002")

        assert detail is not None
        assert detail["id"] == "op_002"
        assert detail["type"] == "compute"
        assert detail["tool"] == "custom_operation"
        assert detail["state"] == "clean"
        assert detail["inputs"] == ["A.field"]
        assert detail["outputs"] == ["A.magnitude"]
        # Full code should be present (not truncated)
        assert detail["code"] == "result = df_field.apply(np.linalg.norm, axis=1)"
        assert detail["description"] == "test compute"
        # Full params dict should be present
        assert "code" in detail["params"]
        assert "output_label" in detail["params"]
        # Parents and children
        assert detail["parents"] == ["op_001"]
        assert detail["children"] == ["op_003"]

    def test_node_detail_fetch(self):
        """node_detail for a fetch node returns params but no code/description fields."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        detail = pipe.node_detail("op_001")

        assert detail is not None
        assert detail["type"] == "fetch"
        assert detail["params"]["dataset_id"] == "DS1"
        assert "code" not in detail
        assert "description" not in detail
        assert detail["parents"] == []
        assert detail["children"] == ["op_002"]

    def test_node_detail_nonexistent(self):
        """node_detail for a nonexistent node returns None."""
        pipe = Pipeline.from_records(_simple_pipeline_records())
        assert pipe.node_detail("op_999") is None
