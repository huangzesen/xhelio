"""Tests for data_ops.dag — PipelineDAG."""

import pytest
from data_ops.dag import PipelineDAG


class TestPipelineDAGAddNode:
    """Test node addition and automatic edge creation."""

    def test_add_source_node(self):
        dag = PipelineDAG()
        dag.add_node(
            op_id="op_000",
            tool="fetch",
            agent="envoy:ace",
            args={"dataset": "AC_H2_MFI"},
            inputs=[],
            outputs={"AC_H2_MFI.BGSEc": "result"},
            status="success",
        )
        assert "op_000" in dag
        assert dag.node("op_000")["tool"] == "fetch"
        assert dag.node("op_000")["agent"] == "envoy:ace"

    def test_add_transform_node_creates_edge(self):
        dag = PipelineDAG()
        dag.add_node(
            op_id="op_000",
            tool="fetch",
            agent="envoy:ace",
            args={},
            inputs=[],
            outputs={"X": "result"},
            status="success",
        )
        dag.add_node(
            op_id="op_001",
            tool="run_code",
            agent="data_ops",
            args={"code": "..."},
            inputs=["X"],
            outputs={"Y": "magnitude"},
            status="success",
        )
        assert dag.predecessors("op_001") == ["op_000"]
        assert dag.successors("op_000") == ["op_001"]

    def test_add_sink_node(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        dag.add_node(op_id="op_001", tool="render_plotly_json", agent="viz",
                     args={}, inputs=["X"], outputs={}, status="success")
        assert dag.node("op_001")["inputs"] == ["X"]
        assert dag.node("op_001")["outputs"] == {}

    def test_label_ownership_updated(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        assert dag.producer_of("X") == "op_000"

    def test_label_overwrite(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        dag.add_node(op_id="op_001", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        assert dag.producer_of("X") == "op_001"

    def test_multi_output_node(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        dag.add_node(op_id="op_001", tool="run_code", agent="d", args={},
                     inputs=["X"], outputs={"Y": "mag", "Z": "angle"},
                     status="success")
        assert dag.producer_of("Y") == "op_001"
        assert dag.producer_of("Z") == "op_001"

    def test_edge_has_label_attribute(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        dag.add_node(op_id="op_001", tool="run_code", agent="d", args={},
                     inputs=["X"], outputs={"Y": "m"}, status="success")
        edge_data = dag._graph.edges["op_000", "op_001"]
        assert "X" in edge_data["labels"]

    def test_error_node_does_not_update_label_owners(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="run_code", agent="d", args={},
                     inputs=[], outputs={"X": "r"}, status="error",
                     error="failed")
        assert dag.producer_of("X") is None

    def test_unknown_input_label_no_edge(self):
        """Input label with no known producer creates no edge."""
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="run_code", agent="d", args={},
                     inputs=["nonexistent"], outputs={"Y": "r"},
                     status="success")
        assert dag.predecessors("op_000") == []

    def test_op_id_counter(self):
        dag = PipelineDAG()
        assert dag.next_op_id() == "op_000"
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        assert dag.next_op_id() == "op_001"


class TestPipelineDAGNodeKind:
    """Test node kind derivation."""

    def test_source_kind(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        assert dag.node_kind("op_000") == "source"

    def test_transform_kind(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        dag.add_node(op_id="op_001", tool="run_code", agent="d", args={},
                     inputs=["X"], outputs={"Y": "m"}, status="success")
        assert dag.node_kind("op_001") == "transform"

    def test_sink_kind(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        dag.add_node(op_id="op_001", tool="render", agent="v", args={},
                     inputs=["X"], outputs={}, status="success")
        assert dag.node_kind("op_001") == "sink"


class TestPipelineDAGGraphQueries:
    """Test traversal and query methods."""

    @pytest.fixture
    def chain_dag(self):
        """Build a 3-node chain: fetch -> transform -> render."""
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        dag.add_node(op_id="op_001", tool="run_code", agent="d", args={},
                     inputs=["X"], outputs={"Y": "m"}, status="success")
        dag.add_node(op_id="op_002", tool="render", agent="v", args={},
                     inputs=["Y"], outputs={}, status="success")
        return dag

    def test_ancestors(self, chain_dag):
        assert chain_dag.ancestors("op_002") == {"op_000", "op_001"}

    def test_descendants(self, chain_dag):
        assert chain_dag.descendants("op_000") == {"op_001", "op_002"}

    def test_roots(self, chain_dag):
        assert chain_dag.roots() == ["op_000"]

    def test_leaves(self, chain_dag):
        assert chain_dag.leaves() == ["op_002"]

    def test_path(self, chain_dag):
        assert chain_dag.path("op_000", "op_002") == [
            "op_000", "op_001", "op_002"
        ]

    def test_path_no_connection(self, chain_dag):
        assert chain_dag.path("op_002", "op_000") == []

    def test_topological_order(self, chain_dag):
        order = chain_dag.topological_order()
        assert order.index("op_000") < order.index("op_001")
        assert order.index("op_001") < order.index("op_002")

    def test_consumers_of(self, chain_dag):
        assert chain_dag.consumers_of("X") == ["op_001"]
        assert chain_dag.consumers_of("Y") == ["op_002"]

    def test_nodes_by_kind(self, chain_dag):
        assert chain_dag.nodes_by_kind("source") == ["op_000"]
        assert chain_dag.nodes_by_kind("transform") == ["op_001"]
        assert chain_dag.nodes_by_kind("sink") == ["op_002"]

    def test_subgraph(self, chain_dag):
        sub = chain_dag.subgraph("op_001")
        assert sub.node_count() == 2  # op_000 + op_001
        assert "op_000" in sub
        assert "op_001" in sub
        assert "op_002" not in sub
        assert sub._session_dir is None  # no persistence

    def test_subgraph_preserves_edges(self, chain_dag):
        sub = chain_dag.subgraph("op_002")
        assert sub.predecessors("op_002") == ["op_001"]
        assert sub.predecessors("op_001") == ["op_000"]


class TestPipelineDAGDiamondGraph:
    """Test DAG with diamond shape: A -> B, A -> C, B -> D, C -> D."""

    @pytest.fixture
    def diamond_dag(self):
        dag = PipelineDAG()
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        dag.add_node(op_id="op_001", tool="run_code", agent="d", args={},
                     inputs=["X"], outputs={"Y": "m"}, status="success")
        dag.add_node(op_id="op_002", tool="run_code", agent="d", args={},
                     inputs=["X"], outputs={"Z": "a"}, status="success")
        dag.add_node(op_id="op_003", tool="run_code", agent="d", args={},
                     inputs=["Y", "Z"], outputs={"W": "c"}, status="success")
        return dag

    def test_diamond_ancestors(self, diamond_dag):
        assert diamond_dag.ancestors("op_003") == {"op_000", "op_001", "op_002"}

    def test_diamond_subgraph(self, diamond_dag):
        sub = diamond_dag.subgraph("op_003")
        assert sub.node_count() == 4


class TestPipelineDAGPersistence:
    """Test save/load round-trip."""

    def test_save_and_load(self, tmp_path):
        dag = PipelineDAG(session_dir=tmp_path)
        dag.add_node(op_id="op_000", tool="fetch", agent="e",
                     args={"dataset": "ACE"}, inputs=[],
                     outputs={"X": "r"}, status="success")
        dag.add_node(op_id="op_001", tool="run_code", agent="d",
                     args={"code": "y=x*2"}, inputs=["X"],
                     outputs={"Y": "m"}, status="success")
        dag.save()

        loaded = PipelineDAG.load(tmp_path)
        assert loaded.node_count() == 2
        assert loaded.producer_of("X") == "op_000"
        assert loaded.producer_of("Y") == "op_001"
        assert loaded.predecessors("op_001") == ["op_000"]
        assert loaded.next_op_id() == "op_002"

    def test_load_missing_file(self, tmp_path):
        dag = PipelineDAG.load(tmp_path)
        assert dag.node_count() == 0

    def test_load_drops_orphan_label_owners(self, tmp_path):
        """label_owners referencing non-existent op_ids are dropped."""
        import json
        data = {
            "version": 1,
            "nodes": [
                {"op_id": "op_000", "tool": "fetch", "agent": "e",
                 "args": {}, "inputs": [], "outputs": {"X": "r"},
                 "status": "success", "timestamp": "t", "error": None}
            ],
            "label_owners": {"X": "op_000", "Y": "op_999"},
        }
        (tmp_path / "pipeline.json").write_text(json.dumps(data))
        dag = PipelineDAG.load(tmp_path)
        assert dag.producer_of("X") == "op_000"
        assert dag.producer_of("Y") is None

    def test_save_no_session_dir(self):
        """save() is a no-op when session_dir is None."""
        dag = PipelineDAG(session_dir=None)
        dag.add_node(op_id="op_000", tool="fetch", agent="e", args={},
                     inputs=[], outputs={"X": "r"}, status="success")
        dag.save()  # should not raise


class TestPipelineDAGListenerIntegration:
    """Test that PipelineDAGListener creates DAG nodes from events."""

    def test_pipeline_event_creates_dag_node(self):
        from agent.event_bus import PipelineDAGListener, SessionEvent
        dag = PipelineDAG()
        listener = PipelineDAGListener(lambda: dag)
        event = SessionEvent(
            id="evt_0001",
            type="data_computed",
            ts="2026-01-01T00:00:00Z",
            tags=frozenset({"pipeline"}),
            agent="data_ops",
            data={
                "tool": "run_code",
                "args": {"code": "y=x*2", "description": "double"},
                "inputs": ["X"],
                "outputs": {"Y": "y"},
                "status": "success",
            },
        )
        listener(event)
        assert dag.node_count() == 1
        node = dag.node("op_000")
        assert node["agent"] == "data_ops"
        assert node["tool"] == "run_code"

    def test_non_pipeline_event_ignored(self):
        from agent.event_bus import PipelineDAGListener, SessionEvent
        dag = PipelineDAG()
        listener = PipelineDAGListener(lambda: dag)
        event = SessionEvent(
            id="evt_0002",
            type="debug",
            ts="2026-01-01T00:00:00Z",
            tags=frozenset({"console"}),
            agent="unknown",
            data={},
        )
        listener(event)
        assert dag.node_count() == 0
