"""Tests for DAG → Plotly visualization."""

from data_ops.dag import PipelineDAG
from data_ops.dag_viz import dag_to_plotly


def test_dag_to_plotly_empty():
    """Empty DAG should return a figure with no data traces."""
    dag = PipelineDAG()
    fig = dag_to_plotly(dag)
    assert "data" in fig
    assert "layout" in fig
    assert fig["data"] == []


def test_dag_to_plotly_single_node():
    """Single node DAG should produce a figure with one node marker."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="run_code", agent="test",
        args={}, inputs=[], outputs={"A": "a"}, status="success",
    )
    fig = dag_to_plotly(dag)
    assert len(fig["data"]) > 0
    # Should have only the node trace (no edges)
    assert len(fig["data"]) == 1
    assert fig["data"][0]["type"] == "scatter"
    assert fig["data"][0]["mode"] == "markers+text"


def test_dag_to_plotly_with_edges():
    """DAG with edges should include edge traces."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="run_code", agent="test",
        args={}, inputs=[], outputs={"A": "a"}, status="success",
    )
    dag.add_node(
        "op_001", tool="render_plotly_json", agent="test",
        args={}, inputs=["A"], outputs={}, status="success",
    )
    fig = dag_to_plotly(dag)
    # Should have edge trace + node trace
    assert len(fig["data"]) == 2
    assert fig["data"][0]["mode"] == "lines"
    assert fig["data"][1]["mode"] == "markers+text"


def test_dag_to_plotly_highlight_subgraph():
    """Highlighting a target should mark ancestor nodes differently."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="run_code", agent="test",
        args={}, inputs=[], outputs={"A": "a"}, status="success",
    )
    dag.add_node(
        "op_001", tool="render_plotly_json", agent="test",
        args={}, inputs=["A"], outputs={}, status="success",
    )
    fig = dag_to_plotly(dag, highlight_op_id="op_001")
    assert "data" in fig
    # The highlighted node should be larger
    node_trace = fig["data"][-1]
    sizes = node_trace["marker"]["size"]
    # op_001 is second node, should have size 16
    assert 16 in sizes


def test_dag_to_plotly_failed_node():
    """Failed node should be colored red."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="run_code", agent="test",
        args={}, inputs=[], outputs={}, status="error",
        error="something broke",
    )
    fig = dag_to_plotly(dag)
    node_trace = fig["data"][-1]
    assert node_trace["marker"]["color"][0] == "#ef4444"


def test_dag_to_plotly_layout_structure():
    """Layout should have expected axis and margin config."""
    dag = PipelineDAG()
    fig = dag_to_plotly(dag)
    layout = fig["layout"]
    assert layout["showlegend"] is False
    assert layout["hovermode"] == "closest"
    assert layout["xaxis"]["showgrid"] is False
    assert layout["margin"]["l"] == 20
