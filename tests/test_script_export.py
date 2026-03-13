"""Tests for pipeline script export."""

from data_ops.dag import PipelineDAG
from data_ops.script_export import export_script


def test_export_script_basic():
    """Exported script should import ReplayContext and call TOOL_REGISTRY."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="run_code", agent="orchestrator",
        args={"code": "x = 1", "inputs": [], "outputs": {"result": "x"}},
        inputs=[], outputs={"result": "result_id"}, status="success",
    )
    script = export_script(dag, "op_000")
    assert "from data_ops.store import DataStore" in script
    assert "from agent.tool_context import ReplayContext" in script
    assert "from agent.tool_handlers import TOOL_REGISTRY" in script
    assert "TOOL_REGISTRY['run_code']" in script


def test_export_script_skips_failed():
    """Failed original steps should be commented out."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="run_code", agent="test",
        args={}, inputs=[], outputs={}, status="error", error="boom",
    )
    script = export_script(dag, "op_000")
    assert "SKIPPED" in script
    # TOOL_REGISTRY should only appear in the import line, not as a call
    lines_with_registry = [
        line for line in script.splitlines()
        if "TOOL_REGISTRY" in line and not line.startswith("from ")
    ]
    assert all(line.lstrip().startswith("#") for line in lines_with_registry)


def test_export_script_multi_step():
    """Multi-step subgraph should produce steps in topological order."""
    dag = PipelineDAG()
    dag.add_node(
        "op_000", tool="run_code", agent="test",
        args={"code": "a = 1"}, inputs=[], outputs={"A": "a"},
        status="success",
    )
    dag.add_node(
        "op_001", tool="render_plotly_json", agent="test",
        args={"figure_json": {}}, inputs=["A"], outputs={},
        status="success",
    )
    script = export_script(dag, "op_001")
    # run_code should appear before render_plotly_json
    rc_pos = script.index("run_code")
    rp_pos = script.index("render_plotly_json")
    assert rc_pos < rp_pos
