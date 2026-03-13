"""PipelineDAG — graph-native pipeline tracking backed by networkx."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

# ---------------------------------------------------------------------------
# Pipeline-relevant tools registry
# ---------------------------------------------------------------------------

PIPELINE_TOOLS: set[str] = {
    "run_code", "fetch", "fetch_data",
    "render_plotly_json", "manage_plot",
    "load_file",
}


def register_pipeline_tool(tool_name: str) -> None:
    """Register a tool as pipeline-relevant (for envoy tools)."""
    PIPELINE_TOOLS.add(tool_name)


# ---------------------------------------------------------------------------
# PipelineDAG
# ---------------------------------------------------------------------------

_PIPELINE_VERSION = 1


class PipelineDAG:
    """Directed acyclic graph of data pipeline operations.

    Wraps a networkx.DiGraph. Nodes are operations, edges are implicit
    from label flow (output label of A appears in inputs of B).

    Thread-safe: all mutations and consistency-dependent reads are locked.
    """

    def __init__(self, session_dir: Path | None = None) -> None:
        self._graph = nx.DiGraph()
        self._label_owners: dict[str, str] = {}  # label -> op_id
        self._counter = 0
        self._session_dir = session_dir
        self._lock = threading.Lock()

    # -- Mutation ----------------------------------------------------------

    def next_op_id(self) -> str:
        """Return the next op_id without consuming it."""
        with self._lock:
            return f"op_{self._counter:03d}"

    def add_node_auto(self, **kwargs) -> str:
        """Atomically allocate an op_id and add a node.

        Prevents race between ``next_op_id()`` and ``add_node()`` when
        called from concurrent listeners.
        """
        with self._lock:
            op_id = f"op_{self._counter:03d}"
            self._add_node_locked(op_id, **kwargs)
        return op_id

    def add_node(
        self,
        op_id: str,
        tool: str,
        agent: str,
        args: dict[str, Any],
        inputs: list[str],
        outputs: dict[str, str],
        status: str,
        error: str | None = None,
    ) -> None:
        """Add an operation node to the DAG.

        Edges are created automatically: for each label in ``inputs``,
        if a producer exists in ``label_owners``, an edge is added from
        the producer's op_id to this node.

        ``label_owners`` is updated only on success.
        """
        with self._lock:
            self._add_node_locked(
                op_id, tool=tool, agent=agent, args=args,
                inputs=inputs, outputs=outputs, status=status, error=error,
            )

    def _add_node_locked(
        self,
        op_id: str,
        tool: str,
        agent: str,
        args: dict[str, Any],
        inputs: list[str],
        outputs: dict[str, str],
        status: str,
        error: str | None = None,
    ) -> None:
        """Internal: add node while lock is already held."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self._graph.add_node(
            op_id,
            tool=tool,
            agent=agent,
            args=args,
            inputs=inputs,
            outputs=outputs,
            status=status,
            timestamp=timestamp,
            error=error,
        )
        # Create edges from producers of input labels
        for label in inputs:
            producer = self._label_owners.get(label)
            if producer is not None and producer in self._graph:
                if self._graph.has_edge(producer, op_id):
                    self._graph.edges[producer, op_id]["labels"].add(label)
                else:
                    self._graph.add_edge(producer, op_id, labels={label})

        # Update label ownership on success only
        if status == "success":
            for label in outputs:
                self._label_owners[label] = op_id

            # Advance counter
            op_num = int(op_id.split("_", 1)[1])
            if op_num >= self._counter:
                self._counter = op_num + 1

    # -- Queries -----------------------------------------------------------

    def __contains__(self, op_id: str) -> bool:
        with self._lock:
            return op_id in self._graph

    def node(self, op_id: str) -> dict[str, Any]:
        """Return attributes of a node."""
        with self._lock:
            return dict(self._graph.nodes[op_id])

    def node_kind(self, op_id: str) -> str:
        """Derive node kind: 'source', 'transform', or 'sink'."""
        with self._lock:
            attrs = self._graph.nodes[op_id]
            has_inputs = bool(attrs["inputs"])
            has_outputs = bool(attrs["outputs"])
            if not has_inputs and has_outputs:
                return "source"
            elif has_inputs and has_outputs:
                return "transform"
            else:
                return "sink"

    def nodes_by_kind(self, kind: str) -> list[str]:
        """Return all op_ids of a given kind."""
        with self._lock:
            result = []
            for op_id in self._graph.nodes:
                attrs = self._graph.nodes[op_id]
                has_inputs = bool(attrs["inputs"])
                has_outputs = bool(attrs["outputs"])
                if kind == "source" and not has_inputs and has_outputs:
                    result.append(op_id)
                elif kind == "transform" and has_inputs and has_outputs:
                    result.append(op_id)
                elif kind == "sink" and has_inputs and not has_outputs:
                    result.append(op_id)
            return result

    def predecessors(self, op_id: str) -> list[str]:
        """Direct parent node IDs."""
        with self._lock:
            return list(self._graph.predecessors(op_id))

    def successors(self, op_id: str) -> list[str]:
        """Direct child node IDs."""
        with self._lock:
            return list(self._graph.successors(op_id))

    def ancestors(self, op_id: str) -> set[str]:
        """All upstream node IDs (transitive)."""
        with self._lock:
            return nx.ancestors(self._graph, op_id)

    def descendants(self, op_id: str) -> set[str]:
        """All downstream node IDs (transitive)."""
        with self._lock:
            return nx.descendants(self._graph, op_id)

    def roots(self) -> list[str]:
        """All source node IDs (in-degree 0)."""
        with self._lock:
            return [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]

    def leaves(self) -> list[str]:
        """All terminal node IDs (out-degree 0)."""
        with self._lock:
            return [n for n in self._graph.nodes if self._graph.out_degree(n) == 0]

    def path(self, from_op: str, to_op: str) -> list[str]:
        """Shortest path between two operations."""
        with self._lock:
            try:
                return nx.shortest_path(self._graph, from_op, to_op)
            except nx.NetworkXNoPath:
                return []

    def producer_of(self, label: str) -> str | None:
        """Which op_id most recently produced this label."""
        with self._lock:
            return self._label_owners.get(label)

    def consumers_of(self, label: str) -> list[str]:
        """All op_ids whose inputs list contains this label."""
        with self._lock:
            return [
                op_id for op_id in self._graph.nodes
                if label in self._graph.nodes[op_id]["inputs"]
            ]

    def topological_order(self) -> list[str]:
        """Execution order for replay."""
        with self._lock:
            return list(nx.topological_sort(self._graph))

    def subgraph(self, op_id: str) -> "PipelineDAG":
        """Extract minimal DAG containing full ancestry for replay.

        Returns a PipelineDAG with session_dir=None (no persistence).
        """
        with self._lock:
            ancestor_ids = nx.ancestors(self._graph, op_id) | {op_id}
            sub = PipelineDAG(session_dir=None)
            sub._graph = self._graph.subgraph(ancestor_ids).copy()
            sub._label_owners = {
                label: owner
                for label, owner in self._label_owners.items()
                if owner in ancestor_ids
            }
            sub._counter = self._counter
            return sub

    def node_count(self) -> int:
        with self._lock:
            return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        with self._lock:
            return self._graph.number_of_edges()

    # -- Persistence -------------------------------------------------------

    def save(self) -> None:
        """Save DAG to pipeline.json in the session directory."""
        if self._session_dir is None:
            return
        # Snapshot under lock
        with self._lock:
            data = self._serialize()
        # Write outside lock
        path = Path(self._session_dir) / "pipeline.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

    def _serialize(self) -> dict:
        """Serialize DAG to a dict (must hold lock)."""
        nodes = []
        for op_id in self._graph.nodes:
            attrs = dict(self._graph.nodes[op_id])
            nodes.append({"op_id": op_id, **attrs})
        return {
            "version": _PIPELINE_VERSION,
            "nodes": nodes,
            "label_owners": dict(self._label_owners),
        }

    @classmethod
    def load(cls, session_dir: Path) -> "PipelineDAG":
        """Load DAG from pipeline.json, or return empty DAG."""
        path = Path(session_dir) / "pipeline.json"
        dag = cls(session_dir=session_dir)
        if not path.exists():
            return dag
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Reconstruct nodes
        for node_data in data.get("nodes", []):
            op_id = node_data.pop("op_id")
            dag._graph.add_node(op_id, **node_data)
        # Reconstruct label_owners (drop refs to missing nodes)
        for label, owner in data.get("label_owners", {}).items():
            if owner in dag._graph:
                dag._label_owners[label] = owner
        # Reconstruct edges from inputs + label_owners
        for op_id in dag._graph.nodes:
            for label in dag._graph.nodes[op_id].get("inputs", []):
                producer = dag._label_owners.get(label)
                if producer is not None and producer in dag._graph:
                    if dag._graph.has_edge(producer, op_id):
                        dag._graph.edges[producer, op_id]["labels"].add(label)
                    else:
                        dag._graph.add_edge(producer, op_id, labels={label})
        # Restore counter from highest op_id
        if dag._graph.nodes:
            max_num = max(
                int(op.split("_", 1)[1]) for op in dag._graph.nodes
            )
            dag._counter = max_num + 1
        return dag
