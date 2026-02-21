"""
Pipeline DAG with staleness tracking and mutation.

Asset-centric DAG (inspired by Dagster): nodes represent data artifacts in the
DataStore, not just tasks.  Lazy staleness (inspired by marimo): mutations mark
nodes stale but don't recompute until explicitly triggered.  Backdating (inspired
by Salsa): after re-executing a node, if output is unchanged, skip all descendants.

The Pipeline wraps OperationsLog records — it doesn't replace the recording mechanism.

Architecture:

    OperationsLog (append-only recording, unchanged)
          │
          │ .get_pipeline(final_labels) or raw records
          ▼
    Pipeline (DAG with nodes, edges, state, mutation)
          │
          │ .execute_stale(store, cache_store, renderer)
          ▼
    DataStore + PlotlyRenderer (live session state)
"""

import hashlib
from collections import deque
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pandas as pd

from agent.event_bus import get_event_bus, DEBUG


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class NodeState(Enum):
    CLEAN = "clean"
    STALE = "stale"
    ERROR = "error"
    PENDING = "pending"  # newly inserted, never executed


class NodeType(Enum):
    FETCH = "fetch"
    COMPUTE = "compute"
    CREATE = "create"
    RENDER = "render"


_TOOL_TO_NODE_TYPE = {
    "fetch_data": NodeType.FETCH,
    "custom_operation": NodeType.COMPUTE,
    "store_dataframe": NodeType.CREATE,
    "render_plotly_json": NodeType.RENDER,
}


class PipelineNode:
    """A node in the pipeline DAG representing a data artifact or plot."""

    __slots__ = (
        "id", "node_type", "tool", "params", "inputs", "outputs",
        "state", "output_hash", "input_producers",
    )

    def __init__(
        self,
        id: str,
        node_type: NodeType,
        tool: str,
        params: dict,
        inputs: list[str],
        outputs: list[str],
        state: NodeState = NodeState.CLEAN,
        output_hash: str | None = None,
        input_producers: dict[str, str] | None = None,
    ):
        self.id = id
        self.node_type = node_type
        self.tool = tool
        self.params = dict(params)
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.state = state
        self.output_hash = output_hash
        self.input_producers = dict(input_producers) if input_producers else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "tool": self.tool,
            "params": self.params,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "state": self.state.value,
            "output_hash": self.output_hash,
            "input_producers": self.input_producers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineNode":
        return cls(
            id=d["id"],
            node_type=NodeType(d["node_type"]),
            tool=d["tool"],
            params=d["params"],
            inputs=d["inputs"],
            outputs=d["outputs"],
            state=NodeState(d["state"]),
            output_hash=d.get("output_hash"),
            input_producers=d.get("input_producers"),
        )


class PipelineEdge:
    """A directed edge in the pipeline DAG."""

    __slots__ = ("source_id", "target_id", "label")

    def __init__(self, source_id: str, target_id: str, label: str):
        self.source_id = source_id
        self.target_id = target_id
        self.label = label

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineEdge":
        return cls(d["source_id"], d["target_id"], d["label"])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """DAG with staleness tracking and mutation for data pipeline operations."""

    def __init__(self):
        self._nodes: dict[str, PipelineNode] = {}
        self._edges: list[PipelineEdge] = {}
        self._label_producer: dict[str, str] = {}  # label → producing node_id
        self._counter: int = 0

    # ---- Construction ----

    @classmethod
    def from_operations_log(cls, log, final_labels: set[str]) -> "Pipeline":
        """Construct a Pipeline from a live OperationsLog.

        Uses ``log.get_pipeline(final_labels)`` to extract the connected,
        chronologically-ordered records, then builds the DAG.
        """
        records = log.get_pipeline(final_labels)
        return cls.from_records(records)

    @classmethod
    def from_records(cls, records: list[dict]) -> "Pipeline":
        """Construct a Pipeline from raw operation records.

        Filters to successful records with known tools.  Builds nodes and
        edges based on input/output labels and input_producers provenance.
        """
        pipe = cls()

        for rec in records:
            # Skip errors, dedup fetches, manage_plot resets
            if rec.get("status") != "success":
                continue
            tool = rec.get("tool", "")
            if tool not in _TOOL_TO_NODE_TYPE:
                continue
            if tool == "fetch_data" and rec.get("args", {}).get("already_loaded"):
                continue

            node_type = _TOOL_TO_NODE_TYPE[tool]
            node_id = rec["id"]

            node = PipelineNode(
                id=node_id,
                node_type=node_type,
                tool=tool,
                params=dict(rec.get("args", {})),
                inputs=list(rec.get("inputs", [])),
                outputs=list(rec.get("outputs", [])),
                state=NodeState.CLEAN,
                input_producers=rec.get("input_producers"),
            )
            pipe._nodes[node_id] = node

            # Track which node produces each label
            for label in node.outputs:
                pipe._label_producer[label] = node_id

            # Extract counter from "op_NNN" IDs
            if node_id.startswith("op_"):
                try:
                    num = int(node_id[3:])
                    pipe._counter = max(pipe._counter, num)
                except ValueError:
                    pass

        # Build edges from input_producers (provenance) with fallback to label_producer
        pipe._edges = []
        for node in pipe._nodes.values():
            ip = node.input_producers or {}
            for label in node.inputs:
                # Prefer input_producers snapshot for exact provenance
                producer_id = ip.get(label) or pipe._label_producer.get(label)
                if producer_id and producer_id in pipe._nodes:
                    edge = PipelineEdge(producer_id, node.id, label)
                    pipe._edges.append(edge)

        return pipe

    def _next_id(self) -> str:
        """Generate the next node ID."""
        self._counter += 1
        return f"op_{self._counter:03d}"

    # ---- Topology ----

    def topological_order(self) -> list[str]:
        """Return node IDs in topological order (Kahn's algorithm)."""
        # Build adjacency and in-degree
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        children: dict[str, list[str]] = {nid: [] for nid in self._nodes}

        for edge in self._edges:
            if edge.source_id in self._nodes and edge.target_id in self._nodes:
                in_degree[edge.target_id] += 1
                children[edge.source_id].append(edge.target_id)

        # Deduplicate in-degree counts (multiple edges between same pair)
        in_degree = {nid: 0 for nid in self._nodes}
        seen_pairs: set[tuple[str, str]] = set()
        for edge in self._edges:
            if edge.source_id in self._nodes and edge.target_id in self._nodes:
                pair = (edge.source_id, edge.target_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    in_degree[edge.target_id] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in set(children[nid]):  # deduplicate
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

    def children(self, node_id: str) -> set[str]:
        """Direct downstream node IDs."""
        return {
            e.target_id for e in self._edges
            if e.source_id == node_id and e.target_id in self._nodes
        }

    def parents(self, node_id: str) -> set[str]:
        """Direct upstream node IDs."""
        return {
            e.source_id for e in self._edges
            if e.target_id == node_id and e.source_id in self._nodes
        }

    def descendants(self, node_id: str) -> set[str]:
        """All transitive downstream node IDs (excluding self)."""
        visited: set[str] = set()
        queue = deque(self.children(node_id))
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(self.children(nid))
        return visited

    def ancestors(self, node_id: str) -> set[str]:
        """All transitive upstream node IDs (excluding self)."""
        visited: set[str] = set()
        queue = deque(self.parents(node_id))
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            queue.extend(self.parents(nid))
        return visited

    # ---- Staleness ----

    def propagate_staleness(self, node_id: str) -> set[str]:
        """Mark *node_id* and all descendants STALE.  Returns the affected set."""
        if node_id not in self._nodes:
            return set()
        affected = {node_id}
        self._nodes[node_id].state = NodeState.STALE

        topo = self.topological_order()
        # Walk in topological order starting after node_id
        stale_set = {node_id}
        for nid in topo:
            if nid in stale_set:
                continue
            # If any parent is stale, this node is stale too
            if self.parents(nid) & stale_set:
                stale_set.add(nid)
                self._nodes[nid].state = NodeState.STALE
                affected.add(nid)

        return affected

    def get_stale_nodes(self) -> list[str]:
        """Return STALE and PENDING node IDs in topological order."""
        topo = self.topological_order()
        return [
            nid for nid in topo
            if self._nodes[nid].state in (NodeState.STALE, NodeState.PENDING)
        ]

    # ---- Mutation ----

    def update_node_params(self, node_id: str, new_params: dict) -> set[str]:
        """Merge *new_params* into a node's params and propagate staleness.

        Returns the set of affected (stale) node IDs.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found")
        node = self._nodes[node_id]
        node.params.update(new_params)
        return self.propagate_staleness(node_id)

    def remove_node(self, node_id: str) -> dict:
        """Remove a node and its edges.  Returns info about orphaned labels.

        Returns:
            Dict with "removed" (node_id) and "orphaned_labels" (labels that
            no longer have a producer).
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = self._nodes.pop(node_id)

        # Remove edges involving this node
        self._edges = [
            e for e in self._edges
            if e.source_id != node_id and e.target_id != node_id
        ]

        # Find orphaned labels (labels this node produced that no other node produces)
        orphaned = []
        for label in node.outputs:
            if self._label_producer.get(label) == node_id:
                del self._label_producer[label]
                orphaned.append(label)

        return {"removed": node_id, "orphaned_labels": orphaned}

    def insert_node(
        self,
        after_id: str,
        tool: str,
        params: dict,
        output_label: str,
    ) -> str:
        """Insert a new node after *after_id*, rewiring consumers.

        The new node consumes after_id's outputs and produces *output_label*.
        All nodes that previously consumed after_id's outputs are rewired to
        consume *output_label* instead.

        Returns the new node's ID.
        """
        if after_id not in self._nodes:
            raise KeyError(f"Node '{after_id}' not found")

        after_node = self._nodes[after_id]
        new_id = self._next_id()

        node_type = _TOOL_TO_NODE_TYPE.get(tool, NodeType.COMPUTE)

        # New node consumes after_id's outputs
        new_inputs = list(after_node.outputs)
        new_node = PipelineNode(
            id=new_id,
            node_type=node_type,
            tool=tool,
            params=params,
            inputs=new_inputs,
            outputs=[output_label],
            state=NodeState.PENDING,
            input_producers={lbl: after_id for lbl in new_inputs},
        )
        self._nodes[new_id] = new_node

        # Find consumers of after_id's outputs (edges where source=after_id)
        old_consumer_edges = [
            e for e in self._edges
            if e.source_id == after_id
        ]

        # Remove old edges from after_id to its consumers
        self._edges = [
            e for e in self._edges
            if e.source_id != after_id or e.target_id == new_id
        ]

        # Add edge: after_id → new_node (for each consumed label)
        for lbl in new_inputs:
            self._edges.append(PipelineEdge(after_id, new_id, lbl))

        # Rewire old consumers to consume output_label from new_node
        rewired_consumers: set[str] = set()
        for old_edge in old_consumer_edges:
            consumer_id = old_edge.target_id
            if consumer_id == new_id:
                continue
            consumer = self._nodes.get(consumer_id)
            if consumer is None:
                continue

            # Replace old label references with output_label
            old_label = old_edge.label
            if old_label in consumer.inputs:
                idx = consumer.inputs.index(old_label)
                consumer.inputs[idx] = output_label
                # Update input_producers
                if old_label in consumer.input_producers:
                    del consumer.input_producers[old_label]
                consumer.input_producers[output_label] = new_id

            # Add new edge: new_node → consumer
            self._edges.append(PipelineEdge(new_id, consumer_id, output_label))
            rewired_consumers.add(consumer_id)

        # Update label_producer
        self._label_producer[output_label] = new_id

        # Mark new node as PENDING, rewired consumers as STALE
        for consumer_id in rewired_consumers:
            self.propagate_staleness(consumer_id)

        return new_id

    # ---- Execution ----

    def execute_stale(
        self,
        store,
        cache_store=None,
        renderer=None,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """Re-execute stale/pending nodes with backdating.

        Uses replay handlers from scripts/replay.py for actual execution.

        Args:
            store: DataStore to write results into.
            cache_store: Optional DataStore for cached fetch data.
            renderer: Optional PlotlyRenderer for render nodes.
            progress_cb: Optional (step, total, node_id, tool) callback.

        Returns:
            Dict with execution summary: executed, skipped, errors, backdated.
        """
        from scripts.replay import (
            _replay_fetch, _replay_custom_op, _replay_store_df, _replay_render,
        )

        stale_ids = self.get_stale_nodes()
        if not stale_ids:
            return {"executed": 0, "skipped": 0, "errors": [], "backdated": 0}

        total = len(stale_ids)
        executed = 0
        skipped = 0
        backdated = 0
        errors: list[dict] = []
        skip_set: set[str] = set()  # nodes to skip due to backdating

        for i, node_id in enumerate(stale_ids):
            if node_id in skip_set:
                skipped += 1
                self._nodes[node_id].state = NodeState.CLEAN
                continue

            node = self._nodes[node_id]

            if progress_cb:
                progress_cb(i + 1, total, node_id, node.tool)

            # Skip if any parent is in ERROR state
            parent_states = {
                self._nodes[pid].state for pid in self.parents(node_id)
                if pid in self._nodes
            }
            if NodeState.ERROR in parent_states:
                node.state = NodeState.ERROR
                errors.append({
                    "node_id": node_id,
                    "tool": node.tool,
                    "error": "Skipped: parent node in ERROR state",
                })
                skip_set.update(self.descendants(node_id))
                continue

            # Build a record dict compatible with replay handlers
            record = {
                "id": node_id,
                "tool": node.tool,
                "args": node.params,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "status": "success",
                "input_producers": node.input_producers,
            }

            try:
                if node.tool == "fetch_data":
                    _replay_fetch(record, store, cache_store=cache_store)
                elif node.tool == "custom_operation":
                    _replay_custom_op(record, store)
                elif node.tool == "store_dataframe":
                    _replay_store_df(record, store)
                elif node.tool == "render_plotly_json":
                    fig = _replay_render(record, store)
                    if fig is not None and renderer is not None:
                        renderer._figure = fig
                        renderer._trace_labels = []
                        renderer._panel_count = 1
                else:
                    get_event_bus().emit(DEBUG, agent="pipeline", level="warning", msg=f"Pipeline: unknown tool '{node.tool}' in node {node_id}")
                    node.state = NodeState.CLEAN
                    continue

                # Compute output hash for backdating
                new_hash = self._compute_output_hash(node, store)

                if node.output_hash is not None and new_hash == node.output_hash:
                    # Output unchanged → backdate: skip all descendants
                    backdated += 1
                    node.state = NodeState.CLEAN
                    desc = self.descendants(node_id)
                    # Only skip descendants that are in our stale list
                    for d in desc:
                        if d in stale_ids:
                            skip_set.add(d)
                else:
                    node.output_hash = new_hash
                    node.state = NodeState.CLEAN

                executed += 1

            except Exception as e:
                get_event_bus().emit(DEBUG, agent="pipeline", level="warning", msg=f"Pipeline: node {node_id} ({node.tool}) failed: {e}")
                node.state = NodeState.ERROR
                errors.append({
                    "node_id": node_id,
                    "tool": node.tool,
                    "error": str(e),
                })
                skip_set.update(self.descendants(node_id))

        return {
            "executed": executed,
            "skipped": skipped,
            "errors": errors,
            "backdated": backdated,
        }

    @staticmethod
    def _compute_output_hash(node: PipelineNode, store) -> str | None:
        """Compute a hash of the node's output data for backdating comparison."""
        if not node.outputs:
            # Render nodes: hash the figure JSON params
            return hashlib.md5(
                str(sorted(node.params.items())).encode()
            ).hexdigest()

        parts = []
        for label in node.outputs:
            entry = store.get(label)
            if entry is None:
                return None
            try:
                if hasattr(entry.data, "values") and isinstance(entry.data, pd.DataFrame):
                    # DataFrame: hash via pd.util.hash_pandas_object
                    h = pd.util.hash_pandas_object(entry.data).values.tobytes()
                    parts.append(h)
                elif hasattr(entry.data, "values"):
                    # xarray DataArray
                    parts.append(entry.data.values.tobytes())
                else:
                    parts.append(str(entry.data).encode())
            except Exception:
                return None

        if not parts:
            return None
        combined = b"".join(p if isinstance(p, bytes) else p.encode() for p in parts)
        return hashlib.md5(combined).hexdigest()

    # ---- Serialization ----

    def to_dict(self) -> dict:
        """Serialize the pipeline to a dict."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
            "label_producer": dict(self._label_producer),
            "counter": self._counter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        """Deserialize a Pipeline from a dict."""
        pipe = cls()
        pipe._nodes = {
            nid: PipelineNode.from_dict(nd)
            for nid, nd in d["nodes"].items()
        }
        pipe._edges = [PipelineEdge.from_dict(ed) for ed in d["edges"]]
        pipe._label_producer = dict(d.get("label_producer", {}))
        pipe._counter = d.get("counter", 0)
        return pipe

    def node_detail(self, node_id: str) -> dict | None:
        """Return full detail for one node, or None if not found."""
        node = self._nodes.get(node_id)
        if node is None:
            return None

        _TOOL_DISPLAY = {
            "fetch_data": "fetch",
            "custom_operation": "compute",
            "store_dataframe": "create",
            "render_plotly_json": "render",
        }

        detail = {
            "id": node.id,
            "type": _TOOL_DISPLAY.get(node.tool, node.tool),
            "tool": node.tool,
            "state": node.state.value,
            "inputs": node.inputs,
            "outputs": node.outputs,
            "params": dict(node.params),
            "parents": sorted(self.parents(node_id)),
            "children": sorted(self.children(node_id)),
        }

        # Extract code and description for compute nodes
        if node.tool == "custom_operation":
            detail["code"] = node.params.get("code", "")
            detail["description"] = node.params.get("description", "")

        return detail

    def to_summary(self) -> dict:
        """Compact summary for LLM consumption."""
        _TOOL_DISPLAY = {
            "fetch_data": "fetch",
            "custom_operation": "compute",
            "store_dataframe": "create",
            "render_plotly_json": "render",
        }

        nodes_summary = []
        for nid in self.topological_order():
            node = self._nodes[nid]
            summary = {
                "id": nid,
                "type": _TOOL_DISPLAY.get(node.tool, node.tool),
                "state": node.state.value,
                "inputs": node.inputs,
                "outputs": node.outputs,
            }
            # Add key params based on tool type
            if node.tool == "fetch_data":
                summary["dataset"] = node.params.get("dataset_id", "")
                summary["parameter"] = node.params.get("parameter_id", "")
                tr = node.params.get("time_range_resolved") or node.params.get("time_range", "")
                summary["time_range"] = tr
            elif node.tool == "custom_operation":
                code = node.params.get("code", "")
                # Show first line of code as preview
                first_line = code.split("\n")[0] if code else ""
                if len(first_line) > 60:
                    first_line = first_line[:57] + "..."
                summary["code_preview"] = first_line
                desc = node.params.get("description", "")
                if desc:
                    summary["description"] = desc
            elif node.tool == "render_plotly_json":
                fig_json = node.params.get("figure_json", {})
                n_traces = len(fig_json.get("data", []))
                title = fig_json.get("layout", {}).get("title", {})
                if isinstance(title, dict):
                    title = title.get("text", "")
                summary["traces"] = n_traces
                summary["title"] = str(title)[:60] if title else ""

            nodes_summary.append(summary)

        edges_summary = [
            {"from": e.source_id, "to": e.target_id, "label": e.label}
            for e in self._edges
        ]

        stale = self.get_stale_nodes()
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "stale_count": len(stale),
            "stale_nodes": stale,
            "nodes": nodes_summary,
            "edges": edges_summary,
        }

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> PipelineNode | None:
        return self._nodes.get(node_id)
