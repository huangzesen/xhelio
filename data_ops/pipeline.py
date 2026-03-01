"""
Pipeline — unified module for live session DAG and saved replayable pipelines.

Two classes with different data models and lifecycles, one canonical location:

**Pipeline** — live session DAG with OOP nodes/edges, staleness tracking,
backdating, and mutation.  Asset-centric DAG (inspired by Dagster): nodes
represent data artifacts in the DataStore.  Lazy staleness (inspired by marimo):
mutations mark nodes stale but don't recompute until explicitly triggered.
Backdating (inspired by Salsa): after re-executing a node, if output is
unchanged, skip all descendants.

**SavedPipeline** — persisted replayable pipeline.  Extracts a session's
operation pipeline into a named, parameterized template that can be replayed
with any time range — no LLM needed.  Steps are classified into two phases:

- **Appropriation**: fetches and transforms data (chainable).
- **Presentation**: consumes data, produces nothing further (terminal).

Storage: ``~/.xhelio/pipelines/{pipeline_id}.json``
Index:   ``~/.xhelio/pipelines/_index.json``
"""

import copy
import hashlib
import json
import os
import re
import uuid
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from agent.event_bus import get_event_bus, DEBUG
from data_ops.operations_log import OperationsLog, _lookup_with_suffix_fallback


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_PIPELINES_DIR_NAME = "pipelines"
_INDEX_FILENAME = "_index.json"
_PIPELINE_VERSION = 1

_PRESENTATION_TOOLS = frozenset({"render_plotly_json"})
_SKIP_TOOLS = frozenset({"manage_plot"})
_RELEVANT_TOOLS = frozenset({
    "fetch_data", "custom_operation", "store_dataframe", "render_plotly_json",
})

# Keys to strip from fetch_data args (time injected at execution)
_FETCH_TIME_KEYS = frozenset({
    "time_range", "time_range_resolved", "time_start", "time_end",
    "time_min", "time_max", "already_loaded",
})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pipelines_dir() -> Path:
    """Return (and create) the pipelines storage directory."""
    from config import get_data_dir
    d = get_data_dir() / _PIPELINES_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gen_pipeline_id() -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"pl_{date}_{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scrub_xaxis_ranges(layout: dict) -> dict:
    """Remove hardcoded date ranges from xaxis*.range in a Plotly layout."""
    layout = copy.deepcopy(layout)
    for key in list(layout.keys()):
        if key.startswith("xaxis") or key == "xaxis":
            ax = layout[key]
            if isinstance(ax, dict):
                ax.pop("range", None)
                ax.pop("autorange", None)
    return layout


def _extract_time_range_from_fetch(args: dict) -> tuple[str, str] | None:
    """Extract the resolved time range from a fetch_data args dict."""
    tr = args.get("time_range_resolved")
    if isinstance(tr, list) and len(tr) == 2:
        return (str(tr[0]), str(tr[1]))
    if isinstance(tr, str) and " to " in tr:
        parts = tr.split(" to ", 1)
        return (parts[0], parts[1])
    ts, te = args.get("time_start"), args.get("time_end")
    if ts and te:
        return (str(ts), str(te))
    tr_raw = args.get("time_range", "")
    if isinstance(tr_raw, str) and " to " in tr_raw:
        parts = tr_raw.split(" to ", 1)
        return (parts[0], parts[1])
    return None


def topological_sort_steps(steps: list[dict]) -> list[dict]:
    """Sort steps by dependency order (inputs → outputs).

    Each step has ``step_id`` and ``inputs`` (list of step_ids).
    Returns a new list in execution order.

    Raises ValueError if cycles are detected.
    """
    id_to_step = {s["step_id"]: s for s in steps}
    in_degree: dict[str, int] = {s["step_id"]: 0 for s in steps}
    children: dict[str, list[str]] = {s["step_id"]: [] for s in steps}

    for s in steps:
        for inp in s.get("inputs", []):
            if inp in id_to_step:
                in_degree[s["step_id"]] += 1
                children[inp].append(s["step_id"])

    queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
    result = []
    while queue:
        sid = queue.popleft()
        result.append(id_to_step[sid])
        for child in children[sid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(result) != len(steps):
        raise ValueError("Cycle detected in pipeline steps")
    return result


def is_vanilla(steps: list[dict]) -> bool:
    """Return True if pipeline is vanilla (no transforms, <3 fetches).

    A vanilla pipeline has zero custom_operation/store_dataframe steps AND
    fewer than 3 fetch_data steps.  These are trivial fetch-and-render
    workflows that add noise to the search index.
    """
    fetch_count = 0
    for step in steps:
        tool = step.get("tool", "")
        if tool in ("custom_operation", "store_dataframe"):
            return False
        if tool == "fetch_data":
            fetch_count += 1
    return fetch_count < 3


def appropriation_fingerprint(steps: list[dict]) -> str:
    """Compute a SHA-256 fingerprint of a pipeline's appropriation phase.

    Two pipelines with the same data fetches and transforms but different
    visualizations produce the same fingerprint, enabling family grouping.

    The fingerprint is computed from:
    - Tool name
    - dataset_id / parameter_id (for fetches)
    - code and units (for compute / store_dataframe)
    - Canonical (position-based) input references

    Excluded: time ranges, descriptions, presentation steps, output_labels.
    """
    # Filter to appropriation steps only
    appro_steps = [s for s in steps if s.get("phase") == "appropriation"]
    if not appro_steps:
        return hashlib.sha256(b"empty").hexdigest()

    # Topologically sort them
    sorted_steps = topological_sort_steps(appro_steps)

    # Build canonical ID mapping: original step_id → position-based ID
    canonical_map: dict[str, str] = {}
    for i, step in enumerate(sorted_steps):
        canonical_map[step["step_id"]] = f"f{i:03d}"

    # Build canonical representation of each step
    canonical_steps = []
    for step in sorted_steps:
        tool = step.get("tool", "")
        params = step.get("params", {})

        canon: dict = {"tool": tool}

        # Canonical input references
        canonical_inputs = sorted(
            canonical_map.get(inp, inp)
            for inp in step.get("inputs", [])
        )
        if canonical_inputs:
            canon["inputs"] = canonical_inputs

        # Tool-specific identity keys
        if tool == "fetch_data":
            for key in ("dataset_id", "parameter_id"):
                if key in params:
                    canon[key] = params[key]
        elif tool in ("custom_operation", "store_dataframe"):
            for key in ("code", "units"):
                if key in params:
                    canon[key] = params[key]

        canonical_steps.append(canon)

    # Deterministic serialization → SHA-256
    blob = json.dumps(canonical_steps, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Enums and data classes (Pipeline DAG)
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
# Pipeline (live session DAG)
# ---------------------------------------------------------------------------

class Pipeline:
    """DAG with staleness tracking and mutation for data pipeline operations."""

    def __init__(self):
        self._nodes: dict[str, PipelineNode] = {}
        self._edges: list[PipelineEdge] = []
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

        # Build edges from input_producers (provenance) with fallback to label_producer.
        # Use _lookup_with_suffix_fallback so subcolumn labels (e.g.
        # DATASET.column) and dedup-suffixed labels (.1, .2) resolve to
        # the parent-label producer.
        pipe._edges = []
        for node in pipe._nodes.values():
            ip = node.input_producers or {}
            for label in node.inputs:
                producer_id = (
                    _lookup_with_suffix_fallback(ip, label)
                    or _lookup_with_suffix_fallback(pipe._label_producer, label)
                )
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


# ---------------------------------------------------------------------------
# SavedPipeline (persisted replayable pipeline)
# ---------------------------------------------------------------------------

class SavedPipeline:
    """In-memory saved pipeline with extraction, validation, mutation,
    execution, and persistence."""

    def __init__(self, data: dict):
        self._data = data

    # ── Properties ────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self._data["id"]

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @name.setter
    def name(self, value: str):
        self._data["name"] = value

    @property
    def description(self) -> str:
        return self._data.get("description", "")

    @description.setter
    def description(self, value: str):
        self._data["description"] = value

    @property
    def tags(self) -> list[str]:
        return self._data.get("tags", [])

    @tags.setter
    def tags(self, value: list[str]):
        self._data["tags"] = value

    @property
    def steps(self) -> list[dict]:
        return self._data.get("steps", [])

    @property
    def time_range_original(self) -> list[str]:
        return self._data.get("time_range_original", ["", ""])

    @property
    def is_vanilla(self) -> bool:
        return is_vanilla(self.steps)

    @property
    def family_id(self) -> str:
        return appropriation_fingerprint(self.steps)

    # ── Extraction ────────────────────────────────────────────────────

    @classmethod
    def from_session(
        cls,
        session_id: str,
        render_op_id: str | None = None,
        name: str = "",
        description: str = "",
        tags: list[str] | None = None,
    ) -> "SavedPipeline":
        """Extract a saved pipeline from a session's operations log.

        Args:
            session_id: Session directory name.
            render_op_id: If given, extract only the pipeline for this
                specific render.  Otherwise use the full pipeline.
            name: Human-readable name for the pipeline.
            description: Description of what the pipeline produces.
            tags: Optional tags for categorization.

        Returns:
            A new SavedPipeline instance.

        Raises:
            FileNotFoundError: If session operations.json doesn't exist.
        """
        from config import get_data_dir
        session_dir = get_data_dir() / "sessions" / session_id
        ops_path = session_dir / "operations.json"
        if not ops_path.exists():
            raise FileNotFoundError(f"No operations.json at {ops_path}")

        with open(ops_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)

        log = OperationsLog()
        log.load_from_records(all_records)

        # Collect all successful output labels
        all_labels: set[str] = set()
        for rec in all_records:
            if rec.get("status") == "success":
                all_labels.update(rec.get("outputs", []))

        # Extract pipeline
        if render_op_id:
            pipeline_records = log.get_state_pipeline(render_op_id, all_labels)
        else:
            pipeline_records = log.get_pipeline(all_labels)

        # Filter to relevant tools with success status
        filtered = []
        for rec in pipeline_records:
            tool = rec.get("tool", "")
            if tool in _SKIP_TOOLS:
                continue
            if tool not in _RELEVANT_TOOLS:
                continue
            if rec.get("status") != "success":
                continue
            if tool == "fetch_data" and rec.get("args", {}).get("already_loaded"):
                continue
            filtered.append(rec)

        # Build op_id → step_id mapping and label → op_id mapping
        op_to_step: dict[str, str] = {}
        label_producer: dict[str, str] = {}   # label → op_id (fallback)
        steps: list[dict] = []
        time_ranges: list[tuple[str, str]] = []

        for i, rec in enumerate(filtered):
            step_id = f"s{i + 1:03d}"
            op_id = rec["id"]
            op_to_step[op_id] = step_id

            # Track which op produced each output label (for fallback)
            for out_label in rec.get("outputs", []):
                label_producer[out_label] = op_id

            tool = rec["tool"]
            args = dict(rec.get("args", {}))

            # Determine phase
            phase = "presentation" if tool in _PRESENTATION_TOOLS else "appropriation"

            # Build inputs from input_producers with parent-label fallback.
            # Render inputs may be subcolumn labels (e.g. DATASET.column)
            # or dedup-suffixed (.1, .2); fall back to label_producer map.
            inputs: list[str] = []
            ip = rec.get("input_producers", {})
            for inp_label in rec.get("inputs", []):
                producer_id = (
                    _lookup_with_suffix_fallback(ip, inp_label)
                    or _lookup_with_suffix_fallback(label_producer, inp_label)
                )
                if producer_id and producer_id in op_to_step:
                    sid = op_to_step[producer_id]
                    if sid not in inputs:
                        inputs.append(sid)

            # Determine output_label
            outputs = rec.get("outputs", [])
            output_label = outputs[0] if outputs else None
            if phase == "presentation":
                output_label = None

            # Clean args per tool type
            step_description = ""
            if tool == "fetch_data":
                tr = _extract_time_range_from_fetch(args)
                if tr:
                    time_ranges.append(tr)
                # Strip time keys
                clean_args = {
                    k: v for k, v in args.items()
                    if k not in _FETCH_TIME_KEYS
                }
                step_description = f"Fetch {args.get('dataset_id', '?')}.{args.get('parameter_id', '?')}"

            elif tool == "custom_operation":
                clean_args = {}
                for k in ("code", "description", "units"):
                    if k in args:
                        clean_args[k] = args[k]
                step_description = args.get("description", "Custom operation")

            elif tool == "store_dataframe":
                clean_args = {}
                for k in ("code", "description", "units"):
                    if k in args:
                        clean_args[k] = args[k]
                step_description = args.get("description", "Create DataFrame")

            elif tool == "render_plotly_json":
                fig_json = args.get("figure_json", {})
                # Scrub hardcoded date ranges from layout
                if "layout" in fig_json:
                    fig_json["layout"] = _scrub_xaxis_ranges(fig_json["layout"])
                clean_args = {"figure_json": fig_json}
                step_description = "Render plot"

            else:
                clean_args = args

            step = {
                "step_id": step_id,
                "phase": phase,
                "tool": tool,
                "params": clean_args,
                "inputs": inputs,
                "output_label": output_label,
                "description": step_description,
            }
            steps.append(step)

        # Determine time_range_original from widest fetch range
        if time_ranges:
            t_start = min(tr[0] for tr in time_ranges)
            t_end = max(tr[1] for tr in time_ranges)
            time_range_original = [t_start, t_end]
        else:
            time_range_original = ["", ""]

        pipeline_data = {
            "version": _PIPELINE_VERSION,
            "id": _gen_pipeline_id(),
            "name": name,
            "description": description,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "source_session_id": session_id,
            "source_render_op_id": render_op_id,
            "tags": tags or [],
            "time_range_original": time_range_original,
            "steps": steps,
        }

        return cls(pipeline_data)

    # ── Validation ────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate the pipeline structure and contents.

        Returns:
            List of issue descriptions.  Empty means valid.
        """
        issues: list[str] = []
        data = self._data

        # 1. Schema checks
        for key in ("version", "id", "steps"):
            if key not in data:
                issues.append(f"Missing required key: {key}")
        if data.get("version") != _PIPELINE_VERSION:
            issues.append(f"Unsupported version: {data.get('version')} (expected {_PIPELINE_VERSION})")
        if not isinstance(data.get("steps"), list):
            issues.append("'steps' must be a list")
            return issues  # can't continue without steps

        steps = data["steps"]
        step_ids = {s["step_id"] for s in steps}

        # 2-3. Dataset / parameter existence
        for step in steps:
            if step["tool"] == "fetch_data":
                dataset_id = step["params"].get("dataset_id", "")
                if not dataset_id:
                    issues.append(f"Step {step['step_id']}: fetch_data missing dataset_id")
                param_id = step["params"].get("parameter_id", "")
                if not param_id:
                    issues.append(f"Step {step['step_id']}: fetch_data missing parameter_id")

        # 4. AST safety for custom_operation code
        for step in steps:
            if step["tool"] == "custom_operation":
                code = step["params"].get("code", "")
                if code:
                    try:
                        from data_ops.custom_ops import validate_code
                        violations = validate_code(code)
                        for v in violations:
                            issues.append(f"Step {step['step_id']}: code violation: {v}")
                    except Exception as e:
                        issues.append(f"Step {step['step_id']}: code validation error: {e}")

        # 5. DAG connectivity — all inputs reference existing step_ids, no cycles
        for step in steps:
            for inp in step.get("inputs", []):
                if inp not in step_ids:
                    issues.append(
                        f"Step {step['step_id']}: input '{inp}' references non-existent step"
                    )

        try:
            topological_sort_steps(steps)
        except ValueError as e:
            issues.append(str(e))

        # 6. Presentation is terminal — no step should reference a presentation step
        presentation_ids = {
            s["step_id"] for s in steps if s.get("phase") == "presentation"
        }
        for step in steps:
            for inp in step.get("inputs", []):
                if inp in presentation_ids:
                    issues.append(
                        f"Step {step['step_id']}: references presentation step '{inp}' as input"
                    )

        # 7. No orphans — every appropriation step should be transitively consumed
        # Build reverse reachability from presentation steps
        consumed: set[str] = set()
        queue = deque(presentation_ids)
        while queue:
            sid = queue.popleft()
            if sid in consumed:
                continue
            consumed.add(sid)
            step = next((s for s in steps if s["step_id"] == sid), None)
            if step:
                for inp in step.get("inputs", []):
                    if inp not in consumed:
                        queue.append(inp)

        for step in steps:
            if step.get("phase") == "appropriation" and step["step_id"] not in consumed:
                issues.append(
                    f"Step {step['step_id']}: orphan appropriation step "
                    f"(not consumed by any presentation)"
                )

        # 8. Label uniqueness
        labels = [
            s["output_label"] for s in steps
            if s.get("output_label") is not None
        ]
        seen: set[str] = set()
        for label in labels:
            if label in seen:
                issues.append(f"Duplicate output_label: '{label}'")
            seen.add(label)

        return issues

    # ── Mutation ──────────────────────────────────────────────────────

    def add_step(
        self, step: dict, after_step_id: str | None = None
    ) -> str:
        """Insert a step into the pipeline.

        Args:
            step: Step dict (must have tool, params, phase, inputs, output_label).
            after_step_id: Insert after this step.  If None, appends at end.

        Returns:
            The generated step_id.
        """
        # Generate next step_id
        existing_nums = []
        for s in self.steps:
            m = re.match(r"s(\d+)", s["step_id"])
            if m:
                existing_nums.append(int(m.group(1)))
        next_num = max(existing_nums, default=0) + 1
        step_id = f"s{next_num:03d}"
        step["step_id"] = step_id

        if after_step_id is None:
            self.steps.append(step)
        else:
            idx = next(
                (i for i, s in enumerate(self.steps) if s["step_id"] == after_step_id),
                None,
            )
            if idx is None:
                raise KeyError(f"Step '{after_step_id}' not found")
            self.steps.insert(idx + 1, step)

        self._data["updated_at"] = _now_iso()
        return step_id

    def remove_step(self, step_id: str) -> dict:
        """Remove a step by ID.

        Returns:
            The removed step dict.

        Raises:
            KeyError: If step not found.
        """
        idx = next(
            (i for i, s in enumerate(self.steps) if s["step_id"] == step_id),
            None,
        )
        if idx is None:
            raise KeyError(f"Step '{step_id}' not found")

        removed = self.steps.pop(idx)

        # Clean up references to the removed step in other steps' inputs
        for s in self.steps:
            if step_id in s.get("inputs", []):
                s["inputs"].remove(step_id)

        self._data["updated_at"] = _now_iso()
        return removed

    def update_step_params(self, step_id: str, params: dict) -> None:
        """Update parameters for a step.

        Args:
            step_id: Step to update.
            params: New parameter values to merge into existing params.

        Raises:
            KeyError: If step not found.
        """
        step = next(
            (s for s in self.steps if s["step_id"] == step_id), None
        )
        if step is None:
            raise KeyError(f"Step '{step_id}' not found")
        step["params"].update(params)
        self._data["updated_at"] = _now_iso()

    # ── Execution ─────────────────────────────────────────────────────

    def execute(
        self,
        time_start: str,
        time_end: str,
        progress_cb: Callable | None = None,
    ):
        """Execute the pipeline with a new time range.

        Args:
            time_start: Start time (ISO 8601).
            time_end: End time (ISO 8601).
            progress_cb: Optional (step, total, tool) -> None callback.

        Returns:
            ReplayResult with the DataStore, figure, and execution info.
        """
        from scripts.replay import (
            ReplayResult,
            _replay_fetch,
            _replay_custom_op,
            _replay_store_df,
            _replay_render,
        )
        from data_ops.store import DataStore

        # Topologically sort steps
        sorted_steps = topological_sort_steps(self.steps)

        # Build step_id → output_label mapping
        step_to_label: dict[str, str] = {}
        for s in sorted_steps:
            if s.get("output_label"):
                step_to_label[s["step_id"]] = s["output_label"]

        import tempfile
        store = DataStore(Path(tempfile.mkdtemp()))
        result = ReplayResult(store=store, steps_total=len(sorted_steps))

        for i, step in enumerate(sorted_steps):
            tool = step["tool"]
            step_id = step["step_id"]

            if progress_cb:
                progress_cb(i + 1, len(sorted_steps), tool)

            try:
                # Build a replay-compatible record dict
                record = self._build_replay_record(
                    step, step_to_label, time_start, time_end
                )

                if tool == "fetch_data":
                    _replay_fetch(record, store)
                elif tool == "custom_operation":
                    _replay_custom_op(record, store)
                elif tool == "store_dataframe":
                    _replay_store_df(record, store)
                elif tool == "render_plotly_json":
                    fig = _replay_render(record, store)
                    if fig is not None:
                        result.figure = fig
                else:
                    result.steps_completed += 1
                    continue

                result.steps_completed += 1

            except Exception as e:
                import logging
                logger = logging.getLogger("xhelio")
                logger.warning(
                    "Pipeline execute: step %s (%s) failed: %s",
                    step_id, tool, e,
                )
                result.errors.append({
                    "op_id": step_id,
                    "tool": tool,
                    "error": str(e),
                })

        return result

    def _build_replay_record(
        self,
        step: dict,
        step_to_label: dict[str, str],
        time_start: str,
        time_end: str,
    ) -> dict:
        """Build a replay-engine-compatible record from a pipeline step."""
        tool = step["tool"]
        params = copy.deepcopy(step.get("params", {}))

        # Resolve inputs from step_ids to data labels
        input_labels = []
        for inp_step_id in step.get("inputs", []):
            label = step_to_label.get(inp_step_id)
            if label:
                input_labels.append(label)

        outputs = []
        if step.get("output_label"):
            outputs = [step["output_label"]]

        if tool == "fetch_data":
            # Inject time range
            params["time_range_resolved"] = [time_start, time_end]

        elif tool == "custom_operation":
            # source_labels are reconstructed from input step_ids
            params["source_labels"] = input_labels

        record = {
            "id": step["step_id"],
            "tool": tool,
            "status": "success",
            "args": params,
            "inputs": input_labels,
            "outputs": outputs,
        }
        return record

    # ── Persistence ───────────────────────────────────────────────────

    # ── Feedback ─────────────────────────────────────────────────────

    @property
    def feedback(self) -> list[dict]:
        """Return the list of user feedback entries."""
        return self._data.get("feedback", [])

    def add_feedback(self, comment: str) -> dict:
        """Append a user feedback entry and update timestamp."""
        entry = {"comment": comment.strip(), "timestamp": _now_iso(), "source": "user"}
        self._data.setdefault("feedback", []).append(entry)
        self._data["updated_at"] = _now_iso()
        return entry

    # ── Persistence ───────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the pipeline to disk (atomic write) and git-commit."""
        # Integrity check — warn on orphans but don't block the save
        issues = self.validate()
        if issues:
            import logging
            logger = logging.getLogger("xhelio")
            orphans = [i for i in issues if "orphan" in i.lower()]
            if orphans:
                logger.warning(
                    "SavedPipeline %s: saving with %d orphan issue(s): %s",
                    self.id, len(orphans), "; ".join(orphans),
                )

        self._data["updated_at"] = _now_iso()
        pl_dir = _pipelines_dir()

        # Write pipeline file
        pl_path = pl_dir / f"{self.id}.json"
        tmp_path = pl_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)
        os.replace(tmp_path, pl_path)

        # Update index
        _update_index(self._build_index_entry())

        # Git-commit the data directory (memory.json + pipelines/)
        from agent.memory import _git_commit_data
        _git_commit_data(pl_dir.parent)

    @classmethod
    def load(cls, pipeline_id: str) -> "SavedPipeline":
        """Load a pipeline from disk.

        Raises:
            FileNotFoundError: If pipeline doesn't exist.
        """
        pl_path = _pipelines_dir() / f"{pipeline_id}.json"
        if not pl_path.exists():
            raise FileNotFoundError(f"Pipeline '{pipeline_id}' not found")
        with open(pl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def list_all(cls) -> list[dict]:
        """Return index summaries for all saved pipelines."""
        index_path = _pipelines_dir() / _INDEX_FILENAME
        if not index_path.exists():
            return []
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    @classmethod
    def delete(cls, pipeline_id: str) -> bool:
        """Soft-delete (archive) a pipeline.

        The pipeline file is kept on disk with ``archived=True`` so it can
        be recovered from git history or via the restore endpoint.  The
        pipeline is removed from the active index.

        Returns:
            True if archived, False if not found.
        """
        pl_dir = _pipelines_dir()
        pl_path = pl_dir / f"{pipeline_id}.json"
        if not pl_path.exists():
            return False

        # Load, mark as archived, and re-save the file
        try:
            with open(pl_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        data["archived"] = True
        data["archived_at"] = _now_iso()

        tmp_path = pl_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, pl_path)

        # Remove from active index
        index_path = pl_dir / _INDEX_FILENAME
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                items = [it for it in items if it.get("id") != pipeline_id]
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(items, f, indent=2, default=str)
            except (json.JSONDecodeError, OSError):
                pass

        # Git-commit the archive
        from agent.memory import _git_commit_data
        _git_commit_data(pl_dir.parent)

        return True

    @classmethod
    def list_archived(cls) -> list[dict]:
        """Return summary dicts for all archived pipelines.

        Scans all ``*.json`` files in the pipelines directory and returns
        entries where ``archived=True``.
        """
        pl_dir = _pipelines_dir()
        results = []
        for path in pl_dir.glob("*.json"):
            if path.name == _INDEX_FILENAME:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("archived"):
                    results.append({
                        "id": data.get("id", path.stem),
                        "name": data.get("name", ""),
                        "description": data.get("description", ""),
                        "archived_at": data.get("archived_at", ""),
                        "created_at": data.get("created_at", ""),
                        "tags": data.get("tags", []),
                    })
            except (json.JSONDecodeError, OSError):
                continue
        return results

    @classmethod
    def restore(cls, pipeline_id: str) -> "SavedPipeline":
        """Restore an archived pipeline back to active status.

        Raises:
            FileNotFoundError: If the pipeline file doesn't exist.
            ValueError: If the pipeline is not archived.
        """
        pl_path = _pipelines_dir() / f"{pipeline_id}.json"
        if not pl_path.exists():
            raise FileNotFoundError(f"Pipeline '{pipeline_id}' not found")
        with open(pl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("archived"):
            raise ValueError(f"Pipeline '{pipeline_id}' is not archived")
        data["archived"] = False
        data.pop("archived_at", None)
        pipeline = cls(data)
        pipeline.save()  # re-adds to index + git commits
        return pipeline

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a deep copy of the pipeline data."""
        return copy.deepcopy(self._data)

    @classmethod
    def from_dict(cls, d: dict) -> "SavedPipeline":
        """Create a pipeline from a dict."""
        return cls(copy.deepcopy(d))

    # ── Internal ──────────────────────────────────────────────────────

    def _build_index_entry(self) -> dict:
        """Build a summary dict for the index file."""
        datasets = set()
        for step in self.steps:
            if step["tool"] == "fetch_data":
                ds = step["params"].get("dataset_id", "")
                param = step["params"].get("parameter_id", "")
                if ds:
                    datasets.add(f"{ds}.{param}" if param else ds)

        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "created_at": self._data.get("created_at", ""),
            "updated_at": self._data.get("updated_at", ""),
            "step_count": len(self.steps),
            "datasets": sorted(datasets),
            "source_session_id": self._data.get("source_session_id", ""),
        }


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def _update_index(entry: dict) -> None:
    """Add or update an entry in the index file."""
    index_path = _pipelines_dir() / _INDEX_FILENAME
    items: list[dict] = []
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except (json.JSONDecodeError, OSError):
            items = []

    # Replace existing or append
    updated = False
    for i, it in enumerate(items):
        if it.get("id") == entry["id"]:
            items[i] = entry
            updated = True
            break
    if not updated:
        items.append(entry)

    tmp_path = index_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, default=str)
    os.replace(tmp_path, index_path)
