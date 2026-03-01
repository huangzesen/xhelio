"""
Operations log for pipeline reproducibility.

Records every data-producing operation as a JSON-serializable dict so the
full pipeline can eventually be replayed.  This phase implements recording
only — replay comes later.

Storage: ``~/.xhelio/sessions/{session_id}/operations.json``
"""

import contextvars
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _resolve_producer_from(
    rec: dict,
    label: str,
    rec_by_id: dict[str, dict],
    last_producer: dict[str, Any],
) -> dict | None:
    """Find the producer of *label* for a given record.

    Uses the record's ``input_producers`` snapshot if available,
    falling back to ``last_producer`` for backward compatibility.
    Both paths apply dedup-suffix stripping as a fallback.
    """
    ip = rec.get("input_producers", {})
    producer_id = _lookup_with_suffix_fallback(ip, label)
    if producer_id is not None:
        prod = rec_by_id.get(producer_id)
        if prod is not None:
            return prod
    return _lookup_with_suffix_fallback(last_producer, label)


def _lookup_with_suffix_fallback(mapping: dict[str, Any], label: str) -> Any:
    """Look up *label* in *mapping*, falling back to progressive parent-label splitting.

    Handles both dedup suffixes (.1, .2) and arbitrary subcolumn labels
    (e.g. 'DATASET.column_name').  Tries the exact label first, then
    progressively strips rightmost dot-separated segments: for 'A.B.C',
    tries 'A.B', then 'A'.

    Returns None if no parent matches.
    """
    val = mapping.get(label)
    if val is not None:
        return val
    # Progressive split from right: A.B.C → try A.B, then A
    parts = label.split(".")
    for i in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:i])
        val = mapping.get(parent)
        if val is not None:
            return val
    return None


class OperationsLog:
    """Ordered, thread-safe log of data-producing operations."""

    def __init__(self, session_id: str = ""):
        self._session_id = session_id
        self._records: list[dict] = []
        self._counter: int = 0
        self._lock = threading.Lock()
        # Maps label → op_id of the most recent successful producer
        self._current_producers: dict[str, str] = {}

    def _find_producer(self, label: str) -> str | None:
        """Find the producer op_id for a label, using progressive parent-label splitting.

        Handles both dedup suffixes (.1, .2) and arbitrary subcolumn labels
        (e.g. 'DATASET.column_name'). Tries exact label first, then
        progressively strips rightmost dot-separated segments.
        """
        return _lookup_with_suffix_fallback(self._current_producers, label)

    def record(
        self,
        tool: str,
        args: dict[str, Any],
        outputs: list[str],
        inputs: Optional[list[str]] = None,
        status: str = "success",
        error: Optional[str] = None,
    ) -> dict:
        """Append an operation record and return it.

        Args:
            tool: Tool name (e.g. "fetch_data", "custom_operation").
            args: Tool-specific arguments dict.
            outputs: Labels produced by this operation.
            inputs: Labels consumed by this operation.
            status: "success" or "error".
            error: Error message if status is "error".

        Returns:
            The recorded operation dict.
        """
        with self._lock:
            self._counter += 1
            input_list = inputs or []
            # Snapshot: which op produced each input at this moment
            input_producers = {
                label: pid
                for label in input_list
                if (pid := self._find_producer(label)) is not None
            }
            prefix = f"{self._session_id}:" if self._session_id else ""
            record = {
                "id": f"{prefix}op_{self._counter:03d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "status": status,
                "inputs": input_list,
                "outputs": outputs,
                "args": args,
                "error": error,
                "input_producers": input_producers,
            }
            self._records.append(record)
            # Update current producers for successful outputs
            if status == "success":
                for label in outputs:
                    self._current_producers[label] = record["id"]
            return record

    def get_records(self) -> list[dict]:
        """Return a copy of all records."""
        with self._lock:
            return list(self._records)

    def save_to_file(self, path: Path) -> None:
        """Write records to a JSON file."""
        import os
        with self._lock:
            data = list(self._records)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)

    def load_from_records(self, records: list[dict]) -> int:
        """Restore from an in-memory list and resume the counter.

        When ``session_id`` is set, any record whose ID lacks the session
        prefix is migrated to ``{session_id}:op_NNN`` format.  References
        inside ``input_producers`` values are migrated as well.

        Returns:
            Number of records loaded.
        """
        with self._lock:
            self._records = list(records)

            # Migrate old-style IDs to scoped format when session_id is set
            if self._session_id:
                prefix = f"{self._session_id}:"
                # Build old→new mapping for ID migration
                id_map: dict[str, str] = {}
                for rec in self._records:
                    old_id = rec["id"]
                    if ":" not in old_id:
                        new_id = f"{prefix}{old_id}"
                        id_map[old_id] = new_id
                        rec["id"] = new_id
                # Migrate input_producers values
                if id_map:
                    for rec in self._records:
                        ip = rec.get("input_producers")
                        if ip:
                            rec["input_producers"] = {
                                label: id_map.get(pid, pid)
                                for label, pid in ip.items()
                            }

            self._counter = self._max_counter_from_records(self._records)
            # Rebuild _current_producers by replaying all records' outputs
            self._current_producers.clear()
            for rec in self._records:
                if rec.get("status") == "success":
                    for label in rec.get("outputs", []):
                        self._current_producers[label] = rec["id"]
            return len(self._records)

    def load_from_file(self, path: Path) -> int:
        """Load records from a JSON file and resume the counter.

        Returns:
            Number of records loaded.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.load_from_records(data)

    @staticmethod
    def _max_counter_from_records(records: list[dict]) -> int:
        """Extract the highest numeric ID from op_NNN-style IDs.

        Handles both plain ``op_NNN`` and scoped ``session_id:op_NNN`` formats.
        """
        max_id = 0
        for rec in records:
            op_id = rec.get("id", "")
            # Strip session prefix if present (e.g. "sess_abc:op_001" → "op_001")
            if ":" in op_id:
                op_id = op_id.rsplit(":", 1)[-1]
            if op_id.startswith("op_"):
                try:
                    max_id = max(max_id, int(op_id[3:]))
                except ValueError:
                    pass
        return max_id

    def get_pipeline(self, final_labels: set[str]) -> list[dict]:
        """Extract the minimal ordered operation chain to reproduce *final_labels*.

        Each returned record is a shallow copy of the original with added fields:

        - ``contributes_to``: list of product IDs (op IDs of end-state
          operations like ``render_plotly_json``) that this operation
          transitively feeds into. Empty means "orphan".
        - ``product_family`` (renders only): op ID of the first render in
          the family (family identifier).
        - ``state_index`` (renders only): 0-based index within the family.
        - ``state_count`` (renders only): total states in the family.

        Args:
            final_labels: Labels present in the DataStore (the desired outputs).

        Returns:
            Filtered, chronologically ordered list of operation records.
        """
        with self._lock:
            records = list(self._records)

        if not final_labels or not records:
            return []

        # Build index: for each label, the last successful producer record.
        # Skip noise records (dedup fetches, manage_plot resets) so they
        # can't shadow a real producer and then get filtered out later.
        last_producer: dict[str, dict] = {}
        for rec in records:
            if rec["status"] != "success":
                continue
            if rec["tool"] == "fetch_data" and rec["args"].get("already_loaded"):
                continue
            if rec["tool"] == "manage_plot" and rec["args"].get("action") == "reset":
                continue
            for label in rec["outputs"]:
                last_producer[label] = rec

        # Also build an id-based lookup for input_producers references
        rec_by_id: dict[str, dict] = {r["id"]: r for r in records}

        # Transitively resolve all required labels (Pass 1 — data labels)
        selected_ids: set[str] = set()
        queue = list(final_labels)
        visited: set[str] = set()
        while queue:
            label = queue.pop()
            if label in visited:
                continue
            visited.add(label)
            rec = _lookup_with_suffix_fallback(last_producer, label)
            if rec is None:
                continue
            selected_ids.add(rec["id"])
            for inp in rec["inputs"]:
                if inp not in visited:
                    queue.append(inp)

        # ── Pass 2: per-render provenance using input_producers ─────────
        # Each successful render_plotly_json is a product.  BFS backwards
        # using the render's input_producers snapshot (not the global
        # last_producer) so each render traces to the *specific* ops that
        # were current when it ran.
        contributes_to: dict[str, set[str]] = {}

        for rec in reversed(records):
            if rec["tool"] == "render_plotly_json" and rec["status"] == "success":
                product_id = rec["id"]
                selected_ids.add(product_id)
                contributes_to.setdefault(product_id, set()).add(product_id)
                # BFS backwards using this render's input_producers
                product_queue: list[tuple[str, dict]] = [
                    (label, rec) for label in rec["inputs"]
                ]
                product_visited: set[str] = set()
                while product_queue:
                    label, context_rec = product_queue.pop()
                    if label in product_visited:
                        continue
                    product_visited.add(label)
                    prod = _resolve_producer_from(context_rec, label, rec_by_id, last_producer)
                    if prod is None:
                        continue
                    selected_ids.add(prod["id"])
                    contributes_to.setdefault(prod["id"], set()).add(product_id)
                    for inp in prod["inputs"]:
                        if inp not in product_visited:
                            product_queue.append((inp, prod))

        # ── Product families: group renders by input label set ──────────
        render_records = [
            r for r in records
            if r["tool"] == "render_plotly_json" and r["status"] == "success"
        ]
        # family key → list of render records in chronological order
        families: dict[frozenset[str], list[dict]] = {}
        for r in render_records:
            key = frozenset(r["inputs"])
            families.setdefault(key, []).append(r)

        # Build annotations: render op_id → (product_family, state_index, state_count)
        family_annotations: dict[str, tuple[str, int, int]] = {}
        for members in families.values():
            family_id = members[0]["id"]
            count = len(members)
            for idx, r in enumerate(members):
                family_annotations[r["id"]] = (family_id, idx, count)

        # Filter: chronological order, only selected.
        pipeline = []
        for rec in records:
            if rec["id"] in selected_ids:
                rec_copy = dict(rec)
                rec_copy["contributes_to"] = sorted(
                    contributes_to.get(rec["id"], set())
                )
                if rec["id"] in family_annotations:
                    fam_id, si, sc = family_annotations[rec["id"]]
                    rec_copy["product_family"] = fam_id
                    rec_copy["state_index"] = si
                    rec_copy["state_count"] = sc
                pipeline.append(rec_copy)

        return pipeline

    def get_state_pipeline(self, render_op_id: str, final_labels: set[str]) -> list[dict]:
        """Extract the operation chain for a single render (one product state).

        Walks backwards from the render identified by *render_op_id* using its
        ``input_producers`` snapshot to collect only the operations that
        contributed to that specific render.

        Args:
            render_op_id: The op ID of a ``render_plotly_json`` record.
            final_labels: Labels present in the DataStore (used for the
                ``last_producer`` fallback when ``input_producers`` is absent).

        Returns:
            Chronologically ordered list of operation records with
            ``contributes_to`` set to ``[render_op_id]`` for every included op.
            Returns ``[]`` if the render is not found or not a successful
            ``render_plotly_json``.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return []

        rec_by_id: dict[str, dict] = {r["id"]: r for r in records}

        # Find the render record
        render_rec = rec_by_id.get(render_op_id)
        if render_rec is None:
            return []
        if render_rec["tool"] != "render_plotly_json" or render_rec["status"] != "success":
            return []

        # Build last_producer using only records *before* the render.
        # This prevents the BFS from selecting a re-fetch that happened
        # after the render, which would appear later in the chronological
        # pipeline and fail during replay execution.
        last_producer: dict[str, dict] = {}
        for rec in records:
            if rec["id"] == render_op_id:
                break
            if rec["status"] != "success":
                continue
            if rec["tool"] == "fetch_data" and rec["args"].get("already_loaded"):
                continue
            if rec["tool"] == "manage_plot" and rec["args"].get("action") == "reset":
                continue
            for label in rec["outputs"]:
                last_producer[label] = rec

        # BFS backwards from the render
        selected_ids: set[str] = {render_op_id}
        queue: list[tuple[str, dict]] = [
            (label, render_rec) for label in render_rec["inputs"]
        ]
        visited: set[str] = set()
        while queue:
            label, context_rec = queue.pop()
            if label in visited:
                continue
            visited.add(label)
            prod = _resolve_producer_from(context_rec, label, rec_by_id, last_producer)
            if prod is None:
                continue
            selected_ids.add(prod["id"])
            for inp in prod["inputs"]:
                if inp not in visited:
                    queue.append((inp, prod))

        # Filter chronologically, annotate contributes_to
        pipeline = []
        for rec in records:
            if rec["id"] in selected_ids:
                rec_copy = dict(rec)
                rec_copy["contributes_to"] = [render_op_id]
                pipeline.append(rec_copy)

        return pipeline

    def get_pipeline_mermaid(self, final_labels: set[str]) -> str:
        """Return a Mermaid flowchart string for the pipeline.

        Multi-state product families are collapsed into a single node
        labeled ``plot (N states)``.

        Args:
            final_labels: Labels present in the DataStore (the desired outputs).

        Returns:
            Mermaid ``graph TD`` source text, or empty string if pipeline is empty.
        """
        pipeline = self.get_pipeline(final_labels)
        if not pipeline:
            return ""

        _TOOL_DISPLAY = {
            "fetch_data": "fetch",
            "custom_operation": "compute",
            "store_dataframe": "create",
            "render_plotly_json": "plot",
            "manage_plot": "export",
        }

        # Identify collapsed families: family_id → list of member op IDs
        collapsed: dict[str, list[str]] = {}
        for rec in pipeline:
            fam = rec.get("product_family")
            if fam and rec.get("state_count", 1) > 1:
                collapsed.setdefault(fam, []).append(rec["id"])

        # For collapsed families, the representative node ID is the family_id.
        # Map each member op_id → representative node ID.
        member_to_rep: dict[str, str] = {}
        for fam_id, members in collapsed.items():
            for m in members:
                member_to_rep[m] = fam_id

        # Build a lookup: label -> id of the record that produced it
        label_producer: dict[str, str] = {}
        for rec in pipeline:
            for label in rec["outputs"]:
                label_producer[label] = rec["id"]

        lines = ["graph TD"]
        emitted_nodes: set[str] = set()

        for rec in pipeline:
            rid = rec["id"]
            rep = member_to_rep.get(rid)

            if rep:
                # Collapsed family member — emit the family node once
                if rep not in emitted_nodes:
                    state_count = rec["state_count"]
                    node_label = f"plot ({state_count} states)"
                    lines.append(f'    {rep}["{node_label}"]')
                    emitted_nodes.add(rep)
                # Edges: connect to the representative node
                target = rep
            else:
                # Normal node
                tool_short = _TOOL_DISPLAY.get(rec["tool"], rec["tool"])
                outputs = ", ".join(rec["outputs"]) if rec["outputs"] else ""
                if outputs:
                    node_label = f"{tool_short}\\n{outputs}"
                else:
                    node_label = tool_short
                lines.append(f'    {rid}["{node_label}"]')
                emitted_nodes.add(rid)
                target = rid

            # Edges from input producers to this node.
            # Use input_producers snapshot when available for correct provenance,
            # fall back to label_producer for old records.  Both paths apply
            # dedup-suffix stripping so .1/.2 labels resolve correctly.
            seen_edges: set[tuple[str, str]] = set()
            ip = rec.get("input_producers", {})
            for inp in rec["inputs"]:
                src = _lookup_with_suffix_fallback(ip, inp) or _lookup_with_suffix_fallback(label_producer, inp)
                if src:
                    # Remap source if it's also a collapsed member
                    src_node = member_to_rep.get(src, src)
                    edge = (src_node, target)
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        lines.append(f"    {src_node} -->|{inp}| {target}")

        return "\n".join(lines)

    def set_pipeline_status(self, op_id: str, status: str) -> bool:
        """Set ``pipeline_status`` on a record.

        Args:
            op_id: The operation ID (plain or scoped).
            status: One of ``"fresh"``, ``"registered"``, ``"discarded"``.

        Returns:
            True if the record was found and updated, False otherwise.
        """
        with self._lock:
            for rec in self._records:
                if rec["id"] == op_id:
                    rec["pipeline_status"] = status
                    return True
            return False

    def get_render_ops_by_status(self, status: str = "fresh") -> list[dict]:
        """Return successful render ops with the given ``pipeline_status``.

        Records without a ``pipeline_status`` key are treated as ``"fresh"``.

        Args:
            status: The pipeline_status to filter by (default ``"fresh"``).

        Returns:
            List of matching render operation records (copies).
        """
        with self._lock:
            return [
                dict(rec) for rec in self._records
                if rec["tool"] == "render_plotly_json"
                and rec["status"] == "success"
                and rec.get("pipeline_status", "fresh") == status
            ]

    def clear(self) -> None:
        """Reset records and counter."""
        with self._lock:
            self._records.clear()
            self._counter = 0
            self._current_producers.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# ContextVar-based log — each context gets its own instance.
_log_var: contextvars.ContextVar[Optional[OperationsLog]] = contextvars.ContextVar(
    "_log_var", default=None
)


def get_operations_log() -> OperationsLog:
    """Return the OperationsLog for the current context, creating one if needed."""
    log = _log_var.get()
    if log is None:
        log = OperationsLog()
        _log_var.set(log)
    return log


def set_operations_log(log: OperationsLog) -> None:
    """Explicitly set the OperationsLog for the current context."""
    _log_var.set(log)


def reset_operations_log() -> None:
    """Reset the OperationsLog for the current context (mainly for testing)."""
    _log_var.set(None)
