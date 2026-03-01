"""
Replay engine for operations pipelines.

Replays a recorded pipeline (from OperationsLog) to reproduce data and plots.
Creates a fresh DataStore — never touches the global singleton.

Usage (CLI):
    python scripts/replay.py                         # latest session, cached
    python scripts/replay.py SESSION_ID              # specific session, cached
    python scripts/replay.py SESSION_ID --fresh      # re-fetch from remote
    python scripts/replay.py --list                  # list sessions
    python scripts/replay.py SESSION_ID -o out.html  # save to file

Public API:
    replay_pipeline(records, ...) -> ReplayResult
    replay_session(session_id, ...) -> ReplayResult
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Add project root to path so this works as both a script and a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go

from data_ops.custom_ops import run_dataframe_creation, run_multi_source_operation
from data_ops.fetch import fetch_data
from data_ops.operations_log import OperationsLog
from data_ops.store import DataEntry, DataStore, build_source_map
from rendering.plotly_renderer import RenderResult, fill_figure_data

logger = logging.getLogger("xhelio")


@dataclass
class ReplayResult:
    """Result of replaying a pipeline."""

    store: DataStore
    figure: Optional[go.Figure] = None
    steps_completed: int = 0
    steps_total: int = 0
    errors: list[dict] = field(default_factory=list)


def replay_pipeline(
    records: list[dict],
    *,
    cache_store: Optional[DataStore] = None,
    connected_only: bool = True,
    progress_callback: Optional[Callable] = None,
) -> ReplayResult:
    """Replay a list of operation records, reproducing data and plots.

    Args:
        records: Operation records (from OperationsLog.get_pipeline() or raw).
        cache_store: Optional pre-loaded DataStore.  When provided, fetch_data
            operations copy entries from it instead of hitting remote servers.
            Custom operations are always re-executed from code.
        connected_only: If True (default), skip orphan operations (those with
            empty ``contributes_to``).  Only meaningful when records come from
            ``get_pipeline()`` which annotates each record.
        progress_callback: Optional (step, total, tool) -> None callback.

    Returns:
        ReplayResult with the fresh DataStore, final figure, and error info.
    """
    if connected_only:
        # Skip records explicitly annotated as orphans (contributes_to=[]).
        # Records without the field (raw records) are always kept.
        # If no records have non-empty contributes_to (e.g., fetch-only
        # pipelines with no renders), keep everything.
        has_connected = any(r.get("contributes_to") for r in records)
        if has_connected:
            records = [
                r for r in records
                if "contributes_to" not in r or r["contributes_to"]
            ]

    import tempfile
    store = DataStore(Path(tempfile.mkdtemp()))
    result = ReplayResult(store=store, steps_total=len(records))

    for i, rec in enumerate(records):
        tool = rec.get("tool", "")
        op_id = rec.get("id", f"step_{i}")

        if progress_callback:
            progress_callback(i + 1, len(records), tool)

        try:
            if tool == "fetch_data":
                _replay_fetch(rec, store, cache_store=cache_store)
            elif tool == "custom_operation":
                _replay_custom_op(rec, store)
            elif tool == "store_dataframe":
                _replay_store_df(rec, store)
            elif tool == "render_plotly_json":
                fig = _replay_render(rec, store)
                if fig is not None:
                    result.figure = fig
            elif tool == "manage_plot":
                logger.debug("Replay: skipping manage_plot op %s", op_id)
                result.steps_completed += 1
                continue
            else:
                logger.warning("Replay: unknown tool '%s' in op %s, skipping", tool, op_id)
                result.steps_completed += 1
                continue

            result.steps_completed += 1

        except Exception as e:
            logger.warning("Replay: op %s (%s) failed: %s", op_id, tool, e)
            result.errors.append({
                "op_id": op_id,
                "tool": tool,
                "error": str(e),
            })

    return result


def replay_session(
    session_id: str,
    *,
    session_dir: Optional[Path] = None,
    use_cache: bool = True,
    connected_only: bool = True,
    progress_callback: Optional[Callable] = None,
) -> ReplayResult:
    """Load a session's operations log and replay the pipeline.

    Args:
        session_id: Session ID (directory name under session_dir).
        session_dir: Base sessions directory. Defaults to ~/.xhelio/sessions/.
        use_cache: If True (default), load saved data from the session's
            ``data/`` directory and use it for fetch_data steps instead of
            re-fetching from remote servers.  Custom operations are always
            re-executed from code regardless of this flag.
            If False, all data is fetched fresh from remote servers.
        connected_only: If True (default), skip orphan operations that don't
            contribute to any end-state product.
        progress_callback: Optional (step, total, tool) -> None callback.

    Returns:
        ReplayResult from replaying the extracted pipeline.

    Raises:
        FileNotFoundError: If operations.json doesn't exist for the session.
    """
    if session_dir is None:
        from config import get_data_dir
        session_dir = get_data_dir() / "sessions"

    session_path = Path(session_dir) / session_id
    ops_path = session_path / "operations.json"
    if not ops_path.exists():
        raise FileNotFoundError(f"No operations.json found at {ops_path}")

    with open(ops_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    # Build OperationsLog and extract the minimal pipeline
    log = OperationsLog()
    log.load_from_records(all_records)

    # Collect all output labels from successful records
    final_labels: set[str] = set()
    for rec in all_records:
        if rec.get("status") == "success":
            final_labels.update(rec.get("outputs", []))

    pipeline = log.get_pipeline(final_labels)

    # Load cached data from session directory if requested
    cache_store: Optional[DataStore] = None
    if use_cache:
        data_dir = session_path / "data"
        if data_dir.exists() and (data_dir / "_labels.json").exists():
            cache_store = DataStore(data_dir)
            logger.info("Replay: loaded %d cached entries from %s", len(cache_store), data_dir)
        else:
            logger.warning(
                "Replay: use_cache=True but no data/ directory for session %s, "
                "falling back to fresh fetch",
                session_id,
            )

    return replay_pipeline(
        pipeline,
        cache_store=cache_store,
        connected_only=connected_only,
        progress_callback=progress_callback,
    )


def replay_state(
    session_id: str,
    render_op_id: str,
    *,
    session_dir: Optional[Path] = None,
    use_cache: bool = True,
    progress_callback: Optional[Callable] = None,
) -> ReplayResult:
    """Replay only the operations needed for a single render (product state).

    Args:
        session_id: Session ID (directory name under session_dir).
        render_op_id: The op ID of the ``render_plotly_json`` to reproduce.
        session_dir: Base sessions directory. Defaults to ~/.xhelio/sessions/.
        use_cache: If True (default), use saved session data for fetch_data steps.
        progress_callback: Optional (step, total, tool) -> None callback.

    Returns:
        ReplayResult from replaying the state-specific pipeline.

    Raises:
        FileNotFoundError: If operations.json doesn't exist for the session.
    """
    if session_dir is None:
        from config import get_data_dir
        session_dir = get_data_dir() / "sessions"

    session_path = Path(session_dir) / session_id
    ops_path = session_path / "operations.json"
    if not ops_path.exists():
        raise FileNotFoundError(f"No operations.json found at {ops_path}")

    with open(ops_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    log = OperationsLog()
    log.load_from_records(all_records)

    # Collect all output labels for the last_producer fallback
    final_labels: set[str] = set()
    for rec in all_records:
        if rec.get("status") == "success":
            final_labels.update(rec.get("outputs", []))

    pipeline = log.get_state_pipeline(render_op_id, final_labels)
    if not pipeline:
        logger.warning(
            "replay_state: no pipeline extracted for render %s", render_op_id
        )

    # Load cached data
    cache_store: Optional[DataStore] = None
    if use_cache:
        data_dir = session_path / "data"
        if data_dir.exists() and (data_dir / "_labels.json").exists():
            cache_store = DataStore(data_dir)
            logger.info("Replay: loaded %d cached entries from %s", len(cache_store), data_dir)

    return replay_pipeline(
        pipeline,
        cache_store=cache_store,
        connected_only=False,  # state pipeline is already scoped
        progress_callback=progress_callback,
    )


# ---------------------------------------------------------------------------
# Per-tool handlers
# ---------------------------------------------------------------------------


def _replay_fetch(
    rec: dict,
    store: DataStore,
    *,
    cache_store: Optional[DataStore] = None,
) -> None:
    """Replay a fetch_data operation.

    If *cache_store* is provided and contains the needed label, the entry is
    copied from cache instead of fetching from a remote server.
    """
    # Try cache first
    if cache_store is not None:
        all_cached = True
        for label in rec.get("outputs", []):
            entry = cache_store.get(label)
            if entry is not None:
                store.put(entry)
                logger.debug("Replay: loaded %s from cache", label)
            else:
                all_cached = False
        if all_cached:
            return
        # Some labels missing from cache — fall through to fresh fetch
        logger.debug(
            "Replay: partial cache miss for %s, fetching fresh",
            rec.get("id"),
        )

    args = rec["args"]
    dataset_id = args["dataset_id"]
    parameter_id = args["parameter_id"]

    # Prefer resolved time range (actual dates), fall back to raw args
    time_range_resolved = args.get("time_range_resolved")
    if isinstance(time_range_resolved, list) and len(time_range_resolved) == 2:
        time_min, time_max = time_range_resolved
    elif isinstance(time_range_resolved, str) and " to " in time_range_resolved:
        time_min, time_max = time_range_resolved.split(" to ", 1)
    elif args.get("time_start") and args.get("time_end"):
        time_min, time_max = args["time_start"], args["time_end"]
    elif args.get("time_range") and " to " in args.get("time_range", ""):
        time_min, time_max = args["time_range"].split(" to ", 1)
    else:
        time_min = args.get("time_min", "")
        time_max = args.get("time_max", "")

    fetch_result = fetch_data(dataset_id, parameter_id, time_min, time_max)

    # Store each output label
    for label in rec.get("outputs", []):
        data = fetch_result["data"]
        entry = DataEntry(
            label=label,
            data=data,
            units=fetch_result.get("units", ""),
            description=fetch_result.get("description", ""),
            source="cdf",
        )
        store.put(entry)


def _replay_custom_op(rec: dict, store: DataStore) -> None:
    """Replay a custom_operation."""
    args = rec["args"]
    code = args["code"]
    source_labels = rec.get("inputs", [])

    sources, err = build_source_map(store, source_labels)
    if err is not None:
        raise ValueError(f"Cannot build source map: {err}")

    # Build source_timeseries map from store entries
    source_ts: dict[str, bool] = {}
    for label in source_labels:
        entry = store.get(label)
        if entry is not None:
            suffix = label.rsplit(".", 1)[-1]
            prefix = "da" if entry.is_xarray else "df"
            var_name = f"{prefix}_{suffix}"
            source_ts[var_name] = entry.is_timeseries

    result_data, _warnings = run_multi_source_operation(
        sources, code, source_timeseries=source_ts,
    )

    for label in rec.get("outputs", []):
        import xarray as xr
        entry = DataEntry(
            label=label,
            data=result_data,
            units=args.get("units", ""),
            description=args.get("description", ""),
            source="computed",
            is_timeseries=not isinstance(result_data, xr.DataArray) or "time" in result_data.dims,
        )
        store.put(entry)


def _replay_store_df(rec: dict, store: DataStore) -> None:
    """Replay a store_dataframe operation."""
    args = rec["args"]
    code = args["code"]

    result_data = run_dataframe_creation(code)

    for label in rec.get("outputs", []):
        import pandas as pd
        is_ts = isinstance(result_data.index, pd.DatetimeIndex)
        entry = DataEntry(
            label=label,
            data=result_data,
            units=args.get("units", ""),
            description=args.get("description", ""),
            source="created",
            is_timeseries=is_ts,
        )
        store.put(entry)


def _replay_render(rec: dict, store: DataStore) -> Optional[go.Figure]:
    """Replay a render_plotly_json operation. Returns the Figure or None."""
    args = rec["args"]
    fig_json = args.get("figure_json")
    if fig_json is None:
        raise ValueError("render_plotly_json record missing 'figure_json' in args")

    # Build entry map from all labels referenced in the figure's data traces
    from data_ops.store import resolve_entry

    entry_map: dict[str, DataEntry] = {}
    for trace in fig_json.get("data", []):
        label = trace.get("data_label")
        if label is None:
            continue
        entry, _ = resolve_entry(store, label)
        if entry is None:
            raise ValueError(f"data_label '{label}' not found in replay store")
        entry_map[label] = entry

    render_result: RenderResult = fill_figure_data(fig_json, entry_map)
    return render_result.figure


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_sessions() -> None:
    """Print sessions that have operations.json files."""
    from config import get_data_dir

    sessions_dir = get_data_dir() / "sessions"
    if not sessions_dir.exists():
        print("No sessions directory found.")
        return

    rows: list[tuple[str, int, int, bool]] = []
    for d in sorted(sessions_dir.iterdir()):
        ops_path = d / "operations.json"
        if not ops_path.exists():
            continue
        with open(ops_path, "r", encoding="utf-8") as f:
            ops = json.load(f)
        n_ops = len(ops)
        n_render = sum(1 for op in ops if op.get("tool") == "render_plotly_json")
        has_data = (d / "data" / "_labels.json").exists()
        rows.append((d.name, n_ops, n_render, has_data))

    if not rows:
        print("No sessions with operations.json found.")
        return

    print(f"{'Session ID':<40} {'Ops':>4} {'Renders':>7} {'Cached':>6}")
    print("-" * 60)
    for sid, n_ops, n_render, has_data in rows:
        cached = "yes" if has_data else "no"
        print(f"{sid:<40} {n_ops:>4} {n_render:>7} {cached:>6}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay a session pipeline and output the final plot.",
    )
    parser.add_argument(
        "session_id", nargs="?", default=None,
        help="Session ID to replay.  Omit to replay the latest session.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List sessions with operations logs and exit.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file (html/png/pdf).  Default: opens in browser.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Fetch data fresh from remote servers instead of using cached "
             "session data.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Replay all pipeline operations including orphans (by default "
             "only operations connected to the final plot are replayed).",
    )
    parser.add_argument(
        "--state", default=None, metavar="OP_ID",
        help="Replay only the operations for a specific render state "
             "(e.g. op_004). Uses get_state_pipeline() to extract the "
             "minimal chain for that render.",
    )
    args = parser.parse_args()

    if args.list:
        _list_sessions()
        return

    from config import get_data_dir

    sessions_dir = get_data_dir() / "sessions"

    # Default to latest session with operations
    session_id = args.session_id
    if session_id is None:
        candidates = []
        for d in sorted(sessions_dir.iterdir(), reverse=True):
            if (d / "operations.json").exists():
                candidates.append(d.name)
        if not candidates:
            print("No sessions with operations.json found.")
            return
        session_id = candidates[0]
        print(f"Using latest session: {session_id}")

    use_cache = not args.fresh
    connected_only = not args.all

    def progress(step: int, total: int, tool: str) -> None:
        mode = "cache" if use_cache else "fresh"
        print(f"  [{step}/{total}] {tool} ({mode})")

    if args.state:
        print(
            f"Replaying state {args.state} from session {session_id} "
            f"({'cached' if use_cache else 'fresh'})..."
        )
        result = replay_state(
            session_id,
            args.state,
            session_dir=sessions_dir,
            use_cache=use_cache,
            progress_callback=progress,
        )
    else:
        scope = "connected" if connected_only else "all"
        print(f"Replaying session {session_id} ({'cached' if use_cache else 'fresh'}, {scope})...")
        result = replay_session(
            session_id,
            session_dir=sessions_dir,
            use_cache=use_cache,
            connected_only=connected_only,
            progress_callback=progress,
        )

    print(f"\nCompleted: {result.steps_completed}/{result.steps_total} steps")
    if result.errors:
        print(f"Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  {err['op_id']} ({err['tool']}): {err['error']}")

    if result.figure is None:
        print("No figure produced.")
        return

    output = args.output
    if output is None:
        # Open in browser
        result.figure.show()
    elif output.endswith(".html"):
        result.figure.write_html(output, include_plotlyjs=True)
        print(f"Saved to {output}")
    elif output.endswith(".png"):
        result.figure.write_image(output, scale=2)
        print(f"Saved to {output}")
    elif output.endswith(".pdf"):
        result.figure.write_image(output)
        print(f"Saved to {output}")
    else:
        print(f"Unknown output format: {output}")
        print("Supported: .html, .png, .pdf")


if __name__ == "__main__":
    main()
