#!/usr/bin/env python3
"""LOC history — count lines of code at every commit on master and chart the result.

Uses git plumbing (ls-tree + cat-file --batch) to count lines without checkout.
Caches results per commit hash in .loc_history_cache.json for fast re-runs.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: plotly is required. Install with: pip install plotly", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXTENSIONS = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TSX",
    ".css": "CSS",
    ".yaml": "YAML",
    ".yml": "YAML",
}

# .claude/skills/**/*.md counted as "Skills (MD)"
SKILLS_PREFIX = ".claude/skills/"
SKILLS_EXT = ".md"
SKILLS_LANG = "Skills (MD)"

EXCLUDE_DIRS = {
    "venv/",
    "node_modules/",
    "__pycache__/",
    "dist/",
    ".pytest_cache/",
    "cdaweb_data/",
    "ppi_data/",
    "spikes/",
}

# GitHub-style colors
COLORS = {
    "Python": "#3572A5",
    "TypeScript": "#3178C6",
    "TSX": "#61DAFB",
    "CSS": "#563D7C",
    "YAML": "#CB171E",
    SKILLS_LANG: "#83CD29",
}

# Stable ordering for consistent stacking
LANG_ORDER = ["Python", "TypeScript", "TSX", "CSS", "YAML", SKILLS_LANG]

CACHE_FILE = ".loc_history_cache.json"
MAX_WORKERS = 4

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def get_repo_root() -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )


def get_commits(repo_root: str) -> list[tuple[str, str]]:
    """Return [(commit_hash, date_str), ...] — one per hour on master, chronological.

    Uses ISO 8601 hourly buckets (YYYY-MM-DDTHH).  For each hour that has
    commits, we keep only the *last* commit (the state at the end of that hour).
    """
    raw = subprocess.check_output(
        ["git", "log", "master", "--first-parent",
         "--format=%H %ad",
         "--date=format:%Y-%m-%dT%H",
         "--reverse"],
        cwd=repo_root,
        text=True,
    ).strip()
    if not raw:
        return []

    # Walk chronologically; for each hourly bucket keep the last commit
    bucket: dict[str, tuple[str, str]] = {}  # hour_key -> (hash, hour_key)
    order: list[str] = []
    for line in raw.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        commit_hash, hour_key = parts
        if hour_key not in bucket:
            order.append(hour_key)
        bucket[hour_key] = (commit_hash, hour_key)

    return [bucket[k] for k in order]


def _classify_path(path: str) -> str | None:
    """Return language name for a tracked path, or None to skip."""
    # Check exclusions
    for excl in EXCLUDE_DIRS:
        if path.startswith(excl) or f"/{excl}" in path:
            return None

    # Skills markdown
    if path.startswith(SKILLS_PREFIX) and path.endswith(SKILLS_EXT):
        return SKILLS_LANG

    ext = os.path.splitext(path)[1]
    return EXTENSIONS.get(ext)


def count_loc_at_commit(commit: str, repo_root: str) -> dict[str, int]:
    """Count LOC per language at a commit using git plumbing."""
    # Get tree listing: mode type hash\tpath
    tree_raw = subprocess.check_output(
        ["git", "ls-tree", "-r", commit],
        cwd=repo_root,
        text=True,
    )

    # Classify paths and collect unique blob hashes
    blob_to_langs: dict[str, list[str]] = defaultdict(list)
    for line in tree_raw.splitlines():
        # format: <mode> <type> <hash>\t<path>
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if len(parts) < 3:
            continue
        blob_hash = parts[2]
        obj_type = parts[1]
        if obj_type != "blob":
            continue

        lang = _classify_path(path)
        if lang:
            blob_to_langs[blob_hash].append(lang)

    if not blob_to_langs:
        return {}

    # Batch-count lines via git cat-file --batch
    # Send blob hashes, read sizes, count newlines
    input_data = "\n".join(blob_to_langs.keys()) + "\n"
    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    stdout, _ = proc.communicate(input_data.encode())

    counts: dict[str, int] = defaultdict(int)
    pos = 0
    data = stdout
    blob_keys = list(blob_to_langs.keys())
    idx = 0

    while pos < len(data) and idx < len(blob_keys):
        # Read header line: "<hash> <type> <size>\n"
        nl = data.index(b"\n", pos)
        header = data[pos : nl].decode("ascii", errors="replace")
        pos = nl + 1

        header_parts = header.split()
        if len(header_parts) < 3 or header_parts[1] == "missing":
            idx += 1
            continue

        blob_hash = header_parts[0]
        size = int(header_parts[2])

        # Read blob content
        blob_content = data[pos : pos + size]
        pos += size + 1  # +1 for trailing newline after blob

        # Count lines (number of newlines; add 1 if file doesn't end with newline
        # and has content)
        if size == 0:
            line_count = 0
        else:
            line_count = blob_content.count(b"\n")
            if not blob_content.endswith(b"\n"):
                line_count += 1

        # Attribute lines to all languages this blob maps to
        # (a blob can map to multiple files of different languages if hashes collide,
        # which is extremely rare, but we handle it correctly)
        for lang in blob_to_langs[blob_hash]:
            counts[lang] += line_count

        idx += 1

    return dict(counts)


# ---------------------------------------------------------------------------
# Parallel runner with cache
# ---------------------------------------------------------------------------


def load_cache(repo_root: str) -> dict:
    cache_path = os.path.join(repo_root, CACHE_FILE)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(repo_root: str, cache: dict) -> None:
    cache_path = os.path.join(repo_root, CACHE_FILE)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def count_loc_parallel(
    commits: list[tuple[str, str]],
    repo_root: str,
    use_cache: bool = True,
) -> list[dict]:
    """Count LOC for each commit, returning [{date, commit, langs...}, ...]."""
    cache = load_cache(repo_root) if use_cache else {}
    results = [None] * len(commits)
    to_compute: list[tuple[int, str, str]] = []

    for i, (commit, date_str) in enumerate(commits):
        if commit in cache:
            results[i] = {"date": date_str, "commit": commit, **cache[commit]}
        else:
            to_compute.append((i, commit, date_str))

    if not to_compute:
        print("All commits cached — nothing to compute.")
        return results

    cached_count = len(commits) - len(to_compute)
    print(
        f"Computing LOC for {len(to_compute)} commits "
        f"({cached_count} cached)..."
    )

    def _worker(item):
        idx, commit, date_str = item
        loc = count_loc_at_commit(commit, repo_root)
        return idx, commit, date_str, loc

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, item): item for item in to_compute}
        for future in as_completed(futures):
            idx, commit, date_str, loc = future.result()
            results[idx] = {"date": date_str, "commit": commit, **loc}
            cache[commit] = loc
            done += 1
            if done % 50 == 0 or done == len(to_compute):
                print(f"  {done}/{len(to_compute)} done")

    if use_cache:
        save_cache(repo_root, cache)

    return results


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------


def build_figure(data: list[dict]) -> go.Figure:
    """Build stacked area (cumulative) + stacked bar (per-commit delta) figure."""
    dates = [d["date"] for d in data]
    x_idx = list(range(len(data)))

    # Build tick values/labels — show date at roughly 10-15 evenly spaced points
    n = len(data)
    tick_step = max(1, n // 12)
    tick_vals = list(range(0, n, tick_step))
    if tick_vals[-1] != n - 1:
        tick_vals.append(n - 1)
    tick_labels = [dates[i] for i in tick_vals]

    # Custom hover text: "YYYY-MM-DDTHH"
    hover_dates = [dates[i] for i in range(n)]

    # Collect per-language series
    series: dict[str, list[int]] = {}
    for lang in LANG_ORDER:
        vals = [d.get(lang, 0) for d in data]
        if any(v > 0 for v in vals):
            series[lang] = vals

    # Compute per-commit deltas
    deltas: dict[str, list[int]] = {}
    for lang, vals in series.items():
        d = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        deltas[lang] = d

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
        subplot_titles=("Lines of Code", "Hourly Net Change"),
    )

    # Row 1: stacked area
    for lang in LANG_ORDER:
        if lang not in series:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=series[lang],
                name=lang,
                mode="lines",
                stackgroup="loc",
                line=dict(width=0.5, color=COLORS.get(lang)),
                fillcolor=COLORS.get(lang),
                customdata=hover_dates,
                hovertemplate="%{customdata}: %{y:,} lines<extra>" + lang + "</extra>",
            ),
            row=1,
            col=1,
        )

    # Row 2: stacked bar (delta)
    for lang in LANG_ORDER:
        if lang not in deltas:
            continue
        fig.add_trace(
            go.Bar(
                x=x_idx,
                y=deltas[lang],
                name=lang,
                marker_color=COLORS.get(lang),
                showlegend=False,
                customdata=hover_dates,
                hovertemplate="%{customdata}: %{y:+,} lines<extra>" + lang + "</extra>",
            ),
            row=2,
            col=1,
        )

    total_loc = sum(data[-1].get(lang, 0) for lang in LANG_ORDER) if data else 0
    date_range = f"{dates[0]} — {dates[-1]}" if dates else ""

    fig.update_layout(
        title=dict(
            text=(
                f"xhelio LOC History ({date_range}) — "
                f"{total_loc:,} total lines, {n} hourly snapshots"
            ),
            x=0.5,
        ),
        template="plotly_white",
        barmode="relative",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
        width=1100,
        height=700,
        margin=dict(t=100),
    )

    # Apply date tick labels on both axes
    for row in [1, 2]:
        fig.update_xaxes(
            tickvals=tick_vals,
            ticktext=tick_labels,
            row=row,
            col=1,
        )

    fig.update_yaxes(title_text="Lines", row=1, col=1)
    fig.update_yaxes(title_text="Delta", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LOC history chart for xhelio")
    parser.add_argument(
        "--png", action="store_true", help="Save HTML + PNG instead of showing"
    )
    parser.add_argument("-o", "--output", help="Output file path (PNG)")
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore cached results"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print raw data as JSON and exit"
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    commits = get_commits(repo_root)
    if not commits:
        print("No commits found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(commits)} hourly snapshots on master ({commits[0][1]} — {commits[-1][1]})")

    data = count_loc_parallel(commits, repo_root, use_cache=not args.no_cache)

    if args.json:
        json.dump(data, sys.stdout, indent=2)
        print()
        return

    fig = build_figure(data)

    if args.png or args.output:
        png_path = args.output or os.path.join(repo_root, "loc_history.png")
        html_path = os.path.join(repo_root, "loc_history.html")

        fig.write_html(html_path)
        print(f"Saved HTML: {html_path}")

        fig.write_image(png_path, scale=2)
        print(f"Saved PNG:  {png_path}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
