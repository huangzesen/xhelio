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

# Distinct colors — avoid similar blues
COLORS = {
    "Python": "#3572A5",
    "TypeScript": "#F0C040",
    "TSX": "#61DAFB",
    "CSS": "#A855F7",
    "YAML": "#EF4444",
    SKILLS_LANG: "#22C55E",
}

# Stable ordering for consistent stacking (largest at bottom)
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


def _format_date_label(iso_hour: str) -> str:
    """Convert '2026-02-14T09' to 'Feb 14'."""
    try:
        date_part = iso_hour.split("T")[0]  # '2026-02-14'
        parts = date_part.split("-")
        months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{months[int(parts[1])]} {int(parts[2])}"
    except (IndexError, ValueError):
        return iso_hour


def _aggregate_deltas_6h(
    data: list[dict], series: dict[str, list[int]],
) -> tuple[list[str], dict[str, list[int]]]:
    """Aggregate per-hour LOC into 6-hour delta buckets.

    Returns (bucket_labels, {lang: [delta_per_bucket, ...]}).
    Each bucket spans 6 consecutive hourly snapshots.  The delta is computed
    as LOC_at_end_of_bucket minus LOC_at_start_of_bucket.
    """
    n = len(data)
    bucket_size = 6
    bucket_labels: list[str] = []
    agg_deltas: dict[str, list[int]] = {lang: [] for lang in series}

    for start in range(0, n, bucket_size):
        end = min(start + bucket_size, n) - 1  # last index in this bucket
        label = data[start]["date"]
        bucket_labels.append(label)
        for lang, vals in series.items():
            prev = vals[start - 1] if start > 0 else 0
            agg_deltas[lang].append(vals[end] - prev)

    return bucket_labels, agg_deltas


def build_figure(data: list[dict]) -> go.Figure:
    """Build stacked area (hourly) + stacked bar (6-hour delta) figure."""
    dates = [d["date"] for d in data]
    n = len(data)
    x_idx = list(range(n))

    # Build tick values/labels — use readable "Mon DD" format, ~12 ticks
    tick_step = max(1, n // 12)
    tick_vals = list(range(0, n, tick_step))
    if tick_vals[-1] != n - 1:
        tick_vals.append(n - 1)
    tick_labels = [_format_date_label(dates[i]) for i in tick_vals]

    # Collect per-language series
    series: dict[str, list[int]] = {}
    for lang in LANG_ORDER:
        vals = [d.get(lang, 0) for d in data]
        if any(v > 0 for v in vals):
            series[lang] = vals

    # Aggregate deltas into 6-hour buckets
    bucket_labels, deltas_6h = _aggregate_deltas_6h(data, series)
    bucket_x = list(range(len(bucket_labels)))

    # Tick labels for the bottom panel — match top panel density
    nb = len(bucket_labels)
    btick_step = max(1, nb // 12)
    btick_vals = list(range(0, nb, btick_step))
    if btick_vals[-1] != nb - 1:
        btick_vals.append(nb - 1)
    btick_labels = [_format_date_label(bucket_labels[i]) for i in btick_vals]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        row_heights=[0.65, 0.35],
    )

    # Row 1: stacked area (hourly) with semi-transparent fill
    for lang in LANG_ORDER:
        if lang not in series:
            continue
        color = COLORS.get(lang, "#888")
        # Make fill semi-transparent
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_rgba = f"rgba({r},{g},{b},0.65)"
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=series[lang],
                name=lang,
                mode="lines",
                stackgroup="loc",
                line=dict(width=0.5, color=color),
                fillcolor=fill_rgba,
                customdata=[dates[i] for i in range(n)],
                hovertemplate="%{customdata}: %{y:,}<extra>" + lang + "</extra>",
            ),
            row=1,
            col=1,
        )

    # Row 2: stacked bar (6-hour delta) with semi-transparent bars
    for lang in LANG_ORDER:
        if lang not in deltas_6h:
            continue
        color = COLORS.get(lang, "#888")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        bar_rgba = f"rgba({r},{g},{b},0.75)"
        fig.add_trace(
            go.Bar(
                x=bucket_x,
                y=deltas_6h[lang],
                name=lang,
                marker_color=bar_rgba,
                marker_line_width=0,
                showlegend=False,
                customdata=bucket_labels,
                hovertemplate="%{customdata}: %{y:+,}<extra>" + lang + "</extra>",
            ),
            row=2,
            col=1,
        )

    # 3-day rolling average of total delta (window=12 buckets of 6h = 3 days)
    total_deltas = [
        sum(deltas_6h[lang][i] for lang in deltas_6h)
        for i in range(len(bucket_x))
    ]
    window = 12  # 3 days / 6 hours
    rolling_avg = []
    for i in range(len(total_deltas)):
        start = max(0, i - window + 1)
        rolling_avg.append(sum(total_deltas[start:i + 1]) / (i - start + 1))
    fig.add_trace(
        go.Scatter(
            x=bucket_x,
            y=rolling_avg,
            name="3-day avg",
            mode="lines",
            line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dot"),
            showlegend=True,
            hovertemplate="%{y:+,.0f}<extra>3-day avg</extra>",
        ),
        row=2,
        col=1,
    )

    total_loc = sum(data[-1].get(lang, 0) for lang in LANG_ORDER) if data else 0
    start_label = _format_date_label(dates[0]) if dates else ""
    end_label = _format_date_label(dates[-1]) if dates else ""

    fig.update_layout(
        title=dict(
            text=(
                f"<b>xhelio</b> — {total_loc:,} lines of code"
                f"<br><span style='font-size:13px;color:#666'>"
                f"{start_label} – {end_label} 2026 · {n} snapshots</span>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        template="plotly_white",
        barmode="relative",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        hovermode="x unified",
        width=1200,
        height=750,
        margin=dict(t=80, b=60, l=60, r=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
    )

    # Top panel axes
    fig.update_xaxes(
        tickvals=tick_vals,
        ticktext=tick_labels,
        tickangle=0,
        tickfont=dict(size=10),
        showgrid=False,
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Lines of Code",
        title_font=dict(size=11),
        tickfont=dict(size=10),
        gridcolor="rgba(0,0,0,0.06)",
        row=1, col=1,
    )

    # Bottom panel axes
    fig.update_xaxes(
        tickvals=btick_vals,
        ticktext=btick_labels,
        tickangle=0,
        tickfont=dict(size=10),
        showgrid=False,
        title_text="",
        row=2, col=1,
    )
    fig.update_yaxes(
        title_text="6h Delta",
        title_font=dict(size=11),
        tickfont=dict(size=10),
        gridcolor="rgba(0,0,0,0.06)",
        dtick=1000,
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.15)",
        zerolinewidth=1,
        row=2, col=1,
    )

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
