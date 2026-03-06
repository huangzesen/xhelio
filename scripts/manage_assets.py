#!/usr/bin/env python3
"""
Disk asset management CLI for xhelio.

Monitor and clean up cached data files, sessions, and SPICE kernels.

Usage:
  python scripts/manage_assets.py status                       # Overview of all assets
  python scripts/manage_assets.py status --category cdf        # One category detail
  python scripts/manage_assets.py clean cdf                    # Clean all CDF cache
  python scripts/manage_assets.py clean cdf --mission ace psp  # Clean specific missions
  python scripts/manage_assets.py clean cdf --older-than 30    # Files older than 30 days
  python scripts/manage_assets.py clean sessions --empty        # Remove 0-turn sessions
  python scripts/manage_assets.py clean sessions --older-than 7
  python scripts/manage_assets.py clean ppi
  python scripts/manage_assets.py clean spice
  python scripts/manage_assets.py clean all --dry-run          # Preview cleanup
  python scripts/manage_assets.py clean all --yes              # Skip confirmation
  python scripts/manage_assets.py status --json                # Machine-readable output
"""

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path (same pattern as scripts/check_session.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from data_ops.asset_manager import (
    AssetCategory,
    AssetOverview,
    clean_cdf_cache,
    clean_ppi_cache,
    clean_sessions,
    clean_spice_kernels,
    format_bytes,
    get_asset_overview,
    get_cdf_cache_detail,
    get_ppi_cache_detail,
    get_sessions_detail,
    get_spice_kernels_detail,
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

CATEGORY_LABELS = {
    "cdf_cache": "CDF Cache",
    "ppi_cache": "PPI Cache",
    "sessions": "Sessions",
    "spice_kernels": "SPICE Kernels",
}

CATEGORY_MAP = {
    "cdf": "cdf_cache",
    "ppi": "ppi_cache",
    "sessions": "sessions",
    "spice": "spice_kernels",
}


def _epoch_to_str(epoch: float | None) -> str:
    if epoch is None:
        return "-"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _print_overview(overview: AssetOverview) -> None:
    print(f"\nDisk Usage Overview  (scanned in {overview.scan_time_ms} ms)")
    print(f"Total: {format_bytes(overview.total_bytes)}\n")
    for cat in overview.categories:
        label = CATEGORY_LABELS.get(cat.name, cat.name)
        print(f"  {label:<16} {format_bytes(cat.total_bytes):>10}   {cat.file_count:>6} files   {cat.path}")
    print()


def _print_category(cat: AssetCategory) -> None:
    label = CATEGORY_LABELS.get(cat.name, cat.name)
    print(f"\n{label}  ({format_bytes(cat.total_bytes)}, {cat.file_count} files)")
    print(f"Path: {cat.path}\n")

    if not cat.subcategories:
        print("  (empty)")
        return

    # Column widths
    name_w = max(len(s.name) for s in cat.subcategories)
    name_w = max(name_w, 4)

    header = f"  {'Name':<{name_w}}  {'Size':>10}  {'Files':>6}  {'Oldest':>16}  {'Newest':>16}"
    if cat.name == "sessions":
        header += f"  {'Turns':>5}  {'Name'}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for s in cat.subcategories:
        line = f"  {s.name:<{name_w}}  {format_bytes(s.total_bytes):>10}  {s.file_count:>6}  {_epoch_to_str(s.oldest_mtime):>16}  {_epoch_to_str(s.newest_mtime):>16}"
        if cat.name == "sessions":
            turns = str(s.turn_count) if s.turn_count is not None else "-"
            sname = s.session_name or ""
            line += f"  {turns:>5}  {sname}"
        print(line)
    print()


def _print_result(result: dict) -> None:
    if result["dry_run"]:
        print(f"\n[DRY RUN] Would delete {result['deleted_count']} files, freeing {result['freed_human']}")
    else:
        print(f"\nDeleted {result['deleted_count']} files, freed {result['freed_human']}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    if args.category:
        cat_name = CATEGORY_MAP.get(args.category)
        if not cat_name:
            print(f"Unknown category: {args.category}", file=sys.stderr)
            print(f"Valid categories: {', '.join(CATEGORY_MAP.keys())}", file=sys.stderr)
            sys.exit(1)
        detail_fn = {
            "cdf_cache": get_cdf_cache_detail,
            "ppi_cache": get_ppi_cache_detail,
            "sessions": get_sessions_detail,
            "spice_kernels": get_spice_kernels_detail,
        }[cat_name]
        cat = detail_fn()
        if args.json:
            print(json.dumps(dataclasses.asdict(cat), indent=2))
        else:
            _print_category(cat)
    else:
        overview = get_asset_overview()
        if args.json:
            print(json.dumps(dataclasses.asdict(overview), indent=2))
        else:
            _print_overview(overview)
            # Also show subcategory summaries
            for cat in overview.categories:
                if cat.subcategories:
                    _print_category(cat)


def cmd_clean(args: argparse.Namespace) -> None:
    category = args.category
    dry_run = args.dry_run
    older_than = args.older_than
    yes = args.yes

    targets = {
        "cdf": ("CDF cache", lambda: clean_cdf_cache(
            missions=args.mission or None,
            older_than_days=older_than,
            dry_run=dry_run,
        )),
        "ppi": ("PPI cache", lambda: clean_ppi_cache(
            collections=args.mission or None,
            older_than_days=older_than,
            dry_run=dry_run,
        )),
        "sessions": ("sessions", lambda: clean_sessions(
            session_ids=args.mission or None,
            older_than_days=older_than,
            empty_only=args.empty,
            dry_run=dry_run,
        )),
        "spice": ("SPICE kernels", lambda: clean_spice_kernels(
            missions=args.mission or None,
            dry_run=dry_run,
        )),
    }

    if category == "all":
        cats_to_clean = list(targets.keys())
    elif category in targets:
        cats_to_clean = [category]
    else:
        print(f"Unknown category: {category}", file=sys.stderr)
        print(f"Valid categories: {', '.join(targets.keys())}, all", file=sys.stderr)
        sys.exit(1)

    # Confirmation
    if not dry_run and not yes:
        desc = ", ".join(targets[c][0] for c in cats_to_clean)
        filters = []
        if args.mission:
            filters.append(f"targets={args.mission}")
        if older_than:
            filters.append(f"older than {older_than} days")
        if getattr(args, "empty", False):
            filters.append("empty only")
        filter_str = f" ({', '.join(filters)})" if filters else ""
        response = input(f"Clean {desc}{filter_str}? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Cancelled.")
            return

    for cat_key in cats_to_clean:
        label, fn = targets[cat_key]
        print(f"Cleaning {label}...")
        result = fn()
        _print_result(result)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disk asset management for xhelio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # status
    sp_status = subparsers.add_parser("status", help="Show disk usage overview")
    sp_status.add_argument("--category", "-c", choices=list(CATEGORY_MAP.keys()),
                           help="Show detail for one category")
    sp_status.add_argument("--json", action="store_true", help="Output as JSON")

    # clean
    sp_clean = subparsers.add_parser("clean", help="Clean up cached data")
    sp_clean.add_argument("category", choices=["cdf", "ppi", "sessions", "spice", "all"],
                          help="Category to clean (or 'all')")
    sp_clean.add_argument("--mission", "-m", nargs="+",
                          help="Specific missions/collections/session IDs to clean")
    sp_clean.add_argument("--older-than", type=int, metavar="DAYS",
                          help="Only delete files older than N days")
    sp_clean.add_argument("--empty", action="store_true",
                          help="Only delete empty sessions (0 turns)")
    sp_clean.add_argument("--dry-run", action="store_true",
                          help="Preview what would be deleted without actually deleting")
    sp_clean.add_argument("--yes", "-y", action="store_true",
                          help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "clean":
        cmd_clean(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
