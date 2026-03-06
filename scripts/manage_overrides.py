#!/usr/bin/env python3
"""
Manage mission override files.

Override files live in ``{data_dir}/mission_overrides/`` and let the agent
persist learned knowledge (caveats, notes, corrections) separately from
the auto-generated mission JSON.  Overrides are sparse patch files that
are deep-merged at load time.

Usage:
    python scripts/manage_overrides.py list                # Show all override files
    python scripts/manage_overrides.py show <mission>      # Pretty-print an override
    python scripts/manage_overrides.py validate <mission>  # Check for issues in override file
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for knowledge imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.mission_loader import (
    _get_overrides_dir,
    _load_override,
    _CDAWEB_DIR,
)


def _load_base(stem: str) -> dict | None:
    """Load the raw base mission JSON (no overrides applied)."""
    filepath = _CDAWEB_DIR / f"{stem}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _cache_key(mission_id: str) -> str:
    return mission_id.lower().replace("-", "_")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    """List all override files with a brief summary."""
    overrides_dir = _get_overrides_dir()
    if not overrides_dir.exists():
        print(f"No overrides directory: {overrides_dir}")
        return

    # Mission-level overrides
    files = sorted(overrides_dir.glob("*.json"))
    # Dataset-level overrides (in subdirectories)
    ds_files = sorted(overrides_dir.glob("*/*.json"))

    if not files and not ds_files:
        print(f"No override files in {overrides_dir}")
        return

    print(f"Override directory: {overrides_dir}\n")

    if files:
        print("Mission-level overrides:")
        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                keys = list(data.keys())
                print(f"  {filepath.stem}.json  — keys: {', '.join(keys)}")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  {filepath.stem}.json  — ERROR: {exc}")

    if ds_files:
        if files:
            print()
        print("Dataset-level overrides:")
        for filepath in ds_files:
            stem = filepath.parent.name
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                keys = list(data.keys())
                print(f"  {stem}/{filepath.stem}.json  — keys: {', '.join(keys)}")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  {stem}/{filepath.stem}.json  — ERROR: {exc}")


def cmd_show(args: argparse.Namespace) -> None:
    """Pretty-print an override file."""
    stem = _cache_key(args.mission)
    override = _load_override(stem)
    if override is None:
        overrides_dir = _get_overrides_dir()
        print(f"No override file for {args.mission!r} at {overrides_dir / f'{stem}.json'}")
        return
    print(json.dumps(override, indent=2))


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate an override file against the base mission JSON."""
    stem = _cache_key(args.mission)
    base = _load_base(stem)
    if base is None:
        print(f"No base mission file found for {args.mission!r}")
        sys.exit(1)

    override = _load_override(stem)
    if override is None:
        overrides_dir = _get_overrides_dir()
        print(f"No override file for {args.mission!r} at {overrides_dir / f'{stem}.json'}")
        sys.exit(1)

    issues: list[str] = []

    # Check that the override is a valid JSON object (already guaranteed
    # by _load_override, but check structure)
    if not isinstance(override, dict):
        issues.append("Override is not a JSON object")
    else:
        # Warn about overriding instruments that don't exist in base
        if "instruments" in override and isinstance(override["instruments"], dict):
            base_instruments = base.get("instruments", {})
            for inst_id in override["instruments"]:
                if inst_id not in base_instruments:
                    issues.append(
                        f"Instrument {inst_id!r} not found in base mission "
                        f"(will be added as new)"
                    )

    if issues:
        print(f"Validation notes for {args.mission!r}:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"Override for {args.mission!r} is valid.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage mission override files",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="Show all override files with summary")

    # show
    p_show = sub.add_parser("show", help="Pretty-print override content")
    p_show.add_argument("mission", help="Mission ID (e.g., PSP, ACE)")

    # validate
    p_validate = sub.add_parser(
        "validate", help="Check override file for issues"
    )
    p_validate.add_argument("mission", help="Mission ID")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "show": cmd_show,
        "validate": cmd_validate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
