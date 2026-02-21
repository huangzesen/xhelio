#!/usr/bin/env python3
"""Extract learned annotations from metadata cache to mission_overrides.

Scans all knowledge/missions/*/metadata/*.json and ppi/*/metadata/*.json
files for learned fields (_validated, _note) and:
1. Writes dataset overrides to ~/.xhelio/mission_overrides/{stem}/{dataset_id}.json
2. Strips those fields from the original metadata cache files

This is a one-time migration script to separate auto-generated metadata
from learned annotations.

Usage:
    python scripts/extract_annotations.py          # dry-run (preview)
    python scripts/extract_annotations.py --apply  # actually write changes
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_data_dir

MISSIONS_DIR = Path(__file__).parent.parent / "knowledge" / "missions"


def _scan_metadata_files() -> list[tuple[Path, str, str]]:
    """Find all metadata cache files with annotations.

    Returns:
        List of (cache_path, mission_stem, dataset_id) tuples.
    """
    results = []

    for source_dir in [MISSIONS_DIR, MISSIONS_DIR / "ppi"]:
        if not source_dir.exists():
            continue
        for mission_dir in sorted(source_dir.iterdir()):
            if not mission_dir.is_dir():
                continue
            # Skip ppi dir when scanning top-level (we handle it separately)
            if source_dir == MISSIONS_DIR and mission_dir.name == "ppi":
                continue
            metadata_dir = mission_dir / "metadata"
            if not metadata_dir.exists():
                continue
            for cache_file in sorted(metadata_dir.glob("*.json")):
                if cache_file.name.startswith("_"):
                    continue
                results.append((cache_file, mission_dir.name, cache_file.stem))

    return results


def _extract_annotations(info: dict) -> dict | None:
    """Extract learned fields from a metadata info dict.

    Returns:
        A sparse override dict, or None if no annotations found.
    """
    override = {}

    # Top-level _validated
    if info.get("_validated"):
        override["_validated"] = True

    # Parameters with _note
    params_with_notes = {}
    for param in info.get("parameters", []):
        name = param.get("name", "")
        if param.get("_note"):
            params_with_notes[name] = {"_note": param["_note"]}

    if params_with_notes:
        override["parameters_annotations"] = params_with_notes

    return override if override else None


def _strip_annotations(info: dict) -> bool:
    """Remove learned fields from a metadata info dict in-place.

    Returns:
        True if any fields were removed.
    """
    changed = False

    if "_validated" in info:
        del info["_validated"]
        changed = True

    for param in info.get("parameters", []):
        if "_note" in param:
            del param["_note"]
            changed = True

    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually write changes (default is dry-run)")
    args = parser.parse_args()

    overrides_dir = get_data_dir() / "mission_overrides"

    files = _scan_metadata_files()
    print(f"Scanning {len(files)} metadata cache files...")

    extracted = 0
    cleaned = 0

    for cache_path, stem, dataset_id in files:
        try:
            info = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  SKIP {cache_path}: {exc}")
            continue

        override = _extract_annotations(info)
        if override is None:
            continue

        ds_dir = overrides_dir / stem
        override_path = ds_dir / f"{dataset_id}.json"

        print(f"  {stem}/{dataset_id}:")
        if override.get("_validated"):
            print(f"    _validated: true")
        notes = override.get("parameters_annotations", {})
        for pname, pannot in notes.items():
            print(f"    {pname}: {pannot.get('_note', '')}")

        if args.apply:
            # Write override (merge with existing if any)
            ds_dir.mkdir(parents=True, exist_ok=True)
            existing = {}
            if override_path.exists():
                try:
                    existing = json.loads(override_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    existing = {}

            # Deep merge override into existing
            for key, value in override.items():
                if isinstance(existing.get(key), dict) and isinstance(value, dict):
                    existing[key].update(value)
                else:
                    existing[key] = value

            override_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            extracted += 1

            # Clean the cache file
            if _strip_annotations(info):
                cache_path.write_text(
                    json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                cleaned += 1

    print()
    if args.apply:
        print(f"Done: {extracted} overrides written, {cleaned} cache files cleaned.")
    else:
        print(f"Dry run: {extracted + cleaned if extracted or cleaned else 'no'} "
              f"files would be modified. Run with --apply to write changes.")
        if override:
            print(f"Override dir: {overrides_dir}")


if __name__ == "__main__":
    main()
