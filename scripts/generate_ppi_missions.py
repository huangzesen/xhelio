#!/usr/bin/env python3
"""
Generate PPI mission JSON files from the PDS PPI Metadex.

Thin wrapper around knowledge.bootstrap.populate_ppi_missions().

Usage:
    python scripts/generate_ppi_missions.py                   # All PPI missions
    python scripts/generate_ppi_missions.py --mission cassini  # One mission
    python scripts/generate_ppi_missions.py --list             # Show missions without generating
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for knowledge imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)

from knowledge.mission_prefixes import match_dataset_to_mission


def _fetch_and_list_missions():
    """Fetch Metadex and list available PPI missions."""
    from knowledge.metadex_client import fetch_all_ppi_collections, metadex_id_to_dataset_id

    collections = fetch_all_ppi_collections()
    groups: dict[str, list] = {}
    for coll in collections:
        dataset_id = metadex_id_to_dataset_id(coll["id"], coll["archive_type"])
        stem, _ = match_dataset_to_mission(dataset_id)
        if stem:
            groups.setdefault(stem, []).append(coll)

    print(f"\nPPI missions ({len(groups)}) — {len(collections)} total collections:")
    for stem in sorted(groups):
        pds3 = sum(1 for c in groups[stem] if c["archive_type"] == 3)
        pds4 = sum(1 for c in groups[stem] if c["archive_type"] == 4)
        parts = []
        if pds3:
            parts.append(f"{pds3} PDS3")
        if pds4:
            parts.append(f"{pds4} PDS4")
        print(f"  {stem}: {len(groups[stem])} datasets ({', '.join(parts)})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PPI mission JSONs from Metadex"
    )
    parser.add_argument(
        "--mission", type=str, default=None,
        help="Generate only one mission (e.g., 'cassini')"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List missions from Metadex without generating"
    )
    parser.add_argument(
        "--metadata", action="store_true",
        help="Fetch parameter metadata from PDS label files "
             "(can be run independently of mission JSON generation)"
    )
    args = parser.parse_args()

    if args.list:
        _fetch_and_list_missions()
        return

    # Set up logging so bootstrap messages are visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    only_stems = None
    if args.mission:
        only_stems = {args.mission.lower().replace("-", "_")}

    if args.metadata:
        from knowledge.bootstrap import populate_ppi_metadata
        populate_ppi_metadata(only_stems=only_stems)
        return

    from knowledge.bootstrap import populate_ppi_missions
    populate_ppi_missions(only_stems=only_stems)


if __name__ == "__main__":
    main()
