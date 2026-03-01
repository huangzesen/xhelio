#!/usr/bin/env python3
"""Batch validation of PPI archive fetch across all missions.

For each dataset in each PPI mission JSON, runs the fetch pipeline
through file discovery.  Optionally downloads and parses the first
file per dataset (``--deep``).

Usage:
    # All PPI missions (discovery only)
    venv/bin/python scripts/validate_ppi_archive.py

    # Deep validation (download + parse first file)
    venv/bin/python scripts/validate_ppi_archive.py --deep

    # Filter by mission or dataset
    venv/bin/python scripts/validate_ppi_archive.py --mission insight
    venv/bin/python scripts/validate_ppi_archive.py --dataset "urn:nasa:pds:insight-ifg-mars:data-ifg-calibrated"
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PPI_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "missions" / "ppi"


def _collect_datasets(mission_filter: str | None, dataset_filter: str | None):
    """Collect (mission_stem, dataset_id) pairs from PPI JSONs."""
    datasets = []
    for fp in sorted(_PPI_DIR.glob("*.json")):
        stem = fp.stem
        if mission_filter and mission_filter.lower() not in stem.lower():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for inst_key, inst_data in data.get("instruments", {}).items():
            for ds_id in inst_data.get("datasets", {}):
                if dataset_filter and dataset_filter not in ds_id:
                    continue
                datasets.append((stem, ds_id))
    return datasets


def validate(
    mission_filter: str | None,
    dataset_filter: str | None,
    deep: bool,
):
    datasets = _collect_datasets(mission_filter, dataset_filter)
    print(f"Validating {len(datasets)} dataset(s)...\n")

    results = {"ok": [], "resolution_failed": [], "no_files": [], "parse_error": []}

    for i, (stem, ds_id) in enumerate(datasets, 1):
        status = "ok"
        detail = ""
        try:
            from data_ops.fetch_ppi_archive import _resolve_collection_url
            collection_url = _resolve_collection_url(ds_id)
        except Exception as e:
            status = "resolution_failed"
            detail = str(e)[:80]
            results[status].append((stem, ds_id, detail))
            print(f"  [{i}/{len(datasets)}] {status:20s} {ds_id}")
            continue

        try:
            from data_ops.fetch_ppi_archive import _discover_data_files
            # Use a broad time range to check if any files exist at all
            pairs = _discover_data_files(collection_url, "2000-01-01", "2030-01-01")
            if not pairs:
                status = "no_files"
                detail = f"url={collection_url}"
            else:
                detail = f"{len(pairs)} file(s)"
        except Exception as e:
            status = "no_files"
            detail = str(e)[:80]

        if status != "ok":
            results[status].append((stem, ds_id, detail))
            print(f"  [{i}/{len(datasets)}] {status:20s} {ds_id}  ({detail})")
            continue

        # Deep: download + parse first file
        if deep and pairs:
            try:
                from data_ops.fetch_ppi_archive import (
                    _download_file, _parse_xml_label, _read_table,
                )
                data_url, label_url = pairs[0]
                local_data = _download_file(data_url)
                local_label = _download_file(label_url)
                label = _parse_xml_label(
                    local_label.read_text(encoding="utf-8", errors="replace")
                )
                # Try reading with the first non-time field
                field_names = [
                    f["name"] for f in label.get("fields", [])
                    if f["name"].lower() not in ("time", "epoch", "scet")
                ]
                if field_names:
                    df = _read_table(local_data, label, field_names[0])
                    if df is None or len(df) == 0:
                        status = "parse_error"
                        detail = "read_table returned None/empty"
                    else:
                        detail += f", {len(df)} rows"
                else:
                    status = "parse_error"
                    detail = "no non-time fields in label"
            except Exception as e:
                status = "parse_error"
                detail = str(e)[:80]

        results[status].append((stem, ds_id, detail))
        print(f"  [{i}/{len(datasets)}] {status:20s} {ds_id}  ({detail})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for cat in ("ok", "resolution_failed", "no_files", "parse_error"):
        count = len(results[cat])
        if count:
            print(f"  {cat:20s}: {count}")
    total = len(datasets)
    ok_count = len(results["ok"])
    print(f"\n  Total: {total}, OK: {ok_count}, Failed: {total - ok_count}")

    if results["resolution_failed"]:
        print(f"\n--- Resolution failures ---")
        for stem, ds_id, detail in results["resolution_failed"]:
            print(f"  [{stem}] {ds_id}: {detail}")

    if results["no_files"]:
        print(f"\n--- No files found ---")
        for stem, ds_id, detail in results["no_files"]:
            print(f"  [{stem}] {ds_id}: {detail}")

    if results["parse_error"]:
        print(f"\n--- Parse errors ---")
        for stem, ds_id, detail in results["parse_error"]:
            print(f"  [{stem}] {ds_id}: {detail}")

    return len(results["ok"]) == total


def main():
    parser = argparse.ArgumentParser(
        description="Validate PPI archive fetch across all missions.",
    )
    parser.add_argument("--mission", help="Filter by mission stem (e.g., 'insight')")
    parser.add_argument("--dataset", help="Filter by dataset ID substring")
    parser.add_argument(
        "--deep", action="store_true",
        help="Download and parse first file per dataset",
    )
    args = parser.parse_args()

    ok = validate(args.mission, args.dataset, args.deep)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
