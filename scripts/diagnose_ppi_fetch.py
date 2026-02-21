#!/usr/bin/env python3
"""Step-by-step diagnostic for PPI archive data fetching.

Exercises each stage of the fetch pipeline independently so failures
are easy to pinpoint:

  1. Collection URL  — URN → archive directory resolution
  2. File discovery  — data+label file pairs
  3. Download+parse  — first file pair: label parsing → table read

Usage:
    venv/bin/python scripts/diagnose_ppi_fetch.py \\
        "urn:nasa:pds:insight-ifg-mars:data-ifg-calibrated" \\
        "Bx_SC" "2019-01-01" "2019-01-07"
"""

import argparse
import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def diagnose(dataset_id: str, parameter_id: str, time_min: str, time_max: str) -> bool:
    """Run diagnostics and return True if all steps pass."""
    all_ok = True

    # --- Step 1: Collection URL ---
    print("\n1. Resolve collection URL")
    try:
        from data_ops.fetch_ppi_archive import _resolve_collection_url
        collection_url = _resolve_collection_url(dataset_id)
        _ok(f"URL: {collection_url}")
    except Exception as e:
        _fail(f"Exception: {e}")
        traceback.print_exc()
        return False

    # --- Step 2: File discovery ---
    print("\n2. Discover data files")
    try:
        from data_ops.fetch_ppi_archive import _discover_data_files
        file_pairs = _discover_data_files(collection_url, time_min, time_max)
        if not file_pairs:
            _fail("No data file pairs found")
            all_ok = False
        else:
            _ok(f"{len(file_pairs)} file pair(s) found")
            for i, (durl, lurl) in enumerate(file_pairs[:5]):
                dname = durl.rsplit("/", 1)[-1]
                _info(f"  [{i+1}] {dname}")
            if len(file_pairs) > 5:
                _info(f"  ... and {len(file_pairs) - 5} more")
    except Exception as e:
        _fail(f"Exception: {e}")
        traceback.print_exc()
        return False

    if not file_pairs:
        return False

    # --- Step 3: Download + parse first file ---
    print("\n3. Download and parse first file pair")
    data_url, label_url = file_pairs[0]
    try:
        from data_ops.fetch_ppi_archive import (
            _download_file, _parse_label, _read_table,
        )

        # Download
        local_data = _download_file(data_url)
        local_label = _download_file(label_url)
        _ok(f"Downloaded: {local_data.name} ({local_data.stat().st_size / 1024:.1f} KB)")
        _ok(f"Downloaded: {local_label.name} ({local_label.stat().st_size / 1024:.1f} KB)")

        # Parse label
        label = _parse_label(local_label)
        _ok(f"Table type: {label['table_type']}, "
            f"{len(label['fields'])} fields, "
            f"delimiter: {label.get('delimiter', 'N/A')}")
        field_names = [f["name"] for f in label["fields"]]
        _info(f"Fields: {', '.join(field_names[:10])}"
              + ("..." if len(field_names) > 10 else ""))

        # Read table
        df = _read_table(local_data, label, parameter_id)
        if df is None:
            _fail(f"_read_table returned None for parameter '{parameter_id}'")
            _info(f"Available fields: {', '.join(field_names)}")
            all_ok = False
        elif len(df) == 0:
            _fail("DataFrame is empty (0 rows)")
            all_ok = False
        else:
            _ok(f"DataFrame: {len(df)} rows, {len(df.columns)} columns")
            _info(f"Time range: {df.index[0]} to {df.index[-1]}")
            _info(f"Columns: {list(df.columns)}")
            _info(f"Sample values:\n{df.head(3).to_string()}")

    except Exception as e:
        _fail(f"Exception: {e}")
        traceback.print_exc()
        all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose PPI archive data fetching step by step.",
    )
    parser.add_argument("dataset_id", help="PDS URN dataset ID")
    parser.add_argument("parameter_id", help="Parameter name (from label metadata)")
    parser.add_argument("time_min", help="Start time (ISO format)")
    parser.add_argument("time_max", help="End time (ISO format)")
    args = parser.parse_args()

    print(f"Diagnosing PPI fetch for:")
    print(f"  Dataset:   {args.dataset_id}")
    print(f"  Parameter: {args.parameter_id}")
    print(f"  Range:     {args.time_min} to {args.time_max}")

    ok = diagnose(args.dataset_id, args.parameter_id, args.time_min, args.time_max)

    print("\n" + "=" * 60)
    if ok:
        print("Result: ALL STEPS PASSED")
    else:
        print("Result: SOME STEPS FAILED (see above)")
    print("=" * 60)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
