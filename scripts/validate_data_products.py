#!/usr/bin/env python3
"""Validate data products by fetching 1 hour of data from each dataset.

Enumerates datasets from mission JSONs, loads metadata cache to pick the
first non-Time parameter, and calls fetch_data() with a 1-hour window
starting from the dataset's start_date.

Usage:
    # Single mission
    python scripts/validate_data_products.py --mission dscovr --output validation_results/dscovr_results.json

    # Multiple missions
    python scripts/validate_data_products.py --mission ace,wind,dscovr --output validation_results/batch_results.json

    # All missions
    python scripts/validate_data_products.py --all --output validation_results/all_results.json
"""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MISSIONS_DIR = PROJECT_ROOT / "knowledge" / "missions"


def find_mission_json(stem: str) -> Path | None:
    """Find mission JSON file across cdaweb and ppi directories."""
    for subdir in ("cdaweb", "ppi"):
        path = MISSIONS_DIR / subdir / f"{stem}.json"
        if path.exists():
            return path
    return None


def find_metadata_cache(stem: str, dataset_id: str, backend: str) -> Path | None:
    """Find the metadata cache JSON for a dataset."""
    # PPI datasets use underscore-escaped filenames
    if backend == "ppi":
        safe_name = dataset_id.replace(":", "_")
        path = MISSIONS_DIR / "ppi" / stem / "metadata" / f"{safe_name}.json"
    else:
        path = MISSIONS_DIR / "cdaweb" / stem / "metadata" / f"{dataset_id}.json"

    if path.exists():
        return path
    return None


def load_first_parameter(metadata_path: Path) -> str | None:
    """Load metadata JSON and return the first non-Time parameter name."""
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        for param in data.get("parameters", []):
            name = param.get("name", "")
            if name.lower() not in ("time", ""):
                return name
    except (json.JSONDecodeError, OSError):
        pass
    return None


def enumerate_datasets(mission_stems: list[str]) -> list[dict]:
    """Enumerate all datasets for the given mission stems.

    Returns list of dicts with keys:
        mission_stem, backend, dataset_id, start_date, parameter_id
    """
    results = []

    for stem in mission_stems:
        # Check both cdaweb and ppi
        for backend in ("cdaweb", "ppi"):
            json_path = MISSIONS_DIR / backend / f"{stem}.json"
            if not json_path.exists():
                continue

            try:
                mission_data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            for _inst_key, inst in mission_data.get("instruments", {}).items():
                for dataset_id, ds_info in inst.get("datasets", {}).items():
                    start_date = ds_info.get("start_date")
                    if not start_date:
                        continue

                    # Find metadata to get first parameter
                    meta_path = find_metadata_cache(stem, dataset_id, backend)
                    if meta_path is None:
                        results.append({
                            "mission_stem": stem,
                            "backend": backend,
                            "dataset_id": dataset_id,
                            "start_date": start_date,
                            "parameter_id": None,
                            "skip_reason": "no_metadata_cache",
                        })
                        continue

                    param = load_first_parameter(meta_path)
                    if param is None:
                        results.append({
                            "mission_stem": stem,
                            "backend": backend,
                            "dataset_id": dataset_id,
                            "start_date": start_date,
                            "parameter_id": None,
                            "skip_reason": "no_valid_parameter",
                        })
                        continue

                    results.append({
                        "mission_stem": stem,
                        "backend": backend,
                        "dataset_id": dataset_id,
                        "start_date": start_date,
                        "parameter_id": param,
                        "skip_reason": None,
                    })

    return results


def validate_one_dataset(entry: dict) -> dict:
    """Validate a single dataset by fetching 1 hour of data.

    Returns a result dict with status, error info, timing, etc.
    """
    dataset_id = entry["dataset_id"]
    parameter_id = entry["parameter_id"]
    start_date = entry["start_date"]

    result = {
        "dataset_id": dataset_id,
        "mission_stem": entry["mission_stem"],
        "backend": entry["backend"],
        "parameter_id": parameter_id,
        "start_date": start_date,
    }

    # Skip if no parameter
    if entry.get("skip_reason"):
        result["status"] = "skipped"
        result["skip_reason"] = entry["skip_reason"]
        result["elapsed_s"] = 0.0
        result["rows_returned"] = 0
        return result

    # Compute 1-hour window
    # Parse start_date, add 1 hour
    try:
        # Handle various ISO formats
        sd = start_date.rstrip("Z").split("+")[0]
        if "T" in sd:
            dt = datetime.fromisoformat(sd)
        else:
            dt = datetime.fromisoformat(sd + "T00:00:00")
        time_min = dt.isoformat() + "Z"
        time_max = (dt + timedelta(hours=1)).isoformat() + "Z"
    except ValueError as e:
        result["status"] = "error"
        result["error"] = f"Bad start_date format: {e}"
        result["elapsed_s"] = 0.0
        result["rows_returned"] = 0
        return result

    # Try fetch
    t0 = time.monotonic()
    try:
        from data_ops.fetch import fetch_data
        data = fetch_data(dataset_id, parameter_id, time_min, time_max)
        elapsed = time.monotonic() - t0

        # Check for confirmation_required (large download)
        if isinstance(data, dict) and data.get("status") == "confirmation_required":
            result["status"] = "skipped"
            result["skip_reason"] = "confirmation_required"
            result["elapsed_s"] = round(elapsed, 2)
            result["rows_returned"] = 0
            return result

        # Count rows
        df = data.get("data")
        if df is not None:
            if hasattr(df, "sizes"):  # xarray
                rows = df.sizes.get("time", 0)
            else:
                rows = len(df)
        else:
            rows = 0

        result["status"] = "ok"
        result["elapsed_s"] = round(elapsed, 2)
        result["rows_returned"] = rows
        result["units"] = data.get("units", "")

    except Exception as e:
        elapsed = time.monotonic() - t0
        result["status"] = "error"
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        result["elapsed_s"] = round(elapsed, 2)
        result["rows_returned"] = 0
        # Include traceback for debugging
        result["traceback"] = traceback.format_exc().split("\n")[-4:-1]

    return result


def get_all_mission_stems() -> list[str]:
    """Get all unique mission stems across cdaweb and ppi."""
    stems = set()
    for subdir in ("cdaweb", "ppi"):
        d = MISSIONS_DIR / subdir
        if d.exists():
            for f in d.glob("*.json"):
                stems.add(f.stem)
    return sorted(stems)


def main():
    parser = argparse.ArgumentParser(
        description="Validate data products by fetching 1 hour of data"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mission", type=str,
                       help="Comma-separated mission stems (e.g., 'ace,wind')")
    group.add_argument("--all", action="store_true",
                       help="Validate all missions")

    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of parallel workers (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just enumerate datasets without fetching")

    args = parser.parse_args()

    # Determine mission stems
    if args.all:
        stems = get_all_mission_stems()
    else:
        stems = [s.strip() for s in args.mission.split(",")]

    print(f"Missions to validate: {', '.join(stems)}")

    # Enumerate datasets
    datasets = enumerate_datasets(stems)
    fetchable = [d for d in datasets if d.get("skip_reason") is None]
    skipped = [d for d in datasets if d.get("skip_reason") is not None]

    print(f"Total datasets: {len(datasets)} "
          f"(fetchable: {len(fetchable)}, skipped: {len(skipped)})")

    if args.dry_run:
        # Just write the enumeration
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = []
        for d in datasets:
            r = {
                "dataset_id": d["dataset_id"],
                "mission_stem": d["mission_stem"],
                "backend": d["backend"],
                "parameter_id": d["parameter_id"],
                "start_date": d["start_date"],
                "status": "skipped" if d.get("skip_reason") else "pending",
            }
            if d.get("skip_reason"):
                r["skip_reason"] = d["skip_reason"]
            results.append(r)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Dry run: wrote {len(results)} entries to {output_path}")
        return

    # Validate datasets
    results = []

    # Process skipped ones first
    for d in skipped:
        results.append({
            "dataset_id": d["dataset_id"],
            "mission_stem": d["mission_stem"],
            "backend": d["backend"],
            "parameter_id": d["parameter_id"],
            "start_date": d["start_date"],
            "status": "skipped",
            "skip_reason": d["skip_reason"],
            "elapsed_s": 0.0,
            "rows_returned": 0,
        })

    # Fetch data in parallel
    n_workers = max(1, args.workers)
    completed = 0
    total = len(fetchable)

    if n_workers == 1:
        for d in fetchable:
            result = validate_one_dataset(d)
            results.append(result)
            completed += 1
            status_char = "." if result["status"] == "ok" else "X" if result["status"] == "error" else "S"
            print(f"  [{completed}/{total}] {status_char} {d['dataset_id']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_entry = {
                pool.submit(validate_one_dataset, d): d for d in fetchable
            }
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "dataset_id": entry["dataset_id"],
                        "mission_stem": entry["mission_stem"],
                        "backend": entry["backend"],
                        "parameter_id": entry["parameter_id"],
                        "start_date": entry["start_date"],
                        "status": "error",
                        "error": f"Worker exception: {e}",
                        "error_type": type(e).__name__,
                        "elapsed_s": 0.0,
                        "rows_returned": 0,
                    }
                results.append(result)
                completed += 1
                status_char = "." if result["status"] == "ok" else "X" if result["status"] == "error" else "S"
                print(f"  [{completed}/{total}] {status_char} {entry['dataset_id']}", flush=True)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print summary
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = sum(1 for r in results if r["status"] == "error")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    print(f"\nResults: {n_ok} ok, {n_err} errors, {n_skip} skipped")
    print(f"Written to: {output_path}")

    if n_err > 0:
        print(f"\nTop errors:")
        errors = [r for r in results if r["status"] == "error"]
        # Group by error type
        by_type: dict[str, int] = {}
        for r in errors:
            et = r.get("error_type", "unknown")
            by_type[et] = by_type.get(et, 0) + 1
        for et, count in sorted(by_type.items(), key=lambda x: -x[1])[:10]:
            print(f"  {et}: {count}")


if __name__ == "__main__":
    main()
