#!/usr/bin/env python3
"""
Query CDAWeb REST API to get file counts and total sizes for all datasets
defined in knowledge/missions/*.json.

Uses the orig_data endpoint which returns file URLs with Length (bytes)
for each dataset's full time range. Much faster and more accurate than
crawling SPDF directory listings.

Usage:
    python scripts/query_dataset_sizes.py
    python scripts/query_dataset_sizes.py --workers 8
    python scripts/query_dataset_sizes.py --output dataset_sizes.json
"""

import argparse
import json
import glob
import logging
import os
import signal
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests

CDAWEB_API = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets"

_stop_flag = threading.Event()


def format_size(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} EB"


def load_all_datasets(missions_dir):
    """Load all datasets from mission JSON files. Returns list of (mission_id, dataset_id, start, stop)."""
    datasets = []
    for path in sorted(glob.glob(os.path.join(missions_dir, "*.json"))):
        with open(path) as f:
            mission = json.load(f)
        mission_id = mission["id"]
        for inst in mission["instruments"].values():
            for ds_id, ds_info in inst["datasets"].items():
                # Strip @N suffix used for duplicate dataset IDs
                clean_id = ds_id.split("@")[0]
                start = ds_info.get("start_date", "")
                stop = ds_info.get("stop_date", "")
                if start and stop:
                    datasets.append((mission_id, clean_id, start, stop))
    # Deduplicate by clean dataset ID
    seen = set()
    unique = []
    for mission_id, ds_id, start, stop in datasets:
        if ds_id not in seen:
            seen.add(ds_id)
            unique.append((mission_id, ds_id, start, stop))
    return unique


def iso_to_cdaweb(iso_str):
    """Convert ISO date to CDAWeb API format: YYYYMMDDTHHmmSSZ"""
    # Strip fractional seconds and timezone
    clean = iso_str.replace(".000Z", "Z").replace(".000", "")
    if clean.endswith("Z"):
        clean = clean[:-1]
    # Parse and reformat
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(clean, fmt)
            return dt.strftime("%Y%m%dT%H%M%SZ")
        except ValueError:
            continue
    return clean.replace("-", "").replace(":", "") + "Z"


def query_dataset(session, mission_id, dataset_id, start, stop, retries=3):
    """Query CDAWeb for file listing of a dataset. Returns (mission_id, dataset_id, file_count, total_bytes, error)."""
    start_fmt = iso_to_cdaweb(start)
    stop_fmt = iso_to_cdaweb(stop)
    url = f"{CDAWEB_API}/{dataset_id}/orig_data/{start_fmt},{stop_fmt}"

    for attempt in range(retries):
        if _stop_flag.is_set():
            return (mission_id, dataset_id, 0, 0, "interrupted")
        try:
            r = session.get(url, headers={"Accept": "application/json"}, timeout=60)
            if r.status_code == 404:
                return (mission_id, dataset_id, 0, 0, "not_found")
            if r.status_code == 400:
                return (mission_id, dataset_id, 0, 0, f"bad_request: {r.text[:200]}")
            r.raise_for_status()
            data = r.json()
            files = data.get("FileDescription", [])
            total_bytes = sum(f.get("Length", 0) for f in files)
            return (mission_id, dataset_id, len(files), total_bytes, None)
        except Exception as e:
            if attempt == retries - 1:
                return (mission_id, dataset_id, 0, 0, str(e)[:200])
            time.sleep(2 * (attempt + 1))

    return (mission_id, dataset_id, 0, 0, "max_retries")


def setup_logger(log_path):
    logger = logging.getLogger("dataset_sizes")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s\n%(message)s\n"))
    logger.addHandler(handler)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Query CDAWeb for dataset file sizes")
    parser.add_argument("--workers", type=int, default=6,
                        help="Concurrent API requests (default: 6, be nice to CDAWeb)")
    parser.add_argument("--output", type=str, default="dataset_sizes.json",
                        help="Output JSON file (default: dataset_sizes.json)")
    parser.add_argument("--log", type=str,
                        default=f"dataset_sizes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                        help="Log file for periodic snapshots")
    parser.add_argument("--missions-dir", type=str, default="knowledge/missions",
                        help="Path to missions JSON directory")
    args = parser.parse_args()

    # Load datasets
    datasets = load_all_datasets(args.missions_dir)
    print(f"Loaded {len(datasets)} unique datasets from {args.missions_dir}")
    print(f"Using {args.workers} concurrent workers")
    print(f"Output: {os.path.abspath(args.output)}")
    print(f"Log: {os.path.abspath(args.log)}\n")

    # State
    lock = threading.Lock()
    mission_files = defaultdict(int)
    mission_bytes = defaultdict(int)
    dataset_results = {}
    done_count = 0
    error_count = 0
    start_time = time.time()

    # Logger for periodic snapshots
    sc_logger = setup_logger(args.log)

    def log_snapshot():
        with lock:
            elapsed = str(timedelta(seconds=int(time.time() - start_time)))
            total_bytes = sum(mission_bytes.values())
            total_files = sum(mission_files.values())
            lines = [f"=== Dataset Size Snapshot [{elapsed}] ==="]
            lines.append(
                f"Datasets queried: {done_count}/{len(datasets)}  |  "
                f"Total files: {total_files:,}  |  "
                f"Total size: {format_size(total_bytes)}  |  "
                f"Errors: {error_count}"
            )
            lines.append(f"{'Mission':<25s} {'Datasets':>10s} {'Files':>12s} {'Size':>14s}")
            lines.append("-" * 63)
            # Count datasets per mission
            mission_ds_count = defaultdict(int)
            for key in dataset_results:
                m = dataset_results[key]["mission"]
                mission_ds_count[m] += 1
            sorted_missions = sorted(mission_bytes.items(), key=lambda x: x[1], reverse=True)
            for m, size in sorted_missions:
                lines.append(
                    f"{m:<25s} {mission_ds_count.get(m, 0):>10,} "
                    f"{mission_files[m]:>12,} {format_size(size):>14s}"
                )
            lines.append("=" * 63)
            sc_logger.info("\n".join(lines))

    def log_loop():
        while not _stop_flag.is_set():
            _stop_flag.wait(5)
            log_snapshot()

    log_thread = threading.Thread(target=log_loop, daemon=True)
    log_thread.start()

    # Signal handler
    def _sigterm_handler(signum, frame):
        _stop_flag.set()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Query all datasets
    session = requests.Session()
    session.headers["User-Agent"] = "helio-agent-dataset-sizer/1.0 (research)"

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for mission_id, ds_id, start, stop in datasets:
                f = executor.submit(query_dataset, session, mission_id, ds_id, start, stop)
                futures[f] = (mission_id, ds_id)

            for future in as_completed(futures):
                if _stop_flag.is_set():
                    break
                m_id, ds_id, n_files, n_bytes, err = future.result()
                with lock:
                    done_count += 1
                    if err:
                        error_count += 1
                    dataset_results[ds_id] = {
                        "mission": m_id,
                        "files": n_files,
                        "bytes": n_bytes,
                        "size_human": format_size(n_bytes),
                        "error": err,
                    }
                    if n_files > 0:
                        mission_files[m_id] += n_files
                        mission_bytes[m_id] += n_bytes

                    # Print progress
                    elapsed = time.time() - start_time
                    total_b = sum(mission_bytes.values())
                    pct = done_count / len(datasets) * 100
                    line = (
                        f"\r[{timedelta(seconds=int(elapsed))}] "
                        f"{done_count}/{len(datasets)} ({pct:.0f}%) | "
                        f"{format_size(total_b)} | "
                        f"errors: {error_count} | "
                        f"last: {ds_id[:40]}"
                    )
                    sys.stderr.write(line + " " * 20)
                    sys.stderr.flush()
    except KeyboardInterrupt:
        _stop_flag.set()

    _stop_flag.set()
    log_thread.join(timeout=2)

    # Final snapshot
    log_snapshot()

    # Print summary
    print("\n\n" + "=" * 70)
    print("Dataset Size Summary (from CDAWeb API)")
    print("=" * 70)
    total_files = sum(mission_files.values())
    total_bytes = sum(mission_bytes.values())
    print(f"Datasets queried:  {done_count}/{len(datasets)}")
    print(f"Total data files:  {total_files:,}")
    print(f"Total data size:   {format_size(total_bytes)}")
    print(f"Errors:            {error_count}")
    elapsed = time.time() - start_time
    print(f"Time elapsed:      {timedelta(seconds=int(elapsed))}")
    print()
    print(f"{'Mission':<25s} {'Datasets':>10s} {'Files':>12s} {'Size':>14s}")
    print("-" * 63)
    mission_ds_count = defaultdict(int)
    for key in dataset_results:
        m = dataset_results[key]["mission"]
        mission_ds_count[m] += 1
    sorted_missions = sorted(mission_bytes.items(), key=lambda x: x[1], reverse=True)
    for m, size in sorted_missions:
        print(f"{m:<25s} {mission_ds_count.get(m, 0):>10,} {mission_files[m]:>12,} {format_size(size):>14s}")
    print("=" * 70)

    # Save JSON
    result = {
        "generated_at": datetime.now().isoformat(),
        "datasets_queried": done_count,
        "datasets_total": len(datasets),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_size_human": format_size(total_bytes),
        "errors": error_count,
        "missions": {
            m: {
                "datasets": mission_ds_count.get(m, 0),
                "files": mission_files[m],
                "bytes": mission_bytes[m],
                "size_human": format_size(mission_bytes[m]),
            }
            for m, _ in sorted_missions
        },
        "datasets": dataset_results,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {os.path.abspath(args.output)}")
    print(f"Log saved to {os.path.abspath(args.log)}")


if __name__ == "__main__":
    main()
