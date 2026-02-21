#!/usr/bin/env python3
"""Merge per-mission validation results into a single report.

Reads all *_results.json files from a results directory and produces
a unified validation_report.json with summary stats.

Usage:
    python scripts/merge_validation_results.py --input-dir validation_results/ --output validation_results/validation_report.json
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge validation result files into a single report"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing *_results.json files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output report JSON path")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    all_results = []

    # Read all result files
    result_files = sorted(input_dir.glob("*_results.json"))
    if not result_files:
        print(f"No *_results.json files found in {input_dir}")
        return

    print(f"Found {len(result_files)} result files")

    for rf in result_files:
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_results.extend(data)
            else:
                print(f"  Warning: {rf.name} is not a list, skipping")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not read {rf.name}: {e}")

    if not all_results:
        print("No results to merge")
        return

    # Compute stats
    total = len(all_results)
    n_ok = sum(1 for r in all_results if r.get("status") == "ok")
    n_err = sum(1 for r in all_results if r.get("status") == "error")
    n_skip = sum(1 for r in all_results if r.get("status") == "skipped")

    # By mission
    by_mission: dict[str, dict] = {}
    for r in all_results:
        ms = r.get("mission_stem", "unknown")
        if ms not in by_mission:
            by_mission[ms] = {"total": 0, "ok": 0, "error": 0, "skipped": 0}
        by_mission[ms]["total"] += 1
        status = r.get("status", "unknown")
        if status in by_mission[ms]:
            by_mission[ms][status] += 1

    # Sort by mission name
    by_mission = dict(sorted(by_mission.items()))

    # Failures list
    failures = []
    for r in all_results:
        if r.get("status") == "error":
            failures.append({
                "dataset_id": r.get("dataset_id"),
                "mission_stem": r.get("mission_stem"),
                "backend": r.get("backend"),
                "parameter_id": r.get("parameter_id"),
                "error": r.get("error", ""),
                "error_type": r.get("error_type", ""),
                "elapsed_s": r.get("elapsed_s", 0),
            })

    # Skipped list
    skipped = []
    for r in all_results:
        if r.get("status") == "skipped":
            skipped.append({
                "dataset_id": r.get("dataset_id"),
                "mission_stem": r.get("mission_stem"),
                "backend": r.get("backend"),
                "skip_reason": r.get("skip_reason", ""),
            })

    # Error type summary
    error_types = Counter(r.get("error_type", "unknown") for r in all_results if r.get("status") == "error")
    error_type_summary = dict(error_types.most_common())

    # Skip reason summary
    skip_reasons = Counter(r.get("skip_reason", "unknown") for r in all_results if r.get("status") == "skipped")
    skip_reason_summary = dict(skip_reasons.most_common())

    # Timing stats for successful fetches
    ok_times = [r.get("elapsed_s", 0) for r in all_results if r.get("status") == "ok"]
    timing = {}
    if ok_times:
        timing = {
            "mean_s": round(sum(ok_times) / len(ok_times), 2),
            "max_s": round(max(ok_times), 2),
            "min_s": round(min(ok_times), 2),
            "total_s": round(sum(ok_times), 2),
        }

    # Build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "ok": n_ok,
            "error": n_err,
            "skipped": n_skip,
            "error_rate": f"{n_err / max(total - n_skip, 1) * 100:.1f}%",
        },
        "error_type_summary": error_type_summary,
        "skip_reason_summary": skip_reason_summary,
        "timing": timing,
        "by_mission": by_mission,
        "failures": failures,
        "skipped_datasets": skipped,
    }

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Validation Report Summary")
    print(f"{'='*60}")
    print(f"Total datasets: {total}")
    print(f"  OK:      {n_ok}")
    print(f"  Errors:  {n_err}")
    print(f"  Skipped: {n_skip}")
    print(f"  Error rate (of attempted): {report['summary']['error_rate']}")

    if timing:
        print(f"\nTiming (successful fetches):")
        print(f"  Mean: {timing['mean_s']}s, Max: {timing['max_s']}s, Min: {timing['min_s']}s")

    if error_type_summary:
        print(f"\nError types:")
        for et, count in error_type_summary.items():
            print(f"  {et}: {count}")

    if skip_reason_summary:
        print(f"\nSkip reasons:")
        for sr, count in skip_reason_summary.items():
            print(f"  {sr}: {count}")

    print(f"\nMissions with highest error counts:")
    mission_errors = [(m, s["error"]) for m, s in by_mission.items() if s["error"] > 0]
    mission_errors.sort(key=lambda x: -x[1])
    for m, errs in mission_errors[:15]:
        ms = by_mission[m]
        print(f"  {m}: {errs}/{ms['total']} errors")

    print(f"\nFull report: {output_path}")


if __name__ == "__main__":
    main()
