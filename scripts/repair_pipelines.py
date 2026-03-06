#!/usr/bin/env python3
"""One-shot repair: re-extract saved pipelines that have orphan nodes.

Scans all saved pipeline JSON files, validates each for orphan nodes,
and re-extracts from the source session's operations.json.  Preserves
the original id, name, description, tags, and created_at.

Usage:
    venv/bin/python scripts/repair_pipelines.py          # dry-run
    venv/bin/python scripts/repair_pipelines.py --apply   # overwrite stale files
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ops.pipeline import SavedPipeline


def repair_orphan_pipelines(*, apply: bool = False) -> None:
    """Scan all saved pipelines and re-extract those with orphan nodes."""
    entries = SavedPipeline.list_all()
    if not entries:
        print("No saved pipelines found.")
        return

    repaired = []
    deleted = []
    skipped = []
    failed = []

    for entry in entries:
        pid = entry["id"]
        try:
            pipeline = SavedPipeline.load(pid)
        except FileNotFoundError:
            print(f"  {pid}: index entry but file missing — skipping")
            skipped.append(pid)
            continue

        issues = pipeline.validate()
        orphans = [i for i in issues if "orphan" in i.lower()]

        if not orphans:
            skipped.append(pid)
            continue

        print(f"\n  {pid} ({pipeline.name})")
        print(f"    Orphan issues: {len(orphans)}")
        for o in orphans:
            print(f"      {o}")

        source_session = pipeline._data.get("source_session_id", "")
        source_render_op = pipeline._data.get("source_render_op_id")

        if not source_session:
            print(f"    No source_session_id — will delete")
            if apply:
                SavedPipeline.delete(pid)
            deleted.append(pid)
            continue

        # Check if source session exists
        from config import get_data_dir
        ops_path = get_data_dir() / "sessions" / source_session / "operations.json"
        if not ops_path.exists():
            print(f"    Source session {source_session} missing — will delete")
            if apply:
                SavedPipeline.delete(pid)
            deleted.append(pid)
            continue

        # Re-extract from source session
        try:
            new_pipeline = SavedPipeline.from_session(
                source_session,
                render_op_id=source_render_op,
            )
        except Exception as e:
            print(f"    Re-extraction failed: {e}")
            if apply:
                SavedPipeline.delete(pid)
            deleted.append(pid)
            continue

        # Re-validate the newly extracted pipeline
        new_issues = new_pipeline.validate()
        new_orphans = [i for i in new_issues if "orphan" in i.lower()]

        if new_orphans:
            print(f"    Re-extracted pipeline still has {len(new_orphans)} orphans — will delete")
            for o in new_orphans:
                print(f"      {o}")
            if apply:
                SavedPipeline.delete(pid)
            deleted.append(pid)
            continue

        # Preserve original metadata
        new_pipeline._data["id"] = pipeline._data["id"]
        new_pipeline._data["name"] = pipeline._data.get("name", "")
        new_pipeline._data["description"] = pipeline._data.get("description", "")
        new_pipeline._data["tags"] = pipeline._data.get("tags", [])
        new_pipeline._data["created_at"] = pipeline._data.get("created_at", "")

        old_step_count = len(pipeline.steps)
        new_step_count = len(new_pipeline.steps)
        print(f"    Re-extracted: {old_step_count} → {new_step_count} steps, 0 orphans")

        if apply:
            new_pipeline.save()
        repaired.append(pid)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary {'(DRY RUN)' if not apply else '(APPLIED)'}:")
    print(f"  Total scanned: {len(entries)}")
    print(f"  Skipped (clean): {len(skipped)}")
    print(f"  Repaired: {len(repaired)}")
    print(f"  Deleted: {len(deleted)}")
    if repaired:
        print(f"\n  Repaired pipelines:")
        for pid in repaired:
            print(f"    {pid}")
    if deleted:
        print(f"\n  Deleted pipelines:")
        for pid in deleted:
            print(f"    {pid}")
    if not apply and (repaired or deleted):
        print(f"\n  Run with --apply to make changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repair saved pipelines with orphan nodes by re-extracting from source sessions."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually overwrite stale files (default is dry-run).",
    )
    args = parser.parse_args()
    repair_orphan_pipelines(apply=args.apply)
