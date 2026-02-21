"""Mission data startup utilities.

Provides mission status checking, interactive refresh menu, and
CLI flag resolution — shared by main.py and the API server.
"""

import json
from datetime import datetime
from pathlib import Path


def get_mission_status() -> dict:
    """Scan mission JSONs and return a status summary.

    Returns:
        Dict with keys: mission_count, mission_names, total_datasets, oldest_date.
    """
    missions_dir = Path(__file__).parent / "missions"
    # Scan both cdaweb/ and ppi/ subdirectories
    mission_files = sorted(
        list((missions_dir / "cdaweb").glob("*.json"))
        + list((missions_dir / "ppi").glob("*.json"))
    )

    # Deduplicate stems (same mission may exist in both cdaweb/ and ppi/)
    seen_stems = set()
    unique_names = []
    total_datasets = 0
    seen_dataset_ids = set()
    oldest_date = None

    for f in mission_files:
        stem = f.stem
        if stem not in seen_stems:
            seen_stems.add(stem)
            unique_names.append(stem)

        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for inst in data.get("instruments", {}).values():
                for ds_id in inst.get("datasets", {}):
                    if ds_id not in seen_dataset_ids:
                        seen_dataset_ids.add(ds_id)
                        total_datasets += 1
            gen_at = data.get("_meta", {}).get("generated_at", "")
            if gen_at:
                try:
                    dt = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
                    if oldest_date is None or dt < oldest_date:
                        oldest_date = dt
                except ValueError:
                    pass
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "mission_count": len(unique_names),
        "mission_names": unique_names,
        "total_datasets": total_datasets,
        "oldest_date": oldest_date,
    }


def show_mission_menu() -> str:
    """Print mission data status and show an interactive menu.

    Returns:
        Action string: "continue", "refresh", or "all".
    """
    status = get_mission_status()

    print("-" * 60)
    print("  Mission Data Status")
    print("-" * 60)

    if status["mission_count"] == 0:
        print("  No mission data found. Will download on first use.")
        print()
        return "continue"

    names = status["mission_names"]
    print(f"  Missions loaded: {status['mission_count']} ({', '.join(names)})")
    print(f"  Total datasets:  {status['total_datasets']}")
    if status["oldest_date"]:
        age_str = status["oldest_date"].strftime("%Y-%m-%d %H:%M UTC")
        print(f"  Last refreshed:  {age_str}")
    print()
    print("  [Enter] Continue with current data")
    print("  [r]     Refresh time ranges (fast — updates start/stop dates only)")
    print("  [f]     Full rebuild (delete and re-download everything)")
    print()

    try:
        choice = input("  Choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "continue"

    if choice in ("r", "refresh"):
        return "refresh"
    elif choice in ("f", "full"):
        return "rebuild"
    return "continue"


def run_mission_refresh(action: str):
    """Execute mission data refresh based on chosen action.

    Args:
        action: "refresh" for lightweight time-range update,
                "rebuild" for destructive full rebuild of all missions.
    """
    from knowledge.bootstrap import (
        populate_missions, populate_ppi_missions,
        clean_all_missions, refresh_time_ranges,
    )
    import knowledge.bootstrap as bootstrap_mod

    if action == "refresh":
        print("\nRefreshing dataset time ranges...")
        refresh_time_ranges()
    elif action == "rebuild":
        print("\nFull rebuild of all missions...")
        clean_all_missions()
        bootstrap_mod._bootstrap_checked = False
        populate_missions()
        populate_ppi_missions()
        try:
            from knowledge.bootstrap import populate_ppi_metadata
            populate_ppi_metadata()
        except Exception:
            pass  # Non-fatal — metadata populated lazily as fallback

    from knowledge.mission_loader import clear_cache
    from knowledge.metadata_client import clear_cache as clear_metadata_cache
    clear_cache()
    clear_metadata_cache()
    print()


def run_cdaweb_rebuild(progress_callback=None):
    """Rebuild CDAWeb mission data only (destructive)."""
    from knowledge.bootstrap import (
        populate_missions, clean_all_missions,
    )
    import knowledge.bootstrap as bootstrap_mod

    clean_all_missions(source="cdaweb")
    bootstrap_mod._bootstrap_checked = False
    populate_missions(progress_callback=progress_callback)

    from knowledge.mission_loader import clear_cache
    from knowledge.metadata_client import clear_cache as clear_metadata_cache
    clear_cache()
    clear_metadata_cache()


def run_ppi_rebuild(progress_callback=None):
    """Rebuild PPI mission data only (destructive)."""
    from knowledge.bootstrap import (
        populate_ppi_missions, populate_ppi_metadata, clean_all_missions,
    )

    clean_all_missions(source="ppi")
    populate_ppi_missions(progress_callback=progress_callback)
    try:
        populate_ppi_metadata()
    except Exception:
        pass  # Non-fatal — metadata populated lazily as fallback

    from knowledge.mission_loader import clear_cache
    from knowledge.metadata_client import clear_cache as clear_metadata_cache
    clear_cache()
    clear_metadata_cache()


def run_refresh(progress_callback=None):
    """Refresh time ranges for both CDAWeb and PPI missions."""
    from knowledge.bootstrap import refresh_time_ranges

    result = refresh_time_ranges(progress_callback=progress_callback)

    from knowledge.mission_loader import clear_cache
    from knowledge.metadata_client import clear_cache as clear_metadata_cache
    clear_cache()
    clear_metadata_cache()
    return result


def resolve_refresh_flags(
    refresh: bool = False,
    refresh_full: bool = False,
    refresh_all: bool = False,
):
    """Map CLI flags to an action, or show interactive menu.

    Args:
        refresh: True if --refresh was passed (lightweight time-range update).
        refresh_full: True if --refresh-full was passed (destructive rebuild).
        refresh_all: True if --refresh-all was passed (rebuild, same as refresh_full).
    """
    if refresh:
        run_mission_refresh("refresh")
    elif refresh_full or refresh_all:
        run_mission_refresh("rebuild")
    else:
        action = show_mission_menu()
        if action != "continue":
            run_mission_refresh(action)
