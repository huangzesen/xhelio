"""
Auto-download mission data on first run + time-range refresh.

When no CDAWeb mission JSON files exist (fresh clone), fetches the CDAWeb
catalog via CDAS REST API and Master CDF files, and populates all mission
files + metadata cache automatically.

PPI missions are populated via the PDS PPI Metadex Solr API, which indexes
all PDS3 and PDS4 data collections in a single HTTP call.

Also provides ``refresh_time_ranges()`` which updates start/stop dates
for both CDAWeb (via CDAS REST API) and PPI missions (via Metadex).

This module is lazy-imported by mission_loader.load_all_missions() only
when no *.json files are found in knowledge/missions/cdaweb/.
"""

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from agent.event_bus import get_event_bus, DEBUG, BOOTSTRAP_PROGRESS, METADATA_FETCH

try:
    import requests
except ImportError:
    requests = None

from .mission_prefixes import (
    match_dataset_to_mission,
    create_mission_skeleton,
    get_all_mission_stems,
    get_mission_name,
    get_canonical_id,
    PRIMARY_MISSIONS,
)


# Constants
MISSIONS_DIR = Path(__file__).parent / "missions" / "cdaweb"
PPI_DIR = Path(__file__).parent / "missions" / "ppi"
DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 2)
MAX_RETRIES = 2

# Module-level flag: only check once per process
_bootstrap_checked = False

def ensure_missions_populated(progress_callback=None):
    """Check if any mission data exists; run full populate if missing.

    On first run, downloads all missions from CDAWeb REST API + Master CDF
    files with full parameter metadata (~5-10 minutes), then PPI missions
    from PDS PPI Metadex. Only runs once per process.

    Args:
        progress_callback: Optional callable(dict) for streaming progress
            events to the caller (e.g. background loader).
    """
    global _bootstrap_checked
    if _bootstrap_checked:
        return
    _bootstrap_checked = True

    cdaweb_exists = any(MISSIONS_DIR.glob("*.json"))
    ppi_exists = any(PPI_DIR.glob("*.json"))

    if cdaweb_exists and ppi_exists:
        return  # Already bootstrapped

    if not cdaweb_exists:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                    msg="First run detected — downloading full mission catalog + "
                    "parameter metadata (all missions, typically 5-10 minutes). "
                    "This only happens once.")
        try:
            populate_missions(progress_callback=progress_callback)
        except Exception as e:
            get_event_bus().emit(BOOTSTRAP_PROGRESS, level="warning",
                           msg=f"CDAWeb auto-download failed: {e}. "
                           "The agent will start with a partial catalog. "
                           "Retry by restarting, or run: "
                           "python scripts/generate_mission_data.py --force")

    if not ppi_exists:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                    msg="Downloading PPI mission catalog + parameter metadata...")
        try:
            populate_ppi_missions(progress_callback=progress_callback)
        except Exception as e:
            get_event_bus().emit(BOOTSTRAP_PROGRESS, level="warning",
                           msg=f"PPI auto-download failed: {e}. "
                           "PPI missions will not be available. "
                           "Retry by restarting, or run: "
                           "python scripts/generate_ppi_missions.py")

        # Eagerly fetch PPI parameter metadata from label files
        try:
            populate_ppi_metadata()
        except Exception as e:
            get_event_bus().emit(BOOTSTRAP_PROGRESS, level="warning",
                           msg=f"PPI metadata population failed: {e}. "
                           "Metadata will be populated lazily on first fetch.")


def populate_missions(only_stems: set[str] | None = None, progress_callback=None):
    """Download and populate mission data from CDAWeb.

    Args:
        only_stems: If provided, only download these mission stems.
                    If None, download all missions found in the catalog.
        progress_callback: Optional callable(dict) for streaming progress events.

    Steps:
      1. Fetch dataset catalog (CDAS REST API)
      2. Group datasets by mission via prefix matching
      3. Filter to only_stems if specified
      4. Create skeleton mission JSONs for missing ones
      5. Parallel-fetch metadata for all datasets (Master CDF)
      6. Merge metadata into mission JSONs
      7. Generate _index.json and _calibration_exclude.json per mission
    """
    if requests is None:
        raise RuntimeError(
            "'requests' package is required for auto-download. "
            "Install with: pip install requests"
        )

    start_time = time.time()

    # Step 1+1b: Fetch CDAWeb REST API metadata (single HTTP call)
    from .cdaweb_metadata import fetch_dataset_metadata
    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg="Fetching CDAWeb dataset metadata (catalog + instrument types)...")
    if progress_callback:
        progress_callback({"type": "progress", "phase": "cdaweb", "step": "catalog",
                           "message": "Fetching CDAWeb dataset catalog..."})
    cdaweb_meta = fetch_dataset_metadata()
    if cdaweb_meta:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                    msg=f"Got metadata for {len(cdaweb_meta)} datasets")
    else:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="warning",
                    msg="CDAWeb metadata unavailable, falling back to prefix hints")

    # Build catalog from CDAS REST metadata
    catalog = _fetch_catalog(cdaweb_meta)

    # Step 2: Group datasets by mission
    mission_datasets = _group_by_mission(catalog)

    # Step 3: Filter to requested stems
    if only_stems:
        mission_datasets = {
            stem: ds for stem, ds in mission_datasets.items()
            if stem in only_stems
        }

    total_datasets = sum(len(ds) for ds in mission_datasets.values())
    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg=f"Grouped {total_datasets} datasets into {len(mission_datasets)} missions")
    if progress_callback:
        progress_callback({"type": "progress", "phase": "cdaweb", "step": "grouped",
                           "message": f"Grouped {total_datasets} datasets into {len(mission_datasets)} missions"})

    # Step 4: Create skeleton JSONs for missions that don't exist yet
    # Include all requested stems, even those with no CDAWeb datasets
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stems_to_skeleton = set(mission_datasets.keys())
    if only_stems:
        stems_to_skeleton |= only_stems
    for stem in sorted(stems_to_skeleton):
        filepath = MISSIONS_DIR / f"{stem}.json"
        if not filepath.exists():
            skeleton = create_mission_skeleton(stem)
            _save_json(filepath, skeleton)

    # Step 5: Parallel-fetch metadata for all datasets
    all_fetch_items = []
    for stem, datasets in mission_datasets.items():
        cache_dir = MISSIONS_DIR / stem / "metadata"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for ds_id, instrument_hint in datasets:
            all_fetch_items.append((ds_id, instrument_hint, stem, cache_dir))

    # Shuffle so slow/failing missions don't cluster together in the progress bar
    random.shuffle(all_fetch_items)

    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg=f"Fetching parameter metadata for {len(all_fetch_items)} datasets across "
                f"{len(mission_datasets)} missions (Master CDF, timeout=10s, 3 retries)...")
    results = _fetch_all_info(all_fetch_items, cdaweb_meta=cdaweb_meta,
                              source_label="CDAWeb",
                              progress_callback=progress_callback)

    # Step 6: Merge results into mission JSONs
    if progress_callback:
        progress_callback({"type": "progress", "phase": "cdaweb", "step": "merging",
                           "message": "Merging metadata into mission files..."})
    _merge_into_missions(mission_datasets, results, cdaweb_meta)

    # Step 7: Generate per-mission index and calibration exclude files
    for stem in mission_datasets:
        _generate_index(stem)
        _ensure_calibration_exclude(stem)

    elapsed = time.time() - start_time
    n_success = sum(1 for r in results.values() if r is not None)
    n_failed = len(results) - n_success
    msg = (f"Bootstrap complete in {elapsed:.0f}s: "
           f"{len(mission_datasets)} missions, {n_success} datasets fetched")
    if n_failed:
        msg += f", {n_failed} failed"
    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info", msg=msg)
    if progress_callback:
        progress_callback({"type": "progress", "phase": "cdaweb", "step": "done",
                           "message": msg, "pct": 100})


def populate_ppi_missions(only_stems: set[str] | None = None, progress_callback=None):
    """Download and populate PPI mission data from PDS PPI Metadex.

    Fetches all data collections from the Metadex Solr API in a single
    HTTP call, groups by mission, and writes mission JSONs to
    knowledge/missions/ppi/{stem}.json.

    No parallel fetching needed — Metadex returns everything in one
    response (no HAPI server dependency).

    Args:
        only_stems: If provided, only generate these mission stems.
                    If None, generate all PPI missions.
        progress_callback: Optional callable(dict) for streaming progress events.
    """
    if requests is None:
        raise RuntimeError(
            "'requests' package is required for PPI download. "
            "Install with: pip install requests"
        )

    start_time = time.time()

    # Step 1: Fetch all collections from Metadex (single HTTP call)
    from .metadex_client import fetch_all_ppi_collections, metadex_id_to_dataset_id

    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg="Fetching PPI collections from Metadex...")
    if progress_callback:
        progress_callback({"type": "progress", "phase": "ppi", "step": "catalog",
                           "message": "Fetching PPI collections from Metadex..."})
    collections = fetch_all_ppi_collections()
    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg=f"Found {len(collections)} PPI data collections in Metadex")

    # Step 2: Group by mission
    groups: dict[str, list[dict]] = {}
    for coll in collections:
        metadex_id = coll["id"]
        archive_type = coll["archive_type"]
        dataset_id = metadex_id_to_dataset_id(metadex_id, archive_type)
        coll["_dataset_id"] = dataset_id

        mission_stem, _ = match_dataset_to_mission(dataset_id)
        if mission_stem:
            groups.setdefault(mission_stem, []).append(coll)

    if only_stems:
        groups = {s: ds for s, ds in groups.items() if s in only_stems}

    total_datasets = sum(len(ds) for ds in groups.values())
    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg=f"Grouped {total_datasets} PPI datasets into {len(groups)} missions")
    if progress_callback:
        progress_callback({"type": "progress", "phase": "ppi", "step": "grouped",
                           "message": f"Grouped {total_datasets} PPI datasets into {len(groups)} missions"})

    PPI_DIR.mkdir(parents=True, exist_ok=True)

    # Step 3: Build and save mission JSONs per stem
    sorted_groups = sorted(groups)
    for i, stem in enumerate(sorted_groups):
        stem_collections = groups[stem]
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                    msg=f"PPI {stem}: {len(stem_collections)} datasets")
        if progress_callback:
            pct = (i + 1) * 100 // len(sorted_groups)
            progress_callback({
                "type": "progress", "phase": "ppi", "step": "building",
                "message": f"Building {stem} ({i+1}/{len(sorted_groups)})",
                "done": i + 1, "total": len(sorted_groups), "pct": pct,
                "current": stem,
            })

        mission = _build_ppi_mission_json(stem, stem_collections)
        filepath = PPI_DIR / f"{stem}.json"
        _save_json(filepath, mission)

        # Create metadata dir (populated lazily on first data fetch)
        cache_dir = PPI_DIR / stem / "metadata"
        cache_dir.mkdir(parents=True, exist_ok=True)

        _generate_ppi_index(stem)

    elapsed = time.time() - start_time
    msg = (f"PDS PPI populate complete in {elapsed:.0f}s: "
           f"{len(groups)} missions, {total_datasets} datasets")
    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info", msg=msg)
    if progress_callback:
        progress_callback({"type": "progress", "phase": "ppi", "step": "done",
                           "message": msg, "pct": 100})


def populate_ppi_metadata(only_stems: set[str] | None = None):
    """Eagerly fetch parameter metadata for PPI datasets from label files.

    For each PPI dataset that has no metadata cache file yet, downloads
    one label file from the PDS archive, parses it, and saves the
    metadata to the cache directory.  Runs in parallel with
    ThreadPoolExecutor.

    After all fetches, regenerates ``_index.json`` for affected missions
    so ``parameter_count`` is non-zero.

    Args:
        only_stems: If provided, only process these mission stems.
                    If None, process all PPI missions.
    """
    from data_ops.fetch_ppi_archive import fetch_ppi_label_metadata
    from .metadata_client import _dataset_id_to_cache_filename

    start_time = time.time()

    # Collect (dataset_id, slot, stem, cache_dir) for datasets missing metadata
    work_items: list[tuple[str, str, str, Path]] = []
    affected_stems: set[str] = set()

    for filepath in sorted(PPI_DIR.glob("*.json")):
        stem = filepath.stem
        if only_stems and stem not in only_stems:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                mission_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        cache_dir = PPI_DIR / stem / "metadata"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for inst in mission_data.get("instruments", {}).values():
            for ds_id, ds_entry in inst.get("datasets", {}).items():
                slot = ds_entry.get("slot")
                if not slot:
                    continue  # No archive path — can't fetch label

                cache_filename = _dataset_id_to_cache_filename(ds_id)
                cache_file = cache_dir / cache_filename
                if cache_file.exists():
                    continue  # Already cached

                work_items.append((ds_id, slot, stem, cache_dir))
                affected_stems.add(stem)

    if not work_items:
        get_event_bus().emit(METADATA_FETCH, level="info",
                    msg="PPI metadata: all datasets already cached, nothing to do")
        return

    get_event_bus().emit(METADATA_FETCH, level="info",
                msg=f"Fetching PPI label metadata for {len(work_items)} datasets "
                f"across {len(affected_stems)} missions...")

    # Shuffle so slow datasets don't cluster
    random.shuffle(work_items)

    # Try tqdm for interactive terminal progress
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    _use_tqdm = has_tqdm and not any(
        type(h).__name__ == "_ListHandler"
        for h in logging.getLogger("xhelio").handlers
    )

    success_count = 0
    fail_count = 0
    batch_start = time.time()

    def _fetch_one(item):
        ds_id, slot, stem, cache_dir = item
        try:
            info = fetch_ppi_label_metadata(ds_id, slot)
        except Exception as e:
            get_event_bus().emit(METADATA_FETCH, level="debug",
                        msg=f"[PPI-meta] Failed {ds_id}: {e}")
            return ds_id, None
        if info is None:
            return ds_id, None
        # Save to cache
        cache_filename = _dataset_id_to_cache_filename(ds_id)
        cache_file = cache_dir / cache_filename
        try:
            cache_file.write_text(
                json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return ds_id, info

    if _use_tqdm:
        pbar = tqdm(total=len(work_items), desc="PPI metadata", unit="ds")

    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, item): item for item in work_items}
        for future in as_completed(futures):
            ds_id, info = future.result()
            if info is not None:
                success_count += 1
            else:
                fail_count += 1

            if _use_tqdm:
                pbar.set_postfix_str(
                    f"{success_count} ok, {fail_count} err", refresh=False
                )
                pbar.update(1)

    if _use_tqdm:
        pbar.close()

    # Regenerate _index.json for affected missions
    for stem in affected_stems:
        _generate_ppi_index(stem)

    elapsed = time.time() - start_time
    get_event_bus().emit(METADATA_FETCH, level="info",
                msg=f"PPI metadata complete in {elapsed:.0f}s: {success_count} succeeded, "
                f"{fail_count} failed (no label or unsupported format)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_catalog(cdaweb_meta: dict | None = None) -> list[dict]:
    """Fetch the dataset catalog from CDAWeb via CDAS REST API.

    Uses CDAS REST API metadata if provided, otherwise fetches it.
    """
    # If we already have CDAS REST metadata, convert to catalog format
    if cdaweb_meta:
        catalog = [{"id": ds_id} for ds_id in cdaweb_meta]
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                    msg=f"Built catalog from CDAS REST metadata: {len(catalog)} datasets")
        return catalog

    # Fetch from CDAS REST API
    from .cdaweb_metadata import fetch_dataset_metadata
    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg="Fetching dataset catalog from CDAS REST API...")
    meta = fetch_dataset_metadata()
    if meta:
        catalog = [{"id": ds_id} for ds_id in meta]
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                    msg=f"Found {len(catalog)} datasets in CDAS REST catalog")
        return catalog

    raise RuntimeError("Could not fetch dataset catalog from CDAS REST API")


def _group_by_mission(catalog: list[dict]) -> dict[str, list[tuple[str, str | None]]]:
    """Group catalog entries by mission stem.

    Returns:
        Dict mapping mission_stem -> list of (dataset_id, instrument_hint).
    """
    groups: dict[str, list[tuple[str, str | None]]] = {}
    for entry in catalog:
        ds_id = entry.get("id", "")
        mission_stem, instrument_hint = match_dataset_to_mission(ds_id)
        if mission_stem:
            groups.setdefault(mission_stem, []).append((ds_id, instrument_hint))
    return groups


def _fetch_single_info(
    ds_id: str,
    cdaweb_meta: dict | None = None,
) -> dict | None:
    """Fetch metadata for a single dataset from Master CDF.

    Returns parsed info dict or None on failure.
    """
    from .master_cdf import fetch_dataset_metadata_from_master

    # Get start/stop dates from cdaweb_meta if available
    start_date = ""
    stop_date = ""
    if cdaweb_meta:
        entry = cdaweb_meta.get(ds_id)
        if entry:
            start_date = entry.get("start_date", "")
            stop_date = entry.get("stop_date", "")

    return fetch_dataset_metadata_from_master(
        ds_id, start_date=start_date, stop_date=stop_date
    )


def _fetch_and_save(
    ds_id: str,
    cache_dir: Path,
    cdaweb_meta: dict | None = None,
) -> dict | None:
    """Fetch a single dataset's metadata and save to cache. Thread-safe.

    Returns parsed info dict or None on failure.
    """
    info = _fetch_single_info(ds_id, cdaweb_meta=cdaweb_meta)
    if info is None:
        return None

    cache_file = cache_dir / f"{ds_id}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError:
        pass  # Cache write failure is non-fatal

    return info


def _fetch_all_info(
    items: list[tuple[str, str | None, str, Path]],
    cdaweb_meta: dict | None = None,
    source_label: str = "",
    progress_callback=None,
) -> dict[str, dict | None]:
    """Parallel-fetch metadata for all datasets, with retries.

    Args:
        items: List of (dataset_id, instrument_hint, mission_stem, cache_dir).
        cdaweb_meta: Optional CDAS REST metadata dict for date injection.
        source_label: Display label for progress (e.g., "CDAWeb", "PDS PPI").

    Returns:
        Dict mapping dataset_id -> info dict (or None if all retries failed).
    """
    # Try tqdm for interactive terminal progress, fall back to logger.
    # The logger-based path also ensures the web UI's live log can display
    # progress (tqdm writes to stderr which bypasses the logging system).
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # Detect if a _ListHandler is attached (i.e. we're inside the
    # web UI background thread).  In that case prefer logger-based progress
    # so the lines appear in the live log panel.
    _use_tqdm = has_tqdm and not any(
        type(h).__name__ == "_ListHandler"
        for h in logging.getLogger("xhelio").handlers
    )

    total_items = len(items)
    results: dict[str, dict | None] = {}
    pending = [(ds_id, cache_dir) for ds_id, _, _, cache_dir in items]

    for attempt in range(1, MAX_RETRIES + 1):
        if not pending:
            break

        if attempt > 1:
            get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                        msg=f"Retry {attempt}/{MAX_RETRIES}: {len(pending)} datasets remaining...")

        failed = []
        attempt_success = 0
        batch_start = time.time()

        if _use_tqdm:
            prefix = f"{source_label} " if source_label else ""
            desc = (f"{prefix}Retry {attempt}/{MAX_RETRIES} ({len(pending)} left)"
                    if attempt > 1
                    else f"{prefix}metadata ({len(pending)} datasets)")
            pbar = tqdm(
                total=len(pending),
                desc=desc,
                unit="ds",
            )
        else:
            counter = {"done": 0, "total": len(pending)}
            # Log every N items (more frequent for smaller batches)
            _log_every = max(1, len(pending) // 10)

        with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as pool:
            futures = {
                pool.submit(_fetch_and_save, ds_id, cache_dir, cdaweb_meta): (ds_id, cache_dir)
                for ds_id, cache_dir in pending
            }

            for future in as_completed(futures):
                ds_id, cache_dir = futures[future]
                info = future.result()
                results[ds_id] = info

                if info is None:
                    failed.append((ds_id, cache_dir))
                else:
                    attempt_success += 1

                if _use_tqdm:
                    n_ok = attempt_success
                    n_fail = len(failed)
                    pbar.set_postfix_str(f"{n_ok} ok, {n_fail} err", refresh=False)
                    pbar.update(1)
                else:
                    counter["done"] += 1
                    if counter["done"] % _log_every == 0 or counter["done"] == counter["total"]:
                        pct = counter["done"] * 100 // counter["total"]
                        elapsed = time.time() - batch_start
                        rate = counter["done"] / elapsed if elapsed > 0 else 0
                        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                                    msg=f"Downloading: {counter['done']}/{counter['total']} "
                                    f"({pct}%) — {rate:.1f} ds/s, {len(failed)} failed")
                        if progress_callback:
                            phase = source_label.lower().replace(" ", "_") if source_label else "metadata"
                            progress_callback({
                                "type": "progress",
                                "phase": phase,
                                "step": "metadata",
                                "message": f"Downloading metadata: {counter['done']}/{counter['total']} ({pct}%)",
                                "done": counter["done"],
                                "total": counter["total"],
                                "failed": len(failed),
                                "pct": pct,
                            })

        if _use_tqdm:
            pbar.close()

        batch_elapsed = time.time() - batch_start
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                     msg=f"Attempt {attempt}: {attempt_success} succeeded, "
                     f"{len(failed)} failed in {batch_elapsed:.0f}s")

        pending = failed

    if pending:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="warning",
                       msg=f"{len(pending)} datasets failed after {MAX_RETRIES} attempts "
                       "(these datasets may not exist as Master CDFs on CDAWeb)")

    return results


def _merge_dataset_info(metadata_info: dict, cdaweb_entry: dict | None = None) -> dict:
    """Extract dataset entry from metadata + CDAWeb metadata.

    Only stores lightweight catalog info (description, dates, PI, DOI).
    Full parameter details stay in the per-dataset metadata cache files at
    knowledge/missions/{mission}/metadata/{dataset_id}.json — loaded on demand
    by metadata_client.py when the agent needs them.

    Args:
        metadata_info: Metadata response (from Master CDF).
        cdaweb_entry: Optional metadata from CDAWeb REST API for this dataset.
    """
    entry = {
        "description": "",
        "start_date": metadata_info.get("startDate", ""),
        "stop_date": metadata_info.get("stopDate", ""),
    }

    # Enrich with CDAWeb metadata
    if cdaweb_entry:
        entry["description"] = cdaweb_entry.get("label", "")
        if cdaweb_entry.get("pi_name"):
            entry["pi_name"] = cdaweb_entry["pi_name"]
        if cdaweb_entry.get("pi_affiliation"):
            entry["pi_affiliation"] = cdaweb_entry["pi_affiliation"]
        if cdaweb_entry.get("doi"):
            entry["doi"] = cdaweb_entry["doi"]
        if cdaweb_entry.get("notes_url"):
            entry["notes_url"] = cdaweb_entry["notes_url"]

    # Fall back to metadata description if no CDAWeb label
    if not entry["description"]:
        entry["description"] = metadata_info.get("description", "")

    return entry


def _merge_into_missions(
    mission_datasets: dict[str, list[tuple[str, str | None]]],
    results: dict[str, dict | None],
    cdaweb_meta: dict[str, dict] | None = None,
):
    """Merge fetched metadata results into mission JSON files.

    Uses CDAWeb InstrumentType metadata (when available) to group datasets
    into meaningful instrument categories instead of dumping into "General".

    Priority for instrument assignment:
      1. Dataset already exists in a named instrument → keep it
      2. Prefix hint from mission_prefixes → use it (preserves curated structure)
      3. CDAWeb InstrumentType → group by primary type
      4. Fallback → "General"

    After merging, backfills keywords for instruments that have keywords=[].
    """
    if cdaweb_meta is None:
        cdaweb_meta = {}

    from .cdaweb_metadata import pick_primary_type, get_type_info

    for stem, datasets in mission_datasets.items():
        filepath = MISSIONS_DIR / f"{stem}.json"
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            mission_data = json.load(f)

        updated = 0
        for ds_id, instrument_hint in datasets:
            info = results.get(ds_id)
            if info is None:
                continue

            # Priority 1: dataset already exists in a named instrument
            target_instrument = None
            for inst_id, inst in mission_data.get("instruments", {}).items():
                if ds_id in inst.get("datasets", {}):
                    target_instrument = inst_id
                    break

            # Priority 2: prefix hint from mission_prefixes
            if target_instrument is None and instrument_hint:
                if instrument_hint not in mission_data.get("instruments", {}):
                    mission_data.setdefault("instruments", {})[instrument_hint] = {
                        "name": instrument_hint,
                        "keywords": [],
                        "datasets": {},
                    }
                target_instrument = instrument_hint

            # Priority 3: CDAWeb InstrumentType grouping
            if target_instrument is None:
                meta = cdaweb_meta.get(ds_id)
                if meta and meta.get("instrument_types"):
                    primary_type = pick_primary_type(meta["instrument_types"])
                    if primary_type:
                        type_info = get_type_info(primary_type)
                        inst_key = type_info["id"]
                        if inst_key not in mission_data.get("instruments", {}):
                            mission_data.setdefault("instruments", {})[inst_key] = {
                                "name": type_info["name"],
                                "keywords": list(type_info["keywords"]),
                                "datasets": {},
                            }
                        target_instrument = inst_key

            # Priority 4: fallback to "General"
            if target_instrument is None:
                if "General" not in mission_data.get("instruments", {}):
                    mission_data.setdefault("instruments", {})["General"] = {
                        "name": "General",
                        "keywords": [],
                        "datasets": {},
                    }
                target_instrument = "General"

            inst = mission_data["instruments"][target_instrument]
            # Look up CDAWeb metadata; fall back to base ID without @N suffix
            cdaweb_entry = cdaweb_meta.get(ds_id)
            if cdaweb_entry is None and "@" in ds_id:
                cdaweb_entry = cdaweb_meta.get(ds_id.split("@")[0])
            inst.setdefault("datasets", {})[ds_id] = _merge_dataset_info(
                info, cdaweb_entry
            )
            updated += 1

            # Store observatory_group at mission level (from first dataset that has it)
            if cdaweb_entry and cdaweb_entry.get("observatory_group"):
                if "observatory_group" not in mission_data:
                    mission_data["observatory_group"] = cdaweb_entry["observatory_group"]

        # Backfill keywords for instruments that have keywords=[]
        _backfill_instrument_keywords(mission_data, cdaweb_meta)

        # Remove empty "General" if other instruments exist
        instruments = mission_data.get("instruments", {})
        if ("General" in instruments
                and not instruments["General"].get("datasets")
                and len(instruments) > 1):
            del instruments["General"]

        # Update _meta
        mission_data["_meta"] = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "CDAS REST + Master CDF",
        }

        _save_json(filepath, mission_data)

    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg=f"Merged data into {len(mission_datasets)} mission JSON files")


def _backfill_instrument_keywords(
    mission_data: dict,
    cdaweb_meta: dict[str, dict],
):
    """Backfill keywords for instruments that have keywords=[].

    Looks up the InstrumentType of their datasets via CDAWeb metadata
    and sets keywords from the type info.
    """
    from .cdaweb_metadata import pick_primary_type, get_type_info

    for inst_id, inst in mission_data.get("instruments", {}).items():
        if inst.get("keywords"):
            continue  # Already has keywords

        # Collect InstrumentTypes from all datasets in this instrument
        all_types = set()
        for ds_id in inst.get("datasets", {}):
            meta = cdaweb_meta.get(ds_id)
            if meta and meta.get("instrument_types"):
                for t in meta["instrument_types"]:
                    all_types.add(t)

        if not all_types:
            continue

        # Pick primary type and use its keywords
        primary = pick_primary_type(list(all_types))
        if primary:
            type_info = get_type_info(primary)
            if type_info.get("keywords"):
                inst["keywords"] = list(type_info["keywords"])


def _patch_metadata_cache_dates(
    cache_dir: Path, ds_id: str, start_date: str, stop_date: str,
):
    """Patch startDate/stopDate in an individual metadata cache file."""
    cache_file = cache_dir / f"{ds_id}.json"
    if not cache_file.exists():
        return
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        changed = False
        if start_date and info.get("startDate") != start_date:
            info["startDate"] = start_date
            changed = True
        if stop_date and info.get("stopDate") != stop_date:
            info["stopDate"] = stop_date
            changed = True
        if changed:
            _save_json(cache_file, info)
    except (json.JSONDecodeError, OSError):
        pass  # Non-fatal — mission JSON is the primary source of truth


def _generate_index(mission_stem: str):
    """Generate _index.json summary for a mission's metadata cache."""
    cache_dir = MISSIONS_DIR / mission_stem / "metadata"
    if not cache_dir.exists():
        return

    index_entries = []
    for cache_file in sorted(cache_dir.glob("*.json")):
        if cache_file.name.startswith("_"):
            continue
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                info = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        ds_id = cache_file.stem
        param_count = sum(
            1 for p in info.get("parameters", [])
            if p.get("name", "").lower() != "time"
        )
        start_date = info.get("startDate", "")
        stop_date = info.get("stopDate", "")
        if start_date and "T" in start_date:
            start_date = start_date.split("T")[0]
        if stop_date and "T" in stop_date:
            stop_date = stop_date.split("T")[0]

        index_entries.append({
            "id": ds_id,
            "description": info.get("description", ""),
            "start_date": start_date,
            "stop_date": stop_date,
            "parameter_count": param_count,
            "instrument": "",
        })

    # Read mission ID from JSON
    mission_json = MISSIONS_DIR / f"{mission_stem}.json"
    mission_id = mission_stem.upper()
    if mission_json.exists():
        try:
            with open(mission_json, "r", encoding="utf-8") as f:
                mission_id = json.load(f).get("id", mission_id)
        except (json.JSONDecodeError, OSError):
            pass

    index_data = {
        "mission_id": mission_id,
        "dataset_count": len(index_entries),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets": index_entries,
    }
    index_file = cache_dir / "_index.json"
    _save_json(index_file, index_data)


def _ensure_calibration_exclude(mission_stem: str):
    """Create a basic _calibration_exclude.json if one doesn't exist."""
    metadata_dir = MISSIONS_DIR / mission_stem / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    exclude_file = metadata_dir / "_calibration_exclude.json"
    if not exclude_file.exists():
        exclude_data = {
            "description": "Auto-generated exclusion patterns for calibration/housekeeping data",
            "patterns": ["*_K0_*", "*_K1_*", "*_K2_*"],
            "ids": [],
        }
        _save_json(exclude_file, exclude_data)


def refresh_time_ranges(only_stems: set[str] | None = None, progress_callback=None) -> dict:
    """Lightweight refresh: update only start_date/stop_date in mission JSONs.

    Refreshes both CDAWeb and PPI missions:
    - CDAWeb: single HTTP call to CDAS REST API (~3s) for all ~3000 datasets.
    - PPI: parallel HAPI /info calls (~30s) for each PPI dataset.

    Args:
        only_stems: If provided, only refresh these mission stems.
                    If None, refresh every *.json in both source dirs.
        progress_callback: Optional callable(dict) for streaming progress events.

    Returns:
        Dict with keys: missions_updated, datasets_updated,
        datasets_failed, elapsed_seconds.
    """
    if requests is None:
        raise RuntimeError(
            "'requests' package is required for refresh. "
            "Install with: pip install requests"
        )

    start_time = time.time()
    total_updated = 0
    total_failed = 0
    missions_updated = 0
    total_datasets = 0

    # ----- Phase 1: CDAWeb (single HTTP call) -----
    from .cdaweb_metadata import fetch_dataset_metadata
    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg="Fetching CDAWeb catalog for time ranges...")
    if progress_callback:
        progress_callback({"type": "progress", "phase": "cdaweb", "step": "start",
                           "message": "Fetching CDAWeb catalog for time ranges..."})
    cdaweb_meta = fetch_dataset_metadata()
    if cdaweb_meta:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                    msg=f"Got time ranges for {len(cdaweb_meta)} CDAWeb datasets")

        for filepath in sorted(MISSIONS_DIR.glob("*.json")):
            stem = filepath.stem
            if only_stems and stem not in only_stems:
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    mission_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            cache_dir = MISSIONS_DIR / stem / "metadata"
            stem_updated = 0
            for inst in mission_data.get("instruments", {}).values():
                for ds_id, ds_entry in inst.get("datasets", {}).items():
                    total_datasets += 1
                    meta = cdaweb_meta.get(ds_id)
                    if meta is None:
                        total_failed += 1
                        continue
                    new_start = meta.get("start_date", "")
                    new_stop = meta.get("stop_date", "")
                    if new_start:
                        ds_entry["start_date"] = new_start
                    if new_stop:
                        ds_entry["stop_date"] = new_stop
                    stem_updated += 1

                    # Also patch the individual metadata cache file
                    _patch_metadata_cache_dates(cache_dir, ds_id, new_start, new_stop)

            if stem_updated > 0:
                mission_data.setdefault("_meta", {})
                mission_data["_meta"]["generated_at"] = (
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                )
                _save_json(filepath, mission_data)
                missions_updated += 1

            total_updated += stem_updated
            _generate_index(stem)
    else:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="warning",
                    msg="CDAWeb catalog unavailable — skipping CDAWeb refresh")

    if progress_callback:
        progress_callback({"type": "progress", "phase": "cdaweb", "step": "done",
                           "message": f"CDAWeb: {total_updated} datasets updated", "pct": 50})

    # ----- Phase 2: PPI (single Metadex query) -----
    if progress_callback:
        progress_callback({"type": "progress", "phase": "ppi", "step": "start",
                           "message": "Refreshing PPI time ranges from Metadex..."})
    ppi_updated, ppi_failed, ppi_missions = _refresh_ppi_time_ranges(only_stems)
    total_updated += ppi_updated
    total_failed += ppi_failed
    missions_updated += ppi_missions

    elapsed = time.time() - start_time

    if total_failed:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="warning",
                       msg=f"{total_failed} dataset(s) failed to refresh "
                       "(dates left unchanged)")

    msg = (f"Time-range refresh complete in {elapsed:.1f}s: "
           f"{missions_updated} missions, {total_updated} datasets updated")
    if total_failed:
        msg += f", {total_failed} failed"
    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info", msg=msg)
    if progress_callback:
        progress_callback({"type": "progress", "phase": "ppi", "step": "done",
                           "message": msg, "pct": 100})

    return {
        "missions_updated": missions_updated,
        "datasets_updated": total_updated,
        "datasets_failed": total_failed,
        "elapsed_seconds": round(elapsed, 1),
    }


def _refresh_ppi_time_ranges(
    only_stems: set[str] | None = None,
) -> tuple[int, int, int]:
    """Refresh start/stop dates for PPI missions via Metadex.

    Single Metadex query returns fresh dates for all PPI collections.
    Patches the mission JSONs + metadata cache files.

    Returns:
        Tuple of (datasets_updated, datasets_failed, missions_updated).
    """
    if not PPI_DIR.exists():
        return 0, 0, 0

    ppi_jsons = sorted(PPI_DIR.glob("*.json"))
    if not ppi_jsons:
        return 0, 0, 0

    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg="Refreshing PPI time ranges from Metadex...")

    try:
        from .metadex_client import fetch_all_ppi_collections, metadex_id_to_dataset_id
        collections = fetch_all_ppi_collections()
    except Exception as e:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="warning",
                    msg=f"Metadex fetch failed — PPI refresh skipped: {e}")
        return 0, 0, 0

    # Build date map: dataset_id -> (start, stop)
    date_map: dict[str, tuple[str, str]] = {}
    for coll in collections:
        dataset_id = metadex_id_to_dataset_id(coll["id"], coll["archive_type"])
        start = coll.get("start_date_time", "")
        stop = coll.get("stop_date_time", "")
        if start and stop:
            date_map[dataset_id] = (start, stop)

    if not date_map:
        get_event_bus().emit(BOOTSTRAP_PROGRESS, level="warning",
                    msg="No PPI dates from Metadex — PPI refresh skipped")
        return 0, 0, 0

    get_event_bus().emit(BOOTSTRAP_PROGRESS, level="info",
                msg=f"Got time ranges for {len(date_map)} PPI datasets from Metadex")

    from knowledge.metadata_client import _dataset_id_to_cache_filename

    updated_count = 0
    failed_count = 0
    missions_touched = set()

    for filepath in ppi_jsons:
        stem = filepath.stem
        if only_stems and stem not in only_stems:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                mission_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        cache_dir = PPI_DIR / stem / "metadata"
        stem_changed = False
        for inst in mission_data.get("instruments", {}).values():
            for ds_id, ds_entry in inst.get("datasets", {}).items():
                if ds_id not in date_map:
                    failed_count += 1
                    continue

                new_start, new_stop = date_map[ds_id]
                # Truncate to date only for mission JSON
                start_date = new_start.split("T")[0] if "T" in new_start else new_start
                stop_date = new_stop.split("T")[0] if "T" in new_stop else new_stop

                if start_date:
                    ds_entry["start_date"] = start_date
                if stop_date:
                    ds_entry["stop_date"] = stop_date
                updated_count += 1
                stem_changed = True

                # Patch metadata cache file
                if cache_dir.exists():
                    cache_filename = _dataset_id_to_cache_filename(ds_id)
                    cache_file = cache_dir / cache_filename
                    if cache_file.exists():
                        try:
                            with open(cache_file, "r", encoding="utf-8") as f:
                                cached = json.load(f)
                            changed = False
                            if new_start and cached.get("startDate") != new_start:
                                cached["startDate"] = new_start
                                changed = True
                            if new_stop and cached.get("stopDate") != new_stop:
                                cached["stopDate"] = new_stop
                                changed = True
                            if changed:
                                _save_json(cache_file, cached)
                        except (json.JSONDecodeError, OSError):
                            pass

        if stem_changed:
            mission_data.setdefault("_meta", {})
            mission_data["_meta"]["generated_at"] = (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            _save_json(filepath, mission_data)
            missions_touched.add(stem)

            _generate_ppi_index(stem)

    return updated_count, failed_count, len(missions_touched)


def _generate_ppi_index(mission_stem: str):
    """Generate _index.json summary for a PPI mission.

    Reads parameter counts from metadata cache files when available,
    falling back to 0 for datasets that haven't been cached yet.
    """
    from .metadata_client import _dataset_id_to_cache_filename

    cache_dir = PPI_DIR / mission_stem / "metadata"
    cache_dir.mkdir(parents=True, exist_ok=True)

    mission_json = PPI_DIR / f"{mission_stem}.json"
    mission_id = mission_stem.upper()
    index_entries = []

    if mission_json.exists():
        try:
            with open(mission_json, "r", encoding="utf-8") as f:
                mission_data = json.load(f)
            mission_id = mission_data.get("id", mission_id)

            for inst in mission_data.get("instruments", {}).values():
                for ds_id, ds_entry in inst.get("datasets", {}).items():
                    start_date = ds_entry.get("start_date", "")
                    stop_date = ds_entry.get("stop_date", "")
                    if start_date and "T" in start_date:
                        start_date = start_date.split("T")[0]
                    if stop_date and "T" in stop_date:
                        stop_date = stop_date.split("T")[0]

                    # Read parameter count from cache file if available
                    param_count = 0
                    cache_filename = _dataset_id_to_cache_filename(ds_id)
                    cache_file = cache_dir / cache_filename
                    if cache_file.exists():
                        try:
                            with open(cache_file, "r", encoding="utf-8") as cf:
                                cached = json.load(cf)
                            param_count = sum(
                                1 for p in cached.get("parameters", [])
                                if p.get("name", "").lower() != "time"
                            )
                        except (json.JSONDecodeError, OSError):
                            pass

                    index_entries.append({
                        "id": ds_id,
                        "description": ds_entry.get("description", ""),
                        "start_date": start_date,
                        "stop_date": stop_date,
                        "parameter_count": param_count,
                        "instrument": inst.get("name", ""),
                    })
        except (json.JSONDecodeError, OSError):
            pass

    index_data = {
        "mission_id": mission_id,
        "dataset_count": len(index_entries),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets": index_entries,
    }
    index_file = cache_dir / "_index.json"
    _save_json(index_file, index_data)


def _build_ppi_mission_json(
    stem: str,
    collections: list[dict],
) -> dict:
    """Build a PPI mission JSON from Metadex collection dicts.

    Args:
        stem: Mission stem (e.g., ``"juno"``).
        collections: List of normalized Metadex collection dicts,
            each with a ``_dataset_id`` key added by ``populate_ppi_missions()``.
    """
    name = get_mission_name(stem)

    # Build keywords from stem and name
    keywords = set()
    keywords.add(stem.lower())
    for word in name.split():
        w = word.strip("/()")
        if len(w) > 1:
            keywords.add(w.lower())
    keywords.add("ppi")
    keywords.add("pds")

    # Use a distinct ID when a CDAWeb mission with the same stem exists
    cdaweb_json = MISSIONS_DIR / f"{stem}.json"
    has_cdaweb = cdaweb_json.exists()
    canonical_id = get_canonical_id(stem)
    if has_cdaweb:
        mission_id = canonical_id + "_PPI"
        mission_name = f"{name} (PDS PPI)"
        keywords.add(mission_id.lower().replace("-", "_"))
    else:
        mission_id = canonical_id
        mission_name = name

    mission = {
        "id": mission_id,
        "name": mission_name,
        "keywords": sorted(keywords),
        "profile": {
            "description": f"{name} data from PDS Planetary Plasma Interactions archive.",
            "coordinate_systems": [],
            "typical_cadence": "",
            "data_caveats": [
                "PDS3 datasets use fixed-width ASCII tables (.sts/.TAB files).",
                "PDS4 datasets use XML-labeled ASCII tables.",
            ],
            "analysis_patterns": [],
        },
        "instruments": {},
    }

    # Group datasets into instruments using Metadex instrument names
    instrument_datasets: dict[str, dict] = {}

    for coll in collections:
        dataset_id = coll["_dataset_id"]
        title = coll.get("title", "")
        instruments = coll.get("instruments", [])

        # Derive instrument key from Metadex instrument names
        inst_key = _derive_ppi_instrument_key(
            dataset_id, title, instruments=instruments,
        )

        # Truncate dates to date-only for mission JSON
        start = coll.get("start_date_time", "")
        stop = coll.get("stop_date_time", "")
        if start and "T" in start:
            start = start.split("T")[0]
        if stop and "T" in stop:
            stop = stop.split("T")[0]

        ds_entry = {
            "description": title,
            "start_date": start,
            "stop_date": stop,
        }

        # Store slot for PDS3 URL resolution
        slot = coll.get("slot", "")
        if slot:
            ds_entry["slot"] = slot

        # Store archive type for fetch routing
        archive_type = coll.get("archive_type", 0)
        if archive_type:
            ds_entry["archive_type"] = archive_type

        instrument_datasets.setdefault(inst_key, {})[dataset_id] = ds_entry

    # Build instrument entries
    for inst_key, datasets in sorted(instrument_datasets.items()):
        mission["instruments"][inst_key] = {
            "name": inst_key,
            "keywords": _ppi_instrument_keywords(inst_key),
            "datasets": datasets,
        }

    mission["_meta"] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "PDS PPI Metadex",
    }

    return mission


def _derive_ppi_instrument_key(
    dataset_id: str, title: str,
    instruments: list[str] | None = None,
) -> str:
    """Derive an instrument group key from Metadex instrument names, URN, and title.

    Priority:
    1. Metadex instrument name (from ``observing_system.observing_system_component``)
    2. Keyword heuristic from dataset ID and title (fallback)
    """
    # Priority 1: use Metadex instrument name if available
    if instruments:
        for inst_name in instruments:
            mapped = _MAP_INSTRUMENT_NAME.get(inst_name.lower())
            if mapped:
                return mapped
        # Use the first instrument name as-is if no mapping
        return instruments[0]

    # Priority 2: keyword heuristic (fallback for older data)
    raw_id = dataset_id.replace("urn:nasa:pds:", "").replace("pds3:", "")
    lower = raw_id.lower()
    title_lower = title.lower()

    if "fgm" in lower or ("mag" in lower and "image" not in lower):
        return "MAG"
    elif "pls" in lower or "plasma" in title_lower:
        return "Plasma"
    elif "pws" in lower or "radio" in title_lower or "wave" in title_lower:
        return "Waves"
    elif "crs" in lower or "cosmic" in title_lower:
        return "Cosmic Ray"
    elif "lecp" in lower or "energetic" in title_lower:
        return "Energetic Particles"
    elif "jad" in lower or "jade" in title_lower:
        return "JADE"
    elif "jed" in lower or "jedi" in title_lower:
        return "JEDI"
    elif "asc" in lower and "jno" in lower:
        return "ASC"
    elif "swea" in lower or "solar-wind" in lower:
        return "Solar Wind"
    elif "sep" in lower:
        return "SEP"
    elif "swia" in lower or "swi" in lower:
        return "Solar Wind"
    elif "euv" in lower:
        return "EUV"
    elif "lpw" in lower:
        return "LPW"
    elif "static" in lower:
        return "STATIC"

    return "General"


# Map of lowercase Metadex instrument names → clean group keys
_MAP_INSTRUMENT_NAME: dict[str, str] = {
    "magnetometer": "MAG",
    "fluxgate magnetometer": "MAG",
    "mag": "MAG",
    "waves": "Waves",
    "plasma wave instrument": "Waves",
    "plasma wave science": "Waves",
    "plasma wave subsystem": "Waves",
    "plasma wave spectrometer": "Waves",
    "radio and plasma wave investigation": "Waves",
    "jovian auroral distributions experiment": "JADE",
    "jupiter energetic particle detector instrument": "JEDI",
    "advanced stellar compass": "ASC",
    "plasma science": "Plasma",
    "plasma": "Plasma",
    "plasma instrument": "Plasma",
    "plasma analyzer": "Plasma",
    "solar wind electron analyzer": "Solar Wind",
    "solar wind ion analyzer": "Solar Wind",
    "solar wind around pluto": "Solar Wind",
    "solar energetic particle": "SEP",
    "cosmic ray subsystem": "Cosmic Ray",
    "low energy charged particle": "Energetic Particles",
    "energetic particles detector": "Energetic Particles",
    "extreme ultraviolet monitor": "EUV",
    "langmuir probe and waves": "LPW",
    "suprathermal and thermal ion composition": "STATIC",
}


def _ppi_instrument_keywords(inst_key: str) -> list[str]:
    """Return keywords for known PPI instrument types."""
    kw_map = {
        "MAG": ["magnetic", "field", "mag", "magnetometer"],
        "Plasma": ["plasma", "ion", "electron", "density", "velocity"],
        "Plasma Waves": ["radio", "wave", "plasma wave"],
        "Cosmic Ray": ["particle", "energetic", "cosmic ray"],
        "Energetic Particles": ["particle", "energetic"],
        "Solar Wind": ["plasma", "solar wind", "ion"],
        "SEP": ["particle", "energetic"],
        "EUV": ["imaging", "remote sensing"],
        "LPW": ["electric", "e-field"],
        "STATIC": ["plasma", "ion"],
    }
    return kw_map.get(inst_key, [])


def clean_all_missions(
    only_stems: set[str] | None = None,
    source: str | None = None,
):
    """Delete mission JSONs and metadata cache dirs for a fresh rebuild.

    Args:
        only_stems: If provided, only delete these mission stems.
                    If None, delete everything.
        source: If ``"cdaweb"`` only clean CDAWeb missions.
                If ``"ppi"`` only clean PPI missions.
                If ``None`` (default) clean both.

    Preserves the missions/ directory itself but removes generated content.
    Returns the count of (deleted_files, deleted_dirs) for logging.
    """
    global _bootstrap_checked
    _bootstrap_checked = False

    deleted_files = 0
    deleted_dirs = 0

    import shutil

    if source == "cdaweb":
        source_dirs = (MISSIONS_DIR,)
    elif source == "ppi":
        source_dirs = (PPI_DIR,)
    else:
        source_dirs = (MISSIONS_DIR, PPI_DIR)

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue

        for filepath in source_dir.glob("*.json"):
            if only_stems and filepath.stem not in only_stems:
                continue
            filepath.unlink()
            deleted_files += 1

        for subdir in source_dir.iterdir():
            if subdir.is_dir():
                if only_stems and subdir.name not in only_stems:
                    continue
                shutil.rmtree(subdir)
                deleted_dirs += 1

    return deleted_files, deleted_dirs


def _save_json(filepath: Path, data: dict):
    """Save a dict as JSON with consistent formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")
