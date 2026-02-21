"""
Full dataset catalog fetch, cache, and search.

Provides access to the combined CDAWeb + PPI dataset catalog via the
CDAS REST API and local PPI mission JSONs. The CDAWeb portion is cached
to disk and refreshed every 24 hours; PPI entries are built from the
local mission JSON files on each load.

Search supports two methods (controlled by config.CATALOG_SEARCH_METHOD):
- "semantic": fastembed cosine similarity (default, much better for NL queries)
- "substring": case-insensitive multi-word substring matching (fast fallback)

Used by the `search_full_catalog` tool to let users find datasets
across all missions (CDAWeb and PPI).
"""

import json
import time
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    requests = None

from config import get_data_dir
from agent.event_bus import get_event_bus, DEBUG, CATALOG_SEARCH

CATALOG_CACHE = get_data_dir() / "catalog_cache.json"
EMBEDDINGS_CACHE = get_data_dir() / "catalog_embeddings.npy"
TEXTS_CACHE = get_data_dir() / "catalog_embed_texts.npy"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# PPI mission JSONs directory
_PPI_DIR = Path(__file__).parent / "missions" / "ppi"

# Lazy state for semantic search
_fastembed_available = None  # None = not checked yet
_embedding_model = None
_catalog_embeddings = None
_catalog_texts = None


def get_full_catalog() -> list[dict]:
    """Fetch and cache the full dataset catalog (CDAWeb + PPI).

    CDAWeb entries come from the CDAS REST API (cached to disk, 24h TTL).
    PPI entries are built from local mission JSON files on each call.
    Returns a list of dicts with 'id' and 'title' keys.

    Returns:
        List of catalog entries, each with 'id' and 'title'.
    """
    catalog = _get_cdaweb_catalog()

    # Append PPI datasets from local mission JSONs
    ppi_entries = _get_ppi_entries()
    if ppi_entries:
        # Deduplicate: PPI IDs shouldn't overlap with CDAWeb, but be safe
        existing_ids = {e["id"] for e in catalog}
        for entry in ppi_entries:
            if entry["id"] not in existing_ids:
                catalog.append(entry)

    return catalog


def _get_cdaweb_catalog() -> list[dict]:
    """Fetch and cache CDAWeb catalog entries."""
    # Check cache freshness
    if CATALOG_CACHE.exists():
        age = time.time() - CATALOG_CACHE.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            try:
                with open(CATALOG_CACHE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("catalog", [])
            except (json.JSONDecodeError, KeyError):
                pass  # Re-fetch on corrupt cache

    if requests is None:
        return []

    # Fetch from CDAS REST API
    catalog = _fetch_from_cdas_rest()

    if not catalog:
        # If all fetches fail, use stale cache
        if CATALOG_CACHE.exists():
            try:
                with open(CATALOG_CACHE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("catalog", [])
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    # Save cache
    data = {"catalog": catalog}
    CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return catalog


def _fetch_from_cdas_rest() -> list[dict]:
    """Fetch catalog from CDAS REST API and convert to id/title list."""
    try:
        from .cdaweb_metadata import fetch_dataset_metadata
        cdaweb_meta = fetch_dataset_metadata()
        if not cdaweb_meta:
            return []
        return [
            {"id": ds_id, "title": meta.get("label", "")}
            for ds_id, meta in sorted(cdaweb_meta.items())
        ]
    except Exception:
        return []


def _get_ppi_entries() -> list[dict]:
    """Build catalog entries from local PPI mission JSON files.

    Walks knowledge/missions/ppi/*.json and extracts dataset IDs and
    descriptions from each mission's instruments.
    """
    if not _PPI_DIR.exists():
        return []

    entries = []
    for filepath in sorted(_PPI_DIR.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                mission_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        mission_name = mission_data.get("name", filepath.stem)
        for inst in mission_data.get("instruments", {}).values():
            for ds_id, ds_info in inst.get("datasets", {}).items():
                title = ds_info.get("description", "")
                if not title:
                    title = f"{mission_name} dataset"
                entries.append({"id": ds_id, "title": title})

    return entries


def _ensure_fastembed() -> bool:
    """Lazy check for fastembed availability. Auto-installs if missing."""
    global _fastembed_available
    if _fastembed_available is not None:
        return _fastembed_available
    try:
        import fastembed  # noqa: F401
        _fastembed_available = True
    except ImportError:
        get_event_bus().emit(CATALOG_SEARCH, agent="CatalogSearch", level="warning", msg="fastembed not installed — attempting auto-install...")
        try:
            import subprocess
            import sys
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "fastembed"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import fastembed  # noqa: F401
            _fastembed_available = True
            get_event_bus().emit(CATALOG_SEARCH, agent="CatalogSearch", level="info", msg="fastembed installed successfully")
        except Exception as exc:
            _fastembed_available = False
            get_event_bus().emit(CATALOG_SEARCH, agent="CatalogSearch", level="warning", msg=f"Failed to install fastembed: {exc} — falling back to substring search")
    return _fastembed_available


def _build_or_load_embeddings(catalog: list[dict]):
    """Build or load cached embeddings for catalog entries.

    Returns (embeddings_matrix, texts_list, model) or (None, None, None) on failure.
    """
    global _embedding_model, _catalog_embeddings, _catalog_texts

    # Return cached if already loaded
    if _catalog_embeddings is not None and _catalog_texts is not None:
        return _catalog_embeddings, _catalog_texts, _embedding_model

    texts = [f"{e.get('id', '')} {e.get('title', '')}" for e in catalog]

    # Try loading from disk cache
    if EMBEDDINGS_CACHE.exists() and TEXTS_CACHE.exists():
        try:
            cached_texts = np.load(TEXTS_CACHE, allow_pickle=True).tolist()
            if cached_texts == texts:
                embeddings = np.load(EMBEDDINGS_CACHE)
                get_event_bus().emit(DEBUG, agent="CatalogSearch", msg=f"Loaded cached embeddings ({len(texts)} entries)")
                _catalog_embeddings = embeddings
                _catalog_texts = texts
                # Model still needed for query embedding
                if _embedding_model is None:
                    from fastembed import TextEmbedding
                    _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                return _catalog_embeddings, _catalog_texts, _embedding_model
        except Exception:
            pass  # Rebuild on any cache issue

    # Build fresh embeddings
    try:
        from fastembed import TextEmbedding

        if _embedding_model is None:
            _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        embeddings = np.array(list(_embedding_model.embed(texts, batch_size=256)))

        # Save cache
        EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_CACHE, embeddings)
        np.save(TEXTS_CACHE, np.array(texts, dtype=object))
        get_event_bus().emit(DEBUG, agent="CatalogSearch", msg=f"Built and cached embeddings ({len(texts)} entries, shape {embeddings.shape})")

        _catalog_embeddings = embeddings
        _catalog_texts = texts
        return _catalog_embeddings, _catalog_texts, _embedding_model
    except Exception as exc:
        get_event_bus().emit(CATALOG_SEARCH, agent="CatalogSearch", level="warning", msg=f"Failed to build embeddings: {exc}")
        return None, None, None


def _semantic_search(query: str, catalog: list[dict], max_results: int) -> list[dict]:
    """Search catalog using fastembed cosine similarity.

    Falls back to _substring_search on any failure.
    """
    try:
        embeddings, texts, model = _build_or_load_embeddings(catalog)
        if embeddings is None:
            return _substring_search(query, catalog, max_results)

        query_emb = np.array(list(model.embed([query])))[0]
        similarities = embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:max_results]

        return [
            {"id": catalog[idx]["id"], "title": catalog[idx]["title"]}
            for idx in top_indices
        ]
    except Exception as exc:
        get_event_bus().emit(CATALOG_SEARCH, agent="CatalogSearch", level="warning", msg=f"Semantic search failed: {exc} — falling back to substring")
        return _substring_search(query, catalog, max_results)


def _substring_search(query: str, catalog: list[dict], max_results: int) -> list[dict]:
    """Search catalog using case-insensitive substring matching.

    All query words must appear in the combined id+title text.
    """
    words = query.lower().split()
    if not words:
        return []

    matches = []
    for entry in catalog:
        text = f"{entry.get('id', '')} {entry.get('title', '')}".lower()
        if all(w in text for w in words):
            matches.append({
                "id": entry.get("id", ""),
                "title": entry.get("title", ""),
            })
            if len(matches) >= max_results:
                break

    return matches


def search_catalog(query: str, max_results: int = 20) -> list[dict]:
    """Search the full dataset catalog (CDAWeb + PPI).

    Uses semantic search (fastembed) by default, with automatic fallback
    to substring matching if fastembed is unavailable or fails.
    Controlled by config.CATALOG_SEARCH_METHOD ("semantic" or "substring").

    Args:
        query: Search terms (e.g., "solar wind proton density", "ACE magnetometer").
        max_results: Maximum number of results to return.

    Returns:
        List of matching catalog entries with 'id' and 'title'.
    """
    import config

    catalog = get_full_catalog()
    if not catalog or not query.strip():
        return []

    if config.CATALOG_SEARCH_METHOD == "semantic" and _ensure_fastembed():
        return _semantic_search(query, catalog, max_results)

    return _substring_search(query, catalog, max_results)


def get_catalog_stats() -> dict:
    """Return basic statistics about the cached catalog.

    Returns:
        Dict with 'total_datasets', 'cache_age_hours', 'cache_exists'.
    """
    catalog = get_full_catalog()
    stats = {
        "total_datasets": len(catalog),
        "cache_exists": CATALOG_CACHE.exists(),
        "cache_age_hours": None,
    }
    if CATALOG_CACHE.exists():
        age_hours = (time.time() - CATALOG_CACHE.stat().st_mtime) / 3600
        stats["cache_age_hours"] = round(age_hours, 1)
    return stats
