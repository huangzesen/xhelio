"""
Static mission/instrument catalog with keyword matching.

Provides fast local lookup to map natural language queries to CDAWeb dataset IDs.
Parameter metadata is fetched separately via Master CDF files.

Data is loaded from per-mission JSON files in knowledge/missions/.
"""

import re
import threading
from typing import Optional

from .mission_loader import load_all_missions_from_disk


def _build_mission_dict(missions: dict) -> dict:
    """Transform the raw missions dict into the MISSIONS catalog format.

    Converts datasets from dicts to lists for backward compatibility with
    code that iterates over inst["datasets"] expecting a list of strings.
    """
    result = {}
    for mission_id, mission in missions.items():
        instruments = {}
        for inst_id, inst in mission.get("instruments", {}).items():
            datasets_dict = inst.get("datasets", {})
            dataset_ids = list(datasets_dict.keys())
            instruments[inst_id] = {
                "name": inst["name"],
                "keywords": inst["keywords"],
                "datasets": dataset_ids,
            }
        result[mission_id] = {
            "name": mission["name"],
            "keywords": mission["keywords"],
            "profile": mission.get("profile", {}),
            "instruments": instruments,
        }
    return result


class _LazyMissionDict(dict):
    """Dict subclass that loads mission data lazily on first access.

    - Does NOT load at construction time (instant startup).
    - On first dict access, reads existing JSON files from disk (no bootstrap).
    - Has a ``reload()`` method called when background loading completes.
    - Thread-safe via ``threading.Lock``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._loaded = False

    def _try_load(self) -> None:
        """Load from disk if not already loaded.  No bootstrap."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            missions = load_all_missions_from_disk()
            data = _build_mission_dict(missions)
            super().update(data)
            self._loaded = True

    def reload(self) -> None:
        """Force-reload from disk.  Called after background load completes."""
        with self._lock:
            super().clear()
            missions = load_all_missions_from_disk()
            data = _build_mission_dict(missions)
            super().update(data)
            self._loaded = True

    # -- override all read methods to trigger lazy load --

    def __getitem__(self, key):
        self._try_load()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._try_load()
        return super().__contains__(key)

    def get(self, key, default=None):
        self._try_load()
        return super().get(key, default)

    def items(self):
        self._try_load()
        return super().items()

    def values(self):
        self._try_load()
        return super().values()

    def keys(self):
        self._try_load()
        return super().keys()

    def __iter__(self):
        self._try_load()
        return super().__iter__()

    def __len__(self):
        self._try_load()
        return super().__len__()

    def __bool__(self):
        self._try_load()
        return super().__bool__()


# Module-level mission catalog dict.
# Instruments have datasets as lists of strings, not dicts.
# Loads lazily on first access — does NOT trigger bootstrap.
MISSIONS = _LazyMissionDict()


# Backward-compatible aliases
SPACECRAFT = MISSIONS


def classify_instrument_type(keywords: list[str]) -> str:
    """Classify instrument type from keywords.

    Used by browse_datasets enrichment and prompt generation to categorize
    datasets by physical measurement type.

    Args:
        keywords: List of keyword strings from the mission JSON.

    Returns:
        A type string: magnetic, plasma, particles, electric, waves, indices,
        ephemeris, or other.
    """
    kws = [k.lower() for k in keywords]
    if any(k in kws for k in ("magnetic", "mag", "b-field", "magnetometer")):
        return "magnetic"
    if any(k in kws for k in ("plasma", "solar wind", "ion", "electron")):
        return "plasma"
    if any(k in kws for k in ("particle", "energetic", "cosmic ray")):
        return "particles"
    if any(k in kws for k in ("electric", "e-field")):
        return "electric"
    if any(k in kws for k in ("radio", "wave", "plasma wave")):
        return "waves"
    if any(k in kws for k in ("index", "indices", "geomagnetic")):
        return "indices"
    if any(k in kws for k in ("ephemeris", "orbit", "attitude", "position")):
        return "ephemeris"
    return "other"


def list_missions_catalog() -> list[dict]:
    """List all supported missions.

    Returns:
        List of dicts with 'id' and 'name' keys.
    """
    return [
        {"id": mission_id, "name": info["name"]}
        for mission_id, info in MISSIONS.items()
    ]


# Backward-compatible alias
list_spacecraft = list_missions_catalog


def list_instruments(mission_id: str) -> list[dict]:
    """List instruments for a mission.

    Args:
        mission_id: Mission ID (e.g., "PSP", "ACE")

    Returns:
        List of dicts with 'id' and 'name' keys.
    """
    sc = MISSIONS.get(mission_id)
    if not sc:
        return []
    return [
        {"id": inst_id, "name": info["name"]}
        for inst_id, info in sc["instruments"].items()
    ]


def _keyword_in_query(kw: str, query: str) -> bool:
    """Check if a keyword appears in query as a whole word (word-boundary match).

    This prevents short keywords like 'co' (Cassini) from matching inside
    unrelated words like 'dscovr'.
    """
    return bool(re.search(r"\b" + re.escape(kw) + r"\b", query))


def match_mission(query: str) -> Optional[str]:
    """Match a query string to a mission using keywords.

    Uses word-boundary matching and prefers longer keyword matches to avoid
    false positives from short prefixes (e.g., 'co' matching inside 'dscovr').

    Args:
        query: User's search query

    Returns:
        Mission ID or None if no match.
    """
    query_lower = query.lower()

    # First pass: exact ID match
    for mission_id in MISSIONS:
        if query_lower == mission_id.lower():
            return mission_id

    # Second pass: check if any mission ID or name appears as a word in the query
    # This catches "ace solar wind" → ACE (the ID "ace" is in the query)
    for mission_id, info in MISSIONS.items():
        if _keyword_in_query(mission_id.lower(), query_lower):
            return mission_id
        name_lower = info["name"].lower()
        if _keyword_in_query(name_lower, query_lower):
            return mission_id

    # Third pass: keyword match with word boundaries, prefer longest match
    best_match = None
    best_kw_len = 0
    for mission_id, info in MISSIONS.items():
        for kw in info["keywords"]:
            if len(kw) > best_kw_len and _keyword_in_query(kw, query_lower):
                best_match = mission_id
                best_kw_len = len(kw)

    return best_match


# Backward-compatible alias
match_spacecraft = match_mission


def match_instrument(mission_id: str, query: str) -> Optional[str]:
    """Match a query string to an instrument using keywords.

    Args:
        mission_id: Mission ID to search within
        query: User's search query

    Returns:
        Instrument ID or None if no match.
    """
    sc = MISSIONS.get(mission_id)
    if not sc:
        return None

    query_lower = query.lower()

    # First pass: exact keyword matches (prefer precise over substring)
    for inst_id, info in sc["instruments"].items():
        # Check exact match on ID
        if query_lower == inst_id.lower():
            return inst_id
        # Check exact keyword match
        for kw in info["keywords"]:
            if kw == query_lower:
                return inst_id

    # Second pass: word-boundary keyword matches, prefer longest
    best_match = None
    best_kw_len = 0
    for inst_id, info in sc["instruments"].items():
        for kw in info["keywords"]:
            if len(kw) > best_kw_len and _keyword_in_query(kw, query_lower):
                best_match = inst_id
                best_kw_len = len(kw)

    return best_match


def search_by_keywords(query: str) -> Optional[dict]:
    """Combined search: find mission, instrument, and datasets from a query.

    This is the main entry point for natural language dataset search.

    Args:
        query: User's natural language query (e.g., "parker magnetic field")

    Returns:
        Dict with mission, instrument, datasets, or None if no match.
        Example: {
            "mission": "PSP",
            "mission_name": "Parker Solar Probe",
            "instrument": "FIELDS/MAG",
            "instrument_name": "FIELDS Magnetometer",
            "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"]
        }
    """
    # Step 1: Match mission
    matched = match_mission(query)
    if not matched:
        return None

    mission_info = MISSIONS[matched]

    # Step 2: Match instrument
    instrument = match_instrument(matched, query)
    if not instrument:
        # No instrument match — return all datasets so the agent can choose
        all_datasets = []
        for inst in mission_info["instruments"].values():
            all_datasets.extend(inst["datasets"])
        return {
            "mission": matched,
            "mission_name": mission_info["name"],
            "instrument": None,
            "instrument_name": None,
            "datasets": all_datasets,
            "available_instruments": list_instruments(matched),
        }

    inst_info = mission_info["instruments"][instrument]
    datasets = inst_info["datasets"]

    return {
        "mission": matched,
        "mission_name": mission_info["name"],
        "instrument": instrument,
        "instrument_name": inst_info["name"],
        "datasets": datasets,
    }
