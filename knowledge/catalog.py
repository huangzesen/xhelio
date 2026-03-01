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


# --- SPICE synthetic catalog entry ---
# SPICE is not a CDAWeb/PPI mission — it's injected programmatically.
_SPICE_ENTRY = {
    "name": "SPICE Ephemeris",
    "keywords": [
        "spice", "ephemeris", "trajectory", "orbit", "position",
        "velocity", "distance", "coordinate", "transform",
        "naif", "kernels",
    ],
    "profile": {
        "description": "Ephemeris only: spacecraft position, velocity, trajectory, distance, and coordinate transforms via SPICE/NAIF kernels. Does NOT provide science/instrument data (magnetic field, plasma, particles, etc.) — use the mission-specific agents for science data.",
        "coordinate_systems": ["ECLIPJ2000", "J2000", "GSE", "GSM", "RTN", "HCI", "HAE"],
        "typical_cadence": "configurable (1m to 1d)",
        "data_caveats": [],
        "analysis_patterns": [
            "Use get_spacecraft_ephemeris for position/velocity at a single time or as a timeseries",
            "Use compute_distance for distance between two bodies over a time range",
            "Use transform_coordinates to convert vectors between coordinate frames",
            "Use list_spice_missions to see all supported spacecraft",
            "Use list_coordinate_frames to see available frames before querying",
        ],
    },
    "instruments": {},
}


def _inject_spice_entry(data: dict) -> None:
    """Inject the SPICE synthetic entry into the mission dict."""
    data["SPICE"] = _SPICE_ENTRY


def update_spice_from_mcp(missions: list[dict], frames: list[dict]) -> None:
    """Update the SPICE catalog entry from live MCP server data.

    Derives description and coordinate_systems from the actual MCP server
    responses, replacing the hardcoded defaults. Since ``_SPICE_ENTRY`` is
    the same dict object referenced by ``MISSIONS["SPICE"]``, mutating it
    updates the catalog in-place.

    Args:
        missions: List of mission dicts from ``list_spice_missions`` response
            (each has ``mission_key``, ``has_kernels``, etc.).
        frames: List of frame dicts from ``list_coordinate_frames`` response
            (each has ``frame``, ``full_name``, etc.).
    """
    profile = _SPICE_ENTRY["profile"]

    # --- Update coordinate_systems from live frame list ---
    frame_names = [f["frame"] for f in frames if "frame" in f]
    if frame_names:
        profile["coordinate_systems"] = frame_names

    # --- Derive description from live mission list ---
    with_kernels = [m["mission_key"] for m in missions if m.get("has_kernels")]
    n_with_kernels = len(with_kernels)
    n_total = len(missions)

    if n_total > 0:
        profile["description"] = (
            f"Ephemeris only: spacecraft position, velocity, trajectory, "
            f"distance, and coordinate transforms via SPICE/NAIF kernels "
            f"({n_with_kernels} spacecraft with kernels, {n_total} total). "
            f"Does NOT provide science/instrument data (magnetic field, "
            f"plasma, particles, etc.) — use the mission-specific agents "
            f"for science data."
        )


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
            _inject_spice_entry(data)
            super().update(data)
            self._loaded = True

    def reload(self) -> None:
        """Force-reload from disk.  Called after background load completes."""
        with self._lock:
            super().clear()
            missions = load_all_missions_from_disk()
            data = _build_mission_dict(missions)
            _inject_spice_entry(data)
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
    return bool(re.search(r'\b' + re.escape(kw) + r'\b', query))


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
