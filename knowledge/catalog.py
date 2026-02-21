"""
Static spacecraft/instrument catalog with keyword matching.

Provides fast local lookup to map natural language queries to CDAWeb dataset IDs.
Parameter metadata is fetched separately via Master CDF files.

Data is loaded from per-mission JSON files in knowledge/missions/.
"""

import re
from typing import Optional

from .mission_loader import load_all_missions


def _build_spacecraft_dict() -> dict:
    """Build the SPACECRAFT dict from per-mission JSON files.

    Transforms the JSON structure (datasets as dicts) into the legacy
    format (datasets as lists) for backward compatibility with code that
    iterates over inst["datasets"] expecting a list of strings.
    """
    missions = load_all_missions()
    result = {}
    for mission_id, mission in missions.items():
        instruments = {}
        for inst_id, inst in mission.get("instruments", {}).items():
            # Convert datasets dict to list of dataset ID strings
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


# Backward-compatible module-level dict.
# Instruments have datasets as lists of strings, not dicts.
SPACECRAFT = _build_spacecraft_dict()


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


def list_spacecraft() -> list[dict]:
    """List all supported spacecraft.

    Returns:
        List of dicts with 'id' and 'name' keys.
    """
    return [
        {"id": sc_id, "name": info["name"]}
        for sc_id, info in SPACECRAFT.items()
    ]


def list_instruments(spacecraft: str) -> list[dict]:
    """List instruments for a spacecraft.

    Args:
        spacecraft: Spacecraft ID (e.g., "PSP", "ACE")

    Returns:
        List of dicts with 'id' and 'name' keys.
    """
    sc = SPACECRAFT.get(spacecraft)
    if not sc:
        return []
    return [
        {"id": inst_id, "name": info["name"]}
        for inst_id, info in sc["instruments"].items()
    ]


def get_datasets(spacecraft: str, instrument: str) -> list[str]:
    """Get dataset IDs for a spacecraft/instrument combination.

    Args:
        spacecraft: Spacecraft ID
        instrument: Instrument ID

    Returns:
        List of CDAWeb dataset IDs.
    """
    sc = SPACECRAFT.get(spacecraft)
    if not sc:
        return []
    inst = sc["instruments"].get(instrument)
    if not inst:
        return []
    return inst["datasets"]


def _keyword_in_query(kw: str, query: str) -> bool:
    """Check if a keyword appears in query as a whole word (word-boundary match).

    This prevents short keywords like 'co' (Cassini) from matching inside
    unrelated words like 'dscovr'.
    """
    return bool(re.search(r'\b' + re.escape(kw) + r'\b', query))


def match_spacecraft(query: str) -> Optional[str]:
    """Match a query string to a spacecraft using keywords.

    Uses word-boundary matching and prefers longer keyword matches to avoid
    false positives from short prefixes (e.g., 'co' matching inside 'dscovr').

    Args:
        query: User's search query

    Returns:
        Spacecraft ID or None if no match.
    """
    query_lower = query.lower()

    # First pass: exact ID match
    for sc_id in SPACECRAFT:
        if query_lower == sc_id.lower():
            return sc_id

    # Second pass: check if any spacecraft ID or name appears as a word in the query
    # This catches "ace solar wind" → ACE (the ID "ace" is in the query)
    for sc_id, info in SPACECRAFT.items():
        if _keyword_in_query(sc_id.lower(), query_lower):
            return sc_id
        name_lower = info["name"].lower()
        if _keyword_in_query(name_lower, query_lower):
            return sc_id

    # Third pass: keyword match with word boundaries, prefer longest match
    best_match = None
    best_kw_len = 0
    for sc_id, info in SPACECRAFT.items():
        for kw in info["keywords"]:
            if len(kw) > best_kw_len and _keyword_in_query(kw, query_lower):
                best_match = sc_id
                best_kw_len = len(kw)

    return best_match


def match_instrument(spacecraft: str, query: str) -> Optional[str]:
    """Match a query string to an instrument using keywords.

    Args:
        spacecraft: Spacecraft ID to search within
        query: User's search query

    Returns:
        Instrument ID or None if no match.
    """
    sc = SPACECRAFT.get(spacecraft)
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
    """Combined search: find spacecraft, instrument, and datasets from a query.

    This is the main entry point for natural language dataset search.

    Args:
        query: User's natural language query (e.g., "parker magnetic field")

    Returns:
        Dict with spacecraft, instrument, datasets, or None if no match.
        Example: {
            "spacecraft": "PSP",
            "spacecraft_name": "Parker Solar Probe",
            "instrument": "FIELDS/MAG",
            "instrument_name": "FIELDS Magnetometer",
            "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"]
        }
    """
    # Step 1: Match spacecraft
    spacecraft = match_spacecraft(query)
    if not spacecraft:
        return None

    sc_info = SPACECRAFT[spacecraft]

    # Step 2: Match instrument
    instrument = match_instrument(spacecraft, query)
    if not instrument:
        # No instrument match — return all datasets so the agent can choose
        all_datasets = []
        for inst in sc_info["instruments"].values():
            all_datasets.extend(inst["datasets"])
        return {
            "spacecraft": spacecraft,
            "spacecraft_name": sc_info["name"],
            "instrument": None,
            "instrument_name": None,
            "datasets": all_datasets,
            "available_instruments": list_instruments(spacecraft),
        }

    inst_info = sc_info["instruments"][instrument]
    datasets = inst_info["datasets"]

    return {
        "spacecraft": spacecraft,
        "spacecraft_name": sc_info["name"],
        "instrument": instrument,
        "instrument_name": inst_info["name"],
        "datasets": datasets,
    }
