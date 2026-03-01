"""
CDAWeb REST API client for dataset InstrumentType metadata.

Fetches the CDAWeb dataset catalog (XML) and maps each dataset to its
InstrumentType(s). Used by bootstrap.py to group datasets into meaningful
instrument categories instead of dumping everything into "General".

The CDAWeb REST API returns ~3000 datasets with authoritative taxonomy:
18+ InstrumentType categories like "Magnetic Fields (space)", "Plasma and
Solar Wind", etc.
"""

try:
    import requests
except ImportError:
    requests = None

try:
    import xml.etree.ElementTree as ET
except ImportError:
    ET = None


CDAWEB_REST_URL = (
    "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets"
)

# XML namespace used by CDAWeb REST API
_NS = {"cda": "http://cdaweb.gsfc.nasa.gov/schema"}


# ---------------------------------------------------------------------------
# Static mapping: CDAWeb InstrumentType string → instrument info
# ---------------------------------------------------------------------------

INSTRUMENT_TYPE_INFO = {
    "Magnetic Fields (space)": {
        "id": "mag",
        "name": "Magnetic Fields",
        "keywords": ["magnetic", "field", "mag", "b-field", "imf", "magnetometer"],
    },
    "Plasma and Solar Wind": {
        "id": "plasma",
        "name": "Plasma and Solar Wind",
        "keywords": ["plasma", "solar wind", "proton", "density", "velocity",
                      "temperature", "ion", "electron"],
    },
    "Particles (space)": {
        "id": "particles",
        "name": "Energetic Particles",
        "keywords": ["particle", "energetic", "cosmic ray"],
    },
    "Electric Fields (space)": {
        "id": "efield",
        "name": "Electric Fields",
        "keywords": ["electric", "e-field"],
    },
    "Radio and Plasma Waves (space)": {
        "id": "waves",
        "name": "Radio and Plasma Waves",
        "keywords": ["radio", "wave", "plasma wave"],
    },
    "Activity Indices": {
        "id": "indices",
        "name": "Activity Indices",
        "keywords": ["index", "indices", "geomagnetic", "sym-h", "dst", "kp", "ae"],
    },
    "Ephemeris/Attitude/Ancillary": {
        "id": "ephemeris",
        "name": "Ephemeris and Attitude",
        "keywords": ["ephemeris", "orbit", "attitude", "position"],
    },
    "Ground-Based Magnetometers, Riometers, Sounders": {
        "id": "ground_mag",
        "name": "Ground-Based Magnetometers",
        "keywords": ["ground", "magnetometer", "riometer"],
    },
    "Imaging and Remote Sensing (Magnetosphere/Earth)": {
        "id": "imaging_mag",
        "name": "Magnetospheric Imaging",
        "keywords": ["imaging", "remote sensing", "magnetosphere"],
    },
    "Imaging and Remote Sensing (ITM/Earth)": {
        "id": "imaging_itm",
        "name": "ITM Imaging",
        "keywords": ["imaging", "ionosphere", "thermosphere", "mesosphere"],
    },
    "Imaging and Remote Sensing (Sun)": {
        "id": "imaging_sun",
        "name": "Solar Imaging",
        "keywords": ["solar", "coronagraph", "euv", "uv", "x-ray"],
    },
    "Engineering": {
        "id": "engineering",
        "name": "Engineering",
        "keywords": ["engineering", "housekeeping"],
    },
    "Housekeeping": {
        "id": "engineering",
        "name": "Engineering",
        "keywords": ["engineering", "housekeeping"],
    },
    "Magnetic Fields (Balloon)": {
        "id": "mag_balloon",
        "name": "Balloon Magnetic Fields",
        "keywords": ["magnetic", "balloon"],
    },
    "Particles (Balloon)": {
        "id": "particles_balloon",
        "name": "Balloon Particles",
        "keywords": ["particle", "balloon"],
    },
    "Energetic Particle Detector": {
        "id": "particles",
        "name": "Energetic Particles",
        "keywords": ["particle", "energetic"],
    },
    "Plasma Composition/Charge State Analyzers": {
        "id": "composition",
        "name": "Plasma Composition",
        "keywords": ["composition", "charge state", "ion"],
    },
    "Coronagraph/Heliograph": {
        "id": "coronagraph",
        "name": "Coronagraph",
        "keywords": ["coronagraph", "heliograph", "solar"],
    },
}


# Priority order: science-first, engineering last.
# Used to pick the "primary" type when a dataset has multiple InstrumentTypes.
INSTRUMENT_TYPE_PRIORITY = [
    "Magnetic Fields (space)",
    "Plasma and Solar Wind",
    "Particles (space)",
    "Electric Fields (space)",
    "Radio and Plasma Waves (space)",
    "Activity Indices",
    "Plasma Composition/Charge State Analyzers",
    "Energetic Particle Detector",
    "Coronagraph/Heliograph",
    "Imaging and Remote Sensing (Sun)",
    "Imaging and Remote Sensing (Magnetosphere/Earth)",
    "Imaging and Remote Sensing (ITM/Earth)",
    "Ground-Based Magnetometers, Riometers, Sounders",
    "Magnetic Fields (Balloon)",
    "Particles (Balloon)",
    "Engineering",
    "Housekeeping",
    "Ephemeris/Attitude/Ancillary",
]


def fetch_dataset_metadata() -> dict[str, dict]:
    """Fetch all dataset metadata from CDAWeb REST API.

    Returns:
        Dict mapping dataset_id to metadata dict:
        {
            "instrument": str,           # CDAWeb instrument name (e.g., "MAG")
            "instrument_types": [str],   # list of InstrumentType strings
            "label": str,                # human-readable label
            "observatory": str,          # e.g., "ACE"
        }

    Returns empty dict on failure (graceful degradation).
    """
    if requests is None:
        return {}

    try:
        from data_ops.http_utils import request_with_retry
        resp = request_with_retry(CDAWEB_REST_URL)
    except Exception as e:
        import logging
        logging.getLogger("xhelio").warning("CDAWeb REST API fetch failed: %s", e)
        return {}

    return _parse_xml_catalog(resp.content)


def _parse_xml_catalog(xml_content: bytes) -> dict[str, dict]:
    """Parse CDAWeb XML catalog into a dict keyed by dataset ID."""
    if ET is None:
        return {}

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        import logging
        logging.getLogger("xhelio").warning("CDAWeb XML parse failed: %s", e)
        return {}

    result = {}
    for ds in root.findall("cda:DatasetDescription", _NS):
        ds_id = _text(ds, "cda:Id")
        if not ds_id:
            continue

        instrument_types = [
            el.text.strip()
            for el in ds.findall("cda:InstrumentType", _NS)
            if el.text and el.text.strip()
        ]

        # Time interval
        ti = ds.find("cda:TimeInterval", _NS)
        start_date = ""
        stop_date = ""
        if ti is not None:
            start_date = _text(ti, "cda:Start")
            stop_date = _text(ti, "cda:End")

        result[ds_id] = {
            "instrument": _text(ds, "cda:Instrument") or "",
            "instrument_types": instrument_types,
            "label": _text(ds, "cda:Label") or "",
            "observatory": _text(ds, "cda:Observatory") or "",
            "observatory_group": _text(ds, "cda:ObservatoryGroup") or "",
            "pi_name": _text(ds, "cda:PiName") or "",
            "pi_affiliation": _text(ds, "cda:PiAffiliation") or "",
            "doi": _text(ds, "cda:Doi") or "",
            "notes_url": _text(ds, "cda:Notes") or "",
            "start_date": start_date,
            "stop_date": stop_date,
        }

    return result


def _text(elem, tag: str) -> str:
    """Extract text from a child element, or empty string."""
    child = elem.find(tag, _NS)
    if child is not None and child.text:
        return child.text.strip()
    return ""


def pick_primary_type(instrument_types: list[str]) -> str | None:
    """Pick the highest-priority InstrumentType from a list.

    Args:
        instrument_types: List of InstrumentType strings from CDAWeb.

    Returns:
        The highest-priority type, or None if the list is empty or
        none of the types are recognized.
    """
    if not instrument_types:
        return None

    for priority_type in INSTRUMENT_TYPE_PRIORITY:
        if priority_type in instrument_types:
            return priority_type

    # Return the first one if none match the priority list
    return instrument_types[0] if instrument_types else None


def get_type_info(instrument_type: str) -> dict:
    """Return {id, name, keywords} for an InstrumentType string.

    Args:
        instrument_type: CDAWeb InstrumentType string.

    Returns:
        Dict with id, name, keywords. Falls back to a generic entry
        if the type is not recognized.
    """
    info = INSTRUMENT_TYPE_INFO.get(instrument_type)
    if info:
        return info

    # Fallback: generate a generic entry from the type string
    clean_id = instrument_type.lower().replace(" ", "_").replace("/", "_")
    clean_id = clean_id.replace("(", "").replace(")", "")
    return {
        "id": clean_id,
        "name": instrument_type,
        "keywords": [],
    }
