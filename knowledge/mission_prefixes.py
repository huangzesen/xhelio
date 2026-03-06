"""
Shared CDAWeb dataset ID prefix mapping.

Maps dataset ID prefixes to (mission_stem, instrument_hint) tuples.
Used by scripts/generate_mission_data.py and scripts/fetch_metadata_cache.py
to determine which mission a CDAWeb dataset belongs to.

Order matters: longer/more-specific prefixes must come before shorter ones
so that e.g. "PSP_FLD" matches before "PSP_".
"""


# Primary missions: auto-downloaded on first run.
# Covers key heliophysics science areas: inner heliosphere, L1 monitoring,
# magnetosphere, combined indices, solar, and operational space weather.
# Users can download additional missions with:
#   python scripts/generate_mission_data.py --all
PRIMARY_MISSIONS = [
    "psp",        # Parker Solar Probe — inner heliosphere flagship
    "solo",       # Solar Orbiter — inner heliosphere
    "ace",        # ACE — L1 solar wind monitor
    "wind",       # Wind — long-running solar wind
    "ulysses",    # Ulysses — out-of-ecliptic heliosphere
    "helios",     # Helios — inner heliosphere pioneer
    "voyager1",   # Voyager 1 — outer heliosphere
    "voyager2",   # Voyager 2 — outer heliosphere
]


# Comprehensive map of CDAWeb dataset ID prefixes to (mission_stem, instrument_hint).
# mission_stem is the lowercase JSON filename in knowledge/missions/.
# instrument_hint is an optional instrument key within the mission JSON.
#
# Sorted: longer prefixes first within each mission group.
MISSION_PREFIX_MAP = {
    # --- Parker Solar Probe ---
    "PSP_FLD": ("psp", "FIELDS/MAG"),
    "PSP_SWP_SPC": ("psp", "SWEAP"),
    "PSP_SWP_SPI": ("psp", "SWEAP/SPAN-I"),
    "PSP_SWP_SPA": ("psp", "SWEAP/SPAN-E"),
    "PSP_SWP_SPB": ("psp", "SWEAP/SPAN-E"),
    "PSP_SWP": ("psp", "SWEAP"),
    "PSP_ISOIS": ("psp", "ISOIS"),
    "PSP_": ("psp", None),

    # --- Solar Orbiter (SOLO_ and SO_ prefixes) ---
    "SOLO_L2_MAG": ("solo", "MAG"),
    "SOLO_L2_SWA": ("solo", "SWA-PAS"),
    "SOLO_": ("solo", None),
    "SO_": ("solo", None),

    # --- ACE ---
    "AC_H": ("ace", None),
    "AC_K": ("ace", None),
    "AC_OR": ("ace", None),
    "AC_AT": ("ace", None),

    # --- OMNI ---
    "OMNI_HRO": ("omni", "Combined"),
    "OMNI_": ("omni", "Combined"),
    "OMNI2_": ("omni", "Combined"),

    # --- Wind (short and full-name) ---
    "WIND_": ("wind", None),
    "WI_H": ("wind", None),
    "WI_K": ("wind", None),
    "WI_OR": ("wind", None),
    "WI_AT": ("wind", None),
    "WI_": ("wind", None),

    # --- DSCOVR ---
    "DSCOVR_H0_MAG": ("dscovr", "MAG"),
    "DSCOVR_H1_FC": ("dscovr", "FC"),
    "DSCOVR_": ("dscovr", None),

    # --- MMS (4 spacecraft) ---
    "MMS1_FGM": ("mms", "FGM"),
    "MMS1_FPI": ("mms", "FPI-DIS"),
    "MMS2_FGM": ("mms", "FGM"),
    "MMS2_FPI": ("mms", "FPI-DIS"),
    "MMS3_FGM": ("mms", "FGM"),
    "MMS3_FPI": ("mms", "FPI-DIS"),
    "MMS4_FGM": ("mms", "FGM"),
    "MMS4_FPI": ("mms", "FPI-DIS"),
    "MMS1_": ("mms", None),
    "MMS2_": ("mms", None),
    "MMS3_": ("mms", None),
    "MMS4_": ("mms", None),

    # --- STEREO-A ---
    "STA_L2_MAG": ("stereo_a", "MAG"),
    "STA_L2_PLA": ("stereo_a", "PLASTIC"),
    "STA_L1_MAG": ("stereo_a", "MAG"),
    "STA_L1_PLA": ("stereo_a", "PLASTIC"),
    "STA_": ("stereo_a", None),

    # --- STEREO-B ---
    "STB_L2_MAG": ("stereo_b", None),
    "STB_L2_PLA": ("stereo_b", None),
    "STB_L1_MAG": ("stereo_b", None),
    "STB_L1_PLA": ("stereo_b", None),
    "STB_": ("stereo_b", None),

    # --- THEMIS (5 probes + combined + ground + predicted orbits) ---
    "THAPRED_": ("themis", None),
    "THBPRED_": ("themis", None),
    "THCPRED_": ("themis", None),
    "THDPRED_": ("themis", None),
    "THEPRED_": ("themis", None),
    "THG_": ("themis", None),
    "THA_L2": ("themis", None),
    "THA_L1": ("themis", None),
    "THB_L2": ("themis", None),
    "THB_L1": ("themis", None),
    "THC_L2": ("themis", None),
    "THC_L1": ("themis", None),
    "THD_L2": ("themis", None),
    "THD_L1": ("themis", None),
    "THE_L2": ("themis", None),
    "THE_L1": ("themis", None),
    "THA_": ("themis", None),
    "THB_": ("themis", None),
    "THC_": ("themis", None),
    "THD_": ("themis", None),
    "THE_": ("themis", None),
    "TH_": ("themis", None),

    # --- Cluster (4 spacecraft + combined) ---
    "C1_CP": ("cluster", None),
    "C2_CP": ("cluster", None),
    "C3_CP": ("cluster", None),
    "C4_CP": ("cluster", None),
    "C1_": ("cluster", None),
    "C2_": ("cluster", None),
    "C3_": ("cluster", None),
    "C4_": ("cluster", None),
    "CL_": ("cluster", None),

    # --- STEREO combined ---
    "STEREO_": ("stereo_a", None),

    # --- Van Allen Probes / RBSP (multiple prefix styles) ---
    "RBSP-A-RBSPICE": ("rbsp", None),
    "RBSP-B-RBSPICE": ("rbsp", None),
    "RBSP-A": ("rbsp", None),
    "RBSP-B": ("rbsp", None),
    "RBSPA_": ("rbsp", None),
    "RBSPB_": ("rbsp", None),
    "RBSP_": ("rbsp", None),

    # --- GOES (multiple spacecraft, numbered and lettered) ---
    "GOES10_": ("goes", None),
    "GOES11_": ("goes", None),
    "GOES12_": ("goes", None),
    "GOES13_": ("goes", None),
    "GOES14_": ("goes", None),
    "GOES15_": ("goes", None),
    "GOES16_": ("goes", None),
    "GOES17_": ("goes", None),
    "GOES18_": ("goes", None),
    "GOES_": ("goes", None),
    "G0_": ("goes", None),
    "G6_": ("goes", None),
    "G7_": ("goes", None),
    "G8_": ("goes", None),
    "G9_": ("goes", None),
    "G10_": ("goes", None),
    "G11_": ("goes", None),
    "G12_": ("goes", None),
    "G13_": ("goes", None),
    "G14_": ("goes", None),
    "G15_": ("goes", None),
    "G16_": ("goes", None),
    "G17_": ("goes", None),
    "G18_": ("goes", None),

    # --- Voyager (short and full-name, with hyphens and underscores) ---
    "VG1_": ("voyager1", None),
    "VG2_": ("voyager2", None),
    "VOYAGER-1": ("voyager1", None),
    "VOYAGER-2": ("voyager2", None),
    "VOYAGER1_": ("voyager1", None),
    "VOYAGER2_": ("voyager2", None),

    # --- Ulysses (short and full-name) ---
    "UY_": ("ulysses", None),
    "ULYSSES_": ("ulysses", None),

    # --- Geotail ---
    "GE_": ("geotail", None),

    # --- Polar (short and full-name) ---
    "PO_": ("polar", None),
    "POLAR_": ("polar", None),

    # --- IMAGE (short and full-name) ---
    "IM_": ("image", None),
    "IMAGE_": ("image", None),

    # --- FAST (short and full-name) ---
    "FA_": ("fast", None),
    "FAST_": ("fast", None),

    # --- SOHO ---
    "SOHO_": ("soho", None),

    # --- Juno ---
    "JUNO_": ("juno", None),

    # --- MAVEN ---
    "MVN_": ("maven", None),

    # --- MESSENGER (short and full-name) ---
    "MESS_": ("messenger", None),
    "MESSENGER_": ("messenger", None),

    # --- Cassini (short and full-name) ---
    "CO_": ("cassini", None),
    "CASSINI_": ("cassini", None),

    # --- New Horizons (short, full-name, hyphenated) ---
    "NEW-HORIZONS": ("new_horizons", None),
    "NEW_HORIZONS": ("new_horizons", None),
    "NH_": ("new_horizons", None),

    # --- IMP (multiple spacecraft) ---
    "I8_": ("imp8", None),
    "I1_": ("imp8", None),
    "I2_": ("imp8", None),
    "IA_": ("imp8", None),

    # --- ISEE ---
    "ISEE": ("isee", None),

    # --- Arase/ERG ---
    "ERG_": ("arase", None),
    "ARASE_": ("arase", None),

    # --- TIMED ---
    "TIMED_": ("timed", None),

    # --- TWINS ---
    "TWINS1_": ("twins", None),
    "TWINS2_": ("twins", None),
    "TWINS_": ("twins", None),

    # --- IBEX ---
    "IBEX_": ("ibex", None),

    # --- SAMPEX ---
    "SAMPEX_": ("sampex", None),
    "SE_": ("sampex", None),

    # --- BARREL (two prefix styles) ---
    "BARREL_": ("barrel", None),
    "BAR_": ("barrel", None),

    # --- C/NOFS ---
    "CNOFS_": ("cnofs", None),

    # --- DMSP ---
    "DMSP": ("dmsp", None),

    # --- LANL (multiple satellites) ---
    "LANL_": ("lanl", None),
    "L0_": ("lanl", None),
    "L1_": ("lanl", None),
    "L4_": ("lanl", None),
    "L7_": ("lanl", None),
    "L9_": ("lanl", None),
    "A1_": ("lanl", None),
    "A2_": ("lanl", None),

    # --- AMPTE (including AMPTECCE prefix) ---
    "AMPTECCE": ("ampte", None),
    "AMPTE_": ("ampte", None),

    # --- HAWKEYE ---
    "HAWKEYE_": ("hawkeye", None),

    # --- Helios ---
    "HELIOS1_": ("helios", None),
    "HELIOS2_": ("helios", None),
    "HEL1_": ("helios", None),
    "HEL2_": ("helios", None),

    # --- Pioneer ---
    "PIONEER10_": ("pioneer", None),
    "PIONEER11_": ("pioneer", None),
    "P10_": ("pioneer", None),
    "P11_": ("pioneer", None),

    # --- SNOE ---
    "SNOE_": ("snoe", None),

    # --- SWARM ---
    "SWARM": ("swarm", None),

    # --- ICON ---
    "ICON_": ("icon", None),

    # --- GOLD ---
    "GOLD_": ("gold", None),

    # --- ELFIN ---
    "EL": ("elfin", None),

    # --- Equator-S ---
    "EQ_": ("equator_s", None),

    # --- MAVEN (full-name alt prefix) ---
    "MAVEN_": ("maven", None),

    # --- Dynamics Explorer ---
    "DE1_": ("de", None),
    "DE2_": ("de", None),
    "DE_": ("de", None),

    # --- Pioneer Venus ---
    "PIONEERVENUS_": ("pioneer_venus", None),

    # --- NOAA POES satellites ---
    "NOAA05_": ("noaa", None),
    "NOAA06_": ("noaa", None),
    "NOAA07_": ("noaa", None),
    "NOAA08_": ("noaa", None),
    "NOAA10_": ("noaa", None),
    "NOAA12_": ("noaa", None),
    "NOAA14_": ("noaa", None),
    "NOAA15_": ("noaa", None),
    "NOAA16_": ("noaa", None),
    "NOAA18_": ("noaa", None),
    "NOAA19_": ("noaa", None),

    # --- METOP ---
    "METOP": ("noaa", None),

    # --- CRRES ---
    "CRRES_": ("crres", None),

    # --- GPS ---
    "GPS_": ("gps", None),

    # --- ISS instruments ---
    "ISS_": ("iss", None),

    # --- CIRBE ---
    "CIRBE_": ("cirbe", None),

    # --- CSSWE ---
    "CSSWE_": ("csswe", None),

    # --- ST5 ---
    "ST5-": ("st5", None),

    # ===================================================================
    # PDS PPI URN prefixes (PDS4 — planetary missions)
    # ===================================================================

    # --- Cassini (PDS4) ---
    "urn:nasa:pds:cassini-": ("cassini", None),

    # --- Voyager (PDS4) ---
    "urn:nasa:pds:voyager1.": ("voyager1", None),
    "urn:nasa:pds:voyager2.": ("voyager2", None),
    "urn:nasa:pds:voyager-pws-": ("voyager1", None),  # PWS shared
    "urn:nasa:pds:vg1-": ("voyager1", None),
    "urn:nasa:pds:vg2-": ("voyager2", None),

    # --- Juno (PDS4) ---
    "urn:nasa:pds:juno": ("juno", None),

    # --- MAVEN (PDS4) ---
    "urn:nasa:pds:maven.": ("maven", None),

    # --- Galileo (PDS4) ---
    "urn:nasa:pds:galileo-": ("galileo", None),
    "urn:nasa:pds:go-pls-": ("galileo", None),

    # --- Pioneer (PDS4) ---
    "urn:nasa:pds:p10-": ("pioneer", None),
    "urn:nasa:pds:p11-": ("pioneer", None),

    # --- Ulysses (PDS4) ---
    "urn:nasa:pds:ulysses-": ("ulysses", None),

    # --- MESSENGER (PDS4) ---
    "urn:nasa:pds:mess-": ("messenger", None),

    # --- Pioneer Venus Orbiter (PDS4) ---
    "urn:nasa:pds:pvo-": ("pioneer_venus", None),

    # --- Mars Global Surveyor (PDS4) ---
    "urn:nasa:pds:mgs-": ("mgs", None),

    # --- Venus Express (PDS4) ---
    "urn:nasa:pds:vex-": ("vex", None),

    # --- InSight (PDS4) ---
    "urn:nasa:pds:insight-": ("insight", None),

    # --- Lunar Prospector (PDS4) ---
    "urn:nasa:pds:lp-": ("lunar_prospector", None),

    # --- LRO (PDS4) ---
    "urn:nasa:pds:lro-": ("lro", None),

    # ===================================================================
    # PDS PPI PDS3 prefixes (pds3: prefix)
    # ===================================================================

    # --- Juno (PDS3) ---
    "pds3:JNO-": ("juno", None),

    # --- Cassini (PDS3) ---
    "pds3:CO-": ("cassini", None),

    # --- Voyager (PDS3) ---
    "pds3:VG1-": ("voyager1", None),
    "pds3:VG2-": ("voyager2", None),

    # --- Ulysses (PDS3) ---
    "pds3:ULY-": ("ulysses", None),

    # --- Galileo (PDS3) ---
    "pds3:GO-": ("galileo", None),

    # --- MESSENGER (PDS3) ---
    "pds3:MESS-": ("messenger", None),

    # --- Pioneer Venus (PDS3) ---
    "pds3:PVO-": ("pioneer_venus", None),

    # --- Mars Global Surveyor (PDS3) ---
    "pds3:MGS-": ("mgs", None),

    # --- Venus Express (PDS3) ---
    "pds3:VEX-": ("vex", None),

    # --- Pioneer (PDS3) ---
    "pds3:P10-": ("pioneer", None),
    "pds3:P11-": ("pioneer", None),

    # --- Mars Express (PDS3) ---
    "pds3:MEX-": ("mex", None),

    # --- New Horizons (PDS3) ---
    "pds3:NH-": ("new_horizons", None),
}


# Human-readable names for auto-generating mission JSON skeletons.
MISSION_NAMES = {
    "psp": "Parker Solar Probe",
    "solo": "Solar Orbiter",
    "ace": "ACE",
    "omni": "OMNI",
    "wind": "Wind",
    "dscovr": "DSCOVR Deep Space Climate Observatory",
    "mms": "MMS Magnetospheric Multiscale",
    "stereo_a": "STEREO-A",
    "stereo_b": "STEREO-B",
    "themis": "THEMIS",
    "cluster": "Cluster",
    "rbsp": "Van Allen Probes",
    "goes": "GOES",
    "voyager1": "Voyager 1",
    "voyager2": "Voyager 2",
    "ulysses": "Ulysses",
    "geotail": "Geotail",
    "polar": "Polar",
    "image": "IMAGE",
    "fast": "FAST",
    "soho": "SOHO",
    "juno": "Juno",
    "maven": "MAVEN",
    "messenger": "MESSENGER",
    "cassini": "Cassini",
    "new_horizons": "New Horizons",
    "imp8": "IMP-8",
    "isee": "ISEE",
    "arase": "Arase/ERG",
    "timed": "TIMED",
    "twins": "TWINS",
    "ibex": "IBEX",
    "sampex": "SAMPEX",
    "barrel": "BARREL",
    "cnofs": "C/NOFS",
    "dmsp": "DMSP",
    "lanl": "LANL",
    "ampte": "AMPTE",
    "hawkeye": "Hawkeye",
    "helios": "Helios",
    "pioneer": "Pioneer",
    "snoe": "SNOE",
    "swarm": "Swarm",
    "icon": "ICON",
    "gold": "GOLD",
    "elfin": "ELFIN",
    "equator_s": "Equator-S",
    "de": "Dynamics Explorer",
    "pioneer_venus": "Pioneer Venus",
    "noaa": "NOAA POES",
    "crres": "CRRES",
    "gps": "GPS",
    "iss": "ISS",
    "cirbe": "CIRBE",
    "csswe": "CSSWE",
    "st5": "ST5",
    # PDS PPI missions (not already in CDAWeb list)
    "galileo": "Galileo",
    "mgs": "Mars Global Surveyor",
    "vex": "Venus Express",
    "insight": "InSight",
    "lunar_prospector": "Lunar Prospector",
    "lro": "Lunar Reconnaissance Orbiter",
    "mex": "Mars Express",
}


def match_dataset_to_mission(dataset_id: str) -> tuple[str | None, str | None]:
    """Map a CDAWeb dataset ID to (mission_stem, instrument_hint).

    Checks prefixes from most specific to least specific.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").

    Returns:
        (mission_stem, instrument_hint) or (None, None) if no match.
        mission_stem is the lowercase JSON filename (e.g., "ace").
        instrument_hint is the instrument key in the JSON, or None.
    """
    for prefix, (mission, instrument) in MISSION_PREFIX_MAP.items():
        if dataset_id.startswith(prefix):
            return mission, instrument
    return None, None


def get_all_mission_stems() -> list[str]:
    """Return sorted list of all unique mission stems from the prefix map."""
    stems = set()
    for _, (mission, _) in MISSION_PREFIX_MAP.items():
        stems.add(mission)
    return sorted(stems)


def get_all_canonical_ids() -> list[str]:
    """Return sorted list of all canonical mission IDs."""
    return sorted({get_canonical_id(stem) for stem in get_all_mission_stems()})


def get_mission_name(mission_stem: str) -> str:
    """Get the human-readable name for a mission stem.

    Args:
        mission_stem: Lowercase mission identifier (e.g., "ace", "themis").

    Returns:
        Human-readable name (e.g., "ACE", "THEMIS").
        Falls back to upper-cased stem if not in MISSION_NAMES.
    """
    return MISSION_NAMES.get(mission_stem, mission_stem.upper())


def get_mission_keywords(mission_stem: str) -> list[str]:
    """Auto-derive search keywords from mission name and prefixes.

    Args:
        mission_stem: Lowercase mission identifier.

    Returns:
        List of keyword strings for catalog search.
    """
    name = MISSION_NAMES.get(mission_stem, mission_stem)
    keywords = set()

    # Add the stem itself
    keywords.add(mission_stem.lower())

    # Add words from the human-readable name
    for word in name.split():
        w = word.strip("/()")
        if len(w) > 1:
            keywords.add(w.lower())

    # Add common prefixes for this mission
    for prefix, (m, _) in MISSION_PREFIX_MAP.items():
        if m == mission_stem:
            # Clean prefix (remove trailing _)
            clean = prefix.rstrip("_").lower()
            if len(clean) > 1:
                keywords.add(clean)

    return sorted(keywords)


def get_canonical_id(mission_stem: str) -> str:
    """Return the canonical mission ID for a stem.

    Most missions use UPPER_CASE, but a few have special casing
    (e.g., "SolO" for Solar Orbiter, "STEREO_A" with underscore).
    """
    _CANONICAL = {
        "solo": "SolO",
        "stereo_a": "STEREO_A",
        "stereo_b": "STEREO_B",
    }
    if mission_stem in _CANONICAL:
        return _CANONICAL[mission_stem]
    # Default: upper-case, replace _ with -
    if "_" in mission_stem:
        return mission_stem.upper().replace("_", "-")
    return mission_stem.upper()


def create_mission_skeleton(mission_stem: str) -> dict:
    """Create a minimal mission JSON skeleton for a new mission.

    The skeleton has the correct structure but minimal content.
    It can be populated by generate_mission_data.py with CDAWeb metadata.

    Args:
        mission_stem: Lowercase mission identifier (e.g., "themis").

    Returns:
        Dict with the mission JSON structure.
    """
    name = get_mission_name(mission_stem)
    keywords = get_mission_keywords(mission_stem)

    return {
        "id": get_canonical_id(mission_stem),
        "name": name,
        "keywords": keywords,
        "profile": {
            "description": f"{name} data from CDAWeb.",
            "coordinate_systems": [],
            "typical_cadence": "",
            "data_caveats": [],
            "analysis_patterns": [],
        },
        "instruments": {
            "General": {
                "name": "General",
                "keywords": [],
                "datasets": {},
            }
        },
    }
