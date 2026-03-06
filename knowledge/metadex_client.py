"""
PDS PPI Metadex Solr API client for dataset discovery.

The Metadex service at ``https://pds-ppi.igpp.ucla.edu/metadex/`` indexes all
PDS PPI data collections (PDS3 + PDS4).

A single HTTP GET returns all ~1,200+ data collections with rich metadata
(title, description, instruments, targets, time ranges, DOI, archive type).
"""

from agent.event_bus import get_event_bus, METADATA_FETCH

METADEX_BASE = "https://pds-ppi.igpp.ucla.edu/metadex/collection/select/"

# Fields we actually need from Solr
_FIELDS = ",".join([
    "id",
    "title",
    "description",
    "bundle_id",
    "slot",
    "archive_type",
    "start_date_time",
    "stop_date_time",
    "observing_system.observing_system_component.name",
    "observing_system.observing_system_component.type",
    "target_identification.name",
    "investigation_area.name",
    "citation_information.doi",
])


def fetch_all_ppi_collections(rows: int = 2000) -> list[dict]:
    """Fetch all data-type collections from the PDS PPI Metadex.

    Single HTTP GET to the Solr endpoint, filtered to data collections only.

    Args:
        rows: Maximum number of rows to return (default 2000, well above
              the ~1,279 currently indexed).

    Returns:
        List of normalized collection dicts with keys:
        ``id``, ``title``, ``description``, ``bundle_id``, ``slot``,
        ``archive_type``, ``start_date_time``, ``stop_date_time``,
        ``instruments``, ``targets``, ``doi``.
    """
    from data_ops.http_utils import request_with_retry

    params = {
        "q": "*:*",
        "fq": "type:Data OR type:DATA OR type:data",
        "rows": rows,
        "wt": "json",
        "fl": _FIELDS,
    }

    resp = request_with_retry(
        METADEX_BASE, timeout=30, params=params,
    )
    data = resp.json()
    docs = data.get("response", {}).get("docs", [])
    get_event_bus().emit(METADATA_FETCH, agent="Metadex", level="info", msg=f"Metadex returned {len(docs)} data collections")

    return [_normalize_doc(doc) for doc in docs]


def fetch_mission_collections(mission_query: str) -> list[dict]:
    """Fetch data collections for a single mission from Metadex.

    Args:
        mission_query: Solr query string for the mission
            (e.g., ``"title:Juno"`` or ``"investigation_area.name:Juno"``).

    Returns:
        List of normalized collection dicts (same shape as
        :func:`fetch_all_ppi_collections`).
    """
    from data_ops.http_utils import request_with_retry

    params = {
        "q": mission_query,
        "fq": "type:Data OR type:DATA OR type:data",
        "rows": 500,
        "wt": "json",
        "fl": _FIELDS,
    }

    resp = request_with_retry(
        METADEX_BASE, timeout=30, params=params,
    )
    data = resp.json()
    docs = data.get("response", {}).get("docs", [])

    return [_normalize_doc(doc) for doc in docs]


def metadex_id_to_dataset_id(metadex_id: str, archive_type: int) -> str:
    """Convert a Metadex collection ID to a dataset ID used internally.

    PDS4 URNs (archive_type=4) pass through as-is since they are already
    recognized by the ``urn:nasa:pds:`` prefix routing.

    PDS3 IDs (archive_type=3) get a ``pds3:`` prefix so they can be
    distinguished from CDAWeb dataset IDs in fetch routing.

    Args:
        metadex_id: The ``id`` field from Metadex
            (e.g., ``"JNO-J-3-FGM-CAL-V1.0:DATA"``).
        archive_type: 3 for PDS3, 4 for PDS4.

    Returns:
        Dataset ID string, e.g.:
        - PDS4: ``"urn:nasa:pds:juno_waves_electron-density:data_jupiter"``
        - PDS3: ``"pds3:JNO-J-3-FGM-CAL-V1.0:DATA"``
    """
    if archive_type == 4:
        # PDS4 URN — already in correct format
        return metadex_id
    # PDS3 — add prefix
    return f"pds3:{metadex_id}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_doc(doc: dict) -> dict:
    """Normalize a Metadex Solr document to a flat dict."""
    # Extract instrument names (skip the spacecraft itself)
    component_names = doc.get(
        "observing_system.observing_system_component.name", []
    )
    component_types = doc.get(
        "observing_system.observing_system_component.type", []
    )

    instruments = []
    for i, name in enumerate(component_names):
        ctype = component_types[i] if i < len(component_types) else ""
        if ctype.lower() not in ("spacecraft", "host"):
            instruments.append(name)

    # Extract targets
    targets = doc.get("target_identification.name", [])
    if isinstance(targets, str):
        targets = [targets]

    # DOI
    doi_list = doc.get("citation_information.doi", [])
    doi = doi_list[0] if doi_list else ""

    return {
        "id": doc.get("id", ""),
        "title": doc.get("title", ""),
        "description": doc.get("description", ""),
        "bundle_id": doc.get("bundle_id", ""),
        "slot": doc.get("slot", ""),
        "archive_type": doc.get("archive_type", 0),
        "start_date_time": doc.get("start_date_time", ""),
        "stop_date_time": doc.get("stop_date_time", ""),
        "instruments": instruments,
        "targets": targets,
        "doi": doi,
    }
