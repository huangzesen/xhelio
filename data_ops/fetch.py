"""
Data fetcher — routes data requests to the appropriate backend.

Supports three backends:
- CDF file download from CDAWeb (default for CDAWeb dataset IDs)
- PDS PPI file archive for PDS4 (urn:nasa:pds:*) and PDS3 (pds3:*) datasets

Errors propagate directly so the agent can learn from failures
(e.g., virtual parameters that exist in Master CDF metadata but
not in actual data files).
"""

import logging

logger = logging.getLogger("xhelio")


def fetch_data(
    dataset_id: str,
    parameter_id: str,
    time_min: str,
    time_max: str,
    force: bool = False,
) -> dict:
    """Fetch timeseries data from the appropriate backend.

    Routes by dataset ID format:
    - urn:nasa:pds:* → PDS PPI file archive (PDS4)
    - pds3:* → PDS PPI file archive (PDS3)
    - Everything else → CDF file download from CDAWeb

    Args:
        dataset_id: Dataset ID (CDAWeb, PDS URN, or pds3: prefixed).
        parameter_id: Parameter name (e.g., "BGSEc", "BR").
        time_min: ISO start time (e.g., "2024-01-15T00:00:00Z").
        time_max: ISO end time (e.g., "2024-01-16T00:00:00Z").
        force: If True, bypass the 1 GB download size limit (CDF only).

    Returns:
        Dict with keys: data (DataFrame), units, description, fill_value.

    Raises:
        ValueError: If no data is available or parameter not found.
        requests.HTTPError: If a download fails.
    """
    if dataset_id.startswith("urn:nasa:pds:") or dataset_id.startswith("pds3:"):
        from data_ops.fetch_ppi_archive import fetch_ppi_archive_data
        return fetch_ppi_archive_data(dataset_id, parameter_id, time_min, time_max)

    from data_ops.fetch_cdf import fetch_cdf_data
    return fetch_cdf_data(dataset_id, parameter_id, time_min, time_max, force=force)
