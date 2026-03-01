"""Knowledge base for mission datasets and metadata integration."""

from .catalog import (
    MISSIONS,
    SPACECRAFT,  # backward-compat alias
    list_missions_catalog,
    list_spacecraft,  # backward-compat alias
    list_instruments,
    match_mission,
    match_spacecraft,  # backward-compat alias
    match_instrument,
    search_by_keywords,
)
from .metadata_client import (
    get_dataset_info,
    list_parameters,
)
from .mission_loader import (
    load_mission,
    load_all_missions,
    get_routing_table,
    get_mission_datasets,
    get_mission_ids,
)
