"""Knowledge base for spacecraft datasets and metadata integration."""

from .catalog import (
    SPACECRAFT,
    list_spacecraft,
    list_instruments,
    get_datasets,
    match_spacecraft,
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
