## Mission Tagging

Tag each task with the "mission" field:
- Use mission IDs: PSP, SolO, ACE, OMNI, WIND, DSCOVR, MMS, STEREO_A
- mission="__visualization__" for visualization tasks (plotting, styling, render changes)
- mission="__data_ops__" for data transformation/analysis (custom_operation, describe_data)
- mission="__data_extraction__" for creating DataFrames from text (store_dataframe, event catalogs)
- mission=null for cross-mission tasks that don't fit the above categories

Note: Ephemeris/trajectory data (spacecraft positions, orbits, distances) is available via SPICE tools that both the orchestrator and envoy agents can call directly. Do NOT create a separate mission task for ephemeris — include it in the envoy instruction (e.g., "Fetch PSP magnetic field and also get PSP ephemeris for the same time range") or the orchestrator will call SPICE tools directly.

**Critical rule:** Each batch must have AT MOST one task per mission ID. If a request needs
multiple data types from the same mission, combine them into a single task instruction.