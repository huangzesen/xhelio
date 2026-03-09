# Mission Data Architecture

How mission data is structured, generated, loaded, searched, and overridden.

## Directory Layout

```
knowledge/envoys/
  cdaweb/                          ← CDAWeb missions (auto-generated)
    {stem}.json                    ← Mission JSON (id, name, keywords, instruments, datasets)
    {stem}/metadata/{DATASET}.json ← Per-dataset metadata cache (parameters, dates)
    {stem}/metadata/_index.json    ← Dataset summary index
    {stem}/metadata/_calibration_exclude.json
  ppi/                             ← PDS PPI missions (auto-generated)
    {stem}.json
    {stem}/metadata/{safe_id}.json ← URN colons replaced with underscores
    {stem}/metadata/_index.json
```

No data lives at the top level of `envoys/` — only kind subdirectories (`cdaweb/`, `ppi/`, etc.).

## Mission JSON Structure

Each `{stem}.json` follows this schema:

```json
{
  "id": "PSP",
  "name": "Parker Solar Probe",
  "keywords": ["parker", "psp", "solar probe"],
  "profile": {
    "description": "...",
    "coordinate_systems": ["RTN", "SC"],
    "typical_cadence": "1 min",
    "data_caveats": ["..."],
    "analysis_patterns": ["..."]
  },
  "instruments": {
    "FIELDS/MAG": {
      "name": "FIELDS Magnetometer",
      "keywords": ["magnetic", "field", "mag"],
      "datasets": {
        "PSP_FLD_L2_MAG_RTN_1MIN": {
          "description": "PSP FIELDS Fluxgate Magnetometer",
          "start_date": "2018-10-06",
          "stop_date": "2025-01-15"
        }
      }
    }
  },
  "_meta": {
    "generated_at": "2026-02-15T20:22:00Z",
    "source": "CDAS REST + Master CDF"
  }
}
```

## Generation Pipeline

### CDAWeb Missions (`bootstrap.py:populate_missions()`)

1. Fetch dataset catalog from CDAS REST API (single HTTP call)
2. Group datasets by mission via prefix matching (`mission_prefixes.py`)
3. Create skeleton JSONs for each mission (`create_mission_skeleton()`)
4. Parallel-fetch metadata from Master CDF files (10 workers, 2 retries)
5. Merge metadata into mission JSONs, grouped by instrument:
   - Priority: existing instrument → prefix hint → CDAWeb InstrumentType → "General"
   - Keywords backfilled from InstrumentType metadata
6. Generate `_index.json` and `_calibration_exclude.json` per mission

### PPI Missions (`bootstrap.py:populate_ppi_missions()`)

1. Query Metadex Solr API for all PDS PPI data collections
2. Group collections by mission via URN prefix matching
3. Derive instrument groups from URN structure (`_derive_ppi_instrument_key()`)
4. Fetch metadata from label files in the PDS PPI file archive
5. Build mission JSON with PPI-specific keywords and profile
6. Generate `_index.json` per mission

Also available as standalone script: `python scripts/generate_ppi_missions.py`

### Full Rebuild

Via startup menu `[f]` or `--refresh-full` flag:

```python
clean_all_missions()      # Deletes both cdaweb/ and ppi/ contents
populate_missions()       # Rebuild CDAWeb (~5-10 min)
populate_ppi_missions()   # Rebuild PPI (~30s)
```

### Time-Range Refresh

Via startup menu `[r]` or `--refresh` flag:

```python
refresh_time_ranges()     # CDAWeb: single CDAS REST call (~3s)
                          # PPI: Metadex query (~5s)
```

Only updates `start_date`/`stop_date` in mission JSONs and metadata cache files.

## Loading and Merging

### Multi-Source Deep Merge (`mission_loader.py`)

When a stem exists in both `cdaweb/` and `ppi/` (e.g., cassini, ulysses, voyager1/2):

1. Load CDAWeb JSON as base
2. Load PPI JSON
3. `_deep_merge(base, patch)`: dicts merge recursively, everything else replaces
4. Result: union of all instruments and datasets from both sources

9 missions currently overlap: cassini, juno, maven, messenger, pioneer, pioneer\_venus, ulysses, voyager1, voyager2.

### Override System (`~/.xhelio/mission_overrides/`)

Learned knowledge (caveats, notes, corrections) is persisted separately from auto-generated data:

- **Mission-level**: `{data_dir}/mission_overrides/{stem}.json`
- **Dataset-level**: `{data_dir}/mission_overrides/{stem}/{dataset_id}.json`
- Applied via `_deep_merge()` on top of the merged base at load time
- Survives full rebuilds
- Written by `update_mission_override()` and `update_dataset_override()`

### Metadata Learning (`fetch_cdf.py`)

On first data fetch per dataset, `_sync_metadata_with_data_cdf()` compares Master CDF metadata against actual data CDF variables and writes annotations (`_validated`, `_note`) to dataset overrides — not to metadata cache files.

## Search and Discovery

### `search_datasets` (keyword matching, `knowledge/catalog.py`)

Fast local lookup. Three-pass matching:
1. Exact spacecraft ID match
2. Word-boundary match on ID/name
3. Keyword match with `\b{keyword}\b` regex, preferring longest match

Returns **one spacecraft + one instrument** (best match).

### `search_full_catalog` (full catalog search, `knowledge/catalog_search.py`)

Searches all CDAWeb + PPI datasets combined (2500+):
- CDAWeb entries: from CDAS REST API, cached to disk (24h TTL)
- PPI entries: built from local mission JSONs on each call
- Supports semantic search (fastembed) or substring matching
- Returns top N results with dataset ID and title

### `browse_datasets` (per-mission browse, `knowledge/metadata_client.py`)

Lists all non-calibration datasets for a mission from `_index.json`. Filters out patterns from `_calibration_exclude.json`.

## Data Fetch Routing (`data_ops/fetch.py`)

- `urn:nasa:pds:*` → Direct file download from PDS PPI archive (`fetch_ppi_archive_data()`)
- Everything else → CDF file download from CDAWeb (`fetch_cdf_data()`)

## Key Source Files

| File | Purpose |
|------|---------|
| `knowledge/mission_prefixes.py` | Mission names, keywords, dataset→mission routing |
| `knowledge/cdaweb_metadata.py` | CDAWeb REST API client, InstrumentType grouping |
| `knowledge/bootstrap.py` | Generation: `populate_missions()`, `populate_ppi_missions()`, `clean_all_missions()`, `refresh_time_ranges()` |
| `knowledge/mission_loader.py` | Loading: `load_mission()`, `load_all_missions()`, multi-source deep merge |
| `knowledge/metadata_client.py` | Metadata resolution: 3-layer (memory → file → network), dataset overrides |
| `knowledge/catalog.py` | Keyword search (`search_by_keywords()`), SPACECRAFT dict |
| `knowledge/catalog_search.py` | Full catalog search (CDAWeb + PPI, semantic/substring) |
| `knowledge/master_cdf.py` | Master CDF skeleton download and parsing |
| `data_ops/fetch.py` | Data fetch router (CDAWeb CDF vs PPI archive) |
| `data_ops/fetch_cdf.py` | CDF download, parsing, metadata learning |
| `data_ops/fetch_ppi_archive.py` | PDS PPI direct file archive fetch |
| `data_ops/http_utils.py` | Shared HTTP utilities (`request_with_retry()`) |

## Curated Data Locations

All mission data is reproducible from code. No hand-edited JSON files.

- `knowledge/mission_prefixes.py`: dataset ID prefix → mission stem mapping, mission names, canonical IDs
- `knowledge/cdaweb_metadata.py`: `INSTRUMENT_TYPE_INFO` keyword mappings for CDAWeb instrument grouping
- `knowledge/bootstrap.py`: PPI instrument derivation logic (`_derive_ppi_instrument_key()`, `_ppi_instrument_keywords()`)
