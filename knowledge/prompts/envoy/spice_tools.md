## SPICE Ephemeris Tools

You have access to SPICE/NAIF ephemeris tools for spacecraft positions, velocities, trajectories, distances, and coordinate transforms. Use these alongside data fetching when the user needs positional context (e.g., heliocentric distance for a time range).

Available SPICE tools:
- **get_spacecraft_ephemeris**: Position/velocity at a single time or as a timeseries. Use `list_spice_missions` to check spacecraft availability. Use `step` to control time resolution (e.g., "1h", "1d").
- **compute_distance**: Distance between two bodies over a time range.
- **transform_coordinates**: Transform a 3D vector between coordinate frames.
- **list_spice_missions**: List all supported spacecraft with NAIF IDs.
- **list_coordinate_frames**: List all available frames with descriptions.
- **manage_kernels**: Check kernel status, download, load, or purge kernels.

When storing ephemeris data, use labels following the pattern: `SPICE.PSP_position`, `SPICE.PSP_SUN_distance` (e.g., `SPICE.{SPACECRAFT}_{suffix}`).

**Step size guidelines:** For ≤1 day use "1m", 1–7 days use "5m"–"10m", 1–4 weeks use "1h", 1–6 months use "6h", 6–12 months use "1d". The server rejects >10,000 points unless `allow_large_response=True`.
