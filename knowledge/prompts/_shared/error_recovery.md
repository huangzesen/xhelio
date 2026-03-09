## Error Recovery

- **Fetch returns 0 points**: Check if the time range is valid for the dataset. Try an alternative dataset or time range. Do not retry the same fetch.
- **run_code file not found**: Verify the label spelling against `list_fetched_data` and check that the filename matches (e.g., `pd.read_parquet('AC_H2_MFI.BGSEc.parquet')` for label `AC_H2_MFI.BGSEc`). Ensure the label is listed in `inputs`.
- **Discovery finds nothing**: Try broader keyword searches across the full catalog.
- **All-NaN parameter**: Skip it and move to the next candidate dataset. Do not retry.
- **Sub-agent spinning**: If a sub-agent makes 2-3 rounds with no progress (repeating the same calls), cancel and try a different approach.
- **Delegation returns error**: The sub-agent failed (e.g., wrong variable names, validation errors). Do NOT say 'Done'. Analyze the error details and try: (1) retry with different parameters or time range, (2) use an alternative dataset, or (3) handle the operation yourself using the relevant tools directly.
- **PPI fetch fails**: PDS PPI datasets use URN IDs (e.g., `urn:nasa:pds:cassini-mag-cal:data-1sec-krtp`). Retry at most 2 times, then skip.
- **SPICE = ephemeris only**: SPICE tools provide ONLY position, velocity, trajectory, distance, and coordinate transforms — NO science data (magnetic field, plasma, particles, etc.). For science data, delegate to the appropriate envoy agent (PSP, ACE, etc.). Call SPICE tools directly for ephemeris data.
- **Ephemeris-only missions**: Juno, Galileo, Pioneer 10/11 only have SPICE ephemeris data in this system. Do not search CDAWeb for their science data — use SPICE tools directly (`get_spacecraft_ephemeris`, `compute_distance`, etc.) for positions/trajectories.