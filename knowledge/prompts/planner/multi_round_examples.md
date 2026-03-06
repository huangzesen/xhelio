## Examples

### Example 1: Single mission, multiple data types + plot

User: "Show me PSP magnetic field, solar wind speed, and electron PAD"

```json
{{
  "reasoning": "Single mission — combine all data into one PSP task, then plot",
  "tasks": [
    {{"description": "Fetch PSP mag + plasma + PAD (2024-01-10 to 2024-01-17)", "instruction": "Fetch magnetic field vector, solar wind proton speed, and electron pitch angle distribution for 2024-01-10 to 2024-01-17", "mission": "PSP", "candidate_datasets": ["PSP_FLD_L2_MAG_RTN_1MIN", "PSP_SWP_SPI_SF0A_L3_MOM"]}},
    {{"description": "Plot PSP data", "instruction": "{viz_instruction_prefix} the fetched magnetic field, solar wind speed, and electron PAD in a multi-panel figure", "mission": "__visualization__"}}
  ],
  "summary": "Fetch PSP magnetic field, solar wind speed, and electron PAD, then plot them.",
  "time_range_validated": true
}}
```

### Example 2: Multi-mission comparison with computation

User: "Compare ACE and Wind magnetic field magnitude"

```json
{{
  "reasoning": "Need data from both missions, compute magnitudes, then plot comparison",
  "tasks": [
    {{"description": "Fetch ACE mag (2024-01-10 to 2024-01-17)", "instruction": "Fetch magnetic field vector components for 2024-01-10 to 2024-01-17", "mission": "ACE", "candidate_datasets": ["AC_H2_MFI"]}},
    {{"description": "Fetch Wind mag (2024-01-10 to 2024-01-17)", "instruction": "Fetch magnetic field vector components for 2024-01-10 to 2024-01-17", "mission": "WIND", "candidate_datasets": ["WI_H2_MFI"]}},
    {{"description": "Compute magnitudes", "instruction": "Compute magnitude of the ACE and Wind magnetic field vectors", "mission": "__data_ops__"}},
    {{"description": "Plot comparison", "instruction": "{viz_instruction_prefix} ACE and Wind magnetic field magnitudes together", "mission": "__visualization__"}}
  ],
  "summary": "Fetch ACE and Wind magnetic field, compute magnitudes, plot comparison.",
  "time_range_validated": true
}}
```

### Example 3: Data fetch only (no plot requested)

User: "Load PSP encounter 22 magnetic field data"

```json
{{
  "reasoning": "User wants data loaded but didn't ask for a plot",
  "tasks": [
    {{"description": "Fetch PSP mag (2024-06-01 to 2024-07-15)", "instruction": "Fetch high cadence magnetic field data for PSP encounter 22 around perihelion, 2024-06-01 to 2024-07-15", "mission": "PSP", "candidate_datasets": ["PSP_FLD_L2_MAG_RTN_4_SA_PER_CYC"]}}
  ],
  "summary": "Fetch PSP E22 magnetic field data.",
  "time_range_validated": true
}}
```