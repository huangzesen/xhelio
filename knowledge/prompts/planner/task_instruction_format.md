## Task Instruction Format

Every fetch instruction MUST list ALL physical quantities needed from that mission and the time range.
Combine multiple data types into one instruction when they come from the same mission.
Do NOT include specific parameter names — the envoy agent selects parameters.
Every custom_operation instruction MUST include the exact source_labels (array of label strings).
Every visualization instruction MUST start with "{viz_instruction_prefix} ...".

Example instructions:
- "Fetch magnetic field vector components and solar wind plasma data (density, speed, temperature) for 2024-01-10 to 2024-01-17" (mission: "ACE")
- "Fetch magnetic field vector, solar wind speed, and electron pitch angle distribution for last week" (mission: "PSP")
- "Compute magnitude of AC_H2_MFI.BGSEc, save as ACE_Bmag" (mission: "__data_ops__")
- "{viz_instruction_prefix} ACE_Bmag and Wind_Bmag" (mission: "__visualization__")