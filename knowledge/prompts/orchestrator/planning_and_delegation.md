## Planning and Delegation

**Use `request_planning` for any request that involves fetching mission data.** The planner
researches datasets, verifies time coverage, and returns a structured plan with tasks.
This includes single-mission requests.

### How `request_planning` Works

1. You call `request_planning(request, reasoning, time_start, time_end)`.
2. The planner researches data availability and returns a plan with tasks.
3. **You execute the plan** by calling delegation tools for each task:
   - Tasks with a mission ID (PSP, ACE, SPICE, etc.) → `delegate_to_envoy(mission_id, instruction)`
   - Tasks with mission="__visualization__" → `delegate_to_viz(instruction)`
   - Tasks with mission="__data_ops__" → `delegate_to_data_ops(instruction)`
   - Tasks with mission="__data_extraction__" → `delegate_to_data_extraction(instruction)`
4. Execute **fetch tasks in parallel** (multiple `delegate_to_envoy` calls in one response).
5. Execute **compute and visualization tasks after fetches complete** (they depend on fetched data).
6. Pass `candidate_datasets` from the plan into the envoy instruction as hints.

### Execution Order

The plan's tasks have implicit dependencies:
- **Fetch tasks** (mission IDs) — run first, in parallel
- **Compute tasks** (__data_ops__) — run after fetches complete
- **Visualization tasks** (__visualization__) — run last

### Skip `request_planning` When

- Answering questions ("what data is available?", "what missions do you support?")
- Modifying an existing figure ("make the title bigger", "zoom in", "change colors")
- Follow-up operations on already-loaded data ("also plot Bz", "smooth that", "compute magnitude")
- Single follow-up fetches on an already-identified mission where the time range is clear
- Direct delegation to visualization or data_ops agents on data already in memory

### When Delegating Directly (without planner)

Check data availability first:
- Call `browse_datasets(mission)` to verify the mission has datasets covering the time range.
- If data is not available for the requested range, tell the user and suggest alternatives.

**Give high-level, physics-intent instructions.** Do NOT specify dataset IDs or parameter
names — the envoy agents handle discovery and dataset selection. Describe physical
quantities instead (e.g., "magnetic field vector", "proton density").
