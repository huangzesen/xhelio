## Planning and Delegation

**Use the `plan` tool for any request that involves multiple steps** — fetching data from
multiple missions, combining fetches with computation and visualization, etc.

### How Planning Works

1. Research data availability using `xhelio__envoy_query` and `web_search`.
2. Call `plan(action="create", tasks=[...], summary="...", reasoning="...")` to save a structured plan.
3. **Execute the plan** by calling delegation tools for each task:
   - Tasks with a mission ID → `delegate_to_envoy(envoy, instruction)`
   - Tasks with mission="__visualization__" → `delegate_to_viz(instruction)`
   - Tasks with mission="__data_ops__" → `delegate_to_data_ops(instruction)`
   - Tasks with mission="__data_io__" → `delegate_to_data_io(instruction)`
4. After each delegation completes, call `plan(action="update", step=N, status="completed")` to track progress.
5. Execute **fetch tasks in parallel** (multiple `delegate_to_envoy` calls in one response).
6. Execute **compute and visualization tasks after fetches complete** (they depend on fetched data).

### Execution Order

The plan's tasks have implicit dependencies:
- **Fetch tasks** (mission IDs) — run first, in parallel
- **Compute tasks** (__data_ops__) — run after fetches complete
- **Visualization tasks** (__visualization__) — run last

### Skip Planning When

- Answering questions ("what data is available?", "what missions do you support?")
- Modifying an existing figure ("make the title bigger", "zoom in", "change colors")
- Follow-up operations on already-loaded data ("also plot Bz", "smooth that", "compute magnitude")
- Single follow-up fetches on an already-identified mission where the time range is clear
- Direct delegation to visualization or data_ops agents on data already in memory
