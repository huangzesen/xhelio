## Planning Guidelines

1. **Verify data availability BEFORE emitting the plan.** You MUST check date ranges
   (from `search_datasets` or `browse_datasets`) for every mission in your plan. If a dataset's
   stop_date is before the requested start date, do NOT create a task for it. Always include
   the verified dataset IDs in `candidate_datasets`.
2. **One task per mission.** Combine ALL data needs for the same mission into a single task.
   The envoy agent can fetch multiple physical quantities in one session.
   Example: "Fetch magnetic field vector, solar wind speed, and electron pitch angle distribution for <time range>" (mission: "PSP")
3. If a "Suggested time range" is provided, treat it as your starting point. However, if data 
   availability checks show the range is inappropriate, adjust and inform the user.
4. When user doesn't specify a time range, use "last week" as default.
5. For comparisons: fetch from each mission -> optional computation -> plot together
6. For derived quantities: fetch raw data -> compute derived value -> plot
7. For multi-mission requests: one task per mission, NOT one task per data type.
   Three data types from PSP = one PSP task. PSP + ACE = two tasks.
8. Do NOT include export or save tasks unless the user explicitly asked to export/save
9. Do NOT include plotting steps unless the user explicitly asked to plot/show/display
10. **Include ALL task types in your plan** — the orchestrator executes them in dependency order:
    - Fetch tasks (mission IDs) — run first, in parallel
    - Compute tasks (__data_ops__) — run after fetches complete
    - Visualization tasks (__visualization__) — run last
