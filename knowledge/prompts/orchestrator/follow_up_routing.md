## Follow-Up Routing

When data is already loaded in memory (shown in [ACTIVE SESSION CONTEXT]):
- Delegate to the corresponding envoy agent immediately for data fetching
- Do NOT call dataset discovery tools (`search_datasets`, `list_parameters`) yourself
- The envoy agent has mission-specific knowledge and will handle discovery far more efficiently
- Only use discovery tools yourself when exploring a NEW mission not yet in memory

When a plot exists (active or restorable in [ACTIVE SESSION CONTEXT]):
- If the user asks to **analyze, check, or interpret** the plot → `restore_plot` (if restorable) then `delegate_to_insight`
- If the user asks to **modify, restyle, zoom, or add traces** → `delegate_to_viz`
- If the user asks to **create a new/different plot** → `delegate_to_viz`
- Do NOT re-create a plot with the viz agent when the user is asking about an existing figure