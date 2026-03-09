## Dataset Selection Workflow

0. **Check session history** — call `events(action='check')` to see what has already happened
   in this session (data fetched, searches performed, errors encountered). If events mention
   prior work relevant to your task, call `events(action='details', event_ids=[...])` to get
   full details. Skip any work that's already been done or that already failed.
1. **Check if data is already in memory** — see 'Data currently in memory' in the request,
   and cross-reference with events. If a label already covers your needs, skip fetching.
2. **Pick a dataset** from the Dataset Catalog in your system prompt. Match on description,
   instrument keywords, and time coverage.
3. **Browse parameters**: Call `browse_parameters(dataset_id)` (or `browse_parameters(dataset_ids=[...])` for multiple) to see all available variables. Select the best parameters based on name, units, and description.
4. **Fetch data**: Call your fetch tool (`fetch_data_cdaweb` or `fetch_data_ppi`) for each relevant parameter.
5. **If a parameter returns all-NaN**: Skip it and try the next candidate dataset.
6. **Time range format**: '2024-01-15 to 2024-01-20' (use ' to ' separator, NOT '/').
   Also accepts 'last week', 'January 2024', etc.
7. **Labels**: The fetch tool stores data with label `DATASET.PARAM`.
8. **Multi-quantity requests**: When your request contains multiple physical quantities
   (e.g., magnetic field AND plasma data), handle them all in one session:
   - Identify datasets for ALL quantities from the catalog
   - Call `browse_parameters` for all candidates in one call (parallel execution)
   - Then fetch ALL parameters in parallel (call multiple fetch tools in one response)
   - Report ALL stored labels together at the end