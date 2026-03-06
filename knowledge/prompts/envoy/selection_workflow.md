## Dataset Selection Workflow

0. **Check session history** — call `events(action='check')` to see what has already happened
   in this session (data fetched, searches performed, errors encountered). If events mention
   prior work relevant to your task, call `events(action='details', event_ids=[...])` to get
   full details — this tells you exactly which datasets were searched, which parameters were
   tried, and what failed. Skip any work that's already been done or that already failed.
1. **Check if data is already in memory** — see 'Data currently in memory' in the request,
   and cross-reference with events. If a label already covers your needs, skip fetching.
   If events show a prior fetch attempt failed for a dataset, don't retry it.
2. **When given candidate datasets**: Call `list_parameters` for each candidate to see
   available parameters. Select the best dataset based on parameter coverage and relevance.
   Then call `fetch_data` for each relevant parameter (fetch_data auto-syncs metadata).
3. **When given a vague request**: Call `browse_datasets` to see available datasets (each entry includes description, date range, and parameter count). Use `search_datasets` for keyword filtering — it also returns start_date and stop_date for each dataset.
4. **If a parameter returns all-NaN**: Skip it and try the next candidate dataset.
5. **Time range format**: '2024-01-15 to 2024-01-20' (use ' to ' separator, NOT '/').
   Also accepts 'last week', 'January 2024', etc.
6. **Labels**: fetch_data stores data with label `DATASET.PARAM`.
7. **Multi-quantity requests**: When your request contains multiple physical quantities
   (e.g., magnetic field AND plasma data), handle them all in one session:
   - Discover datasets for ALL quantities first (call multiple search_datasets /
     list_parameters in one response for parallel execution)
   - Then fetch ALL parameters in parallel (call multiple fetch_data in one response)
   - Report ALL stored labels together at the end