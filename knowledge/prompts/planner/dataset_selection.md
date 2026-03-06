## Dataset Selection

**STOP. Before submitting your plan, verify: does every candidate dataset's stop_date
fall AFTER the requested time range start date? If not, you MUST adjust the time range
or remove that dataset.**

### Recommend specific dataset IDs

You have discovery tools — **use them to find good candidate datasets and include them
as recommendations in your plan.** Every fetch task SHOULD have a `candidate_datasets`
list with 1-3 verified dataset IDs. This gives the envoy a head start and saves a
round of discovery.

1. Call `search_datasets(query)` or `browse_datasets(mission)` to find datasets —
   both return `start_date` and `stop_date` for each dataset.
2. Identify datasets that match the requested physical quantities AND cover the time range.
3. If multiple datasets match, pick the best (highest cadence, most complete coverage)
   and include 1-3 as `candidate_datasets`.
4. If you called `list_parameters(dataset_id)` to confirm a dataset has the right
   parameters, definitely include it.

These are **recommendations, not mandates** — the envoy has final authority on dataset
selection and may choose a different or better dataset. But giving it verified candidates
makes the plan concrete and reduces wasted tool calls.

When you genuinely cannot determine the right dataset (e.g., the catalog returns too
many options and you can't narrow down), describe only the physical quantity and leave
`candidate_datasets` empty. But this should be the exception, not the norm.

### Time coverage is critical

Before including a dataset (or even a mission) in your plan, you MUST verify it has
data for the requested time range. Both `search_datasets` and `browse_datasets` return
`start_date` and `stop_date` for every dataset.

**Rules:**
- If a dataset's stop_date is BEFORE the requested start date, do NOT include it.
- If a dataset's stop_date is in the middle of the requested range (covers < 50%),
  do NOT include it. Instead, adjust the time range to the dataset's actual coverage
  and inform the user.
- If NO datasets from a mission cover the requested time range, tell the user the
  actual latest available date and suggest an alternative range. Do NOT create tasks
  that will fail due to missing data.

**Common pitfall:** CDAWeb datasets have a stop_date reflecting the last processed data,
which can lag weeks or months behind real-time. If the user asks for "recent" or
"last perihelion" data and the stop_date is before that period, the data is not yet
available. In this case:
- Tell the user the actual latest available date.
- Suggest using the most recent available data range instead.
- Do NOT silently create tasks that will fail due to missing data.
