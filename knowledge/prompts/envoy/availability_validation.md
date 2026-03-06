## Data Availability Validation (CRITICAL)

After discovering datasets with `browse_datasets` or `search_datasets`, check each
candidate's `start_date` / `stop_date` against the requested time range BEFORE fetching.

Estimate the time coverage: what fraction of the requested time range overlaps with
the best candidate dataset's `start_date`–`stop_date` window.

**Reject if ≥90% of the requested time range falls outside all candidate datasets' coverage.**
Do NOT attempt to fetch. Reject with a structured message:
```
**REJECT: Insufficient data coverage**
Requested: <data type> for <requested time range>
Available: <dataset_id_1> (covers <start> to <stop>), <dataset_id_2> (covers <start> to <stop>)
Estimated coverage: <X>% of requested range
To fetch anyway, re-delegate with [FORCE_FETCH] in the request.
```

If coverage is ≥10% of the requested range, proceed normally —
the system auto-clamps to the available window.

**Force fetch override:** If the request contains `[FORCE_FETCH]`, skip this
validation entirely and fetch whatever is available regardless of coverage.