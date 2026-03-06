## When to Stop Searching

- If a search/discovery task fails to find a dataset, do NOT retry.
  The catalog is deterministic — searching again returns the same results.
- After ONE failed search for a data source, give up on it.
- Produce your plan with whatever data you found.
  Partial results are better than infinite searching.
- Call `produce_plan` as soon as you have enough information to create actionable tasks.
