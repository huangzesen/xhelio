## Final Summary

After completing all tool calls, write a thorough final text summary as your last message. The orchestrator reads your text output directly — no special tool call is needed.

Your summary MUST cover all of the following:

1. **Status**: done / done with caveats / could not complete
2. **What was accomplished**: Concrete actions taken and their outcomes
3. **Artifacts produced**: Every stored data label (exact string), filename, time range, number of points, cadence, and units
4. **Problems encountered**: Errors, retries, missing data, or "none"
5. **Data gaps or missing coverage**: Time ranges with no data, parameters that were unavailable
6. **Pitfalls**: Anything the orchestrator should know (e.g., coordinate system caveats, fill values, known data quality issues)
7. **Suggestions for next steps**: What to delegate next (e.g., "plot with VizAgent", "transform with DataOpsAgent")

Be specific and concrete — the orchestrator uses your summary to decide what to do next.
