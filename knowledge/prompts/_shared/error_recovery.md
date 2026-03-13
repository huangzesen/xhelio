## Error Recovery

- **Sub-agent spinning**: If a sub-agent makes 2-3 rounds with no progress (repeating the same calls), cancel and try a different approach.
- **Delegation returns error**: Analyze the error details and try: (1) retry with different parameters or time range, (2) use an alternative dataset, or (3) try a different approach.
- **Envoy scope**: Each envoy has a specific scope. Use `xhelio__envoy_query()` to understand what an envoy can and cannot do before delegating.
