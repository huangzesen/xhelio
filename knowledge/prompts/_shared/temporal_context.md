## Temporal Context in Tool Results

Every tool result includes timing metadata:
- `_ts`: UTC wall clock when the tool completed (ISO 8601)
- `_elapsed_ms`: Execution time in milliseconds

Use these to:
- **Assess data freshness**: Compare `_ts` across results to understand temporal ordering.
- **Detect slow operations**: `_elapsed_ms` > 10000 means the operation was slow — avoid repeating unnecessarily.
- **Avoid stale data**: When following up on earlier results, check `_ts` to determine if data may be outdated.