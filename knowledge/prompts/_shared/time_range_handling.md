## Time Range Handling

All times are in UTC (appropriate for spacecraft data). Tools that accept time parameters require ISO 8601 dates for `time_start` and `time_end`. Resolve any relative or natural-language time expressions (e.g., "last week", "January 2024", "during the Jupiter encounter") into concrete ISO dates before calling tools.

Formats accepted by `time_start` / `time_end`:
- **Date**: `2024-01-15` (day precision)
- **Datetime**: `2024-01-15T06:00:00` (sub-day precision)