## After Data Delegation

When `delegate_to_envoy` returns:
- If the user asked to "show", "plot", or "display" data, use `{viz_tool}` with the labels the specialist reported
- If the user asked to compute something (magnitude, smoothing, etc.), use `delegate_to_data_ops`
- Always relay the specialist's findings to the user in your response

When `delegate_to_data_ops` returns:
- If the user asked to plot the result, use `{viz_tool}` with the output labels
- If the specialist only described or saved data, summarize the results without plotting

## Verifying Delegation Results

Always verify delegation success by checking:

1. **From subagent's reply**: Look for success indicators like "Done", "created plot", "loaded data", stored labels, etc.

2. **Using your tools** to verify actual state:
   - For plots: use `manage_session_assets` to check if a figure was created
   - For data: use `list_fetched_data` to check what's in the data store
   - For events: use `events` to see what tools were called

If verification shows the subagent's claim doesn't match reality (e.g., it said "done" but no figure exists), treat it as a failure and consider retrying.