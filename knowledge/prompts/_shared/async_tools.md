## Tool Execution

All tools block — you get the real result (success or error) inline, never a "started" acknowledgement.
Do NOT poll with `xhelio__events` or any other tool; results arrive automatically.

### Parallel Tool Calling

When you need results from multiple independent tools before deciding next steps, call them all in the same response. The system automatically runs parallel-safe tools concurrently. No special syntax needed — just emit multiple tool calls in one turn.

Example: to run two independent queries simultaneously, call both in one response:

    xhelio__envoy_query()
    xhelio__assets()

When NOT to parallelize: dependent tools (where one needs the other's result — call sequentially).
