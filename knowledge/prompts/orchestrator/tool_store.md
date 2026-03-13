## Tool Store

All your tools are pre-loaded — delegation, discovery (web_search, xhelio__envoy_query),
memory, session, data_ops, and visualization. There is no need to browse or load
additional tools.

### Data Discovery with `xhelio__envoy_query`

To discover what is available without delegating to an envoy:
1. `xhelio__envoy_query()` — see all available envoys
2. `xhelio__envoy_query(envoy="X")` — see an envoy's capabilities
3. `xhelio__envoy_query(search="(?i)keyword")` — search across all envoys by regex
4. `delegate_to_envoy(envoy="X", request="...")` — delegate the actual work
