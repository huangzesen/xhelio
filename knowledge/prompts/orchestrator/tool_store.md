## Tool Store

All your tools are pre-loaded — delegation, discovery (web_search, envoy_query),
memory, session, data_ops, and visualization. There is no need to browse or load
additional tools.

### Data Discovery with `envoy_query`

To discover what data is available without delegating to an envoy:
1. `envoy_query()` — see all available envoys
2. `envoy_query(envoy="PSP")` — see an envoy's instruments and capabilities
3. `envoy_query(envoy="PSP", path="instruments.FIELDS/MAG")` — drill into specifics
4. `envoy_query(search="(?i)magnetic.*parker")` — search across all envoys by regex
5. `delegate_to_envoy(envoy="PSP", request="...")` — delegate the actual work
