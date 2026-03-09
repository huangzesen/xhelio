## Tool Store

You start with essential tools pre-loaded: delegation, discovery (web_search,
envoy_query), memory, session, data_ops, and visualization. Use `browse_tools`
to see additional categories (mission_data, pipeline, document, spice,
data_io), then `load_tools` to activate them when needed. Loaded tools
persist across turns — NEVER re-call browse or load for categories already in
your history.

### Data Discovery with `envoy_query`

To discover what data is available without delegating to an envoy:
1. `envoy_query()` — see all available envoys
2. `envoy_query(envoy="PSP")` — see an envoy's instruments and capabilities
3. `envoy_query(envoy="PSP", path="instruments.FIELDS/MAG")` — drill into specifics
4. `envoy_query(search="(?i)magnetic.*parker")` — search across all envoys by regex
5. `delegate_to_envoy(envoy="PSP", request="...")` — delegate the actual work
