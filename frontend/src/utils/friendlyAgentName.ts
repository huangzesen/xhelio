/** Map backend agent IDs to human-friendly display names.
 *
 * Handles ephemeral overflow suffixes (`#N`) and think-phase suffixes (`/Think`).
 *
 * NOTE: The MAP below is intentionally frontend-only. Agent IDs are internal
 * identifiers that don't change at runtime, and the fallback (line 37) handles
 * unknown agents gracefully by showing the raw ID. If new agent types are
 * added to the backend, add a mapping here for a friendlier display name.
 */
export function friendlyAgentName(agentId: string): string {
  // Strip /Think suffix for display (e.g., "VizAgent[Plotly]/Think" → "VizAgent[Plotly]")
  const baseName = agentId.replace(/\/Think$/, '');
  const thinkSuffix = baseName !== agentId ? ' (Think)' : '';

  // Strip #N ephemeral suffix (e.g., "EnvoyAgent[PSP]#0" → "EnvoyAgent[PSP]")
  const withoutSeq = baseName.replace(/#\d+$/, '');
  const seqMatch = baseName.match(/#(\d+)$/);
  const seqSuffix = seqMatch ? ` #${seqMatch[1]}` : '';

  // Strip :<hex> hash suffix from new-style IDs (e.g., "orchestrator:aa2b87" → "orchestrator")
  const withoutHash = withoutSeq.replace(/:[0-9a-f]{4,8}$/, '');

  // EnvoyAgent[PSP] → Envoy [PSP]
  const missionMatch = withoutHash.match(/^EnvoyAgent\[(.+)]$/);
  if (missionMatch) return `Envoy [${missionMatch[1]}]${seqSuffix}${thinkSuffix}`;

  // VizAgent[Plotly] → Visualization [Plotly], VizAgent[Mpl] → Visualization [Mpl]
  const vizMatch = withoutHash.match(/^VizAgent\[(.+)]$/);
  if (vizMatch) return `Visualization [${vizMatch[1]}]${seqSuffix}${thinkSuffix}`;

  // New-style viz IDs: "viz:plotly" → Visualization [Plotly]
  const newVizMatch = withoutHash.match(/^viz:(.+)$/);
  if (newVizMatch) {
    const backend = newVizMatch[1].charAt(0).toUpperCase() + newVizMatch[1].slice(1);
    return `Visualization [${backend}]${seqSuffix}${thinkSuffix}`;
  }

  const MAP: Record<string, string> = {
    orchestrator: 'Orchestrator',
    VizAgent: 'Visualization',  // backward compat for old sessions
    DataOpsAgent: 'Data Ops',
    DataIOAgent: 'Data I/O',
    InsightAgent: 'Insight',
    Memory: 'Memory',
    Discovery: 'Discovery',
    data_ops: 'Data Ops',
    data_io: 'Data I/O',
    memory: 'Memory',
    eureka: 'Discovery',
  };
  const friendly = MAP[withoutHash];
  if (friendly) return `${friendly}${seqSuffix}${thinkSuffix}`;

  return agentId; // fallback: show raw ID
}
