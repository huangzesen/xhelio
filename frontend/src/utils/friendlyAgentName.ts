/** Map backend agent IDs to human-friendly display names.
 *
 * Handles ephemeral overflow suffixes (`#N`) and think-phase suffixes (`/Think`).
 */
export function friendlyAgentName(agentId: string): string {
  // Strip /Think suffix for display (e.g., "PlannerAgent/Think" → "PlannerAgent")
  const baseName = agentId.replace(/\/Think$/, '');
  const thinkSuffix = baseName !== agentId ? ' (Think)' : '';

  // Strip #N ephemeral suffix (e.g., "MissionAgent[PSP]#0" → "MissionAgent[PSP]")
  const withoutSeq = baseName.replace(/#\d+$/, '');
  const seqMatch = baseName.match(/#(\d+)$/);
  const seqSuffix = seqMatch ? ` #${seqMatch[1]}` : '';

  // MissionAgent[PSP] → Mission [PSP]
  const missionMatch = withoutSeq.match(/^MissionAgent\[(.+)]$/);
  if (missionMatch) return `Mission [${missionMatch[1]}]${seqSuffix}${thinkSuffix}`;

  // VizAgent[Plotly] → Visualization [Plotly], VizAgent[Mpl] → Visualization [Mpl]
  const vizMatch = withoutSeq.match(/^VizAgent\[(.+)]$/);
  if (vizMatch) return `Visualization [${vizMatch[1]}]${seqSuffix}${thinkSuffix}`;

  const MAP: Record<string, string> = {
    orchestrator: 'Orchestrator',
    VizAgent: 'Visualization',  // backward compat for old sessions
    DataOpsAgent: 'Data Ops',
    DataExtractionAgent: 'Extraction',
    InsightAgent: 'Insight',
    Memory: 'Memory',
    Discovery: 'Discovery',
    PlannerAgent: 'Planner',
    SpiceMCP: 'SPICE',
  };
  const friendly = MAP[withoutSeq];
  if (friendly) return `${friendly}${seqSuffix}${thinkSuffix}`;

  return agentId; // fallback: show raw ID
}
