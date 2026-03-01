export function agentModelTier(agentName: string): string {
  if (agentName === 'Orchestrator') return 'smart';
  if (agentName === 'Planner') return 'planner';
  if (agentName === 'Inline') return 'inline';
  if (agentName === 'Memory') return 'inline';
  if (agentName.startsWith('InsightAgent')) return 'insight';
  return 'sub_agent';
}
