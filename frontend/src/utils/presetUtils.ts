import type { PresetConfig, AgentStationConfig } from '../api/types';
import type { BuiltinPreset } from '../constants/builtinPresets';
import { TIER_AGENT_MAP } from '../constants/builtinPresets';

/**
 * Expand a BuiltinPreset (tier-based) into a full PresetConfig (per-agent).
 * Used when the user clicks "Use" or "Duplicate" on a built-in preset.
 */
export function expandBuiltinPreset(
  builtin: BuiltinPreset,
  agentTypes: { id: string }[],
): PresetConfig {
  const agents: Record<string, AgentStationConfig> = {};

  for (const agent of agentTypes) {
    // Find which tier this agent belongs to
    let tierKey: string | null = null;
    for (const [tier, agentIds] of Object.entries(TIER_AGENT_MAP)) {
      if (agentIds.includes(agent.id)) {
        tierKey = tier;
        break;
      }
    }

    const model = tierKey
      ? builtin.tiers[tierKey as keyof typeof builtin.tiers] || builtin.tiers.model
      : builtin.tiers.model;

    const station: AgentStationConfig = {
      provider: builtin.provider,
      model,
    };

    // Apply provider-specific extras
    if (builtin.extras?.base_url) {
      station.base_url = builtin.extras.base_url;
    }
    if (builtin.extras?.thinking) {
      if (tierKey === 'model' || tierKey === 'planner_model') {
        station.thinking = builtin.extras.thinking.model || 'off';
      } else if (tierKey === 'insight_model') {
        station.thinking = builtin.extras.thinking.insight || 'off';
      } else if (tierKey === 'sub_agent_model') {
        station.thinking = builtin.extras.thinking.sub_agent || 'off';
      }
    }
    if (builtin.extras?.api_compat) {
      station.api_compat = builtin.extras.api_compat;
    }
    if (builtin.extras?.rate_limit_interval != null) {
      station.rate_limit_interval = builtin.extras.rate_limit_interval;
    }

    agents[agent.id] = station;
  }

  return {
    name: builtin.name,
    agents,
    capabilities: { ...builtin.capabilities },
  };
}

/**
 * Derive a tier summary from a PresetConfig for display in the Overview card.
 * Looks at representative agents to determine model per tier.
 */
export function deriveTierSummary(
  preset: PresetConfig,
): { provider: string; tiers: Record<string, string> } | null {
  const orchestrator = preset.agents?.orchestrator;
  if (!orchestrator) return null;

  const provider = orchestrator.provider;

  const tiers: Record<string, string> = {
    model: orchestrator.model || '',
    sub_agent_model: preset.agents?.viz_plotly?.model || preset.agents?.data_ops?.model || '',
    insight_model: preset.agents?.insight?.model || '',
    inline_model: preset.agents?.inline?.model || '',
    planner_model: preset.agents?.planner?.model || '',
  };

  return { provider, tiers };
}
