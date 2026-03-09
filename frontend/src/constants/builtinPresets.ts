// Built-in provider presets — hardcoded sensible defaults per provider.
// These are NOT stored in config. When a user activates one, it gets expanded
// into a per-agent PresetConfig and saved to config.presets.

export interface BuiltinPreset {
  id: string;
  name: string;
  icon: string;
  provider: string;
  description: string;
  tiers: {
    model: string;
    sub_agent_model: string;
    insight_model: string;
    inline_model: string;
    planner_model: string;
  };
  capabilities: {
    web_search: string | null;
    vision: string | null;
  };
  extras?: {
    thinking?: Record<string, string>;
    base_url?: string;
    api_compat?: string;
    rate_limit_interval?: number;
  };
}

// Tier → agent type mapping (single source of truth)
export const TIER_AGENT_MAP: Record<string, string[]> = {
  model: ['orchestrator'],
  planner_model: ['planner'],
  insight_model: ['insight'],
  sub_agent_model: ['viz_plotly', 'viz_mpl', 'viz_jsx', 'data_ops', 'data_io', 'envoy', 'eureka', 'memory'],
  inline_model: ['inline'],  // Autocomplete, session titles
};

export const MODEL_TIER_KEYS = ['model', 'sub_agent_model', 'insight_model', 'inline_model', 'planner_model'] as const;

// All 10 supported providers
export const PROVIDERS = [
  { id: 'gemini', name: 'Gemini', icon: '✦' },
  { id: 'openai', name: 'OpenAI', icon: '◎' },
  { id: 'anthropic', name: 'Anthropic', icon: '◈' },
  { id: 'minimax', name: 'MiniMax', icon: '◉' },
  { id: 'grok', name: 'Grok', icon: '◆' },
  { id: 'deepseek', name: 'DeepSeek', icon: '◇' },
  { id: 'qwen', name: 'Qwen', icon: '◍' },
  { id: 'kimi', name: 'Kimi', icon: '◐' },
  { id: 'glm', name: 'GLM', icon: '◑' },
  { id: 'custom', name: 'Custom', icon: '◫' },
] as const;

// MiniMax API endpoint options (LLM API and MCP use different formats)
export const MINIMAX_ENDPOINTS = [
  { value: 'https://api.minimaxi.com', label: 'China (api.minimaxi.com)' },
  { value: 'https://api.minimax.io', label: 'International (api.minimax.io)' },
] as const;

export const BUILTIN_PRESETS: BuiltinPreset[] = [
  {
    id: 'gemini',
    name: 'Gemini',
    icon: '✦',
    provider: 'gemini',
    description: 'Google Gemini — fast, multimodal, great value',
    tiers: {
      model: 'gemini-2.5-flash',
      sub_agent_model: 'gemini-2.5-flash',
      insight_model: 'gemini-2.5-flash',
      inline_model: 'gemini-2.5-flash-lite',
      planner_model: 'gemini-2.5-flash',
    },
    capabilities: { web_search: 'own', vision: 'own' },
    extras: { thinking: { model: 'low', sub_agent: 'low', insight: 'low' } },
  },
  {
    id: 'openai',
    name: 'OpenAI',
    icon: '◎',
    provider: 'openai',
    description: 'OpenAI GPT models — reliable, widely supported',
    tiers: {
      model: 'gpt-4.1',
      sub_agent_model: 'gpt-4.1-mini',
      insight_model: 'gpt-4.1-mini',
      inline_model: 'gpt-4.1-nano',
      planner_model: 'gpt-4.1',
    },
    capabilities: { web_search: null, vision: 'own' },
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    icon: '◈',
    provider: 'anthropic',
    description: 'Claude models — strong reasoning, careful analysis',
    tiers: {
      model: 'claude-sonnet-4-20250514',
      sub_agent_model: 'claude-haiku-4-5-20251001',
      insight_model: 'claude-sonnet-4-20250514',
      inline_model: 'claude-haiku-4-5-20251001',
      planner_model: 'claude-sonnet-4-20250514',
    },
    capabilities: { web_search: null, vision: 'own' },
  },
  {
    id: 'minimax',
    name: 'MiniMax',
    icon: '◉',
    provider: 'minimax',
    description: 'MiniMax — cost-effective, fast inference',
    tiers: {
      model: 'MiniMax-M2.5-Highspeed',
      sub_agent_model: 'MiniMax-M2.5-Highspeed',
      insight_model: 'MiniMax-M2.5-Highspeed',
      inline_model: 'MiniMax-M2.5-Highspeed',
      planner_model: 'MiniMax-M2.5-Highspeed',
    },
    capabilities: { web_search: 'own', vision: 'own' },
    extras: { base_url: 'https://api.minimaxi.com/anthropic', rate_limit_interval: 2 },
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    icon: '◇',
    provider: 'deepseek',
    description: 'DeepSeek — strong reasoning at low cost',
    tiers: {
      model: 'deepseek-chat',
      sub_agent_model: 'deepseek-chat',
      insight_model: 'deepseek-chat',
      inline_model: 'deepseek-chat',
      planner_model: 'deepseek-chat',
    },
    capabilities: { web_search: null, vision: null },
  },
];

// Funky names for user-created combos
export const COMBO_NAME_SUGGESTIONS = [
  'Speedy Gonzales',
  'Big Brain',
  'Frankenstein',
  'Budget Wizard',
  'Pocket Rocket',
  'The Overachiever',
  'Lazy Sunday',
  'Mad Scientist',
  'Swiss Army Knife',
  'Turbo Nerd',
];

export const COMBO_EMOJIS: Record<string, string> = {
  'Speedy Gonzales': '\u26A1',
  'Big Brain': '\uD83E\uDDE0',
  'Frankenstein': '\uD83E\uDDDF',
  'Budget Wizard': '\uD83E\uDDD9',
  'Pocket Rocket': '\uD83D\uDE80',
  'The Overachiever': '\uD83D\uDCAA',
  'Lazy Sunday': '\uD83D\uDE34',
  'Mad Scientist': '\uD83D\uDD2C',
  'Swiss Army Knife': '\uD83D\uDD27',
  'Turbo Nerd': '\uD83E\uDD13',
};

export function getComboEmoji(name: string): string {
  return COMBO_EMOJIS[name] || '\uD83C\uDF9B\uFE0F';
}

export function slugify(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}
