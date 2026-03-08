import { useCallback, useEffect, useRef, useState } from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import { listModels, listApiKeys, getProviders, getAgentTypes } from '../../api/client';
import type { ModelInfo, ProviderInfo } from '../../api/client';
import type {
  ApiKeyEntry,
  PresetConfig,
  AgentStationConfig,
  AgentTypeInfo,
  WorkbenchConfig,
} from '../../api/types';
import {
  Loader2,
  Save,
  Copy,
  Trash2,
  Pencil,
  Plus,
  FlaskConical,
  Info,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

// ---- Constants ----

const COMBO_NAME_SUGGESTIONS = [
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

const MODEL_TIER_KEYS = ['model', 'sub_agent_model', 'insight_model', 'inline_model', 'planner_model'] as const;

// Tier → agent type mapping (matches config.py resolve_agent_model)
const TIER_AGENT_MAP: Record<string, string[]> = {
  model: ['orchestrator'],
  planner_model: ['planner'],
  insight_model: ['insight'],
  sub_agent_model: ['viz_plotly', 'viz_mpl', 'viz_jsx', 'data_ops', 'data_io', 'envoy', 'eureka', 'memory'],
  inline_model: [], // Not a sub-agent — used for autocomplete/titles
};

const MINIMAX_KNOWN_URLS = [
  'https://api.minimaxi.com/anthropic',
  'https://api.minimax.io/anthropic',
];

const COMBO_EMOJIS: Record<string, string> = {
  'Speedy Gonzales': '\u26A1',
  'Big Brain': '\uD83E\uDDE0',
  'Frankenstein': '\uD83E\uDDDF',
  'Budget Wizard': '\uD83E\uDDD9',
  'Pocket Rocket': '\uD83D\uDE80',
  'The Overachiever': '\uD83C\uDFC6',
  'Lazy Sunday': '\uD83D\uDE34',
  'Mad Scientist': '\uD83D\uDD2C',
  'Swiss Army Knife': '\uD83D\uDD27',
  'Turbo Nerd': '\uD83E\uDD13',
  'Default': '\u2699\uFE0F',
};

const FALLBACK_EMOJIS = ['\uD83C\uDFAF', '\uD83D\uDCA1', '\uD83C\uDF1F', '\uD83D\uDD25', '\uD83C\uDF0A', '\uD83C\uDF08', '\uD83D\uDC8E', '\uD83C\uDF3F'];

function getComboEmoji(name: string): string {
  if (COMBO_EMOJIS[name]) return COMBO_EMOJIS[name];
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
  }
  return FALLBACK_EMOJIS[Math.abs(hash) % FALLBACK_EMOJIS.length];
}

function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

// ---- Preset ↔ Tier conversion helpers ----

/** Expand tier-based config into per-agent preset stations. */
function tiersToPreset(
  name: string,
  provider: string,
  tierModels: Record<string, string>,
  capabilities: { web_search: string | null; vision: string | null },
  extras?: {
    base_url?: string | null;
    api_compat?: string | null;
    thinking?: Record<string, string>;
    rate_limit_interval?: number;
  },
): PresetConfig {
  const agents: Record<string, AgentStationConfig> = {};

  for (const [tier, agentIds] of Object.entries(TIER_AGENT_MAP)) {
    const model = tierModels[tier];
    if (!model || agentIds.length === 0) continue;

    for (const agentId of agentIds) {
      const station: AgentStationConfig = { provider, model };
      if (extras?.base_url) station.base_url = extras.base_url;
      if (extras?.api_compat && provider === 'custom') station.api_compat = extras.api_compat;
      if (extras?.rate_limit_interval && provider === 'minimax') {
        station.rate_limit_interval = extras.rate_limit_interval;
      }
      // Per-agent thinking levels for Gemini
      if (extras?.thinking && provider === 'gemini') {
        if (tier === 'model') station.thinking = extras.thinking.model || 'high';
        else if (tier === 'sub_agent_model') station.thinking = extras.thinking.sub_agent || 'low';
        else if (tier === 'insight_model') station.thinking = extras.thinking.insight || 'low';
        else if (tier === 'planner_model') station.thinking = extras.thinking.model || 'high';
      }
      agents[agentId] = station;
    }
  }

  return { name, agents, capabilities };
}

/** Collapse a preset's per-agent stations back into tier-based view. */
function presetToTiers(
  preset: PresetConfig,
): { provider: string; tierModels: Record<string, string>; capabilities: { web_search: string | null; vision: string | null } } {
  // Pick the provider from the first agent that has one
  let provider = '';
  const tierModels: Record<string, string> = {};

  for (const [tier, agentIds] of Object.entries(TIER_AGENT_MAP)) {
    if (agentIds.length === 0) continue;
    // Use the first agent in this tier as representative
    const representative = agentIds[0];
    const station = preset.agents?.[representative];
    if (station) {
      if (!provider) provider = station.provider;
      tierModels[tier] = station.model;
    }
  }

  return {
    provider: provider || 'gemini',
    tierModels,
    capabilities: preset.capabilities || { web_search: null, vision: null },
  };
}

// ---- Combobox Component ----

function Combobox({
  value,
  onChange,
  suggestions,
  placeholder,
}: {
  value: string;
  onChange: (val: string) => void;
  suggestions: string[];
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  const filtered = suggestions.filter((s) =>
    s.toLowerCase().includes((filter || value).toLowerCase()),
  );

  return (
    <div ref={ref} className="relative">
      <input
        type="text"
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          setFilter(e.target.value);
          if (!open) setOpen(true);
        }}
        onFocus={() => {
          setOpen(true);
          setFilter('');
        }}
        placeholder={placeholder}
        className="block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
      />
      {open && filtered.length > 0 && (
        <div className="absolute left-0 right-0 top-full mt-1 bg-panel rounded-lg border border-border shadow-lg z-20 max-h-48 overflow-y-auto py-1">
          {filtered.map((s) => (
            <button
              key={s}
              type="button"
              className="w-full text-left px-3 py-1.5 text-sm text-text hover:bg-surface transition-colors"
              onMouseDown={(e) => {
                e.preventDefault();
                onChange(s);
                setOpen(false);
                setFilter('');
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Tier Agent Hint ----

function TierAgentHint({
  tier,
  agentTypes,
}: {
  tier: string;
  agentTypes: AgentTypeInfo[];
}) {
  const agentIds = TIER_AGENT_MAP[tier] || [];
  if (agentIds.length === 0) return null;

  const agentNames = agentIds
    .map((id) => {
      const agent = agentTypes.find((a) => a.id === id);
      return agent ? `${agent.icon} ${agent.name}` : id;
    });

  return (
    <div className="flex items-start gap-1.5 mt-1 text-[11px] text-text-muted leading-relaxed">
      <Info size={12} className="mt-0.5 shrink-0 opacity-60" />
      <span>{agentNames.join(', ')}</span>
    </div>
  );
}

// ---- Main Component ----

export function ModelsSection() {
  const { t } = useTranslation(['settings', 'common']);
  const { config, updateConfig } = useSettingsStore();

  const MODEL_TIERS = MODEL_TIER_KEYS.map((key) => ({
    key,
    label: t(`models.tiers.${key}`),
  }));

  // Local state
  const [providerList, setProviderList] = useState<ProviderInfo[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [apiKeys, setApiKeys] = useState<ApiKeyEntry[]>([]);
  const [agentTypes, setAgentTypes] = useState<AgentTypeInfo[]>([]);
  const [minimaxCustomMode, setMiniMaxCustomMode] = useState(false);

  const providerName = useCallback(
    (id: string): string => providerList.find((p) => p.id === id)?.name ?? id,
    [providerList],
  );
  const [saveAsOpen, setSaveAsOpen] = useState(false);
  const [saveAsName, setSaveAsName] = useState('');
  const [renamingKey, setRenamingKey] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [hoveredCombo, setHoveredCombo] = useState<string | null>(null);

  const saveAsRef = useRef<HTMLDivElement>(null);

  // Derived state from config
  const presets = (config.presets ?? {}) as Record<string, PresetConfig>;
  const workbench = (config.workbench ?? {}) as WorkbenchConfig;
  const activePresetKey = workbench.preset ?? null;

  // Derive tier view from active preset or global defaults
  const activeProvider = (() => {
    if (activePresetKey && presets[activePresetKey]) {
      return presetToTiers(presets[activePresetKey]).provider;
    }
    return (config.llm_provider as string) ?? 'gemini';
  })();

  const providerConfigMap = (config.providers ?? {}) as Record<string, Record<string, unknown>>;
  const providerConfig = providerConfigMap[activeProvider] ?? {};

  // Active combo state — derive from active preset or build from current config
  const activeTiers: Record<string, string> = (() => {
    if (activePresetKey && presets[activePresetKey]) {
      return presetToTiers(presets[activePresetKey]).tierModels;
    }
    return {
      model: (providerConfig.model as string) ?? '',
      sub_agent_model: (providerConfig.sub_agent_model as string) ?? '',
      insight_model: (providerConfig.insight_model as string) ?? '',
      inline_model: (providerConfig.inline_model as string) ?? '',
      planner_model: (providerConfig.planner_model as string) ?? '',
    };
  })();

  const activeComboName = (() => {
    if (activePresetKey && presets[activePresetKey]) {
      return presets[activePresetKey].name || activePresetKey;
    }
    return 'Unsaved';
  })();

  // Fetch models when provider changes
  useEffect(() => {
    setModelsLoading(true);
    listModels(activeProvider)
      .then((res) => setAvailableModels(res.models || []))
      .catch(() => setAvailableModels([]))
      .finally(() => setModelsLoading(false));
  }, [activeProvider]);

  // Fetch providers, API keys, and agent types on mount
  useEffect(() => {
    getProviders()
      .then(setProviderList)
      .catch(() => {});
    listApiKeys()
      .then((res) => setApiKeys(res.keys))
      .catch(() => setApiKeys([]));
    getAgentTypes()
      .then((res) => setAgentTypes(res.agents))
      .catch(() => {});
  }, []);

  // Close Save As popover on outside click
  useEffect(() => {
    if (!saveAsOpen) return;
    function handleClick(e: MouseEvent) {
      if (saveAsRef.current && !saveAsRef.current.contains(e.target as Node)) {
        setSaveAsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [saveAsOpen]);

  // ---- Helper: get/set tier values in local view ----
  // We track the "editing" state via the active preset. When a tier changes,
  // we update the preset in-place (or create a new one if unsaved).

  const getTierValue = useCallback(
    (key: string): string => activeTiers[key] ?? '',
    [activeTiers],
  );

  const setTierValue = useCallback(
    (key: string, value: string) => {
      if (activePresetKey && presets[activePresetKey]) {
        // Update the preset's agents for this tier
        const newTiers = { ...activeTiers, [key]: value };
        const existing = presets[activePresetKey];
        const updated = tiersToPreset(
          existing.name,
          activeProvider,
          newTiers,
          existing.capabilities || { web_search: null, vision: null },
          extractExtras(activeProvider, providerConfig),
        );
        updateConfig({ presets: { ...presets, [activePresetKey]: updated } });
      } else {
        // No active preset — update legacy provider config
        const updated = {
          ...providerConfigMap,
          [activeProvider]: { ...providerConfig, [key]: value },
        };
        updateConfig({ providers: updated });
      }
    },
    [activePresetKey, presets, activeTiers, activeProvider, providerConfig, providerConfigMap, updateConfig],
  );

  const setActiveProvider = useCallback(
    (provider: string) => {
      if (activePresetKey && presets[activePresetKey]) {
        // Rebuild the preset with the new provider, keeping tier models
        const existing = presets[activePresetKey];
        const updated = tiersToPreset(
          existing.name,
          provider,
          activeTiers,
          existing.capabilities || { web_search: null, vision: null },
        );
        updateConfig({
          presets: { ...presets, [activePresetKey]: updated },
          llm_provider: provider,
        });
      } else {
        updateConfig({ llm_provider: provider });
      }
    },
    [activePresetKey, presets, activeTiers, updateConfig],
  );

  // ---- Preset operations (replacing combo operations) ----

  const snapshotCurrentPreset = useCallback(
    (name: string): PresetConfig => {
      return tiersToPreset(
        name,
        activeProvider,
        {
          model: getTierValue('model'),
          sub_agent_model: getTierValue('sub_agent_model'),
          insight_model: getTierValue('insight_model'),
          inline_model: getTierValue('inline_model'),
          planner_model: getTierValue('planner_model'),
        },
        {
          web_search: getCapabilityValue('web_search'),
          vision: getCapabilityValue('vision'),
        },
        extractExtras(activeProvider, providerConfig),
      );
    },
    [activeProvider, providerConfig], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const handleSavePreset = useCallback(() => {
    if (!activePresetKey) return;
    const preset = snapshotCurrentPreset(activeComboName);
    updateConfig({ presets: { ...presets, [activePresetKey]: preset } });
  }, [activePresetKey, activeComboName, presets, snapshotCurrentPreset, updateConfig]);

  const handleSaveAsPreset = useCallback(
    (name: string) => {
      const slug = slugify(name);
      if (!slug) return;
      const preset = snapshotCurrentPreset(name);
      updateConfig({
        presets: { ...presets, [slug]: preset },
        workbench: { ...workbench, preset: slug, agents: {} },
      });
      setSaveAsOpen(false);
      setSaveAsName('');
    },
    [presets, workbench, snapshotCurrentPreset, updateConfig],
  );

  const loadPreset = useCallback(
    (presetKey: string) => {
      // Activate the preset in workbench and clear per-agent overrides
      updateConfig({
        workbench: { ...workbench, preset: presetKey, agents: {} },
        // Also sync the legacy llm_provider for backward compat
        llm_provider: presetToTiers(presets[presetKey]).provider,
      });
    },
    [presets, workbench, updateConfig],
  );

  const deletePreset = useCallback(
    (presetKey: string) => {
      const newPresets = { ...presets };
      delete newPresets[presetKey];
      const newActive = activePresetKey === presetKey ? null : activePresetKey;
      updateConfig({
        presets: newPresets,
        workbench: { ...workbench, preset: newActive },
      });
    },
    [presets, activePresetKey, workbench, updateConfig],
  );

  const renamePreset = useCallback(
    (presetKey: string, newName: string) => {
      if (!newName.trim()) return;
      const newSlug = slugify(newName);
      if (!newSlug) return;
      const preset = presets[presetKey];
      if (!preset) return;
      const renamedPreset = { ...preset, name: newName };
      const newPresets = { ...presets };
      delete newPresets[presetKey];
      newPresets[newSlug] = renamedPreset;
      const newActive = activePresetKey === presetKey ? newSlug : activePresetKey;
      updateConfig({
        presets: newPresets,
        workbench: { ...workbench, preset: newActive },
      });
      setRenamingKey(null);
      setRenameValue('');
    },
    [presets, activePresetKey, workbench, updateConfig],
  );

  const handleNewPreset = useCallback(() => {
    const usedNames = new Set(Object.values(presets).map((p) => p.name));
    const nextName =
      COMBO_NAME_SUGGESTIONS.find((n) => !usedNames.has(n)) ?? `Preset ${Object.keys(presets).length + 1}`;
    const slug = slugify(nextName);
    const preset = snapshotCurrentPreset(nextName);
    updateConfig({
      presets: { ...presets, [slug]: preset },
      workbench: { ...workbench, preset: slug, agents: {} },
    });
  }, [presets, workbench, snapshotCurrentPreset, updateConfig]);

  const setComboName = useCallback(
    (name: string) => {
      if (!activePresetKey || !presets[activePresetKey]) return;
      const updated = { ...presets[activePresetKey], name };
      updateConfig({ presets: { ...presets, [activePresetKey]: updated } });
    },
    [activePresetKey, presets, updateConfig],
  );

  // ---- Capability helpers ----
  const configuredProviderIds = new Set(
    apiKeys.filter((k) => k.configured).map((k) => k.provider).filter(Boolean) as string[],
  );
  configuredProviderIds.add(activeProvider);

  const capabilityOptions = [
    { value: activeProvider, label: `${providerName(activeProvider)} ${t('models.ownSuffix')}` },
    ...providerList.filter(
      (p) => p.id !== activeProvider && configuredProviderIds.has(p.id),
    ).map((p) => ({ value: p.id, label: p.name })),
    { value: '', label: t('models.disabled') },
  ];

  const getCapabilityValue = useCallback(
    (key: 'web_search' | 'vision'): string | null => {
      if (activePresetKey && presets[activePresetKey]) {
        return presets[activePresetKey].capabilities?.[key] ?? null;
      }
      return (providerConfig[`${key}_provider`] as string) ?? null;
    },
    [activePresetKey, presets, providerConfig],
  );

  const setCapabilityValue = useCallback(
    (key: 'web_search' | 'vision', value: string | null) => {
      if (activePresetKey && presets[activePresetKey]) {
        const existing = presets[activePresetKey];
        const updated = {
          ...existing,
          capabilities: { ...(existing.capabilities || { web_search: null, vision: null }), [key]: value },
        };
        updateConfig({ presets: { ...presets, [activePresetKey]: updated } });
      } else {
        const updated = {
          ...providerConfigMap,
          [activeProvider]: { ...providerConfig, [`${key}_provider`]: value },
        };
        updateConfig({ providers: updated });
      }
    },
    [activePresetKey, presets, activeProvider, providerConfig, providerConfigMap, updateConfig],
  );

  // ---- Provider-specific field helpers ----
  const getProviderField = useCallback(
    (key: string): string => (providerConfig[key] as string) ?? '',
    [providerConfig],
  );

  const setProviderField = useCallback(
    (key: string, value: unknown) => {
      const updated = {
        ...providerConfigMap,
        [activeProvider]: { ...providerConfig, [key]: value },
      };
      updateConfig({ providers: updated });
    },
    [providerConfigMap, activeProvider, providerConfig, updateConfig],
  );

  // ---- MiniMax base URL helpers ----
  const minimaxBaseUrl = getProviderField('base_url') || MINIMAX_KNOWN_URLS[0];
  const isMinimaxCustomUrl = !MINIMAX_KNOWN_URLS.includes(minimaxBaseUrl);
  const showMinimaxCustom = minimaxCustomMode || isMinimaxCustomUrl;

  // ---- Render ----

  const presetKeys = Object.keys(presets);

  return (
    <div className="py-4 space-y-6 max-w-4xl">
      <div>
        <h2 className="text-lg font-medium text-text mb-1">{t('models.title')}</h2>
        <p className="text-xs text-text-muted">
          {t('models.description')}
        </p>
      </div>

      {/* ========== Part 1: Active Combo Editor ========== */}
      <div className="bg-panel rounded-xl border border-border p-6 space-y-6">
        {/* Header + Save controls */}
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-2">
            <FlaskConical size={18} className="text-primary" />
            <h3 className="font-medium text-text">
              {activePresetKey ? activeComboName : t('models.comboEditor')}
            </h3>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={handleSavePreset}
              disabled={!activePresetKey}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium bg-primary text-white hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Save size={14} />
              {t('common:actions.save')}
            </button>
            <div className="relative" ref={saveAsRef}>
              <button
                onClick={() => {
                  setSaveAsOpen(!saveAsOpen);
                  setSaveAsName(activeComboName);
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium border border-border text-text-muted hover:text-text hover:bg-surface transition-colors"
              >
                <Copy size={14} />
                {t('models.saveAs')}
              </button>
              {saveAsOpen && (
                <div className="absolute right-0 top-full mt-2 w-72 bg-panel rounded-xl border border-border shadow-lg z-20 p-4 space-y-3">
                  <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
                    {t('models.saveAsNewCombo')}
                  </span>
                  <Combobox
                    value={saveAsName}
                    onChange={setSaveAsName}
                    suggestions={COMBO_NAME_SUGGESTIONS}
                    placeholder={t('models.nameThisCombo')}
                  />
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => setSaveAsOpen(false)}
                      className="px-3 py-1.5 rounded-lg text-sm text-text-muted hover:text-text transition-colors"
                    >
                      {t('common:actions.cancel')}
                    </button>
                    <button
                      onClick={() => handleSaveAsPreset(saveAsName)}
                      disabled={!saveAsName.trim()}
                      className="px-3 py-1.5 rounded-lg text-sm font-medium bg-primary text-white hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {t('common:actions.save')}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Combo Name */}
        <div className="space-y-1.5">
          <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
            {t('models.comboName')}
          </span>
          <Combobox
            value={activeComboName}
            onChange={setComboName}
            suggestions={COMBO_NAME_SUGGESTIONS}
            placeholder={t('models.nameThisCombo')}
          />
        </div>

        {/* Provider Dropdown */}
        <div className="space-y-1.5">
          <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
            {t('models.provider')}
          </span>
          <select
            value={activeProvider}
            onChange={(e) => setActiveProvider(e.target.value)}
            className="block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
          >
            {providerList.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>

        {/* Model Tiers */}
        <div className="space-y-1.5">
          <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
            {t('models.modelTiers')}
          </span>
          <div className="space-y-3">
            {MODEL_TIERS.map(({ key, label }) => {
              const current = getTierValue(key);
              const inList = availableModels.some((m) => m.id === current);
              return (
                <label key={key} className="block">
                  <span className="text-xs text-text-muted">{label}</span>
                  <TierAgentHint tier={key} agentTypes={agentTypes} />
                  {modelsLoading ? (
                    <div className="mt-1 flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text-muted">
                      <Loader2 size={14} className="animate-spin" /> {t('models.loadingModels')}
                    </div>
                  ) : availableModels.length > 0 && activeProvider !== 'minimax' ? (
                    <select
                      value={current}
                      onChange={(e) => setTierValue(key, e.target.value)}
                      className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm font-mono bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                    >
                      {!current && <option value="">{t('models.selectModel')}</option>}
                      {current && !inList && <option value={current}>{current}</option>}
                      {availableModels.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.display_name || m.id}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={current}
                      onChange={(e) => setTierValue(key, e.target.value)}
                      placeholder={
                        key === 'planner_model'
                          ? t('models.placeholders.planner_model')
                          : key === 'insight_model'
                            ? t('models.placeholders.insight_model')
                            : ''
                      }
                      className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm font-mono bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                    />
                  )}
                </label>
              );
            })}
          </div>
        </div>

        {/* Capabilities Subsection */}
        <div className="space-y-1.5">
          <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
            {t('models.capabilities')}
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <label className="block">
              <span className="text-xs text-text-muted">{t('models.webSearch')}</span>
              <select
                value={getCapabilityValue('web_search') || activeProvider}
                onChange={(e) => setCapabilityValue('web_search', e.target.value || null)}
                className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
              >
                {capabilityOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="text-xs text-text-muted">{t('models.vision')}</span>
              <select
                value={getCapabilityValue('vision') || activeProvider}
                onChange={(e) => setCapabilityValue('vision', e.target.value || null)}
                className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
              >
                {capabilityOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        {/* Provider-Specific: Gemini Thinking */}
        {activeProvider === 'gemini' && (
          <div className="space-y-1.5">
            <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
              {t('models.gemini.thinkingLevel')}
            </span>
            <div className="grid grid-cols-3 gap-3">
              {([
                ['thinking_model', t('models.gemini.main'), 'high'],
                ['thinking_sub_agent', t('models.gemini.subAgent'), 'low'],
                ['thinking_insight', t('models.gemini.insight'), 'low'],
              ] as const).map(([key, label, defaultVal]) => (
                <label key={key} className="block">
                  <span className="text-xs text-text-muted">{label}</span>
                  <select
                    value={getProviderField(key) || defaultVal}
                    onChange={(e) => setProviderField(key, e.target.value)}
                    className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                  >
                    <option value="off">{t('models.gemini.off')}</option>
                    <option value="low">{t('models.gemini.low')}</option>
                    <option value="high">{t('models.gemini.high')}</option>
                  </select>
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Provider-Specific: MiniMax */}
        {activeProvider === 'minimax' && (
          <div className="space-y-4">
            <div className="space-y-1.5">
              <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
                {t('models.minimax.settings')}
              </span>
              <label className="block">
                <span className="text-xs text-text-muted">{t('models.minimax.baseUrl')}</span>
                <select
                  value={showMinimaxCustom ? '__custom__' : minimaxBaseUrl}
                  onChange={(e) => {
                    if (e.target.value === '__custom__') {
                      setMiniMaxCustomMode(true);
                    } else {
                      setMiniMaxCustomMode(false);
                      setProviderField('base_url', e.target.value);
                    }
                  }}
                  className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm font-mono bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                >
                  <option value="https://api.minimaxi.com/anthropic">
                    {t('models.minimax.china')}
                  </option>
                  <option value="https://api.minimax.io/anthropic">
                    {t('models.minimax.international')}
                  </option>
                  <option value="__custom__">{t('models.minimax.customEndpoint')}</option>
                </select>
                {showMinimaxCustom && (
                  <input
                    type="text"
                    value={isMinimaxCustomUrl ? minimaxBaseUrl : ''}
                    onChange={(e) => setProviderField('base_url', e.target.value)}
                    placeholder="https://your-custom-endpoint.com/anthropic"
                    className="mt-2 block w-full rounded-lg border border-border px-3 py-2 text-sm font-mono bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                  />
                )}
              </label>
            </div>
            <label className="block">
              <span className="text-xs text-text-muted">
                {t('models.minimax.rateLimit')} ({getProviderField('rate_limit_interval') || '2.0'}{t('models.minimax.rateLimitUnit')})
              </span>
              <input
                type="range"
                min={0.5}
                max={5}
                step={0.5}
                value={parseFloat(getProviderField('rate_limit_interval') || '2.0')}
                onChange={(e) =>
                  setProviderField('rate_limit_interval', parseFloat(e.target.value))
                }
                className="mt-1 block w-full"
              />
            </label>
          </div>
        )}

        {/* Base URL for providers that support it */}
        {providerList.find((p) => p.id === activeProvider)?.supports_base_url && (
          <div className="space-y-1.5">
            <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
              {providerName(activeProvider)} {t('title')}
            </span>
            <label className="block">
              <span className="text-xs text-text-muted">{t('models.baseUrl')}</span>
              <input
                type="text"
                value={getProviderField('base_url') || ''}
                onChange={(e) => setProviderField('base_url', e.target.value)}
                placeholder={
                  activeProvider === 'anthropic'
                    ? 'https://api.anthropic.com'
                    : 'https://api.openai.com/v1'
                }
                className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm font-mono bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
              />
            </label>
          </div>
        )}

        {/* Custom provider settings */}
        {activeProvider === 'custom' && (
          <div className="space-y-1.5">
            <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
              {t('models.customBotSettings')}
            </span>
            <div className="space-y-3">
              <label className="block">
                <span className="text-xs text-text-muted">{t('models.apiCompat')}</span>
                <select
                  value={getProviderField('api_compat') || 'openai'}
                  onChange={(e) => setProviderField('api_compat', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                >
                  <option value="openai">{t('models.openaiCompat')}</option>
                  <option value="anthropic">{t('models.anthropicCompat')}</option>
                  <option value="gemini">{t('models.geminiCompat')}</option>
                </select>
              </label>
              <label className="block">
                <span className="text-xs text-text-muted">
                  {t('models.baseUrl')} <span className="text-status-error-text">({t('models.baseUrlRequired')})</span>
                </span>
                <input
                  type="text"
                  value={getProviderField('base_url') || ''}
                  onChange={(e) => setProviderField('base_url', e.target.value)}
                  placeholder="https://your-endpoint.com/v1"
                  className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm font-mono bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                />
              </label>
            </div>
          </div>
        )}
      </div>

      {/* ========== Part 2: Saved Presets List ========== */}
      <div className="space-y-3">
        <h3 className="text-xs font-medium text-text-muted uppercase tracking-wider">
          {t('models.savedCombos')}
        </h3>

        {presetKeys.length > 0 ? (
          <div className="bg-panel rounded-xl border border-border overflow-hidden divide-y divide-border">
            {presetKeys.map((key) => {
              const preset = presets[key];
              const isActive = key === activePresetKey;
              const isHovered = hoveredCombo === key;
              const isRenaming = renamingKey === key;
              const displayName = preset.name || key;

              return (
                <div
                  key={key}
                  className="flex items-center justify-between px-4 py-3 transition-colors hover:bg-surface/50"
                  onMouseEnter={() => setHoveredCombo(key)}
                  onMouseLeave={() => setHoveredCombo(null)}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="text-base shrink-0">{getComboEmoji(displayName)}</span>
                    {isRenaming ? (
                      <input
                        type="text"
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') renamePreset(key, renameValue);
                          if (e.key === 'Escape') {
                            setRenamingKey(null);
                            setRenameValue('');
                          }
                        }}
                        onBlur={() => {
                          if (renameValue.trim()) renamePreset(key, renameValue);
                          else {
                            setRenamingKey(null);
                            setRenameValue('');
                          }
                        }}
                        className="rounded border border-border px-2 py-0.5 text-sm bg-input-bg text-text focus:outline-none focus:ring-1 focus:ring-primary w-40"
                        autoFocus
                      />
                    ) : (
                      <>
                        <span className="text-sm text-text font-medium truncate">
                          {displayName}
                        </span>
                        <span className="text-xs text-text-muted hidden sm:inline">
                          {providerName(presetToTiers(preset).provider)}
                        </span>
                      </>
                    )}
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {isHovered && !isRenaming && (
                      <>
                        <button
                          onClick={() => {
                            setRenamingKey(key);
                            setRenameValue(displayName);
                          }}
                          className="p-1 rounded text-text-muted hover:text-text transition-colors"
                          title={t('models.rename')}
                        >
                          <Pencil size={14} />
                        </button>
                        <button
                          onClick={() => deletePreset(key)}
                          className="p-1 rounded text-text-muted hover:text-status-error-text transition-colors"
                          title={t('common:actions.delete')}
                        >
                          <Trash2 size={14} />
                        </button>
                      </>
                    )}

                    {isActive ? (
                      <span className="px-2 py-0.5 rounded-full text-xs bg-primary/10 text-primary font-medium">
                        {t('common:status.active')}
                      </span>
                    ) : (
                      <button
                        onClick={() => loadPreset(key)}
                        className="px-3 py-1 rounded-lg text-sm border border-border hover:bg-hover-bg transition-colors text-text-muted hover:text-text"
                      >
                        {t('common:actions.load')}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="bg-panel rounded-xl border border-border p-6 text-center text-sm text-text-muted">
            {t('models.noSavedCombos')}
          </div>
        )}

        {/* New Preset button */}
        <button
          onClick={handleNewPreset}
          className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-text-muted hover:text-text hover:bg-panel border border-border transition-colors"
        >
          <Plus size={16} />
          {t('models.newCombo')}
        </button>
      </div>
    </div>
  );
}

// ---- Helper: extract provider-specific extras from config ----

function extractExtras(
  provider: string,
  providerConfig: Record<string, unknown>,
): {
  base_url?: string | null;
  api_compat?: string | null;
  thinking?: Record<string, string>;
  rate_limit_interval?: number;
} {
  const extras: ReturnType<typeof extractExtras> = {};
  const baseUrl = providerConfig.base_url as string | undefined;
  if (baseUrl) extras.base_url = baseUrl;

  if (provider === 'custom') {
    extras.api_compat = (providerConfig.api_compat as string) || 'openai';
  }
  if (provider === 'gemini') {
    extras.thinking = {
      model: (providerConfig.thinking_model as string) || 'high',
      sub_agent: (providerConfig.thinking_sub_agent as string) || 'low',
      insight: (providerConfig.thinking_insight as string) || 'low',
    };
  }
  if (provider === 'minimax') {
    extras.rate_limit_interval = parseFloat(
      (providerConfig.rate_limit_interval as string) || '2.0',
    );
  }
  return extras;
}
