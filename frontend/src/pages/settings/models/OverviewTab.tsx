import { useCallback, useEffect, useRef, useState } from 'react';
import { useSettingsStore } from '../../../stores/settingsStore';
import {
  getProviders,
  getAgentTypes,
  listModels,
  listApiKeys,
  getMcpStatus,
  updateMcpConfig,
} from '../../../api/client';
import type { ProviderInfo, ModelInfo, McpStatus } from '../../../api/client';
import type {
  PresetConfig,
  WorkbenchConfig,
  ApiKeyEntry,
  AgentStationConfig,
} from '../../../api/types';
import { Copy, Pencil, Trash2, Plus, X, ChevronDown } from 'lucide-react';
import { deriveTierSummary, expandBuiltinPreset } from '../../../utils/presetUtils';
import {
  PROVIDERS,
  MINIMAX_ENDPOINTS,
  BUILTIN_PRESETS,
  TIER_AGENT_MAP,
  MODEL_TIER_KEYS,
  getComboEmoji,
  slugify,
  COMBO_NAME_SUGGESTIONS,
} from '../../../constants/builtinPresets';
import { useTranslation } from 'react-i18next';

// ---- Tier display labels ----

const TIER_LABELS: Record<string, string> = {
  model: 'Main',
  sub_agent_model: 'Sub-agent',
  insight_model: 'Insight',
  inline_model: 'Inline',
  planner_model: 'Planner',
};

// ---- Thinking level options (Gemini) ----

const THINKING_LEVELS = ['off', 'low', 'high'] as const;
const THINKING_AGENTS = [
  { key: 'model', label: 'Main' },
  { key: 'sub_agent', label: 'Sub-agent' },
  { key: 'insight', label: 'Insight' },
] as const;

// ---- Providers that need base_url ----

const BASE_URL_PROVIDERS = ['openai', 'anthropic', 'grok', 'deepseek', 'qwen', 'kimi', 'glm'];

// ---- Props ----

interface OverviewTabProps {
  onSwitchToCustomize: () => void;
}

// ---- Main Component ----

export function OverviewTab({ onSwitchToCustomize }: OverviewTabProps) {
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKeyEntry[]>([]);
  const [expandedProvider, setExpandedProvider] = useState<string | null>(null);

  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
    listApiKeys().then((resp) => setApiKeys(resp.keys)).catch(() => {});
  }, []);

  const { t } = useTranslation();
  const { config, updateConfig } = useSettingsStore();

  const workbench = (config.workbench ?? {}) as WorkbenchConfig;
  const presets = (config.presets ?? {}) as Record<string, PresetConfig>;
  const activePresetKey = workbench.preset ?? null;
  const activePreset = activePresetKey ? presets[activePresetKey] ?? null : null;

  // Auto-activate a builtin preset when none is set (fresh install).
  const [autoActivated, setAutoActivated] = useState(false);
  useEffect(() => {
    if (activePresetKey || autoActivated) return;
    const activeProvider = (config.llm_provider as string) || 'gemini';
    const builtin = BUILTIN_PRESETS.find((p) => p.provider === activeProvider);
    if (!builtin) return;

    let cancelled = false;
    getAgentTypes().then((data) => {
      if (cancelled) return;
      // Read fresh state from store to avoid stale closure
      const { config: currentConfig } = useSettingsStore.getState();
      const currentPresets = (currentConfig.presets ?? {}) as Record<string, PresetConfig>;
      const currentWorkbench = (currentConfig.workbench ?? {}) as WorkbenchConfig;
      const presetSlug = `builtin-${builtin.id}`;
      const expanded = expandBuiltinPreset(builtin, data.agents);
      updateConfig({
        presets: { ...currentPresets, [presetSlug]: expanded },
        workbench: { ...currentWorkbench, preset: presetSlug, agents: {} },
      });
      setAutoActivated(true);
      setTimeout(() => useSettingsStore.getState().saveConfig(), 0);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [activePresetKey, autoActivated, config.llm_provider, updateConfig]);

  // ---- New combo form state ----
  const [showNewForm, setShowNewForm] = useState(false);
  const [newName, setNewName] = useState('');
  const newNameRef = useRef<HTMLInputElement>(null);

  // ---- Rename state ----
  const [renamingKey, setRenamingKey] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');

  // ---- Helpers ----

  const providerLabel = useCallback(
    (providerId: string) => {
      const found = providers.find((p) => p.id === providerId);
      return found?.name ?? providerId;
    },
    [providers],
  );

  const hasPerAgentOverrides = useCallback(() => {
    if (!workbench.agents) return false;
    return Object.values(workbench.agents).some((v) => v != null);
  }, [workbench.agents]);

  const isApiKeyConfigured = useCallback(
    (providerId: string) => {
      // Check if any API key for this provider is configured
      return apiKeys.some(
        (k) => k.provider === providerId && k.configured,
      );
    },
    [apiKeys],
  );

  const getActiveProviderId = useCallback(() => {
    if (!activePreset) return null;
    const summary = deriveTierSummary(activePreset);
    return summary?.provider ?? null;
  }, [activePreset]);

  // ---- Mutations ----

  const setActivePreset = useCallback(
    (key: string) => {
      const { config: currentConfig } = useSettingsStore.getState();
      const currentWorkbench = (currentConfig.workbench ?? {}) as WorkbenchConfig;
      updateConfig({
        workbench: { ...currentWorkbench, preset: key },
      });
      setTimeout(() => useSettingsStore.getState().saveConfig(), 0);
    },
    [updateConfig],
  );

  const duplicatePreset = useCallback(
    (sourceKey: string) => {
      const { config: currentConfig } = useSettingsStore.getState();
      const currentPresets = (currentConfig.presets ?? {}) as Record<string, PresetConfig>;
      const source = currentPresets[sourceKey];
      if (!source) return;

      let copyName = `${source.name} Copy`;
      let copyKey = slugify(copyName);
      let n = 2;
      while (currentPresets[copyKey]) {
        copyName = `${source.name} Copy ${n}`;
        copyKey = slugify(copyName);
        n++;
      }

      const newPreset: PresetConfig = {
        ...structuredClone(source),
        name: copyName,
      };

      updateConfig({
        presets: { ...currentPresets, [copyKey]: newPreset },
      });
    },
    [updateConfig],
  );

  const deletePreset = useCallback(
    (key: string) => {
      const { config: currentConfig } = useSettingsStore.getState();
      const currentPresets = (currentConfig.presets ?? {}) as Record<string, PresetConfig>;
      const currentWorkbench = (currentConfig.workbench ?? {}) as WorkbenchConfig;
      const next = { ...currentPresets };
      delete next[key];
      const updatedWorkbench = { ...currentWorkbench };
      if (updatedWorkbench.preset === key) {
        updatedWorkbench.preset = null;
      }
      updateConfig({
        presets: next,
        workbench: updatedWorkbench,
      });
    },
    [updateConfig],
  );

  const confirmRename = useCallback(
    (key: string) => {
      const trimmed = renameValue.trim();
      const { config: currentConfig } = useSettingsStore.getState();
      const currentPresets = (currentConfig.presets ?? {}) as Record<string, PresetConfig>;
      const currentWorkbench = (currentConfig.workbench ?? {}) as WorkbenchConfig;
      if (!trimmed || !currentPresets[key]) {
        setRenamingKey(null);
        return;
      }

      const newKey = slugify(trimmed);
      if (newKey !== key && currentPresets[newKey]) {
        setRenamingKey(null);
        return;
      }

      const updated = { ...currentPresets };
      const preset = { ...updated[key], name: trimmed };

      if (newKey !== key) {
        delete updated[key];
        updated[newKey] = preset;

        if (currentWorkbench.preset === key) {
          updateConfig({
            presets: updated,
            workbench: { ...currentWorkbench, preset: newKey },
          });
        } else {
          updateConfig({ presets: updated });
        }
      } else {
        updated[key] = preset;
        updateConfig({ presets: updated });
      }

      setRenamingKey(null);
    },
    [renameValue, updateConfig],
  );

  const createNewCombo = useCallback(
    (name: string) => {
      const trimmed = name.trim();
      if (!trimmed) return;

      const { config: currentConfig } = useSettingsStore.getState();
      const currentPresets = (currentConfig.presets ?? {}) as Record<string, PresetConfig>;
      const currentWorkbench = (currentConfig.workbench ?? {}) as WorkbenchConfig;

      const key = slugify(trimmed);
      if (currentPresets[key]) return;

      const currentActiveKey = currentWorkbench.preset;
      const currentActivePreset = currentActiveKey ? currentPresets[currentActiveKey] : null;

      const baseAgents = currentActivePreset
        ? structuredClone(currentActivePreset.agents)
        : {};
      const baseCapabilities = currentActivePreset
        ? { ...currentActivePreset.capabilities }
        : { web_search: null, vision: null };

      const newPreset: PresetConfig = {
        name: trimmed,
        agents: baseAgents,
        capabilities: baseCapabilities,
      };

      updateConfig({
        presets: { ...currentPresets, [key]: newPreset },
        workbench: { ...currentWorkbench, preset: key },
      });

      setShowNewForm(false);
      setNewName('');
      onSwitchToCustomize();
    },
    [updateConfig, onSwitchToCustomize],
  );

  // ---- Derived ----

  const tierSummary = activePreset ? deriveTierSummary(activePreset) : null;
  const presetEntries = Object.entries(presets);
  const activeProviderId = getActiveProviderId();

  return (
    <div className="space-y-6">
      {/* ---- Section 1: Active Configuration Card ---- */}
      <ActiveConfigCard
        activePreset={activePreset}
        tierSummary={tierSummary}
        providerLabel={providerLabel}
        hasPerAgentOverrides={hasPerAgentOverrides}
        onEdit={onSwitchToCustomize}
        onDuplicate={() => activePresetKey && duplicatePreset(activePresetKey)}
        onCreateFirst={() => setShowNewForm(true)}
        t={t}
      />

      {/* ---- Section 2: Provider Presets Grid ---- */}
      <div>
        <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">
          {t('settings.models.providerPresets', 'Provider Presets')}
        </h3>

        {/* 5x2 grid of provider cards */}
        <div className="grid grid-cols-5 gap-2 mb-1">
          {PROVIDERS.map((prov) => {
            const hasKey = isApiKeyConfigured(prov.id);
            const isActive = activeProviderId === prov.id;
            const isExpanded = expandedProvider === prov.id;

            return (
              <button
                key={prov.id}
                onClick={() =>
                  setExpandedProvider(isExpanded ? null : prov.id)
                }
                aria-expanded={isExpanded}
                aria-label={`${prov.name} settings`}
                className={`relative flex flex-col items-center gap-1 px-2 py-2.5 rounded-lg border transition-all text-center ${
                  isExpanded
                    ? 'border-primary/40 bg-primary/5 ring-1 ring-primary/20'
                    : isActive
                      ? 'border-primary/30 bg-primary/5'
                      : hasKey
                        ? 'border-border bg-panel hover:bg-hover-bg'
                        : 'border-border bg-panel opacity-60 hover:opacity-80 hover:bg-hover-bg'
                }`}
              >
                {/* API key status dot */}
                <span
                  className={`absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full ${
                    hasKey ? 'bg-green-500' : 'bg-gray-400'
                  }`}
                />
                {/* Active badge */}
                {isActive && (
                  <span className="absolute top-1 left-1 px-1 py-0.5 text-[8px] font-bold uppercase tracking-wider rounded bg-primary/20 text-primary leading-none">
                    {t('settings.models.active', 'Active')}
                  </span>
                )}
                <span className="text-lg mt-1">{prov.icon}</span>
                <span className="text-[11px] font-medium text-text truncate w-full">
                  {prov.name}
                </span>
                {isExpanded && (
                  <ChevronDown className="w-3 h-3 text-primary absolute -bottom-0.5" />
                )}
              </button>
            );
          })}
        </div>

        {/* Expanded provider settings panel (accordion) */}
        {expandedProvider && (
          <ProviderSettingsPanel
            key={expandedProvider}
            providerId={expandedProvider}
            hasApiKey={isApiKeyConfigured(expandedProvider)}
            updateConfig={updateConfig}
            t={t}
          />
        )}
      </div>

      {/* ---- Section 3: My Combos ---- */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
            {t('settings.models.myCombos', 'My Combos')}
          </h3>
          {!showNewForm && (
            <button
              onClick={() => {
                setShowNewForm(true);
                setNewName('');
                setTimeout(() => newNameRef.current?.focus(), 0);
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md border border-border text-text-muted hover:bg-hover-bg transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              {t('settings.models.newCombo', 'New Combo')}
            </button>
          )}
        </div>

        {/* New combo inline form */}
        {showNewForm && (
          <NewComboForm
            newName={newName}
            setNewName={setNewName}
            newNameRef={newNameRef}
            onCreate={createNewCombo}
            onCancel={() => { setShowNewForm(false); setNewName(''); }}
            t={t}
          />
        )}

        {/* Combo cards grid */}
        {presetEntries.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {presetEntries.map(([key, preset]) => (
              <ComboCard
                key={key}
                preset={preset}
                isActive={key === activePresetKey}
                isRenaming={renamingKey === key}
                renameValue={renameValue}
                providerLabel={providerLabel}
                onLoad={() => setActivePreset(key)}
                onDuplicate={() => duplicatePreset(key)}
                onDelete={() => deletePreset(key)}
                onStartRename={() => {
                  setRenamingKey(key);
                  setRenameValue(preset.name);
                }}
                onRenameChange={setRenameValue}
                onConfirmRename={() => confirmRename(key)}
                onCancelRename={() => setRenamingKey(null)}
                t={t}
              />
            ))}
          </div>
        ) : (
          !showNewForm && (
            <p className="text-sm text-text-muted text-center py-6">
              {t(
                'settings.models.noCombosYet',
                'No saved combos yet. Create one to get started.',
              )}
            </p>
          )
        )}
      </div>
    </div>
  );
}

// ---- Sub-components ----

// ---- Active Configuration Card ----

function ActiveConfigCard({
  activePreset,
  tierSummary,
  providerLabel,
  hasPerAgentOverrides,
  onEdit,
  onDuplicate,
  onCreateFirst,
  t,
}: {
  activePreset: PresetConfig | null;
  tierSummary: { provider: string; tiers: Record<string, string> } | null;
  providerLabel: (id: string) => string;
  hasPerAgentOverrides: () => boolean;
  onEdit: () => void;
  onDuplicate: () => void;
  onCreateFirst: () => void;
  t: (key: string, fallback: string) => string;
}) {
  return (
    <div>
      <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">
        {t('settings.models.activeConfiguration', 'Active Configuration')}
      </h3>

      {activePreset && tierSummary ? (
        <div className="border-2 border-primary/30 bg-panel rounded-xl p-5">
          {/* Header row */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <span className="text-2xl">{getComboEmoji(activePreset.name)}</span>
              <div>
                <h4 className="text-base font-semibold text-text">
                  {activePreset.name}
                </h4>
                <span className="text-xs text-text-muted">
                  {providerLabel(tierSummary.provider)}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={onEdit}
                className="px-3 py-1.5 text-xs font-medium rounded-md bg-primary text-white hover:bg-primary/90 transition-colors"
              >
                {t('settings.models.edit', 'Edit')}
              </button>
              <button
                onClick={onDuplicate}
                className="px-3 py-1.5 text-xs font-medium rounded-md border border-border text-text-muted hover:bg-hover-bg transition-colors"
                title={t('settings.models.duplicate', 'Duplicate')}
                aria-label={t('settings.models.duplicate', 'Duplicate')}
              >
                <Copy className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Model tiers grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
            {Object.entries(TIER_LABELS).map(([tierKey, label]) => {
              const model = tierSummary.tiers[tierKey];
              return (
                <div
                  key={tierKey}
                  className="bg-surface rounded-lg px-3 py-2"
                >
                  <div className="text-[10px] font-medium text-text-muted uppercase tracking-wider mb-0.5">
                    {label}
                  </div>
                  <div className="text-xs font-mono text-text truncate" title={model || '—'}>
                    {model || '—'}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Capabilities */}
          <div className="flex items-center gap-4 mb-3">
            <CapabilityIndicator
              label={t('settings.models.webSearch', 'Web Search')}
              enabled={!!activePreset.capabilities?.web_search}
            />
            <CapabilityIndicator
              label={t('settings.models.vision', 'Vision')}
              enabled={!!activePreset.capabilities?.vision}
            />
          </div>

          {/* Per-agent overrides indicator */}
          {hasPerAgentOverrides() && (
            <div className="flex items-center gap-2 text-xs text-primary">
              <span className="w-2 h-2 rounded-full bg-primary" />
              {t(
                'settings.models.perAgentOverrides',
                'Per-agent overrides active',
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="border border-border bg-panel rounded-xl p-5 text-center">
          <p className="text-sm text-text-muted">
            {t(
              'settings.models.noActiveCombo',
              'No active combo selected',
            )}
          </p>
          <button
            onClick={onCreateFirst}
            className="mt-3 px-4 py-2 text-xs font-medium rounded-md bg-primary text-white hover:bg-primary/90 transition-colors"
          >
            {t('settings.models.createFirst', 'Create your first combo')}
          </button>
        </div>
      )}
    </div>
  );
}

// ---- Provider Settings Panel (accordion) ----

function ProviderSettingsPanel({
  providerId,
  hasApiKey,
  updateConfig,
  t,
}: {
  providerId: string;
  hasApiKey: boolean;
  updateConfig: (partial: Partial<Record<string, unknown>>) => void;
  t: (key: string, fallback: string) => string;
}) {
  const builtin = BUILTIN_PRESETS.find((p) => p.provider === providerId);

  // Local state for tier model values
  const [tierModels, setTierModels] = useState<Record<string, string>>(() => {
    const defaults: Record<string, string> = {};
    for (const tierKey of MODEL_TIER_KEYS) {
      defaults[tierKey] = builtin?.tiers[tierKey] ?? '';
    }
    return defaults;
  });

  // Available models from the backend
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // Provider-specific state
  const [thinkingLevels, setThinkingLevels] = useState<Record<string, string>>(() => {
    if (providerId === 'gemini' && builtin?.extras?.thinking) {
      return { ...builtin.extras.thinking };
    }
    return { model: 'off', sub_agent: 'off', insight: 'off' };
  });

  const [baseUrl, setBaseUrl] = useState(() => {
    if (providerId === 'minimax') {
      return builtin?.extras?.base_url ?? MINIMAX_ENDPOINTS[0].value + '/anthropic';
    }
    return builtin?.extras?.base_url ?? '';
  });

  // MiniMax-specific: selected API host (without /anthropic path)
  const [minimaxApiHost, setMinimaxApiHost] = useState(() => {
    if (builtin?.extras?.base_url) {
      return builtin.extras.base_url.replace('/anthropic', '');
    }
    return MINIMAX_ENDPOINTS[0].value;
  });
  const [minimaxCustomHost, setMinimaxCustomHost] = useState('');
  const [minimaxHostMode, setMinimaxHostMode] = useState<'china' | 'intl' | 'custom'>(() => {
    if (builtin?.extras?.base_url) {
      const host = builtin.extras.base_url.replace('/anthropic', '');
      if (host === MINIMAX_ENDPOINTS[0].value) return 'china';
      if (host === MINIMAX_ENDPOINTS[1].value) return 'intl';
      return 'custom';
    }
    return 'china';
  });

  // MiniMax MCP state
  const [mcpStatus, setMcpStatus] = useState<McpStatus | null>(null);
  const [mcpEnabled, setMcpEnabled] = useState(false);
  const [mcpHostMode, setMcpHostMode] = useState<'china' | 'intl'>('china');

  // Custom provider state
  const [apiCompat, setApiCompat] = useState(builtin?.extras?.api_compat ?? 'openai');

  // Save as Combo state
  const [showSaveAs, setShowSaveAs] = useState(false);
  const [saveAsName, setSaveAsName] = useState('');

  // Fetch models when panel opens (if API key is configured)
  useEffect(() => {
    if (!hasApiKey) return;
    setModelsLoading(true);
    setModelsError(null);
    listModels(providerId)
      .then((resp) => {
        if (resp.error) {
          setModelsError(resp.error);
        } else {
          setAvailableModels(resp.models);
        }
        setModelsLoading(false);
      })
      .catch((err) => {
        setModelsError((err as Error).message);
        setModelsLoading(false);
      });
  }, [providerId, hasApiKey]);

  // Fetch MCP status for MiniMax
  useEffect(() => {
    if (providerId !== 'minimax') return;
    getMcpStatus()
      .then((status) => {
        setMcpStatus(status);
        setMcpEnabled(status.enabled);
        // Determine MCP host mode from status
        if (status.api_host === MINIMAX_ENDPOINTS[0].value) {
          setMcpHostMode('china');
        } else {
          setMcpHostMode('intl');
        }
      })
      .catch(() => {});
  }, [providerId]);

  // Update MiniMax base_url when host changes
  useEffect(() => {
    if (providerId !== 'minimax') return;
    setBaseUrl(minimaxApiHost + '/anthropic');
  }, [minimaxApiHost, providerId]);

  const handleMinimaxHostChange = (mode: 'china' | 'intl' | 'custom') => {
    setMinimaxHostMode(mode);
    if (mode === 'china') {
      setMinimaxApiHost(MINIMAX_ENDPOINTS[0].value);
    } else if (mode === 'intl') {
      setMinimaxApiHost(MINIMAX_ENDPOINTS[1].value);
    } else {
      setMinimaxApiHost(minimaxCustomHost || '');
    }
  };

  const handleMcpToggle = (enabled: boolean) => {
    setMcpEnabled(enabled);
    updateMcpConfig({ enabled }).then(() => {
      getMcpStatus().then(setMcpStatus).catch(() => {});
    }).catch(() => {});
  };

  const handleMcpHostChange = (mode: 'china' | 'intl') => {
    setMcpHostMode(mode);
    const host = mode === 'china' ? MINIMAX_ENDPOINTS[0].value : MINIMAX_ENDPOINTS[1].value;
    updateMcpConfig({ api_host: host }).then(() => {
      getMcpStatus().then(setMcpStatus).catch(() => {});
    }).catch(() => {});
  };

  // "Apply as Default" handler
  const [applying, setApplying] = useState(false);
  const handleApplyAsDefault = async () => {
    if (applying) return;
    setApplying(true);
    try {
      const data = await getAgentTypes();
      let presetConfig: PresetConfig;

      if (builtin) {
        // Build a modified builtin with current tier models
        const modifiedBuiltin = {
          ...builtin,
          tiers: { ...builtin.tiers, ...tierModels } as typeof builtin.tiers,
          extras: {
            ...builtin.extras,
            ...(providerId === 'gemini' ? { thinking: thinkingLevels } : {}),
            ...(baseUrl ? { base_url: baseUrl } : {}),
            ...(providerId === 'custom' ? { api_compat: apiCompat } : {}),
          },
        };
        presetConfig = expandBuiltinPreset(modifiedBuiltin, data.agents);
      } else {
        // Manually build PresetConfig using TIER_AGENT_MAP
        const agents: Record<string, AgentStationConfig> = {};
        for (const agent of data.agents) {
          let tierKey: string | null = null;
          for (const [tier, agentIds] of Object.entries(TIER_AGENT_MAP)) {
            if (agentIds.includes(agent.id)) {
              tierKey = tier;
              break;
            }
          }

          const model = tierKey ? tierModels[tierKey] || tierModels.model : tierModels.model;
          const station: AgentStationConfig = {
            provider: providerId,
            model,
          };
          if (baseUrl) station.base_url = baseUrl;
          if (providerId === 'custom') station.api_compat = apiCompat;
          agents[agent.id] = station;
        }

        presetConfig = {
          name: PROVIDERS.find((p) => p.id === providerId)?.name ?? providerId,
          agents,
          capabilities: { web_search: null, vision: null },
        };
      }

      // Read fresh state from store to avoid stale closure
      const { config: currentConfig } = useSettingsStore.getState();
      const currentPresets = (currentConfig.presets ?? {}) as Record<string, PresetConfig>;
      const currentWorkbench = (currentConfig.workbench ?? {}) as WorkbenchConfig;
      const presetSlug = `builtin-${providerId}`;
      updateConfig({
        presets: { ...currentPresets, [presetSlug]: presetConfig },
        workbench: { ...currentWorkbench, preset: presetSlug, agents: {} },
        llm_provider: providerId,
      });
      setTimeout(() => useSettingsStore.getState().saveConfig(), 0);
    } catch {
      // Failed to get agent types — silently ignore
    } finally {
      setApplying(false);
    }
  };

  // "Save as Combo" handler — builds preset and saves under user's chosen name
  const handleSaveAsCombo = async () => {
    const trimmed = saveAsName.trim();
    if (!trimmed) return;
    const comboSlug = slugify(trimmed);

    try {
      const data = await getAgentTypes();
      const agents: Record<string, AgentStationConfig> = {};
      for (const agent of data.agents) {
        let tierKey: string | null = null;
        for (const [tier, agentIds] of Object.entries(TIER_AGENT_MAP)) {
          if (agentIds.includes(agent.id)) {
            tierKey = tier;
            break;
          }
        }

        const model = tierKey ? tierModels[tierKey] || tierModels.model : tierModels.model;
        const station: AgentStationConfig = {
          provider: providerId,
          model,
        };
        if (baseUrl) station.base_url = baseUrl;
        if (providerId === 'gemini' && tierKey) {
          // Map tier key to thinking level key (sub_agent_model → sub_agent)
          const thinkingKey = tierKey.replace('_model', '');
          station.thinking = thinkingLevels[thinkingKey] || thinkingLevels.model;
        }
        if (providerId === 'custom') station.api_compat = apiCompat;
        agents[agent.id] = station;
      }

      const presetConfig: PresetConfig = {
        name: trimmed,
        agents,
        capabilities: builtin?.capabilities
          ? { ...builtin.capabilities }
          : { web_search: null, vision: null },
      };

      const { config: currentConfig } = useSettingsStore.getState();
      const currentPresets = (currentConfig.presets ?? {}) as Record<string, PresetConfig>;
      const currentWorkbench = (currentConfig.workbench ?? {}) as WorkbenchConfig;
      updateConfig({
        presets: { ...currentPresets, [comboSlug]: presetConfig },
        workbench: { ...currentWorkbench, preset: comboSlug, agents: {} },
      });
      setTimeout(() => useSettingsStore.getState().saveConfig(), 0);
      setShowSaveAs(false);
      setSaveAsName('');
    } catch {
      // Failed to get agent types
    }
  };

  // Render model selector (dropdown if models available, text input fallback)
  const renderModelSelect = (tierKey: string) => {
    const value = tierModels[tierKey] || '';
    const label = TIER_LABELS[tierKey] || tierKey;

    if (availableModels.length > 0) {
      return (
        <div key={tierKey}>
          <label className="block text-[10px] font-medium text-text-muted uppercase tracking-wider mb-1">
            {label}
          </label>
          <select
            value={value}
            onChange={(e) =>
              setTierModels((prev) => ({ ...prev, [tierKey]: e.target.value }))
            }
            className="w-full px-2 py-1.5 text-xs bg-surface border border-border rounded-md text-text focus:outline-none focus:ring-1 focus:ring-primary/50"
          >
            <option value="">— Select model —</option>
            {availableModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.display_name || m.id}
              </option>
            ))}
          </select>
        </div>
      );
    }

    return (
      <div key={tierKey}>
        <label className="block text-[10px] font-medium text-text-muted uppercase tracking-wider mb-1">
          {label}
        </label>
        <input
          type="text"
          value={value}
          onChange={(e) =>
            setTierModels((prev) => ({ ...prev, [tierKey]: e.target.value }))
          }
          placeholder="model-name"
          className="w-full px-2 py-1.5 text-xs bg-surface border border-border rounded-md text-text placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono"
        />
      </div>
    );
  };

  return (
    <div className="border border-primary/30 bg-panel rounded-lg p-4 mt-1 space-y-4">
      {/* No API key warning */}
      {!hasApiKey && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-md px-3 py-2 text-xs text-yellow-600 dark:text-yellow-400">
          {t(
            'settings.models.apiKeyNotConfigured',
            'API key not configured. Add one in the API Keys section to use this provider.',
          )}
        </div>
      )}

      {/* Model tier selects */}
      <div>
        <h4 className="text-xs font-semibold text-text mb-2">
          {t('settings.models.modelTiers', 'Model Tiers')}
        </h4>
        {modelsLoading && (
          <p className="text-xs text-text-muted mb-2">
            {t('settings.models.loadingModels', 'Loading available models...')}
          </p>
        )}
        {modelsError && (
          <p className="text-xs text-red-500 mb-2">
            {t('settings.models.modelsError', 'Could not load models')}: {modelsError}
          </p>
        )}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {MODEL_TIER_KEYS.map((tierKey) => renderModelSelect(tierKey))}
        </div>
      </div>

      {/* Provider-specific settings */}
      {providerId === 'gemini' && (
        <div>
          <h4 className="text-xs font-semibold text-text mb-2">
            {t('settings.models.thinkingLevels', 'Thinking Levels')}
          </h4>
          <div className="grid grid-cols-3 gap-3">
            {THINKING_AGENTS.map(({ key, label }) => (
              <div key={key}>
                <label className="block text-[10px] font-medium text-text-muted uppercase tracking-wider mb-1">
                  {label}
                </label>
                <select
                  value={thinkingLevels[key] || 'off'}
                  onChange={(e) =>
                    setThinkingLevels((prev) => ({ ...prev, [key]: e.target.value }))
                  }
                  className="w-full px-2 py-1.5 text-xs bg-surface border border-border rounded-md text-text focus:outline-none focus:ring-1 focus:ring-primary/50"
                >
                  {THINKING_LEVELS.map((level) => (
                    <option key={level} value={level}>
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>
      )}

      {providerId === 'minimax' && (
        <>
          <div>
            <h4 className="text-xs font-semibold text-text mb-2">
              {t('settings.models.apiEndpoint', 'API Endpoint')}
            </h4>
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-xs text-text cursor-pointer">
                <input
                  type="radio"
                  name="minimax-host"
                  checked={minimaxHostMode === 'china'}
                  onChange={() => handleMinimaxHostChange('china')}
                  className="accent-primary"
                />
                {MINIMAX_ENDPOINTS[0].label}
              </label>
              <label className="flex items-center gap-2 text-xs text-text cursor-pointer">
                <input
                  type="radio"
                  name="minimax-host"
                  checked={minimaxHostMode === 'intl'}
                  onChange={() => handleMinimaxHostChange('intl')}
                  className="accent-primary"
                />
                {MINIMAX_ENDPOINTS[1].label}
              </label>
              <label className="flex items-center gap-2 text-xs text-text cursor-pointer">
                <input
                  type="radio"
                  name="minimax-host"
                  checked={minimaxHostMode === 'custom'}
                  onChange={() => handleMinimaxHostChange('custom')}
                  className="accent-primary"
                />
                Custom
              </label>
              {minimaxHostMode === 'custom' && (
                <input
                  type="text"
                  value={minimaxCustomHost}
                  onChange={(e) => {
                    setMinimaxCustomHost(e.target.value);
                    setMinimaxApiHost(e.target.value);
                  }}
                  placeholder="https://your-endpoint.com"
                  className="w-full px-2 py-1.5 text-xs bg-surface border border-border rounded-md text-text placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono ml-5"
                />
              )}
            </div>
          </div>

          {/* Divider */}
          <hr className="border-border" />

          {/* MCP Tools section */}
          <div>
            <h4 className="text-xs font-semibold text-text mb-2">
              {t('settings.models.mcpTools', 'MCP Tools')}
            </h4>
            <label className="flex items-center gap-2 text-xs text-text cursor-pointer mb-3">
              <input
                type="checkbox"
                checked={mcpEnabled}
                onChange={(e) => handleMcpToggle(e.target.checked)}
                className="accent-primary"
              />
              {t('settings.models.enableMcp', 'Enable MiniMax MCP tools')}
            </label>

            {mcpEnabled && (
              <div className="space-y-2 ml-5">
                <div className="text-[10px] font-medium text-text-muted uppercase tracking-wider mb-1">
                  MCP Endpoint
                </div>
                <label className="flex items-center gap-2 text-xs text-text cursor-pointer">
                  <input
                    type="radio"
                    name="mcp-host"
                    checked={mcpHostMode === 'china'}
                    onChange={() => handleMcpHostChange('china')}
                    className="accent-primary"
                  />
                  {MINIMAX_ENDPOINTS[0].label}
                </label>
                <label className="flex items-center gap-2 text-xs text-text cursor-pointer">
                  <input
                    type="radio"
                    name="mcp-host"
                    checked={mcpHostMode === 'intl'}
                    onChange={() => handleMcpHostChange('intl')}
                    className="accent-primary"
                  />
                  {MINIMAX_ENDPOINTS[1].label}
                </label>

                {/* Connection status */}
                {mcpStatus && (
                  <div className="flex items-center gap-2 text-xs text-text-muted mt-2">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        mcpStatus.connected ? 'bg-green-500' : 'bg-red-500'
                      }`}
                    />
                    {mcpStatus.connected
                      ? t('settings.models.mcpConnected', 'Connected')
                      : <span title={mcpStatus.error || undefined}>
                          {t('settings.models.mcpDisconnected', 'Disconnected')}
                        </span>}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}

      {BASE_URL_PROVIDERS.includes(providerId) && (
        <div>
          <label className="block text-[10px] font-medium text-text-muted uppercase tracking-wider mb-1">
            {t('settings.models.baseUrl', 'Base URL')}
          </label>
          <input
            type="text"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.example.com/v1"
            className="w-full px-2 py-1.5 text-xs bg-surface border border-border rounded-md text-text placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono"
          />
        </div>
      )}

      {providerId === 'custom' && (
        <div>
          <label className="block text-[10px] font-medium text-text-muted uppercase tracking-wider mb-1">
            {t('settings.models.apiCompatibility', 'API Compatibility')}
          </label>
          <select
            value={apiCompat}
            onChange={(e) => setApiCompat(e.target.value)}
            className="w-full px-2 py-1.5 text-xs bg-surface border border-border rounded-md text-text focus:outline-none focus:ring-1 focus:ring-primary/50"
          >
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="gemini">Gemini</option>
          </select>

          <label className="block text-[10px] font-medium text-text-muted uppercase tracking-wider mb-1 mt-3">
            {t('settings.models.baseUrl', 'Base URL')} *
          </label>
          <input
            type="text"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.example.com/v1 (required)"
            className="w-full px-2 py-1.5 text-xs bg-surface border border-border rounded-md text-text placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono"
          />
        </div>
      )}

      {/* Save as Combo inline form */}
      {showSaveAs && (
        <div className="border border-primary/30 bg-primary/5 rounded-md p-3 space-y-2">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={saveAsName}
              onChange={(e) => setSaveAsName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && saveAsName.trim()) handleSaveAsCombo();
                else if (e.key === 'Escape') { setShowSaveAs(false); setSaveAsName(''); }
              }}
              placeholder={t('settings.models.comboNamePlaceholder', 'Combo name...')}
              className="flex-1 px-2 py-1.5 text-xs bg-surface border border-border rounded-md text-text placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-primary/50"
              autoFocus
            />
            <button
              onClick={handleSaveAsCombo}
              disabled={!saveAsName.trim()}
              className="px-3 py-1.5 text-xs font-medium rounded-md bg-primary text-white hover:bg-primary/90 disabled:opacity-40 transition-colors"
            >
              {t('settings.models.save', 'Save')}
            </button>
            <button
              onClick={() => { setShowSaveAs(false); setSaveAsName(''); }}
              className="p-1 text-text-muted hover:text-text transition-colors"
              aria-label="Cancel"
            >
              <X size={14} />
            </button>
          </div>
          <div className="flex flex-wrap gap-1">
            {COMBO_NAME_SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setSaveAsName(suggestion)}
                className="px-2 py-0.5 text-[10px] rounded-full bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
              >
                {getComboEmoji(suggestion)} {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-2 pt-2">
        <button
          onClick={handleApplyAsDefault}
          disabled={!hasApiKey || applying}
          className="px-3 py-1.5 text-xs font-medium rounded-md bg-primary text-white hover:bg-primary/90 disabled:opacity-40 transition-colors"
        >
          {t('settings.models.applyAsDefault', 'Apply as Default')}
        </button>
        <button
          onClick={() => { setShowSaveAs(true); setSaveAsName(''); }}
          disabled={!hasApiKey}
          className="px-3 py-1.5 text-xs font-medium rounded-md border border-border text-text hover:bg-hover-bg disabled:opacity-40 transition-colors"
        >
          {t('settings.models.saveAsCombo', 'Save as Combo')}
        </button>
      </div>
    </div>
  );
}

// ---- New Combo Form ----

function NewComboForm({
  newName,
  setNewName,
  newNameRef,
  onCreate,
  onCancel,
  t,
}: {
  newName: string;
  setNewName: (v: string) => void;
  newNameRef: React.RefObject<HTMLInputElement | null>;
  onCreate: (name: string) => void;
  onCancel: () => void;
  t: (key: string, fallback: string) => string;
}) {
  return (
    <div className="border border-primary/30 bg-primary/5 rounded-lg p-4 mb-4">
      <div className="flex items-center gap-2 mb-3">
        <input
          ref={newNameRef}
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && newName.trim()) {
              onCreate(newName);
            } else if (e.key === 'Escape') {
              onCancel();
            }
          }}
          placeholder={t(
            'settings.models.comboNamePlaceholder',
            'Combo name...',
          )}
          className="flex-1 px-3 py-1.5 text-sm bg-surface border border-border rounded-md text-text placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-primary/50"
          autoFocus
        />
        <button
          onClick={() => {
            if (newName.trim()) onCreate(newName);
          }}
          disabled={!newName.trim()}
          className="px-3 py-1.5 text-xs font-medium rounded-md bg-primary text-white hover:bg-primary/90 disabled:opacity-40 transition-colors"
        >
          {t('settings.models.create', 'Create')}
        </button>
        <button
          onClick={onCancel}
          className="p-1.5 text-text-muted hover:text-text transition-colors"
          aria-label="Cancel"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Suggestion chips */}
      <div className="flex flex-wrap gap-1.5">
        {COMBO_NAME_SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            onClick={() => {
              setNewName(suggestion);
              newNameRef.current?.focus();
            }}
            className="px-2.5 py-1 text-xs rounded-full bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
          >
            {getComboEmoji(suggestion)} {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}

// ---- Combo Card ----

function ComboCard({
  preset,
  isActive,
  isRenaming,
  renameValue,
  providerLabel,
  onLoad,
  onDuplicate,
  onDelete,
  onStartRename,
  onRenameChange,
  onConfirmRename,
  onCancelRename,
  t,
}: {
  preset: PresetConfig;
  isActive: boolean;
  isRenaming: boolean;
  renameValue: string;
  providerLabel: (id: string) => string;
  onLoad: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onStartRename: () => void;
  onRenameChange: (v: string) => void;
  onConfirmRename: () => void;
  onCancelRename: () => void;
  t: (key: string, fallback: string) => string;
}) {
  const summary = deriveTierSummary(preset);

  return (
    <div
      className={`p-4 rounded-lg border transition-colors ${
        isActive
          ? 'border-2 border-primary/30 bg-primary/5'
          : 'border-border bg-panel hover:bg-hover-bg'
      }`}
    >
      {/* Card header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <span className="text-lg shrink-0">
            {getComboEmoji(preset.name)}
          </span>
          {isRenaming ? (
            <input
              type="text"
              value={renameValue}
              onChange={(e) => onRenameChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') onConfirmRename();
                else if (e.key === 'Escape') onCancelRename();
              }}
              onBlur={onConfirmRename}
              className="flex-1 min-w-0 px-2 py-0.5 text-sm font-medium bg-surface border border-border rounded text-text focus:outline-none focus:ring-1 focus:ring-primary/50"
              autoFocus
            />
          ) : (
            <div className="min-w-0">
              <h4 className="text-sm font-medium text-text truncate">
                {preset.name}
              </h4>
              <span className="text-[10px] text-text-muted">
                {summary
                  ? providerLabel(summary.provider)
                  : '—'}
              </span>
            </div>
          )}
        </div>
        {isActive && (
          <span className="shrink-0 ml-2 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider rounded-full bg-badge-gray-bg text-badge-gray-text">
            {t('settings.models.active', 'Active')}
          </span>
        )}
      </div>

      {/* Card actions */}
      <div className="flex items-center justify-between mt-3">
        <div className="flex items-center gap-1">
          <button
            onClick={onStartRename}
            className="p-1.5 text-text-muted hover:text-text rounded transition-colors"
            title={t('settings.models.rename', 'Rename')}
            aria-label={t('settings.models.rename', 'Rename')}
          >
            <Pencil className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onDuplicate}
            className="p-1.5 text-text-muted hover:text-text rounded transition-colors"
            title={t('settings.models.duplicate', 'Duplicate')}
            aria-label={t('settings.models.duplicate', 'Duplicate')}
          >
            <Copy className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onDelete}
            className="p-1.5 text-text-muted hover:text-red-500 rounded transition-colors"
            title={t('settings.models.delete', 'Delete')}
            aria-label={t('settings.models.delete', 'Delete')}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {!isActive && (
          <button
            onClick={onLoad}
            className="px-3 py-1 text-xs font-medium rounded-md border border-border text-text hover:bg-hover-bg transition-colors"
          >
            {t('settings.models.load', 'Load')}
          </button>
        )}
      </div>
    </div>
  );
}

// ---- Capability Indicator ----

function CapabilityIndicator({
  label,
  enabled,
}: {
  label: string;
  enabled: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5 text-xs text-text-muted">
      <span
        className={`w-2 h-2 rounded-full ${
          enabled ? 'bg-primary' : 'border border-border'
        }`}
      />
      {label}
    </div>
  );
}
