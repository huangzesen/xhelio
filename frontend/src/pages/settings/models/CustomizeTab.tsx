import { useCallback, useEffect, useMemo, useState } from 'react';
import { useSettingsStore } from '../../../stores/settingsStore';
import { getAgentTypes, listApiKeys, listModels, getProviders } from '../../../api/client';
import type { ModelInfo, ProviderInfo } from '../../../api/client';
import type {
  AgentTypeInfo,
  AgentGroupInfo,
  AgentStationConfig,
  WorkbenchConfig,
  PresetConfig,
  ApiKeyEntry,
} from '../../../api/types';
import {
  COMBO_NAME_SUGGESTIONS,
  slugify,
} from '../../../constants/builtinPresets';
import {
  Loader2,
  ChevronDown,
  X,
  RotateCcw,
  Save,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

// ---- Helper: resolve effective config for an agent ----

function resolveAgentConfig(
  agentType: string,
  workbench: WorkbenchConfig | undefined,
  presets: Record<string, PresetConfig> | undefined,
  globalConfig: Record<string, unknown>,
): AgentStationConfig | null {
  // 1. Per-agent override
  const agentOverride = workbench?.agents?.[agentType];
  if (agentOverride) return agentOverride;

  // 2. Preset default
  const presetName = workbench?.preset;
  if (presetName && presets?.[presetName]?.agents?.[agentType]) {
    return presets[presetName].agents[agentType];
  }

  // 3. Global fallback
  const provider = (globalConfig.llm_provider as string) || 'gemini';
  const providers = (globalConfig.providers as Record<string, Record<string, unknown>>) || {};
  const providerConfig = providers[provider] || {};
  return {
    provider,
    model: (providerConfig.model as string) || '',
  };
}

function isCustomized(
  agentType: string,
  workbench: WorkbenchConfig | undefined,
): boolean {
  return workbench?.agents?.[agentType] != null;
}

// ---- Main Component ----

export function CustomizeTab() {
  const { t } = useTranslation(['settings']);
  const { config, updateConfig } = useSettingsStore();

  // Fetch state
  const [agents, setAgents] = useState<AgentTypeInfo[]>([]);
  const [groups, setGroups] = useState<AgentGroupInfo[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKeyEntry[]>([]);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [loadingAgents, setLoadingAgents] = useState(true);

  // UI state
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);
  const [showSaveAs, setShowSaveAs] = useState(false);
  const [saveAsName, setSaveAsName] = useState('');

  // Model lists cache: provider -> ModelInfo[]
  const [modelCache, setModelCache] = useState<Record<string, ModelInfo[]>>({});
  const [loadingModels, setLoadingModels] = useState<string | null>(null);

  // Derived config
  const workbench = config.workbench as WorkbenchConfig | undefined;
  const presets = (config.presets as Record<string, PresetConfig>) || {};

  // ---- Data fetching ----

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [agentData, keysData, providerData] = await Promise.all([
          getAgentTypes(),
          listApiKeys(),
          getProviders(),
        ]);
        if (cancelled) return;
        setAgents(agentData.agents);
        setGroups(agentData.groups.sort((a, b) => a.order - b.order));
        setApiKeys(keysData.keys);
        setProviders(providerData);
      } catch {
        // Silently handle — the page will show empty
      } finally {
        if (!cancelled) setLoadingAgents(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  // Providers that have API keys configured
  const configuredProviders = useMemo(() => {
    const configured = apiKeys
      .filter((k) => k.configured && k.provider)
      .map((k) => k.provider!);
    return [...new Set(configured)];
  }, [apiKeys]);

  // Provider display name lookup
  const providerNameMap = useMemo(() => {
    const map: Record<string, string> = {};
    for (const p of providers) {
      map[p.id] = p.name;
    }
    return map;
  }, [providers]);

  // ---- Fetch models for a provider ----

  const fetchModelsForProvider = useCallback(async (provider: string) => {
    if (modelCache[provider]) return;
    setLoadingModels(provider);
    try {
      const result = await listModels(provider);
      setModelCache((prev) => ({ ...prev, [provider]: result.models }));
    } catch {
      setModelCache((prev) => ({ ...prev, [provider]: [] }));
    } finally {
      setLoadingModels(null);
    }
  }, [modelCache]);

  // ---- Config update helpers ----

  const updateWorkbench = useCallback(
    (partial: Partial<WorkbenchConfig>) => {
      const current: WorkbenchConfig = (config.workbench as WorkbenchConfig) || {
        preset: null,
        agents: {},
        capabilities: { web_search: null, vision: null },
      };
      updateConfig({ workbench: { ...current, ...partial } });
    },
    [config.workbench, updateConfig],
  );

  const updateAgentStation = useCallback(
    (agentType: string, station: AgentStationConfig | null) => {
      const current: WorkbenchConfig = (config.workbench as WorkbenchConfig) || {
        preset: null,
        agents: {},
        capabilities: { web_search: null, vision: null },
      };
      updateConfig({
        workbench: {
          ...current,
          agents: { ...current.agents, [agentType]: station },
        },
      });
    },
    [config.workbench, updateConfig],
  );

  const updateCapability = useCallback(
    (key: 'web_search' | 'vision', value: string | null) => {
      const current: WorkbenchConfig = (config.workbench as WorkbenchConfig) || {
        preset: null,
        agents: {},
        capabilities: { web_search: null, vision: null },
      };
      updateConfig({
        workbench: {
          ...current,
          capabilities: { ...current.capabilities, [key]: value },
        },
      });
    },
    [config.workbench, updateConfig],
  );

  // ---- Preset handlers ----

  const handleSaveAsPreset = useCallback(() => {
    if (!saveAsName.trim()) return;
    const slug = slugify(saveAsName.trim());

    // Snapshot all current effective configs
    const agentConfigs: Record<string, AgentStationConfig> = {};
    for (const agent of agents) {
      const effective = resolveAgentConfig(agent.id, workbench, presets, config);
      if (effective) {
        agentConfigs[agent.id] = effective;
      }
    }

    const newPreset: PresetConfig = {
      name: saveAsName.trim(),
      agents: agentConfigs,
      capabilities: {
        web_search: workbench?.capabilities?.web_search ?? null,
        vision: workbench?.capabilities?.vision ?? null,
      },
    };

    const updatedPresets = { ...presets, [slug]: newPreset };
    updateConfig({ presets: updatedPresets });

    // Set this as active preset and clear overrides
    updateWorkbench({ preset: slug, agents: {} });
    setShowSaveAs(false);
    setSaveAsName('');
  }, [saveAsName, agents, workbench, presets, config, updateConfig, updateWorkbench]);

  // ---- Group agents ----

  const groupedAgents = useMemo(() => {
    const map = new Map<string, AgentTypeInfo[]>();
    for (const group of groups) {
      map.set(group.id, []);
    }
    for (const agent of agents) {
      const list = map.get(agent.group);
      if (list) {
        list.push(agent);
      } else {
        map.set(agent.group, [agent]);
      }
    }
    return map;
  }, [agents, groups]);

  // Check if any agents have overrides
  const hasOverrides = workbench && Object.keys(workbench.agents || {}).some(
    (k) => workbench.agents[k] != null,
  );

  // ---- Render ----

  if (loadingAgents) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 size={24} className="animate-spin text-text-muted" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Per-Agent Fine-Tuning */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
            Per-Agent Fine-Tuning
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSaveAs(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-sm text-text hover:bg-hover-bg transition-colors"
            >
              <Save size={14} />
              Save As...
            </button>
            {hasOverrides && (
              <button
                onClick={() => updateWorkbench({ agents: {} })}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-sm text-text-muted hover:text-text hover:bg-hover-bg transition-colors"
              >
                <RotateCcw size={14} />
                Reset All
              </button>
            )}
          </div>
        </div>

        {/* Save As Dialog */}
        {showSaveAs && (
          <div className="p-4 rounded-lg bg-panel border border-border space-y-3">
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={saveAsName}
                onChange={(e) => setSaveAsName(e.target.value)}
                placeholder="Preset name"
                className="flex-1 px-3 py-1.5 rounded-lg bg-surface border border-border text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSaveAsPreset();
                  if (e.key === 'Escape') setShowSaveAs(false);
                }}
                autoFocus
              />
              <button
                onClick={handleSaveAsPreset}
                disabled={!saveAsName.trim()}
                className="px-3 py-1.5 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Save
              </button>
              <button
                onClick={() => { setShowSaveAs(false); setSaveAsName(''); }}
                className="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-hover-bg transition-colors"
              >
                <X size={16} />
              </button>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {COMBO_NAME_SUGGESTIONS.map((name) => (
                <button
                  key={name}
                  onClick={() => setSaveAsName(name)}
                  className="px-2 py-0.5 rounded-full text-xs bg-badge-gray-bg text-badge-gray-text hover:opacity-80 transition-opacity"
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Station Cards by Group */}
        {groups.map((group) => {
          const groupAgents = groupedAgents.get(group.id);
          if (!groupAgents || groupAgents.length === 0) return null;
          return (
            <div key={group.id} className="space-y-2">
              <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
                {group.name}
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {groupAgents.map((agent) => {
                  const effective = resolveAgentConfig(agent.id, workbench, presets, config);
                  const customized = isCustomized(agent.id, workbench);
                  const isExpanded = expandedAgent === agent.id;

                  if (isExpanded) {
                    return (
                      <StationCardExpanded
                        key={agent.id}
                        agent={agent}
                        station={effective}
                        customized={customized}
                        configuredProviders={configuredProviders}
                        providerNameMap={providerNameMap}
                        modelCache={modelCache}
                        loadingModels={loadingModels}
                        onFetchModels={fetchModelsForProvider}
                        onUpdate={(station) => updateAgentStation(agent.id, station)}
                        onReset={() => updateAgentStation(agent.id, null)}
                        onClose={() => setExpandedAgent(null)}
                        t={t}
                      />
                    );
                  }

                  return (
                    <StationCardCollapsed
                      key={agent.id}
                      agent={agent}
                      station={effective}
                      customized={customized}
                      providerNameMap={providerNameMap}
                      onClick={() => setExpandedAgent(agent.id)}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* Capability Cards */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
            Capabilities
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <CapabilityCard
              label="Web Search"
              value={workbench?.capabilities?.web_search ?? null}
              configuredProviders={configuredProviders}
              providerNameMap={providerNameMap}
              onChange={(v) => updateCapability('web_search', v)}
              t={t}
            />
            <CapabilityCard
              label="Vision"
              value={workbench?.capabilities?.vision ?? null}
              configuredProviders={configuredProviders}
              providerNameMap={providerNameMap}
              onChange={(v) => updateCapability('vision', v)}
              t={t}
            />
          </div>
        </div>
      </section>
    </div>
  );
}

// ---- Station Card (Collapsed) ----

function StationCardCollapsed({
  agent,
  station,
  customized,
  providerNameMap,
  onClick,
}: {
  agent: AgentTypeInfo;
  station: AgentStationConfig | null;
  customized: boolean;
  providerNameMap: Record<string, string>;
  onClick: () => void;
}) {
  const providerLabel = station
    ? providerNameMap[station.provider] || station.provider
    : '';
  const modelLabel = station?.model
    ? station.model.length > 28
      ? station.model.slice(0, 28) + '...'
      : station.model
    : '';

  return (
    <button
      onClick={onClick}
      className="relative flex items-center gap-3 p-3 rounded-lg bg-panel border border-border hover:bg-hover-bg transition-colors text-left w-full"
    >
      {/* Customized dot */}
      {customized && (
        <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-primary" />
      )}
      {/* Icon */}
      <span className="text-xl flex-shrink-0" role="img" aria-label={agent.name}>
        {agent.icon}
      </span>
      {/* Info */}
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium text-text truncate">{agent.name}</div>
        <div className="flex items-center gap-1.5 mt-0.5">
          {providerLabel && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-gray-bg text-badge-gray-text">
              {providerLabel}
            </span>
          )}
          {modelLabel && (
            <span className="text-xs text-text-muted truncate">{modelLabel}</span>
          )}
        </div>
      </div>
      <ChevronDown size={14} className="text-text-muted flex-shrink-0" />
    </button>
  );
}

// ---- Station Card (Expanded) ----

function StationCardExpanded({
  agent,
  station,
  customized,
  configuredProviders,
  providerNameMap,
  modelCache,
  loadingModels,
  onFetchModels,
  onUpdate,
  onReset,
  onClose,
  t,
}: {
  agent: AgentTypeInfo;
  station: AgentStationConfig | null;
  customized: boolean;
  configuredProviders: string[];
  providerNameMap: Record<string, string>;
  modelCache: Record<string, ModelInfo[]>;
  loadingModels: string | null;
  onFetchModels: (provider: string) => void;
  onUpdate: (station: AgentStationConfig) => void;
  onReset: () => void;
  onClose: () => void;
  t: (key: string) => string;
}) {
  const currentProvider = station?.provider || '';
  const currentModel = station?.model || '';
  const currentBaseUrl = station?.base_url || '';
  const currentThinking = station?.thinking || 'off';
  const currentApiCompat = station?.api_compat || 'openai';
  const currentRateLimit = station?.rate_limit_interval ?? 0;

  // Fetch models when provider changes
  useEffect(() => {
    if (currentProvider) {
      onFetchModels(currentProvider);
    }
  }, [currentProvider]); // eslint-disable-line react-hooks/exhaustive-deps

  const models = modelCache[currentProvider] || [];
  const hasModelDropdown = models.length > 0;
  const isLoadingModels = loadingModels === currentProvider;

  const handleProviderChange = (provider: string) => {
    // Prefill model and base_url from provider defaults
    const providers = (useSettingsStore.getState().config.providers as Record<string, Record<string, unknown>>) || {};
    const defaults = providers[provider] || {};
    onUpdate({
      provider,
      model: (defaults.model as string) || '',
      base_url: (defaults.base_url as string) || null,
    });
  };

  const handleModelChange = (model: string) => {
    onUpdate({
      ...buildCurrentStation(),
      model,
    });
  };

  const handleBaseUrlChange = (base_url: string) => {
    onUpdate({
      ...buildCurrentStation(),
      base_url: base_url || null,
    });
  };

  const handleThinkingChange = (thinking: string) => {
    onUpdate({
      ...buildCurrentStation(),
      thinking,
    });
  };

  const handleApiCompatChange = (api_compat: string) => {
    onUpdate({
      ...buildCurrentStation(),
      api_compat,
    });
  };

  const handleRateLimitChange = (rate_limit_interval: number) => {
    onUpdate({
      ...buildCurrentStation(),
      rate_limit_interval,
    });
  };

  function buildCurrentStation(): AgentStationConfig {
    return {
      provider: currentProvider,
      model: currentModel,
      base_url: currentBaseUrl || null,
      ...(currentProvider === 'gemini' ? { thinking: currentThinking } : {}),
      ...(currentProvider === 'custom' ? { api_compat: currentApiCompat } : {}),
      ...(currentProvider === 'minimax' ? { rate_limit_interval: currentRateLimit } : {}),
    };
  }

  return (
    <div className="col-span-full p-4 rounded-lg bg-panel border border-primary/30 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xl" role="img" aria-label={agent.name}>
            {agent.icon}
          </span>
          <div>
            <div className="text-sm font-medium text-text">{agent.name}</div>
            <div className="text-xs text-text-muted">{agent.description}</div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-hover-bg transition-colors"
        >
          <X size={16} />
        </button>
      </div>

      {/* Provider */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-text-muted">{t('workbench.provider')}</label>
        <select
          value={currentProvider}
          onChange={(e) => handleProviderChange(e.target.value)}
          className="w-full px-3 py-1.5 rounded-lg bg-surface border border-border text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          <option value="" disabled>
            {t('workbench.provider')}
          </option>
          {configuredProviders.map((p) => (
            <option key={p} value={p}>
              {providerNameMap[p] || p}
            </option>
          ))}
        </select>
      </div>

      {/* Model */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-text-muted">{t('workbench.model')}</label>
        {isLoadingModels ? (
          <div className="flex items-center gap-2 px-3 py-1.5 text-sm text-text-muted">
            <Loader2 size={14} className="animate-spin" />
            <span>{t('models.loadingModels')}</span>
          </div>
        ) : hasModelDropdown ? (
          <select
            value={currentModel}
            onChange={(e) => handleModelChange(e.target.value)}
            className="w-full px-3 py-1.5 rounded-lg bg-surface border border-border text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="" disabled>
              {t('models.selectModel')}
            </option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.display_name || m.id}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            value={currentModel}
            onChange={(e) => handleModelChange(e.target.value)}
            placeholder={t('workbench.model')}
            className="w-full px-3 py-1.5 rounded-lg bg-surface border border-border text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        )}
      </div>

      {/* Base URL */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-text-muted">{t('workbench.baseUrl')}</label>
        <input
          type="text"
          value={currentBaseUrl}
          onChange={(e) => handleBaseUrlChange(e.target.value)}
          placeholder={t('workbench.baseUrlPlaceholder')}
          className="w-full px-3 py-1.5 rounded-lg bg-surface border border-border text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
      </div>

      {/* Provider-specific fields */}
      {currentProvider === 'gemini' && (
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-text-muted">{t('workbench.thinking')}</label>
          <select
            value={currentThinking}
            onChange={(e) => handleThinkingChange(e.target.value)}
            className="w-full px-3 py-1.5 rounded-lg bg-surface border border-border text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="off">{t('models.gemini.off')}</option>
            <option value="low">{t('models.gemini.low')}</option>
            <option value="high">{t('models.gemini.high')}</option>
          </select>
        </div>
      )}

      {currentProvider === 'custom' && (
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-text-muted">{t('workbench.apiCompat')}</label>
          <select
            value={currentApiCompat}
            onChange={(e) => handleApiCompatChange(e.target.value)}
            className="w-full px-3 py-1.5 rounded-lg bg-surface border border-border text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="openai">{t('models.openaiCompat')}</option>
            <option value="anthropic">{t('models.anthropicCompat')}</option>
            <option value="gemini">{t('models.geminiCompat')}</option>
          </select>
        </div>
      )}

      {currentProvider === 'minimax' && (
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-text-muted">
            {t('workbench.rateLimit')}
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={10}
              step={0.5}
              value={currentRateLimit}
              onChange={(e) => handleRateLimitChange(parseFloat(e.target.value))}
              className="flex-1"
            />
            <span className="text-sm text-text-muted w-16 text-right">
              {currentRateLimit}s
            </span>
          </div>
        </div>
      )}

      {/* Reset button */}
      {customized && (
        <button
          onClick={onReset}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-text-muted hover:text-text hover:bg-hover-bg border border-border transition-colors"
        >
          <RotateCcw size={14} />
          {t('workbench.resetToPreset')}
        </button>
      )}
    </div>
  );
}

// ---- Capability Card ----

function CapabilityCard({
  label,
  value,
  configuredProviders,
  providerNameMap,
  onChange,
  t,
}: {
  label: string;
  value: string | null;
  configuredProviders: string[];
  providerNameMap: Record<string, string>;
  onChange: (value: string | null) => void;
  t: (key: string) => string;
}) {
  return (
    <div className="p-3 rounded-lg bg-panel border border-border space-y-2">
      <div className="text-sm font-medium text-text">{label}</div>
      <select
        value={value || ''}
        onChange={(e) => onChange(e.target.value || null)}
        className="w-full px-3 py-1.5 rounded-lg bg-surface border border-border text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
      >
        <option value="">{t('models.disabled')}</option>
        <option value="own">{t('models.ownSuffix')}</option>
        {configuredProviders.map((p) => (
          <option key={p} value={p}>
            {providerNameMap[p] || p}
          </option>
        ))}
      </select>
    </div>
  );
}
