import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useSettingsStore } from '../stores/settingsStore';
import { Save, Loader2, Check, RefreshCw, Database, Globe, Eye, EyeOff, CheckCircle2, XCircle, HardDrive, ChevronRight } from 'lucide-react';
import { catalogStream } from '../api/sse';
import { listModels, getApiKeyStatus, updateApiKey, type ModelInfo, type ApiKeyStatus } from '../api/client';
import type { CatalogSSEEvent, MissionLoadingState } from '../api/types';
import { useLoadingStateStore } from '../stores/loadingStateStore';

interface MissionStatus {
  mission_count: number;
  mission_names: string[];
  total_datasets: number;
  oldest_date: string | null;
  loading?: MissionLoadingState;
}

type CatalogAction = 'refresh' | 'rebuild-cdaweb' | 'rebuild-ppi';

export function SettingsPage() {
  const { config, loading, saving, error, saved, sessionSwitched, loadConfig, updateConfig, saveConfig } =
    useSettingsStore();

  const [missionStatus, setMissionStatus] = useState<MissionStatus | null>(null);
  const [activeAction, setActiveAction] = useState<CatalogAction | null>(null);
  const [progressPct, setProgressPct] = useState<number | null>(null);
  const [progressMessages, setProgressMessages] = useState<string[]>([]);
  const [resultMsg, setResultMsg] = useState<{ text: string; isError: boolean } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const autoLoadingState = useLoadingStateStore((s) => s.state);

  // Available models for the active provider
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);

  // API key management
  const [keyStatus, setKeyStatus] = useState<ApiKeyStatus | null>(null);
  const [newKey, setNewKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [keySaving, setKeySaving] = useState(false);
  const [keyResult, setKeyResult] = useState<{ valid: boolean; error: string | null } | null>(null);

  // Setting descriptions served inline from GET /config
  const descriptions = (config._descriptions as Record<string, string>) ?? {};
  const activeProvider = (config.llm_provider as string) ?? 'gemini';

  useEffect(() => {
    loadConfig();
    fetch('/api/catalog/status')
      .then((r) => r.json())
      .then(setMissionStatus)
      .catch(() => {});
  }, [loadConfig]);

  // Reset key state when provider changes
  useEffect(() => {
    setNewKey('');
    setKeyResult(null);
    getApiKeyStatus(activeProvider)
      .then(setKeyStatus)
      .catch(() => {});
  }, [activeProvider]);

  // Fetch available models for the active provider
  useEffect(() => {
    setModelsLoading(true);
    listModels(activeProvider)
      .then((res) => setAvailableModels(res.models || []))
      .catch(() => setAvailableModels([]))
      .finally(() => setModelsLoading(false));
  }, [activeProvider]);

  const handleCatalogAction = useCallback(async (action: CatalogAction) => {
    setActiveAction(action);
    setProgressPct(null);
    setProgressMessages([]);
    setResultMsg(null);

    const controller = new AbortController();
    abortRef.current = controller;

    const endpoint = `/api/catalog/${action}`;

    try {
      for await (const event of catalogStream(endpoint, controller.signal)) {
        if (event.type === 'progress') {
          const pe = event as CatalogSSEEvent & { type: 'progress' };
          if (pe.pct != null) setProgressPct(pe.pct);
          if (pe.message) {
            setProgressMessages((prev) => [...prev.slice(-19), pe.message!]);
          }
        } else if (event.type === 'done') {
          setResultMsg({ text: (event as CatalogSSEEvent & { type: 'done' }).message, isError: false });
        } else if (event.type === 'error') {
          setResultMsg({ text: (event as CatalogSSEEvent & { type: 'error' }).message, isError: true });
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        setResultMsg({ text: `Error: ${(e as Error).message}`, isError: true });
      }
    } finally {
      setActiveAction(null);
      setProgressPct(null);
      abortRef.current = null;
      // Reload status
      fetch('/api/catalog/status')
        .then((r) => r.json())
        .then(setMissionStatus)
        .catch(() => {});
    }
  }, []);

  const handleApiKeySave = useCallback(async () => {
    if (!newKey.trim()) return;
    setKeySaving(true);
    setKeyResult(null);
    try {
      const res = await updateApiKey(activeProvider, newKey.trim());
      setKeyResult({ valid: res.valid, error: res.error });
      if (res.valid) {
        setKeyStatus({ configured: true, masked: res.masked });
        setNewKey('');
        // Refresh models list since the key may have changed
        setModelsLoading(true);
        listModels(activeProvider)
          .then((res) => setAvailableModels(res.models || []))
          .catch(() => setAvailableModels([]))
          .finally(() => setModelsLoading(false));
      }
    } catch (err) {
      setKeyResult({ valid: false, error: (err as Error).message });
    } finally {
      setKeySaving(false);
    }
  }, [newKey, activeProvider]);

  const setField = (key: string, value: unknown) => {
    updateConfig({ [key]: value });
  };

  const setNestedField = (parent: string, key: string, value: unknown) => {
    const current = (config[parent] as Record<string, unknown>) ?? {};
    updateConfig({ [parent]: { ...current, [key]: value } });
  };

  // Helper: read/write model fields from providers.<active> section
  const providers = (config.providers ?? {}) as Record<string, Record<string, unknown>>;
  const providerConfig = providers[activeProvider] ?? {};

  const getProviderField = (key: string): string => {
    return (providerConfig[key] as string) ?? '';
  };

  const setProviderField = (key: string, value: unknown) => {
    const updated = {
      ...providers,
      [activeProvider]: { ...providerConfig, [key]: value },
    };
    updateConfig({ providers: updated });
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="animate-spin text-primary" size={24} />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 bg-surface">
      <div className="max-w-5xl mx-auto space-y-6">
        <h1 className="text-xl font-semibold text-text">Settings</h1>

        {error && (
          <div className="bg-status-error-bg border border-status-error-border text-status-error-text rounded-lg px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {saved && (
          <div className="bg-status-success-bg border border-status-success-border text-status-success-text rounded-lg px-4 py-3 text-sm flex items-center gap-2">
            <Check size={16} />
            {sessionSwitched
              ? 'Settings saved. New session started with updated provider.'
              : 'Settings saved successfully'}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* LLM Provider & Models */}
          <div className="bg-panel rounded-xl border border-border p-4 space-y-4">
            <h2 className="font-medium text-text">LLM Provider & Models</h2>

            {/* API Key */}
            <div className="space-y-2">
              <span className="text-xs text-text-muted block">
                {activeProvider === 'gemini' ? 'Gemini' : activeProvider === 'openai' ? 'OpenAI' : 'Anthropic'} API Key
              </span>
              {keyStatus?.configured && (
                <div className="text-xs text-text-muted">
                  Current key: <span className="font-mono text-text">{keyStatus.masked}</span>
                </div>
              )}
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <input
                    type={showKey ? 'text' : 'password'}
                    value={newKey}
                    onChange={(e) => { setNewKey(e.target.value); setKeyResult(null); }}
                    placeholder={keyStatus?.configured ? 'Enter new key to update' : activeProvider === 'gemini' ? 'AIza...' : activeProvider === 'openai' ? 'sk-...' : 'sk-ant-...'}
                    className="block w-full rounded-lg border border-border px-3 py-2 pr-9 text-sm
                      bg-input-bg text-text placeholder:text-text-muted/50
                      focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey((v) => !v)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text"
                    tabIndex={-1}
                  >
                    {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
                <button
                  onClick={handleApiKeySave}
                  disabled={keySaving || !newKey.trim()}
                  className="px-3 py-2 rounded-lg bg-primary text-white text-sm
                    hover:bg-primary-dark transition-colors disabled:opacity-50 shrink-0"
                >
                  {keySaving ? <Loader2 size={14} className="animate-spin" /> : 'Update'}
                </button>
              </div>
              {keyResult && (
                <div className={`flex items-center gap-1.5 text-xs ${
                  keyResult.valid ? 'text-status-success-text' : 'text-status-error-text'
                }`}>
                  {keyResult.valid ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                  {keyResult.valid ? 'Key updated successfully' : `Invalid key${keyResult.error ? `: ${keyResult.error}` : ''}`}
                </div>
              )}
            </div>

            <div className="border-t border-border" />

            <label className="block">
              <span className="text-xs text-text-muted">Provider</span>
              <select
                value={activeProvider}
                onChange={(e) => setField('llm_provider', e.target.value)}
                className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
              >
                <option value="gemini">Gemini</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
              </select>
            </label>

            {([
              ['model', 'Main Model (orchestrator + planner)'],
              ['sub_agent_model', 'Sub-agent Model (mission / viz / data)'],
              ['insight_model', 'Insight Model (plot analysis)'],
              ['inline_model', 'Inline Model (cheapest)'],
              ['planner_model', 'Planner Model'],
              ['fallback_model', 'Fallback Model'],
            ] as const).map(([key, label]) => {
              const current = getProviderField(key);
              const inList = availableModels.some((m) => m.id === current);
              return (
                <label key={key} className="block">
                  <span className="text-xs text-text-muted">{label}</span>
                  {modelsLoading ? (
                    <div className="mt-1 flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text-muted">
                      <Loader2 size={14} className="animate-spin" /> Loading models...
                    </div>
                  ) : availableModels.length > 0 ? (
                    <select
                      value={current}
                      onChange={(e) => setProviderField(key, e.target.value)}
                      className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm font-mono bg-input-bg text-text"
                    >
                      {!current && <option value="">Select a model</option>}
                      {current && !inList && (
                        <option value={current}>{current}</option>
                      )}
                      {availableModels.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.id}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={current}
                      onChange={(e) => setProviderField(key, e.target.value)}
                      placeholder={key === 'planner_model' ? 'defaults to main model' : key === 'insight_model' ? 'defaults to sub-agent model' : key === 'fallback_model' ? 'defaults to sub-agent model' : ''}
                      className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm font-mono bg-input-bg text-text"
                    />
                  )}
                </label>
              );
            })}

            {activeProvider === 'gemini' && (
            <div className="space-y-2">
                <span className="text-xs text-text-muted block">Gemini Thinking Level</span>
                <div className="grid grid-cols-3 gap-2">
                  <label className="block">
                    <span className="text-xs text-text-muted">Main</span>
                    <select
                      value={getProviderField('thinking_model') || 'high'}
                      onChange={(e) => setProviderField('thinking_model', e.target.value)}
                      className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
                    >
                      <option value="off">Off</option>
                      <option value="low">Low</option>
                      <option value="high">High</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-xs text-text-muted">Sub-agent</span>
                    <select
                      value={getProviderField('thinking_sub_agent') || 'low'}
                      onChange={(e) => setProviderField('thinking_sub_agent', e.target.value)}
                      className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
                    >
                      <option value="off">Off</option>
                      <option value="low">Low</option>
                      <option value="high">High</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-xs text-text-muted">Insight</span>
                    <select
                      value={getProviderField('thinking_insight') || 'low'}
                      onChange={(e) => setProviderField('thinking_insight', e.target.value)}
                      className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
                    >
                      <option value="off">Off</option>
                      <option value="low">Low</option>
                      <option value="high">High</option>
                    </select>
                  </label>
                </div>
            </div>
            )}

            {(activeProvider === 'openai' || activeProvider === 'anthropic') && (
            <label className="block">
              <span className="text-xs text-text-muted">Base URL</span>
              <input
                type="text"
                value={getProviderField('base_url') || ''}
                onChange={(e) => setProviderField('base_url', e.target.value)}
                placeholder={activeProvider === 'anthropic' ? 'https://api.anthropic.com' : 'https://api.openai.com/v1'}
                className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm font-mono bg-input-bg text-text"
              />
            </label>
            )}
          </div>

          {/* Data & Search + Memory + Mission Data */}
          <div className="space-y-4">
            <div className="bg-panel rounded-xl border border-border p-4 space-y-4">
              <h2 className="font-medium text-text">Data & Search</h2>

              <label className="block">
                <span className="text-xs text-text-muted">Catalog Search Method</span>
                <select
                  value={(config.catalog_search_method as string) ?? 'semantic'}
                  onChange={(e) => setField('catalog_search_method', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
                >
                  <option value="semantic">Semantic</option>
                  <option value="substring">Substring</option>
                </select>
              </label>

              <div className="flex items-center justify-between">
                <span className="text-xs text-text-muted">Parallel Fetch</span>
                <input
                  type="checkbox"
                  checked={(config.parallel_fetch as boolean) ?? true}
                  onChange={(e) => setField('parallel_fetch', e.target.checked)}
                  className="rounded"
                />
              </div>

              <label className="block">
                <span className="text-xs text-text-muted">Max Workers ({(config.parallel_max_workers as number) ?? 4})</span>
                <input
                  type="range"
                  min={1}
                  max={8}
                  value={(config.parallel_max_workers as number) ?? 4}
                  onChange={(e) => setField('parallel_max_workers', parseInt(e.target.value))}
                  className="mt-1 block w-full"
                />
              </label>

              <label className="block">
                <span className="text-xs text-text-muted">Max Plot Points ({((config.max_plot_points as number) ?? 10000).toLocaleString()})</span>
                <input
                  type="range"
                  min={1000}
                  max={100000}
                  step={1000}
                  value={(config.max_plot_points as number) ?? 10000}
                  onChange={(e) => setField('max_plot_points', parseInt(e.target.value))}
                  className="mt-1 block w-full"
                />
                {descriptions['max_plot_points'] && (
                  <p className="mt-1.5 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['max_plot_points']}</p>
                )}
              </label>

              <label className="block">
                <span className="text-xs text-text-muted">Visualization Backend</span>
                <select
                  value={(config.prefer_viz_backend as string) ?? 'matplotlib'}
                  onChange={(e) => setField('prefer_viz_backend', e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
                >
                  <option value="plotly">Plotly (interactive)</option>
                  <option value="matplotlib">Matplotlib (static / publication-quality)</option>
                  <option value="jsx">JSX / Recharts (dashboards)</option>
                </select>
                {descriptions['prefer_viz_backend'] && (
                  <p className="mt-1.5 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['prefer_viz_backend']}</p>
                )}
              </label>
            </div>

            {/* Mission Data */}
            <div className="bg-panel rounded-xl border border-border p-4 space-y-4">
              <div className="flex items-center gap-2">
                <Database size={16} className="text-text-muted" />
                <h2 className="font-medium text-text">Mission Data</h2>
              </div>

              {missionStatus && missionStatus.mission_count === 0 && !autoLoadingState?.is_loading && (
                <div className="text-xs text-status-warning-text bg-status-warning-bg border border-status-warning-border rounded px-2 py-1.5">
                  No mission data found. Click "Rebuild CDAWeb" or "Rebuild PPI" to download.
                </div>
              )}

              {/* Auto-loading progress (background bootstrap) */}
              {autoLoadingState?.is_loading && !activeAction && (
                <div className="space-y-2">
                  <div className="text-xs text-text-muted">
                    Auto-downloading mission data in the background...
                  </div>
                  {autoLoadingState.progress_pct > 0 && (
                    <div className="w-full bg-border rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all duration-300"
                        style={{ width: `${Math.min(autoLoadingState.progress_pct, 100)}%` }}
                      />
                    </div>
                  )}
                  {autoLoadingState.progress_message && (
                    <div className="text-xs text-text-muted font-mono">
                      {autoLoadingState.progress_message}
                    </div>
                  )}
                </div>
              )}

              {autoLoadingState?.phase === 'failed' && !activeAction && (
                <div className="text-xs text-status-error-text bg-status-error-bg border border-status-error-border rounded px-2 py-1.5">
                  Auto-download failed: {autoLoadingState.error || 'Unknown error'}. Use the buttons below to retry.
                </div>
              )}

              {missionStatus && missionStatus.mission_count > 0 && (
                <div className="text-xs text-text-muted space-y-1">
                  <div>Missions: <span className="font-mono text-text">{missionStatus.mission_count}</span></div>
                  <div>Datasets: <span className="font-mono text-text">{missionStatus.total_datasets}</span></div>
                  {missionStatus.oldest_date && (
                    <div>Last updated: <span className="font-mono text-text">
                      {new Date(missionStatus.oldest_date).toLocaleDateString()}
                    </span></div>
                  )}
                </div>
              )}

              {/* Result message */}
              {resultMsg && !activeAction && (
                <div className={`text-xs rounded px-2 py-1 ${
                  resultMsg.isError
                    ? 'text-status-error-text bg-status-error-bg'
                    : 'text-status-success-text bg-status-success-bg'
                }`}>
                  {resultMsg.text}
                </div>
              )}

              {/* Progress bar + messages during active action */}
              {activeAction && (
                <div className="space-y-2">
                  {progressPct != null && (
                    <div className="w-full bg-border rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all duration-300"
                        style={{ width: `${Math.min(progressPct, 100)}%` }}
                      />
                    </div>
                  )}
                  {progressMessages.length > 0 && (
                    <div className="text-xs text-text-muted font-mono bg-border-subtle rounded p-2 max-h-24 overflow-y-auto space-y-0.5">
                      {progressMessages.map((msg, i) => (
                        <div key={i}>{msg}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Action buttons */}
              <div className="flex gap-2">
                <button
                  onClick={() => handleCatalogAction('refresh')}
                  disabled={activeAction !== null}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg
                    border border-border text-text text-sm hover:bg-hover-bg transition-colors
                    disabled:opacity-50"
                >
                  {activeAction === 'refresh' ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <RefreshCw size={14} />
                  )}
                  Refresh Dates
                </button>
                <button
                  onClick={() => handleCatalogAction('rebuild-cdaweb')}
                  disabled={activeAction !== null}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg
                    border border-status-warning-border text-status-warning-text text-sm hover:bg-status-warning-bg transition-colors
                    disabled:opacity-50"
                >
                  {activeAction === 'rebuild-cdaweb' ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Database size={14} />
                  )}
                  Rebuild CDAWeb
                </button>
                <button
                  onClick={() => handleCatalogAction('rebuild-ppi')}
                  disabled={activeAction !== null}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg
                    border border-purple-300 text-purple-600 text-sm hover:bg-purple-50 transition-colors
                    disabled:opacity-50"
                >
                  {activeAction === 'rebuild-ppi' ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Globe size={14} />
                  )}
                  Rebuild PPI
                </button>
              </div>
            </div>

            <div className="bg-panel rounded-xl border border-border p-4 space-y-4">
              <h2 className="font-medium text-text">Memory</h2>
              {([
                ['memory_token_budget', 'Memory Token Budget', 100000],
                ['ops_library_max_entries', 'Ops Library Max Entries', 50],
              ] as const).map(([key, label, def]) => (
                <label key={key} className="block">
                  <span className="text-xs text-text-muted">{label}</span>
                  <input
                    type="number"
                    min={1}
                    value={(config[key] as number) ?? def}
                    onChange={(e) => setField(key, parseInt(e.target.value))}
                    className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
                  />
                  {descriptions[key] && (
                    <p className="mt-1.5 text-xs italic text-text-muted/70 leading-relaxed">{descriptions[key]}</p>
                  )}
                </label>
              ))}

              <label className="block">
                <span className="text-xs text-text-muted">Memory Extraction Interval</span>
                <input
                  type="number"
                  min={0}
                  value={(config.memory_extraction_interval as number) ?? 2}
                  onChange={(e) => setField('memory_extraction_interval', parseInt(e.target.value))}
                  className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
                />
                {descriptions['memory_extraction_interval'] && (
                  <p className="mt-1.5 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['memory_extraction_interval']}</p>
                )}
              </label>
            </div>

            {/* Reasoning */}
            <div className="bg-panel rounded-xl border border-border p-4 space-y-4">
              <h2 className="font-medium text-text">Reasoning</h2>
              {([
                ['observation_summaries', 'Observation Summaries', true],
                ['self_reflection', 'Self Reflection', true],
                ['show_thinking', 'Show Thinking', false],
                ['insight_feedback', 'Insight Feedback', false],
                ['async_delegation', 'Async Delegation', false],
              ] as const).map(([key, label, def]) => {
                const reasoning = (config.reasoning as Record<string, unknown>) ?? {};
                const val = (reasoning[key] as boolean) ?? def;
                const desc = descriptions[`reasoning.${key}`];
                return (
                  <div key={key}>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-text-muted">{label}</span>
                      <input
                        type="checkbox"
                        checked={val}
                        onChange={(e) => setNestedField('reasoning', key, e.target.checked)}
                        className="rounded"
                      />
                    </div>
                    {desc && (
                      <p className="mt-1 text-xs italic text-text-muted/70 leading-relaxed">{desc}</p>
                    )}
                  </div>
                );
              })}
              {((config.reasoning as Record<string, unknown>)?.insight_feedback as boolean) && (
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-muted">Insight Feedback Max Iterations</span>
                    <span className="text-xs text-text-muted">{(config.reasoning as Record<string, unknown>)?.insight_feedback_max_iterations ?? 2}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={5}
                    value={(config.reasoning as Record<string, unknown>)?.insight_feedback_max_iterations ?? 2}
                    onChange={(e) => setNestedField('reasoning', 'insight_feedback_max_iterations', parseInt(e.target.value))}
                    className="w-full mt-1"
                  />
                  <p className="mt-1 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['reasoning.insight_feedback_max_iterations']}</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Disk Asset Management link */}
        <Link
          to="/settings/assets"
          className="flex items-center gap-4 bg-panel rounded-xl border border-border p-4
            hover:bg-hover-bg transition-colors group"
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-badge-orange-bg">
            <HardDrive size={18} className="text-badge-orange-text" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-text text-sm">Disk Asset Management</div>
            <div className="text-xs text-text-muted mt-0.5">
              Monitor and clean up cached data files, sessions, and SPICE kernels
            </div>
          </div>
          <ChevronRight size={16} className="text-text-muted group-hover:text-text transition-colors shrink-0" />
        </Link>

        {/* Save button */}
        <div className="flex justify-end">
          <button
            onClick={saveConfig}
            disabled={saving}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-primary text-white
              hover:bg-primary-dark transition-colors disabled:opacity-50"
          >
            {saving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}
