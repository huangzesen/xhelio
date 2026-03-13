import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useSettingsStore } from '../../stores/settingsStore';
import { Loader2, RefreshCw, Database, Globe, HardDrive, ChevronRight } from 'lucide-react';
import { catalogStream } from '../../api/sse';
import type { CatalogSSEEvent, MissionLoadingState } from '../../api/types';
import { useLoadingStateStore } from '../../stores/loadingStateStore';
import { useTranslation } from 'react-i18next';

interface MissionStatus {
  mission_count: number;
  mission_names: string[];
  total_datasets: number;
  oldest_date: string | null;
  loading?: MissionLoadingState;
}

type CatalogAction = 'refresh' | 'rebuild-cdaweb' | 'rebuild-ppi';

export function DataSection() {
  const { t } = useTranslation(['settings', 'common']);
  const { config, updateConfig } = useSettingsStore();
  const descriptions = (config._descriptions as Record<string, string>) ?? {};
  const setField = (key: string, value: unknown) => updateConfig({ [key]: value });
  const setNestedField = (parent: string, key: string, value: unknown) => {
    const current = (config[parent] as Record<string, unknown>) ?? {};
    updateConfig({ [parent]: { ...current, [key]: value } });
  };

  const [missionStatus, setMissionStatus] = useState<MissionStatus | null>(null);
  const [activeAction, setActiveAction] = useState<CatalogAction | null>(null);
  const [progressPct, setProgressPct] = useState<number | null>(null);
  const [progressMessages, setProgressMessages] = useState<string[]>([]);
  const [resultMsg, setResultMsg] = useState<{ text: string; isError: boolean } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const autoLoadingState = useLoadingStateStore((s) => s.state);

  useEffect(() => {
    fetch('/api/catalog/status')
      .then((r) => r.json())
      .then(setMissionStatus)
      .catch(() => {});
  }, []);

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

  return (
    <div className="py-4 space-y-8">
      <div>
        <h2 className="text-lg font-medium text-text mb-1">{t('data.title')}</h2>
        <p className="text-sm text-text-muted">{t('data.description')}</p>
      </div>

      {/* Search subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">{t('data.search')}</h3>
        <label className="block">
          <span className="text-xs text-text-muted">{t('data.catalogSearchMethod')}</span>
          <select
            value={(config.catalog_search_method as string) ?? 'semantic'}
            onChange={(e) => setField('catalog_search_method', e.target.value)}
            className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
          >
            <option value="semantic">{t('data.semantic')}</option>
            <option value="substring">{t('data.substring')}</option>
          </select>
        </label>
      </div>

      {/* Fetching subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">{t('data.fetching')}</h3>

        <label className="flex items-center justify-between cursor-pointer">
          <span className="text-xs text-text-muted">{t('data.parallelFetch')}</span>
          <input
            type="checkbox"
            checked={(config.parallel_fetch as boolean) ?? true}
            onChange={(e) => setField('parallel_fetch', e.target.checked)}
            className="rounded"
          />
        </label>

        <label className="block">
          <span className="text-xs text-text-muted">{t('data.maxWorkers')} ({(config.parallel_max_workers as number) ?? 4})</span>
          <input
            type="range"
            min={1}
            max={8}
            value={(config.parallel_max_workers as number) ?? 4}
            onChange={(e) => setField('parallel_max_workers', parseInt(e.target.value))}
            className="mt-1 block w-full"
          />
        </label>
      </div>

      {/* Visualization subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">{t('data.visualization')}</h3>

        <label className="block">
          <span className="text-xs text-text-muted">{t('data.vizBackend')}</span>
          <select
            value={(config.prefer_viz_backend as string) ?? 'matplotlib'}
            onChange={(e) => setField('prefer_viz_backend', e.target.value)}
            className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
          >
            <option value="plotly">{t('data.plotly')}</option>
            <option value="matplotlib">{t('data.matplotlib')}</option>
            <option value="jsx">{t('data.jsx')}</option>
          </select>
          {descriptions['prefer_viz_backend'] && (
            <p className="mt-1.5 text-xs italic text-text-muted/70 leading-relaxed">{descriptions['prefer_viz_backend']}</p>
          )}
        </label>

        <label className="block">
          <span className="text-xs text-text-muted">{t('data.maxPlotPoints')} ({((config.max_plot_points as number) ?? 10000).toLocaleString()})</span>
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
      </div>

      {/* Sandbox subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">{t('data.sandbox')}</h3>

        <div>
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-xs text-text-muted">{t('data.autoInstall')}</span>
            <input
              type="checkbox"
              checked={((config.sandbox as Record<string, unknown>)?.auto_install as boolean) ?? false}
              onChange={(e) => setNestedField('sandbox', 'auto_install', e.target.checked)}
              className="rounded"
            />
          </label>
          <p className="mt-1.5 text-xs italic text-text-muted/70 leading-relaxed">{t('data.autoInstallDescription')}</p>
        </div>
      </div>

      {/* Mission Data subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">{t('data.missionData')}</h3>

        <div className="bg-panel rounded-xl border border-border p-4 space-y-4">
          <div className="flex items-center gap-2">
            <Database size={16} className="text-text-muted" />
            <span className="font-medium text-text text-sm">{t('data.catalogStatus')}</span>
          </div>

          {missionStatus && missionStatus.mission_count === 0 && !autoLoadingState?.is_loading && (
            <div className="text-xs text-status-warning-text bg-status-warning-bg border border-status-warning-border rounded px-2 py-1.5">
              {t('data.noMissionData')}
            </div>
          )}

          {/* Auto-loading progress (background bootstrap) */}
          {autoLoadingState?.is_loading && !activeAction && (
            <div className="space-y-2">
              <div className="text-xs text-text-muted">
                {t('data.autoDownloading')}
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
              {t('data.autoDownloadFailed', { error: autoLoadingState.error || 'Unknown error' })}
            </div>
          )}

          {missionStatus && missionStatus.mission_count > 0 && (
            <div className="text-xs text-text-muted space-y-1">
              <div>{t('data.missions')}: <span className="font-mono text-text">{missionStatus.mission_count}</span></div>
              <div>{t('data.datasets')}: <span className="font-mono text-text">{missionStatus.total_datasets}</span></div>
              {missionStatus.oldest_date && (
                <div>{t('data.lastUpdated')}: <span className="font-mono text-text">
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
              {t('data.refreshDates')}
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
              {t('data.rebuildCdaweb')}
            </button>
            <button
              onClick={() => handleCatalogAction('rebuild-ppi')}
              disabled={activeAction !== null}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg
                border border-primary/40 text-primary text-sm hover:bg-primary/10 transition-colors
                disabled:opacity-50"
            >
              {activeAction === 'rebuild-ppi' ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Globe size={14} />
              )}
              {t('data.rebuildPpi')}
            </button>
          </div>
        </div>
      </div>

      {/* Disk Assets subsection */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">{t('data.diskAssets')}</h3>

        <Link
          to="/settings/assets"
          className="flex items-center gap-4 bg-panel rounded-xl border border-border p-4
            hover:bg-hover-bg transition-colors group"
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-badge-orange-bg">
            <HardDrive size={18} className="text-badge-orange-text" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-text text-sm">{t('data.diskAssetManagement')}</div>
            <div className="text-xs text-text-muted mt-0.5">
              {t('data.diskAssetDescription')}
            </div>
          </div>
          <ChevronRight size={16} className="text-text-muted group-hover:text-text transition-colors shrink-0" />
        </Link>
      </div>
    </div>
  );
}
