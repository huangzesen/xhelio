import { useEffect, useState } from 'react';
import { usePipelineStore } from '../../stores/pipelineStore';
import { PipelineDAG } from './PipelineDAG';
import { StepTable } from './StepTable';
import { CodeViewer } from './CodeViewer';
import { PlotFullscreen } from '../plot/PlotFullscreen';
import { Loader2, Play, RefreshCw, GitBranch, Inbox, ExternalLink, Save, Check, X, ChevronRight, Maximize2 } from 'lucide-react';
import { PipelineSkeleton } from '../common/Skeleton';
import * as api from '../../api/client';
import type { PlotlyFigure } from '../../api/types';

export function ReplayTab() {
  const {
    sessions,
    selectedSessionId,
    pipeline,
    dagFigure,
    selectedStep,
    renderOptions,
    selectedRenderOpId,
    replaying,
    replayResult,
    loading,
    error,
    loadSessions,
    selectSession,
    selectStep,
    selectRender,
    replay,
    reset,
  } = usePipelineStore();

  const [fullscreenFigure, setFullscreenFigure] = useState<PlotlyFigure | null>(null);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  return (
    <div className="flex-1 overflow-hidden flex">
      {/* Left panel: controls */}
      <div className="w-[300px] shrink-0 border-r border-border bg-panel overflow-y-auto p-4 space-y-4">
        <h2 className="font-medium text-text">Pipeline Explorer</h2>

        {/* Session selector */}
        <label className="block">
          <span className="text-xs text-text-muted">Saved Session</span>
          <select
            value={selectedSessionId ?? ''}
            onChange={(e) => e.target.value && selectSession(e.target.value)}
            className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
          >
            <option value="">Select a session...</option>
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name || s.last_message_preview?.slice(0, 40) || s.id.slice(0, 16)} ({s.op_count} ops)
              </option>
            ))}
          </select>
        </label>

        {selectedSessionId && (
          <>
            {/* Pipeline info */}
            <div className="text-xs text-text-muted space-y-1">
              <div>Operations: <span className="font-mono text-text">{pipeline.length}</span></div>
              <div>
                Renders: <span className="font-mono text-text">
                  {pipeline.filter((r) => r.tool === 'render_plotly_json').length}
                </span>
              </div>
            </div>

            {/* Product / render selector — only show when there are render options beyond "All" */}
            {renderOptions.length > 1 && (
              <label className="block">
                <span className="text-xs text-text-muted">Product / State</span>
                <select
                  value={selectedRenderOpId ?? ''}
                  onChange={(e) => selectRender(e.target.value || null)}
                  className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg"
                >
                  {renderOptions.map((opt) => (
                    <option key={opt.opId ?? '__all__'} value={opt.opId ?? ''}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </label>
            )}

            {/* Replay controls */}
            <div className="space-y-2">
              <button
                onClick={() => replay(true)}
                disabled={replaying}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
                  bg-primary text-white hover:bg-primary-dark transition-colors
                  disabled:opacity-50 text-sm"
              >
                {replaying ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Play size={14} />
                )}
                Replay (cached)
              </button>
              <button
                onClick={() => replay(false)}
                disabled={replaying}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
                  border border-border text-text hover:bg-hover-bg transition-colors
                  disabled:opacity-50 text-sm"
              >
                <RefreshCw size={14} />
                Replay (fresh fetch)
              </button>
            </div>

            {/* Replay result */}
            {replayResult && (
              <div className="text-xs space-y-1">
                <div className={replayResult.errors.length > 0 ? 'text-status-error-text' : 'text-status-success-text'}>
                  {replayResult.steps_completed}/{replayResult.steps_total} steps completed
                </div>
                {replayResult.errors.map((err, i) => (
                  <div key={i} className="text-status-error-text font-mono">
                    {err.op_id}: {err.error}
                  </div>
                ))}
              </div>
            )}

            {/* Save to Gallery */}
            {selectedRenderOpId && (
              <SaveToGallery
                sessionId={selectedSessionId}
                renderOpId={selectedRenderOpId}
                defaultName={renderOptions.find((o) => o.opId === selectedRenderOpId)?.label ?? 'Untitled'}
              />
            )}
          </>
        )}

        {error && (
          <div className="text-xs text-status-error-text bg-status-error-bg rounded px-2 py-1">{error}</div>
        )}
      </div>

      {/* Right panel: DAG + table + code */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-4 bg-surface">
        {/* Breadcrumb */}
        {selectedSessionId && (
          <nav className="flex items-center gap-1 text-sm text-text-muted" aria-label="Breadcrumb">
            <button
              onClick={reset}
              className="hover:text-text transition-colors"
            >
              Pipeline
            </button>
            <ChevronRight size={14} className="shrink-0" />
            <span className="text-text font-medium truncate max-w-[200px]">
              {(() => { const s = sessions.find((s) => s.id === selectedSessionId); return s?.name || s?.last_message_preview?.slice(0, 40) || selectedSessionId.slice(0, 16); })()}
            </span>
            {selectedStep && (
              <>
                <ChevronRight size={14} className="shrink-0" />
                <span className="text-text font-medium truncate max-w-[160px]">
                  {selectedStep.tool}
                  <span className="text-text-muted font-normal ml-1">({selectedStep.id})</span>
                </span>
              </>
            )}
          </nav>
        )}

        {loading && <PipelineSkeleton />}

        {!loading && !selectedSessionId && (
          <div className="flex items-center justify-center py-20">
            <div className="text-center space-y-3">
              <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                <GitBranch size={24} className="text-primary" />
              </div>
              <h3 className="text-base font-medium text-text">Pipeline Explorer</h3>
              <p className="text-sm text-text-muted max-w-xs">
                Select a saved session from the sidebar to visualize its pipeline DAG and inspect operations.
              </p>
              {sessions.length === 0 && (
                <div className="flex items-center justify-center gap-1.5 text-xs text-text-muted mt-2">
                  <Inbox size={14} />
                  No saved sessions yet
                </div>
              )}
            </div>
          </div>
        )}

        {!loading && dagFigure && (
          <div className="bg-panel rounded-xl border border-border p-4 overflow-hidden">
            <h3 className="text-sm font-medium text-text mb-2">Pipeline DAG</h3>
            <PipelineDAG figure={dagFigure} onNodeClick={(opId) => {
              const step = pipeline.find((r) => r.id === opId);
              if (step) selectStep(step);
            }} />
          </div>
        )}

        {!loading && pipeline.length > 0 && (
          <div className="bg-panel rounded-xl border border-border p-4">
            <h3 className="text-sm font-medium text-text mb-2">Operations</h3>
            <StepTable
              records={pipeline}
              selectedId={selectedStep?.id ?? null}
              onSelect={selectStep}
            />
          </div>
        )}

        {selectedStep && (
          <div className="bg-panel rounded-xl border border-border p-4 overflow-hidden">
            <CodeViewer
              record={selectedStep}
              allRecords={pipeline}
              onNavigate={selectStep}
            />
          </div>
        )}

        {/* Replay figure */}
        {replayResult?.figure && (
          <div className="bg-panel rounded-xl border border-border p-4 overflow-hidden">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-text">Replayed Figure</h3>
              <button
                onClick={() => setFullscreenFigure(replayResult.figure!)}
                className="p-1 rounded hover:bg-hover-bg text-text-muted hover:text-text transition-colors"
                aria-label="Expand figure"
                title="Expand"
              >
                <Maximize2 size={14} />
              </button>
            </div>
            <PipelineDAG figure={replayResult.figure} />
          </div>
        )}

        {/* Large replay figure — open in new tab */}
        {replayResult && !replayResult.figure && replayResult.figure_url && (
          <div className="bg-panel rounded-xl border border-border p-6 flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-text">Large replayed figure</h4>
              <p className="text-xs text-text-muted mt-1">
                This figure is too large to display inline. Open it in a new tab for full interactivity.
              </p>
            </div>
            <a
              href={replayResult.figure_url}
              target="_blank"
              rel="noopener noreferrer"
              className="shrink-0 ml-4 flex items-center gap-2 px-4 py-2 rounded-lg
                bg-primary text-white text-sm font-medium hover:bg-primary-dark transition-colors"
            >
              <ExternalLink size={14} />
              Open in new tab
            </a>
          </div>
        )}

        {/* Fullscreen overlay */}
        {fullscreenFigure && (
          <PlotFullscreen
            figures={[fullscreenFigure]}
            currentIndex={0}
            onNavigate={() => {}}
            onClose={() => setFullscreenFigure(null)}
          />
        )}
      </div>
    </div>
  );
}


function SaveToGallery({ sessionId, renderOpId, defaultName }: {
  sessionId: string;
  renderOpId: string;
  defaultName: string;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(defaultName);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Reset state when render selection changes
  useEffect(() => {
    setOpen(false);
    setName(defaultName);
    setSaved(false);
    setSaveError(null);
  }, [renderOpId, defaultName]);

  const handleSave = async () => {
    if (!name.trim()) return;
    setSaving(true);
    setSaveError(null);
    try {
      await api.saveToGallery(name.trim(), sessionId, renderOpId);
      setSaved(true);
      setTimeout(() => {
        setOpen(false);
        setSaved(false);
      }, 2000);
    } catch (err) {
      setSaveError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
          border border-border text-text hover:bg-hover-bg transition-colors text-sm"
      >
        <Save size={14} />
        Save to Gallery
      </button>
    );
  }

  return (
    <div className="space-y-2 p-3 rounded-lg border border-border bg-surface">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-text">Save to Gallery</span>
        <button onClick={() => setOpen(false)} className="text-text-muted hover:text-text">
          <X size={14} />
        </button>
      </div>
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Name for this visualization"
        maxLength={200}
        className="block w-full rounded-lg border border-border px-3 py-1.5 text-sm bg-input-bg text-text"
        onKeyDown={(e) => e.key === 'Enter' && handleSave()}
        autoFocus
      />
      <button
        onClick={handleSave}
        disabled={saving || !name.trim()}
        className="w-full flex items-center justify-center gap-2 px-3 py-1.5 rounded-lg
          bg-primary text-white hover:bg-primary-dark transition-colors
          disabled:opacity-50 text-sm"
      >
        {saving ? (
          <Loader2 size={14} className="animate-spin" />
        ) : saved ? (
          <Check size={14} />
        ) : (
          <Save size={14} />
        )}
        {saving ? 'Saving...' : saved ? 'Saved!' : 'Save'}
      </button>
      {saveError && (
        <div className="text-xs text-status-error-text">{saveError}</div>
      )}
    </div>
  );
}
