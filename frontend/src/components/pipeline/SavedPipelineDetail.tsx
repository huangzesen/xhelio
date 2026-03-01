import { useState, useMemo } from 'react';
import type { SavedPipelineDetail as SavedPipelineDetailType, PipelineExecuteResult, PlotlyFigure, PipelineRecord } from '../../api/types';
import { PipelineDAG } from './PipelineDAG';
import { StepTable } from './StepTable';
import { CodeViewer } from './CodeViewer';
import { PlotFullscreen } from '../plot/PlotFullscreen';
import { ChevronRight, Loader2, Play, ExternalLink, Maximize2, Trash2, Pencil, Check, X, MessageSquare, ChevronDown, Send } from 'lucide-react';
import { stepToRecord } from '../../stores/savedPipelineStore';

interface Props {
  detail: SavedPipelineDetailType;
  dagFigure: PlotlyFigure | null;
  selectedStep: PipelineRecord | null;
  executing: boolean;
  executeResult: PipelineExecuteResult | null;
  error: string | null;
  onBack: () => void;
  onSelectStep: (record: PipelineRecord | null) => void;
  onExecute: (id: string, timeStart: string, timeEnd: string) => void;
  onDelete: (id: string) => void;
  onUpdate?: (id: string, updates: { name?: string; description?: string }) => void;
  onFeedback?: (id: string, comment: string) => void;
}

export function SavedPipelineDetail({
  detail,
  dagFigure,
  selectedStep,
  executing,
  executeResult,
  error,
  onBack,
  onSelectStep,
  onExecute,
  onDelete,
  onUpdate,
  onFeedback,
}: Props) {
  // Adapt steps to PipelineRecord for StepTable / CodeViewer reuse
  const records = useMemo(() => detail.steps.map(stepToRecord), [detail.steps]);

  // Pre-fill time range from original
  const [timeStart, setTimeStart] = useState(() =>
    detail.time_range_original?.[0]?.slice(0, 16) ?? '',
  );
  const [timeEnd, setTimeEnd] = useState(() =>
    detail.time_range_original?.[1]?.slice(0, 16) ?? '',
  );

  const [fullscreenFigure, setFullscreenFigure] = useState<PlotlyFigure | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  // Editing state
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState(detail.name);
  const [editDescription, setEditDescription] = useState(detail.description);

  // Feedback state
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackText, setFeedbackText] = useState('');

  const handleDelete = () => {
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    onDelete(detail.id);
  };

  const handleExecute = () => {
    if (!timeStart || !timeEnd) return;
    // Append :00 seconds if datetime-local omitted them (length 16 = "YYYY-MM-DDTHH:MM")
    const ts = timeStart.length === 16 ? timeStart + ':00' : timeStart;
    const te = timeEnd.length === 16 ? timeEnd + ':00' : timeEnd;
    onExecute(detail.id, ts, te);
  };

  const handleEditStart = () => {
    setEditName(detail.name);
    setEditDescription(detail.description);
    setEditing(true);
  };

  const handleEditSave = () => {
    if (onUpdate) {
      const updates: { name?: string; description?: string } = {};
      if (editName !== detail.name) updates.name = editName;
      if (editDescription !== detail.description) updates.description = editDescription;
      if (Object.keys(updates).length > 0) {
        onUpdate(detail.id, updates);
      }
    }
    setEditing(false);
  };

  const handleEditCancel = () => {
    setEditing(false);
  };

  const handleFeedbackSubmit = () => {
    const trimmed = feedbackText.trim();
    if (!trimmed || !onFeedback) return;
    onFeedback(detail.id, trimmed);
    setFeedbackText('');
  };

  return (
    <div className="space-y-4">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-1 text-sm text-text-muted" aria-label="Breadcrumb">
        <button onClick={onBack} className="hover:text-text transition-colors">
          Pipelines
        </button>
        <ChevronRight size={14} className="shrink-0" />
        <span className="text-text font-medium truncate max-w-[300px]">
          {detail.name}
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

      {/* Header card */}
      <div className="bg-panel rounded-xl border border-border p-4 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            {editing ? (
              <div className="space-y-2">
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="w-full rounded-lg border border-border px-3 py-1.5 text-base font-medium bg-input-bg text-text"
                  placeholder="Pipeline name"
                  autoFocus
                />
                <textarea
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                  rows={4}
                  className="w-full rounded-lg border border-border px-3 py-1.5 text-sm bg-input-bg text-text font-mono resize-y"
                  placeholder="Description (supports Source: / Rationale: / Use cases: format)"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleEditSave}
                    className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs
                      bg-primary text-white hover:bg-primary-dark transition-colors"
                  >
                    <Check size={12} />
                    Save
                  </button>
                  <button
                    onClick={handleEditCancel}
                    className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs
                      text-text-muted hover:bg-hover transition-colors"
                  >
                    <X size={12} />
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="group">
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-medium text-text">{detail.name}</h3>
                  {onUpdate && (
                    <button
                      onClick={handleEditStart}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-hover
                        text-text-muted hover:text-text transition-all"
                      title="Edit name and description"
                    >
                      <Pencil size={13} />
                    </button>
                  )}
                </div>
                {detail.description && (
                  <p className="text-sm text-text-muted mt-1 whitespace-pre-line">{detail.description}</p>
                )}
              </div>
            )}
          </div>
          {!editing && (
            <div className="shrink-0 flex items-center gap-2">
              {confirmDelete && (
                <button
                  onClick={() => setConfirmDelete(false)}
                  className="px-2.5 py-1.5 rounded-lg text-xs text-text-muted
                    hover:bg-hover transition-colors"
                >
                  Cancel
                </button>
              )}
              <button
                onClick={handleDelete}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-colors ${
                  confirmDelete
                    ? 'bg-status-error-bg text-status-error-text hover:bg-red-200 dark:hover:bg-red-900'
                    : 'text-text-muted hover:text-status-error-text hover:bg-status-error-bg'
                }`}
              >
                <Trash2 size={13} />
                {confirmDelete ? 'Confirm delete' : 'Delete'}
              </button>
            </div>
          )}
        </div>

        {/* Tags */}
        {detail.tags.length > 0 && (
          <div className="flex gap-1.5 flex-wrap">
            {detail.tags.map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 rounded text-[11px] font-medium bg-badge-gray-bg text-badge-gray-text"
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Metadata grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div>
            <span className="text-text-muted">Steps</span>
            <div className="font-mono text-text">{detail.step_count}</div>
          </div>
          <div>
            <span className="text-text-muted">Source session</span>
            <div className="font-mono text-text truncate" title={detail.source_session_id}>
              {detail.source_session_id.slice(0, 16)}
            </div>
          </div>
          <div>
            <span className="text-text-muted">Original time range</span>
            <div className="font-mono text-text text-[11px]">
              {detail.time_range_original?.[0]?.slice(0, 10) ?? '?'} &ndash;{' '}
              {detail.time_range_original?.[1]?.slice(0, 10) ?? '?'}
            </div>
          </div>
          <div>
            <span className="text-text-muted">Family</span>
            <div className="font-mono text-text truncate" title={detail.family_id}>
              {detail.family_id?.slice(0, 12) ?? '\u2014'}
            </div>
          </div>
        </div>
      </div>

      {/* Feedback card */}
      {onFeedback && (
        <div className="bg-panel rounded-xl border border-border">
          <button
            onClick={() => setFeedbackOpen(!feedbackOpen)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-text hover:bg-hover transition-colors rounded-xl"
          >
            <div className="flex items-center gap-2">
              <MessageSquare size={14} />
              Feedback
              {detail.feedback && detail.feedback.length > 0 && (
                <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-gray-bg text-badge-gray-text">
                  {detail.feedback.length}
                </span>
              )}
            </div>
            <ChevronDown
              size={14}
              className={`transition-transform ${feedbackOpen ? 'rotate-180' : ''}`}
            />
          </button>

          {feedbackOpen && (
            <div className="px-4 pb-4 space-y-3">
              {/* Existing feedback entries */}
              {detail.feedback && detail.feedback.length > 0 && (
                <div className="space-y-2">
                  {detail.feedback.map((fb, i) => (
                    <div key={i} className="text-xs bg-surface rounded-lg px-3 py-2 border border-border/50">
                      <div className="text-text">{fb.comment}</div>
                      <div className="text-text-muted mt-1">
                        {fb.timestamp?.slice(0, 10)} &middot; {fb.source}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* New feedback input */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={feedbackText}
                  onChange={(e) => setFeedbackText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleFeedbackSubmit(); }}
                  placeholder="Suggest a description improvement or leave feedback..."
                  className="flex-1 rounded-lg border border-border px-3 py-1.5 text-sm bg-input-bg text-text"
                />
                <button
                  onClick={handleFeedbackSubmit}
                  disabled={!feedbackText.trim()}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
                    bg-primary text-white hover:bg-primary-dark transition-colors
                    disabled:opacity-50"
                >
                  <Send size={12} />
                  Submit
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* DAG card */}
      {dagFigure && (
        <div className="bg-panel rounded-xl border border-border p-4 overflow-hidden">
          <h3 className="text-sm font-medium text-text mb-2">Pipeline DAG</h3>
          <PipelineDAG
            figure={dagFigure}
            onNodeClick={(stepId) => {
              const rec = records.find((r) => r.id === stepId);
              if (rec) onSelectStep(rec);
            }}
          />
        </div>
      )}

      {/* Steps table card */}
      {records.length > 0 && (
        <div className="bg-panel rounded-xl border border-border p-4">
          <h3 className="text-sm font-medium text-text mb-2">Steps</h3>
          <StepTable
            records={records}
            selectedId={selectedStep?.id ?? null}
            onSelect={onSelectStep}
          />
        </div>
      )}

      {/* Code viewer card */}
      {selectedStep && (
        <div className="bg-panel rounded-xl border border-border p-4 overflow-hidden">
          <CodeViewer
            record={selectedStep}
            allRecords={records}
            onNavigate={onSelectStep}
          />
        </div>
      )}

      {/* Execute panel */}
      <div className="bg-panel rounded-xl border border-border p-4 space-y-3">
        <h3 className="text-sm font-medium text-text">Execute Pipeline</h3>
        <div className="flex items-end gap-3 flex-wrap">
          <label className="block">
            <span className="text-xs text-text-muted">Start</span>
            <input
              type="datetime-local"
              value={timeStart}
              onChange={(e) => setTimeStart(e.target.value)}
              className="mt-1 block rounded-lg border border-border px-3 py-1.5 text-sm bg-input-bg text-text"
            />
          </label>
          <label className="block">
            <span className="text-xs text-text-muted">End</span>
            <input
              type="datetime-local"
              value={timeEnd}
              onChange={(e) => setTimeEnd(e.target.value)}
              className="mt-1 block rounded-lg border border-border px-3 py-1.5 text-sm bg-input-bg text-text"
            />
          </label>
          <button
            onClick={handleExecute}
            disabled={executing || !timeStart || !timeEnd}
            className="flex items-center gap-2 px-4 py-2 rounded-lg
              bg-primary text-white hover:bg-primary-dark transition-colors
              disabled:opacity-50 text-sm"
          >
            {executing ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Play size={14} />
            )}
            {executing ? 'Executing...' : 'Execute'}
          </button>
        </div>

        {/* Execution result */}
        {executeResult && (
          <div className="text-xs space-y-1">
            <div className={executeResult.errors.length > 0 ? 'text-status-error-text' : 'text-status-success-text'}>
              {executeResult.steps_completed}/{executeResult.steps_total} steps completed
            </div>
            {executeResult.data_labels.length > 0 && (
              <div className="text-text-muted">
                Data labels: <span className="font-mono text-text">{executeResult.data_labels.join(', ')}</span>
              </div>
            )}
            {executeResult.errors.map((err, i) => (
              <div key={i} className="text-status-error-text font-mono">
                {err.op_id}: {err.error}
              </div>
            ))}
          </div>
        )}

        {error && (
          <div className="text-xs text-status-error-text bg-status-error-bg rounded px-2 py-1">{error}</div>
        )}
      </div>

      {/* Execute result figure */}
      {executeResult?.figure && (
        <div className="bg-panel rounded-xl border border-border p-4 overflow-hidden">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-text">Result</h3>
            <button
              onClick={() => setFullscreenFigure(executeResult.figure!)}
              className="p-1 rounded hover:bg-hover-bg text-text-muted hover:text-text transition-colors"
              aria-label="Expand figure"
              title="Expand"
            >
              <Maximize2 size={14} />
            </button>
          </div>
          <PipelineDAG figure={executeResult.figure} />
        </div>
      )}

      {/* Large execute figure — open in new tab */}
      {executeResult && !executeResult.figure && executeResult.figure_url && (
        <div className="bg-panel rounded-xl border border-border p-6 flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-text">Large result figure</h4>
            <p className="text-xs text-text-muted mt-1">
              This figure is too large to display inline. Open it in a new tab for full interactivity.
            </p>
          </div>
          <a
            href={executeResult.figure_url}
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
  );
}
