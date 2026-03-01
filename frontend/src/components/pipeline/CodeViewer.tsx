import { Highlight, themes } from 'prism-react-renderer';
import type { PipelineRecord } from '../../api/types';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface Props {
  record: PipelineRecord;
  allRecords: PipelineRecord[];
  onNavigate: (record: PipelineRecord) => void;
}

const toolColor: Record<string, string> = {
  fetch_data: 'bg-badge-blue-bg text-badge-blue-text',
  custom_operation: 'bg-badge-orange-bg text-badge-orange-text',
  render_plotly_json: 'bg-badge-pink-bg text-badge-pink-text',
  manage_plot: 'bg-badge-gray-bg text-badge-gray-text',
  store_dataframe: 'bg-badge-teal-bg text-badge-teal-text',
};

export function CodeViewer({ record, allRecords, onNavigate }: Props) {
  // Get compute steps for prev/next navigation
  const computeSteps = allRecords.filter(
    (r) => r.tool === 'custom_operation' || r.tool === 'render_plotly_json'
  );
  const currentIdx = computeSteps.findIndex((r) => r.id === record.id);
  const hasPrev = currentIdx > 0;
  const hasNext = currentIdx >= 0 && currentIdx < computeSteps.length - 1;

  // Extract code from args — ensure it's always a string
  const rawCode = record.args?.code;
  const rawFigure = record.args?.figure_json;
  const code =
    (typeof rawCode === 'string' ? rawCode : null) ??
    (typeof rawFigure === 'string' ? rawFigure : rawFigure ? JSON.stringify(rawFigure, null, 2) : null) ??
    JSON.stringify(record.args, null, 2);

  const isCode = record.tool === 'custom_operation';

  return (
    <div className="min-w-0">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm text-text">{record.id}</span>
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
            toolColor[record.tool] ?? 'bg-badge-gray-bg text-badge-gray-text'
          }`}>
            {record.tool}
          </span>
          {record.status === 'success' ? (
            <span className="text-status-success-text text-xs">OK</span>
          ) : (
            <span className="text-status-error-text text-xs">ERR</span>
          )}
        </div>

        {computeSteps.length > 0 && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => hasPrev && onNavigate(computeSteps[currentIdx - 1])}
              disabled={!hasPrev}
              className="p-1 rounded hover:bg-hover-bg disabled:opacity-30 text-text-muted"
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={() => hasNext && onNavigate(computeSteps[currentIdx + 1])}
              disabled={!hasNext}
              className="p-1 rounded hover:bg-hover-bg disabled:opacity-30 text-text-muted"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        )}
      </div>

      {/* Metadata */}
      <div className="text-xs text-text-muted space-y-1 mb-3">
        {'description' in (record.args ?? {}) && (
          <div><span className="text-text font-medium">Description:</span> {String(record.args.description)}</div>
        )}
        {record.inputs && record.inputs.length > 0 && (
          <div><span className="text-text font-medium">Inputs:</span> {record.inputs.join(', ')}</div>
        )}
        {record.outputs && record.outputs.length > 0 && (
          <div><span className="text-text font-medium">Outputs:</span> {record.outputs.join(', ')}</div>
        )}
        {'units' in (record.args ?? {}) && (
          <div><span className="text-text font-medium">Units:</span> {String(record.args.units)}</div>
        )}
      </div>

      {/* Code block */}
      <Highlight theme={themes.nightOwl} code={code} language={isCode ? 'python' : 'json'}>
        {({ style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className="rounded-lg p-3 text-xs overflow-auto max-h-[70vh]"
            style={{ ...style, margin: 0 }}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                <span className="select-none text-text-muted mr-3 inline-block w-6 text-right">
                  {i + 1}
                </span>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>

      {record.error && (
        <div className="mt-2 text-xs text-status-error-text bg-status-error-bg rounded p-2 font-mono">
          {record.error}
        </div>
      )}
    </div>
  );
}
