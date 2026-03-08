import type { PipelineRecord } from '../../api/types';
import { TOOL_COLORS } from '../../constants/toolColors';

interface Props {
  records: PipelineRecord[];
  selectedId: string | null;
  onSelect: (record: PipelineRecord) => void;
}

export function StepTable({ records, selectedId, onSelect }: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs table-fixed">
        <thead>
          <tr className="text-left text-text-muted border-b border-border">
            <th className="pb-2 pr-2 w-[40px]">#</th>
            <th className="pb-2 pr-2 w-[80px]">Op ID</th>
            <th className="pb-2 pr-2 w-[120px]">Tool</th>
            <th className="pb-2 pr-2 w-[120px]">Inputs</th>
            <th className="pb-2 pr-2 w-[120px]">Outputs</th>
            <th className="pb-2">Description</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r, i) => (
            <tr
              key={r.id}
              onClick={() => onSelect(r)}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(r); } }}
              tabIndex={0}
              role="button"
              aria-selected={selectedId === r.id}
              className={`border-b border-border/50 cursor-pointer transition-colors focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-inset ${
                selectedId === r.id ? 'bg-primary/10' : 'hover:bg-hover-bg'
              }`}
            >
              <td className="py-1.5 pr-2 text-text-muted">{i + 1}</td>
              <td className="py-1.5 pr-2 font-mono text-text"><div className="truncate">{r.id}</div></td>
              <td className="py-1.5 pr-2">
                <div className="truncate">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                    TOOL_COLORS[r.tool] ?? 'bg-badge-gray-bg text-badge-gray-text'
                  }`}>
                    {r.tool}
                  </span>
                </div>
              </td>
              <td className="py-1.5 pr-2 font-mono text-text-muted">
                <div className="truncate">{r.inputs?.join(', ') || '—'}</div>
              </td>
              <td className="py-1.5 pr-2 font-mono text-text">
                <div className="truncate">{r.outputs?.join(', ') || '—'}</div>
              </td>
              <td className="py-1.5 text-text-muted">
                <div className="truncate">{(r.args?.description as string) ?? '—'}</div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
