import { useState, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';
import type { DataEntrySummary } from '../../api/types';
import * as api from '../../api/client';

interface Props {
  sessionId: string;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="text-text-muted shrink-0">{label}:</span>
      <span className="text-text">{value}</span>
    </div>
  );
}

function DataEntryCard({ e }: { e: DataEntrySummary }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-surface-elevated rounded px-2 py-1.5 space-y-0.5">
      <div
        className="flex justify-between items-center gap-2 cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
        <ChevronDown
          size={12}
          className={`shrink-0 text-text-muted transition-transform ${expanded ? '' : '-rotate-90'}`}
        />
        <span className="font-mono text-text truncate min-w-0">{e.label}</span>
        <span className="font-mono shrink-0">
          {e.num_points.toLocaleString()} pts
          {(e.memory_bytes ?? 0) > 0 && (
            <span className="text-text-muted ml-1">({formatBytes(e.memory_bytes)})</span>
          )}
        </span>
      </div>
      <div className="flex justify-between items-center gap-2">
        <span className="text-text-muted truncate">
          {e.time_min && e.time_max
            ? `${e.time_min.split('T')[0]} → ${e.time_max.split('T')[0]}`
            : e.units || '\u2014'}
        </span>
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0 ${
          e.source === 'cdf'
            ? 'bg-badge-blue-bg text-badge-blue-text'
            : 'bg-badge-orange-bg text-badge-orange-text'
        }`}>
          {e.source}
        </span>
      </div>

      {expanded && (
        <div className="border-t border-border/50 mt-1 pt-1 space-y-0.5 text-[11px]">
          <DetailRow label="Name" value={<span className="font-mono break-all">{e.label}</span>} />
          {e.id && <DetailRow label="ID" value={<span className="font-mono">{e.id}</span>} />}
          {e.shape && <DetailRow label="Shape" value={e.shape} />}
          {e.units && <DetailRow label="Units" value={e.units} />}
          {e.description && <DetailRow label="Description" value={e.description} />}
          <DetailRow label="Timeseries" value={e.is_timeseries ? 'Yes' : 'No'} />
          {e.columns && e.columns.length > 0 && (
            <div className="space-y-0.5">
              <span className="text-text-muted">Columns:</span>
              <div className="flex flex-wrap gap-1 mt-0.5">
                {e.columns.map((col) => (
                  <span
                    key={col}
                    className="font-mono text-[10px] bg-surface px-1.5 py-0.5 rounded text-text"
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
          )}
          {e.dims && Object.keys(e.dims).length > 0 && (
            <div className="space-y-0.5">
              <span className="text-text-muted">Dimensions:</span>
              <div className="flex flex-wrap gap-1 mt-0.5">
                {Object.entries(e.dims).map(([dim, size]) => (
                  <span
                    key={dim}
                    className="font-mono text-[10px] bg-surface px-1.5 py-0.5 rounded text-text"
                  >
                    {dim}: {size.toLocaleString()}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function DataStore({ sessionId }: Props) {
  const [entries, setEntries] = useState<DataEntrySummary[]>([]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await api.getData(sessionId);
        if (!cancelled) setEntries(data);
      } catch {
        // ignore
      }
    };
    load();
    const interval = setInterval(load, 4000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [sessionId]);

  const totalBytes = entries.reduce((s, e) => s + (e.memory_bytes ?? 0), 0);

  return (
    <div className="text-xs text-text-muted space-y-1">
      <div className="flex justify-between">
        <span className="font-medium">Entries</span>
        <span className="font-mono">
          {entries.length}
          {totalBytes > 0 && (
            <span className="text-text-muted ml-1">({formatBytes(totalBytes)})</span>
          )}
        </span>
      </div>

      {entries.length > 0 && (
        <div className="space-y-1.5 max-h-60 overflow-y-auto mt-1">
          {entries.map((e) => (
            <DataEntryCard key={e.id || e.label} e={e} />
          ))}
        </div>
      )}
    </div>
  );
}
