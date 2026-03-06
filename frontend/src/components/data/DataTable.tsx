import { useEffect, useState } from 'react';
import type { DataEntrySummary } from '../../api/types';
import * as api from '../../api/client';
import { Database } from 'lucide-react';

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

export function DataTable({ sessionId }: Props) {
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

  const totalBytes = entries.reduce((sum, e) => sum + (e.memory_bytes ?? 0), 0);

  return (
    <div className="bg-panel rounded-xl border border-border p-4">
      <div className="flex items-center gap-2 mb-3">
        <Database size={16} className="text-primary" />
        <h2 className="font-medium text-text">Data Store</h2>
        <span className="text-xs text-text-muted">
          ({entries.length} {entries.length === 1 ? 'entry' : 'entries'}{totalBytes > 0 ? `, ${formatBytes(totalBytes)}` : ''})
        </span>
      </div>

      {entries.length === 0 ? (
        <div className="text-xs text-text-muted text-center py-4">No data fetched yet</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-text-muted border-b border-border">
                <th className="pb-2 pr-3">Label</th>
                <th className="pb-2 pr-3">Points</th>
                <th className="pb-2 pr-3">Size</th>
                <th className="pb-2 pr-3">Units</th>
                <th className="pb-2 pr-3">Time Range</th>
                <th className="pb-2">Source</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr key={e.label} className="border-b border-border/50 hover:bg-hover-bg">
                  <td className="py-1.5 pr-3 font-mono text-text">{e.label}</td>
                  <td className="py-1.5 pr-3">{e.num_points.toLocaleString()}</td>
                  <td className="py-1.5 pr-3 font-mono text-text-muted">{formatBytes(e.memory_bytes ?? 0)}</td>
                  <td className="py-1.5 pr-3">{e.units || '—'}</td>
                  <td className="py-1.5 pr-3 text-text-muted">
                    {e.time_min && e.time_max
                      ? `${e.time_min.split('T')[0]} → ${e.time_max.split('T')[0]}`
                      : '—'}
                  </td>
                  <td className="py-1.5">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                      e.source === 'cdf' ? 'bg-badge-blue-bg text-badge-blue-text' : 'bg-badge-orange-bg text-badge-orange-text'
                    }`}>
                      {e.source}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
