import { useState, useEffect } from 'react';
import type { DataEntrySummary, DataPreview as DataPreviewType } from '../../api/types';
import * as api from '../../api/client';
import { Table } from 'lucide-react';

interface Props {
  sessionId: string;
}

export function DataPreview({ sessionId }: Props) {
  const [entries, setEntries] = useState<DataEntrySummary[]>([]);
  const [selectedLabel, setSelectedLabel] = useState('');
  const [preview, setPreview] = useState<DataPreviewType | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getData(sessionId).then(setEntries).catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    if (!selectedLabel) { setPreview(null); return; }
    setLoading(true);
    api.getDataPreview(sessionId, selectedLabel)
      .then(setPreview)
      .catch(() => setPreview(null))
      .finally(() => setLoading(false));
  }, [sessionId, selectedLabel]);

  return (
    <div className="bg-panel rounded-xl border border-border p-4">
      <div className="flex items-center gap-2 mb-3">
        <Table size={16} className="text-primary" />
        <h2 className="font-medium text-text">Data Preview</h2>
      </div>

      <select
        value={selectedLabel}
        onChange={(e) => setSelectedLabel(e.target.value)}
        className="block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text mb-3"
      >
        <option value="">Select data entry...</option>
        {entries.map((e) => (
          <option key={e.label} value={e.label}>{e.label}</option>
        ))}
      </select>

      {loading && <div className="text-xs text-text-muted text-center py-4">Loading...</div>}

      {preview && !loading && (
        <div className="overflow-x-auto">
          <div className="text-xs text-text-muted mb-2">
            {preview.total_rows.toLocaleString()} rows total
            {preview.total_rows > 20 && ' (showing first 10 + last 10)'}
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-text-muted border-b border-border">
                <th className="pb-1 pr-2">Time</th>
                {preview.columns.map((col) => (
                  <th key={col} className="pb-1 pr-2">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, i) => (
                <tr key={i} className="border-b border-border/30">
                  <td className="py-1 pr-2 text-text-muted font-mono">{String(row._index).slice(0, 19)}</td>
                  {preview.columns.map((col) => (
                    <td key={col} className="py-1 pr-2 font-mono">
                      {row[col] == null ? '—' : typeof row[col] === 'number'
                        ? (row[col] as number).toFixed(4)
                        : String(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
