import { useState, useEffect } from 'react';
import { ChevronDown, Image, FileText } from 'lucide-react';
import type { DataEntrySummary, SessionAsset } from '../../api/types';
import * as api from '../../api/client';

interface Props {
  sessionId: string;
}

type FilterKind = 'all' | 'data' | 'figure' | 'file';

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

function KindBadge({ kind }: { kind: string }) {
  const styles: Record<string, string> = {
    data: 'bg-badge-blue-bg text-badge-blue-text',
    figure: 'bg-badge-green-bg text-badge-green-text',
    file: 'bg-badge-orange-bg text-badge-orange-text',
  };
  return (
    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0 ${styles[kind] || 'bg-surface text-text-muted'}`}>
      {kind}
    </span>
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
        <div className="flex items-center gap-1 shrink-0">
          <span className="font-mono">
            {e.num_points.toLocaleString()} pts
            {(e.memory_bytes ?? 0) > 0 && (
              <span className="text-text-muted ml-1">({formatBytes(e.memory_bytes)})</span>
            )}
          </span>
          <KindBadge kind="data" />
        </div>
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

function FigureCard({ asset }: { asset: SessionAsset }) {
  return (
    <div className="bg-surface-elevated rounded px-2 py-1.5 space-y-0.5">
      <div className="flex justify-between items-center gap-2">
        <Image size={12} className="shrink-0 text-text-muted" />
        <span className="font-mono text-text truncate min-w-0">{asset.name}</span>
        <KindBadge kind="figure" />
      </div>
      <div className="flex justify-between items-center gap-2">
        <span className="text-text-muted truncate text-[11px]">
          {[
            asset.metadata.op_id ? `op: ${asset.metadata.op_id}` : null,
            asset.metadata.source_url ? 'external' : null,
            asset.metadata.panel_count ? `${asset.metadata.panel_count} panel${(asset.metadata.panel_count as number) > 1 ? 's' : ''}` : null,
          ].filter(Boolean).join(' · ')}
        </span>
      </div>
    </div>
  );
}

function FileCard({ asset }: { asset: SessionAsset }) {
  return (
    <div className="bg-surface-elevated rounded px-2 py-1.5 space-y-0.5">
      <div className="flex justify-between items-center gap-2">
        <FileText size={12} className="shrink-0 text-text-muted" />
        <span className="font-mono text-text truncate min-w-0">{asset.name}</span>
        <div className="flex items-center gap-1 shrink-0">
          {(asset.metadata.size_bytes as number) > 0 && (
            <span className="font-mono text-text-muted text-[11px]">
              {formatBytes(asset.metadata.size_bytes as number)}
            </span>
          )}
          <KindBadge kind="file" />
        </div>
      </div>
    </div>
  );
}

const FILTER_TABS: { key: FilterKind; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'data', label: 'Data' },
  { key: 'figure', label: 'Figures' },
  { key: 'file', label: 'Files' },
];

export function AssetsPanel({ sessionId }: Props) {
  const [filter, setFilter] = useState<FilterKind>('all');
  const [entries, setEntries] = useState<DataEntrySummary[]>([]);
  const [assets, setAssets] = useState<SessionAsset[]>([]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [data, sessionAssets] = await Promise.all([
          api.getData(sessionId),
          api.getSessionAssets(sessionId),
        ]);
        if (!cancelled) {
          setEntries(Array.isArray(data) ? data : []);
          setAssets(Array.isArray(sessionAssets) ? sessionAssets : []);
        }
      } catch {
        // ignore
      }
    };
    load();
    const interval = setInterval(load, 4000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [sessionId]);

  const figures = assets.filter(a => a.kind === 'figure');
  const files = assets.filter(a => a.kind === 'file');

  const totalCount = entries.length + figures.length + files.length;

  return (
    <div className="text-xs text-text-muted space-y-1">
      {/* Filter tabs */}
      <div className="flex gap-1 mb-1">
        {FILTER_TABS.map(tab => (
          <button
            key={tab.key}
            onClick={() => setFilter(tab.key)}
            className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
              filter === tab.key
                ? 'bg-accent text-white'
                : 'bg-surface-elevated text-text-muted hover:text-text'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex justify-between">
        <span className="font-medium">
          {filter === 'all' ? 'All Assets' : FILTER_TABS.find(t => t.key === filter)?.label}
        </span>
        <span className="font-mono">
          {filter === 'all' ? totalCount
            : filter === 'data' ? entries.length
            : filter === 'figure' ? figures.length
            : files.length}
        </span>
      </div>

      <div className="space-y-1.5 max-h-60 overflow-y-auto mt-1">
        {/* Data entries */}
        {(filter === 'all' || filter === 'data') && entries.map((e, i) => (
          <DataEntryCard key={e.id || `${e.label}-${i}`} e={e} />
        ))}

        {/* Figures */}
        {(filter === 'all' || filter === 'figure') && figures.map(a => (
          <FigureCard key={a.asset_id} asset={a} />
        ))}

        {/* Files */}
        {(filter === 'all' || filter === 'file') && files.map(a => (
          <FileCard key={a.asset_id} asset={a} />
        ))}

        {totalCount === 0 && (
          <div className="text-text-muted italic">No assets yet</div>
        )}
      </div>
    </div>
  );
}
