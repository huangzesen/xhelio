import { useEffect, useMemo } from 'react';
import { useEurekaStore } from '../stores/eurekaStore';
import { Loader2 } from 'lucide-react';

const STATUS_OPTIONS = ['proposed', 'reviewed', 'confirmed', 'rejected'] as const;

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    value < 0.4 ? 'bg-gray-400' : value < 0.7 ? 'bg-blue-500' : 'bg-green-500';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-text-muted w-10">{value.toFixed(1)}</span>
    </div>
  );
}

function EurekaCard({
  eureka,
  onStatusChange,
}: {
  eureka: import('../api/types').EurekaEntry;
  onStatusChange: (id: string, status: string) => void;
}) {
  const statusColor = {
    proposed: 'bg-yellow-500/20 text-yellow-400',
    reviewed: 'bg-blue-500/20 text-blue-400',
    confirmed: 'bg-green-500/20 text-green-400',
    rejected: 'bg-red-500/20 text-red-400',
  };

  return (
    <div className="bg-panel rounded-xl border border-border p-4 space-y-3">
      <h3 className="text-base font-medium text-text">{eureka.title}</h3>

      <div className="space-y-2 text-sm">
        <div>
          <span className="text-text-muted">Observation: </span>
          <span className="text-text">{eureka.observation}</span>
        </div>
        <div>
          <span className="text-text-muted">Hypothesis: </span>
          <span className="text-text">{eureka.hypothesis}</span>
        </div>
        {eureka.evidence.length > 0 && (
          <div>
            <span className="text-text-muted">Evidence:</span>
            <ul className="mt-1 ml-4 list-disc text-text">
              {eureka.evidence.map((ev, i) => (
                <li key={i}>{ev}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <ConfidenceBar value={eureka.confidence} />

      <div className="flex items-center gap-2 flex-wrap">
        {eureka.tags.map((tag) => (
          <span
            key={tag}
            className="px-2 py-0.5 text-xs bg-surface rounded-full text-text-muted"
          >
            {tag}
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between text-xs text-text-muted">
        <span>
          Session:{' '}
          {new Date(eureka.timestamp).toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
        <select
          value={eureka.status}
          onChange={(e) => onStatusChange(eureka.id, e.target.value)}
          className={`px-2 py-1 rounded text-xs border border-border bg-surface cursor-pointer ${statusColor[eureka.status]}`}
        >
          {STATUS_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

export function EurekaPage() {
  const { loading, filters, fetchEurekas, updateStatus, setFilter, filteredEurekas } =
    useEurekaStore();

  const displayed = useMemo(() => filteredEurekas(), [filteredEurekas]);

  useEffect(() => {
    fetchEurekas();
  }, [fetchEurekas]);

  const allTags = useMemo(() => {
    const store = useEurekaStore.getState();
    const tags = new Set<string>();
    store.eurekas.forEach((e) => e.tags.forEach((t) => tags.add(t)));
    return Array.from(tags).sort();
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-surface">
        <Loader2 size={20} className="animate-spin text-text-muted" />
        <span className="ml-2 text-text-muted text-sm">Loading eurekas...</span>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-surface">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-xl font-semibold text-text">Eureka Findings</h1>
          <div className="flex items-center gap-2">
            <select
              value={filters.status || ''}
              onChange={(e) => setFilter('status', e.target.value || undefined)}
              className="px-3 py-1.5 rounded-lg border border-border bg-panel text-sm text-text"
            >
              <option value="">All Status</option>
              {STATUS_OPTIONS.map((s) => (
                <option key={s} value={s}>
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </option>
              ))}
            </select>
            <select
              value={filters.tag || ''}
              onChange={(e) => setFilter('tag', e.target.value || undefined)}
              className="px-3 py-1.5 rounded-lg border border-border bg-panel text-sm text-text"
            >
              <option value="">All Tags</option>
              {allTags.map((t: string) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
        </div>

        {displayed.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <p className="text-text-muted text-sm">
              No eurekas yet. Start a session and explore data to generate discoveries.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {displayed.map((eureka) => (
              <EurekaCard
                key={eureka.id}
                eureka={eureka}
                onStatusChange={updateStatus}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
