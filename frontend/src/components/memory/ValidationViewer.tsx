import { useState, useEffect } from 'react';
import { ShieldCheck, Loader2 } from 'lucide-react';
import { getValidationOverview } from '../../api/client';
import type { ValidationOverview } from '../../api/types';
import { MissionValidationCard } from './MissionValidationCard';

export function ValidationViewer() {
  const [data, setData] = useState<ValidationOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getValidationOverview()
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || 'Failed to load validation data');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  // Loading
  if (loading) {
    return (
      <div className="flex items-center gap-2 py-12 justify-center text-text-muted text-sm">
        <Loader2 size={16} className="animate-spin" />
        Loading validation data...
      </div>
    );
  }

  // Error
  if (error) {
    return (
      <div className="text-sm text-status-error-text bg-status-error-bg border border-status-error-border rounded-lg p-3">
        {error}
      </div>
    );
  }

  // Empty
  if (!data || data.missions.length === 0) {
    return (
      <div className="text-center py-16 text-text-muted">
        <ShieldCheck size={40} className="mx-auto mb-3 opacity-30" />
        <p className="text-sm">No validation data yet.</p>
        <p className="text-xs mt-1">
          Validation records are created automatically when datasets are fetched.
        </p>
      </div>
    );
  }

  // Compute global stats
  const totalDatasets = data.missions.reduce((s, m) => s + m.dataset_count, 0);
  const totalValidated = data.missions.reduce((s, m) => s + m.validated_count, 0);
  const totalPhantom = data.missions.reduce((s, m) => s + m.total_phantom, 0);
  const totalUndocumented = data.missions.reduce((s, m) => s + m.total_undocumented, 0);
  const hasIssues = totalPhantom > 0 || totalUndocumented > 0;

  return (
    <div className="space-y-4">
      {/* Global stats bar */}
      <div className="flex flex-wrap items-center gap-3 px-4 py-3 bg-surface-elevated border border-border rounded-lg text-xs">
        <span className="text-text-muted">
          <span className="font-medium text-text">{data.missions.length}</span>{' '}
          {data.missions.length === 1 ? 'mission' : 'missions'}
        </span>
        <span className="text-border">|</span>
        <span className="text-text-muted">
          <span className="font-medium text-text">{totalValidated}</span>/{totalDatasets} datasets validated
        </span>
        <span className="text-border">|</span>
        {hasIssues ? (
          <>
            {totalPhantom > 0 && (
              <span className="px-1.5 py-0.5 rounded bg-badge-red-bg text-badge-red-text font-medium">
                {totalPhantom} phantom
              </span>
            )}
            {totalUndocumented > 0 && (
              <span className="px-1.5 py-0.5 rounded bg-badge-blue-bg text-badge-blue-text font-medium">
                {totalUndocumented} undocumented
              </span>
            )}
          </>
        ) : (
          <span className="px-1.5 py-0.5 rounded bg-badge-green-bg text-badge-green-text font-medium">
            No issues
          </span>
        )}
      </div>

      {/* Mission cards */}
      {data.missions.map((mission) => (
        <MissionValidationCard key={mission.mission_stem} mission={mission} />
      ))}
    </div>
  );
}
