import { useEffect, useRef, useState } from 'react';
import { useCatalogStore } from '../../stores/catalogStore';
import * as api from '../../api/client';
import { Search, Download, Loader2 } from 'lucide-react';

interface Props {
  sessionId: string;
}

export function CatalogBrowser({ sessionId }: Props) {
  const {
    missions, datasets, parameters, timeRange,
    selectedMission, selectedDataset,
    loading, error,
    loadMissions, selectMission, selectDataset,
  } = useCatalogStore();

  const [selectedParam, setSelectedParam] = useState('');
  const [timeMin, setTimeMin] = useState('');
  const [timeMax, setTimeMax] = useState('');
  const [fetchStatus, setFetchStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
  const [fetching, setFetching] = useState(false);

  const hasLoadedMissions = useRef(false);
  useEffect(() => {
    if (missions.length === 0 && !hasLoadedMissions.current) {
      hasLoadedMissions.current = true;
      loadMissions();
    }
  }, [missions.length, loadMissions]);

  useEffect(() => {
    if (timeRange?.start) setTimeMin(timeRange.start.split('T')[0]);
    if (timeRange?.stop) setTimeMax(timeRange.stop.split('T')[0]);
  }, [timeRange]);

  const handleFetch = async () => {
    if (!selectedDataset || !selectedParam || !timeMin || !timeMax) return;
    setFetching(true);
    setFetchStatus(null);
    try {
      await api.fetchData(sessionId, selectedDataset, selectedParam, timeMin, timeMax);
      setFetchStatus({ type: 'success', message: `Fetched ${selectedDataset}.${selectedParam}` });
    } catch (err) {
      setFetchStatus({ type: 'error', message: (err as Error).message });
    } finally {
      setFetching(false);
    }
  };

  return (
    <div className="bg-panel rounded-xl border border-border p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Search size={16} className="text-primary" />
        <h2 className="font-medium text-text">Catalog Browser</h2>
      </div>

      {/* Mission selector */}
      <select
        value={selectedMission ?? ''}
        onChange={(e) => e.target.value && selectMission(e.target.value)}
        className="block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
      >
        <option value="">Select mission...</option>
        {missions.map((m) => (
          <option key={m.id} value={m.id}>{m.name}</option>
        ))}
      </select>

      {/* Dataset selector */}
      {datasets.length > 0 && (
        <select
          value={selectedDataset ?? ''}
          onChange={(e) => {
            if (e.target.value) {
              selectDataset(e.target.value);
              setSelectedParam('');
            }
          }}
          className="block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
        >
          <option value="">Select dataset...</option>
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.id} {d.name ? `— ${d.name}` : ''}
            </option>
          ))}
        </select>
      )}

      {/* Parameter selector */}
      {parameters.length > 0 && (
        <select
          value={selectedParam}
          onChange={(e) => setSelectedParam(e.target.value)}
          className="block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
        >
          <option value="">Select parameter...</option>
          {parameters.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name} {p.units ? `[${p.units}]` : ''} {p.description ? `— ${p.description}` : ''}
            </option>
          ))}
        </select>
      )}

      {/* Date range */}
      {selectedDataset && (
        <div className="grid grid-cols-2 gap-2">
          <label className="block">
            <span className="text-xs text-text-muted">Start</span>
            <input
              type="date"
              value={timeMin}
              onChange={(e) => setTimeMin(e.target.value)}
              className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
            />
          </label>
          <label className="block">
            <span className="text-xs text-text-muted">End</span>
            <input
              type="date"
              value={timeMax}
              onChange={(e) => setTimeMax(e.target.value)}
              className="mt-1 block w-full rounded-lg border border-border px-3 py-2 text-sm bg-input-bg text-text"
            />
          </label>
        </div>
      )}

      {/* Fetch button */}
      {selectedParam && (
        <button
          onClick={handleFetch}
          disabled={fetching || !timeMin || !timeMax}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
            bg-primary text-white hover:bg-primary-dark transition-colors
            disabled:opacity-50 text-sm"
        >
          {fetching ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <Download size={14} />
          )}
          Fetch Data
        </button>
      )}

      {/* Status */}
      {fetchStatus && (
        <div className={`text-xs rounded px-2 py-1.5 ${
          fetchStatus.type === 'success'
            ? 'bg-status-success-bg text-status-success-text border border-status-success-border'
            : 'bg-status-error-bg text-status-error-text border border-status-error-border'
        }`}>
          {fetchStatus.message}
        </div>
      )}

      {loading && (
        <div className="flex items-center gap-2 text-xs text-text-muted">
          <Loader2 size={12} className="animate-spin" />
          Loading...
        </div>
      )}

      {error && <div className="text-xs text-status-error-text">{error}</div>}
    </div>
  );
}
