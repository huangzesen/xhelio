import { useState, useMemo } from 'react';
import type { SavedPipelineIndexEntry } from '../../api/types';
import { Inbox, Search, X, ArrowUp, ArrowDown, ChevronUp, ChevronDown } from 'lucide-react';

type SortField = 'name' | 'steps' | 'created';

interface Props {
  pipelines: SavedPipelineIndexEntry[];
  onSelect: (id: string) => void;
}

/** Extract mission-level scope from a dataset ID (e.g. "AC_H2_MFI" → "ACE", "PSP_FLD_L2_MAG_RTN" → "PSP") */
function extractScope(dataset: string): string {
  // Common prefixes that map to mission names
  const prefixMap: Record<string, string> = {
    AC: 'ACE',
    WI: 'WIND',
    TH: 'THEMIS',
    MMS: 'MMS',
    PSP: 'PSP',
    SO: 'SOLO',
    VG1: 'VOYAGER1',
    VG2: 'VOYAGER2',
    ULY: 'ULYSSES',
    JNO: 'JUNO',
    DSCOVR: 'DSCOVR',
    STA: 'STEREO-A',
    STB: 'STEREO-B',
    OMNI: 'OMNI',
  };
  const upper = dataset.toUpperCase();
  // Try longest prefix first for correct matching
  const sortedPrefixes = Object.keys(prefixMap).sort((a, b) => b.length - a.length);
  for (const prefix of sortedPrefixes) {
    if (upper.startsWith(prefix + '_') || upper === prefix) {
      return prefixMap[prefix];
    }
  }
  // Fallback: take first segment before underscore or dot
  const stem = dataset.split(/[._]/)[0].toUpperCase();
  return stem;
}

const sortOptions: { label: string; value: SortField }[] = [
  { label: 'Name', value: 'name' },
  { label: 'Steps', value: 'steps' },
  { label: 'Created', value: 'created' },
];

export function SavedPipelineList({ pipelines, onSelect }: Props) {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeScopes, setActiveScopes] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<SortField>('created');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Deduplicated, sorted mission scopes from all pipelines
  const allScopes = useMemo(() => {
    const scopes = new Set<string>();
    for (const p of pipelines) {
      for (const ds of p.datasets) {
        scopes.add(extractScope(ds));
      }
    }
    return [...scopes].sort();
  }, [pipelines]);

  // Filter + sort
  const filteredPipelines = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();

    let result = pipelines.filter((p) => {
      // Search filter: match name, description, or any tag
      if (q) {
        const matchesSearch =
          p.name.toLowerCase().includes(q) ||
          p.description.toLowerCase().includes(q) ||
          p.tags.some((t) => t.toLowerCase().includes(q));
        if (!matchesSearch) return false;
      }

      // Scope filter: pipeline must have at least one dataset matching any active scope
      if (activeScopes.length > 0) {
        const pipelineScopes = p.datasets.map(extractScope);
        const matchesScope = activeScopes.some((s) => pipelineScopes.includes(s));
        if (!matchesScope) return false;
      }

      return true;
    });

    // Sort
    result = [...result].sort((a, b) => {
      let cmp = 0;
      switch (sortBy) {
        case 'name':
          cmp = a.name.localeCompare(b.name);
          break;
        case 'steps':
          cmp = a.step_count - b.step_count;
          break;
        case 'created':
          cmp = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          break;
      }
      return sortDirection === 'asc' ? cmp : -cmp;
    });

    return result;
  }, [pipelines, searchQuery, activeScopes, sortBy, sortDirection]);

  const handleScopeToggle = (scope: string) => {
    setActiveScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope],
    );
  };

  const handleColumnSort = (field: SortField) => {
    if (sortBy === field) {
      setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(field);
      setSortDirection(field === 'name' ? 'asc' : 'desc');
    }
  };

  // No pipelines at all — show empty state (no filter bar)
  if (pipelines.length === 0) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center space-y-3">
          <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
            <Inbox size={24} className="text-primary" />
          </div>
          <h3 className="text-base font-medium text-text">No saved pipelines</h3>
          <p className="text-sm text-text-muted max-w-xs">
            Pipelines are registered from session operations.
            Use the Replay tab to explore sessions, then register pipelines from there.
          </p>
        </div>
      </div>
    );
  }

  const SortIndicator = ({ field }: { field: SortField }) => {
    if (sortBy !== field) return null;
    return sortDirection === 'asc'
      ? <ChevronUp size={12} className="inline ml-0.5" />
      : <ChevronDown size={12} className="inline ml-0.5" />;
  };

  return (
    <div>
      {/* Filter bar */}
      <div className="space-y-3 mb-4">
        {/* Search bar */}
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search pipelines..."
            className="w-full pl-9 pr-8 py-2 rounded-lg border border-border text-sm bg-input-bg text-text placeholder:text-text-muted focus:outline-none focus:border-primary transition-colors"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text transition-colors"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Scope chips + Sort controls */}
        <div className="flex items-start justify-between gap-3">
          {/* Scope chips */}
          {allScopes.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {allScopes.map((scope) => {
                const isActive = activeScopes.includes(scope);
                return (
                  <button
                    key={scope}
                    onClick={() => handleScopeToggle(scope)}
                    className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
                      isActive
                        ? 'bg-primary/20 text-primary border-primary'
                        : 'bg-badge-gray-bg text-badge-gray-text border-transparent hover:border-border'
                    }`}
                  >
                    {scope}
                  </button>
                );
              })}
            </div>
          )}

          {/* Sort controls */}
          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={() => setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'))}
              className="p-1 rounded text-text-muted hover:text-text transition-colors"
              title={sortDirection === 'desc' ? 'Descending (click to toggle)' : 'Ascending (click to toggle)'}
            >
              {sortDirection === 'desc' ? <ArrowDown size={12} /> : <ArrowUp size={12} />}
            </button>
            {sortOptions.map((opt) => (
              <button
                key={opt.value}
                onClick={() => {
                  if (sortBy === opt.value) {
                    setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'));
                  } else {
                    setSortBy(opt.value);
                    setSortDirection(opt.value === 'name' ? 'asc' : 'desc');
                  }
                }}
                className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                  sortBy === opt.value
                    ? 'bg-primary/15 text-primary'
                    : 'text-text-muted hover:text-text'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Table */}
      {filteredPipelines.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <div className="text-center space-y-2">
            <p className="text-sm text-text-muted">No pipelines match your filters.</p>
            <button
              onClick={() => { setSearchQuery(''); setActiveScopes([]); }}
              className="text-xs text-primary hover:text-primary/80 transition-colors"
            >
              Clear filters
            </button>
          </div>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs table-fixed">
            <thead>
              <tr className="text-left text-text-muted border-b border-border">
                <th
                  className="pb-2 pr-2 w-[180px] cursor-pointer select-none hover:text-text transition-colors"
                  onClick={() => handleColumnSort('name')}
                >
                  Name<SortIndicator field="name" />
                </th>
                <th className="pb-2 pr-2">Description</th>
                <th className="pb-2 pr-2 w-[140px]">Tags</th>
                <th
                  className="pb-2 pr-2 w-[50px] text-right cursor-pointer select-none hover:text-text transition-colors"
                  onClick={() => handleColumnSort('steps')}
                >
                  Steps<SortIndicator field="steps" />
                </th>
                <th className="pb-2 pr-2 w-[160px]">Datasets</th>
                <th
                  className="pb-2 w-[160px] cursor-pointer select-none hover:text-text transition-colors"
                  onClick={() => handleColumnSort('created')}
                >
                  Created<SortIndicator field="created" />
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredPipelines.map((p) => (
                <tr
                  key={p.id}
                  onClick={() => onSelect(p.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelect(p.id);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  className="border-b border-border/50 cursor-pointer transition-colors hover:bg-hover-bg
                    focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-inset"
                >
                  <td className="py-2 pr-2 font-medium text-text">
                    <div className="truncate">{p.name}</div>
                  </td>
                  <td className="py-2 pr-2 text-text-muted">
                    <div className="truncate">{p.description || '—'}</div>
                  </td>
                  <td className="py-2 pr-2">
                    <div className="flex gap-1 flex-wrap">
                      {p.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-gray-bg text-badge-gray-text"
                        >
                          {tag}
                        </span>
                      ))}
                      {p.tags.length > 3 && (
                        <span className="text-[10px] text-text-muted">+{p.tags.length - 3}</span>
                      )}
                    </div>
                  </td>
                  <td className="py-2 pr-2 text-right font-mono text-text">{p.step_count}</td>
                  <td className="py-2 pr-2 font-mono text-text-muted">
                    <div className="truncate">{p.datasets.join(', ') || '—'}</div>
                  </td>
                  <td className="py-2 text-text-muted">
                    {new Date(p.created_at).toLocaleString(undefined, {
                      year: 'numeric', month: 'short', day: 'numeric',
                      hour: '2-digit', minute: '2-digit',
                    })}
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
