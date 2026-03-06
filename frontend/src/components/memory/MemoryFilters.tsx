import { useEffect, useRef, useCallback } from 'react';
import { Search, X, ArrowUp, ArrowDown } from 'lucide-react';
import type { MemoryStats } from '../../api/types';

const typeLabels = ['All', 'Preference', 'Pitfall', 'Summary', 'Reflection'] as const;
const typeValues: Record<string, string | null> = {
  All: null,
  Preference: 'preference',
  Pitfall: 'pitfall',
  Summary: 'summary',
  Reflection: 'reflection',
};

const sortOptions: { label: string; value: 'recency' | 'rating' | 'access_count' | 'reviews' }[] = [
  { label: 'Recent', value: 'recency' },
  { label: 'Rating', value: 'rating' },
  { label: 'Reviews', value: 'reviews' },
  { label: 'Access', value: 'access_count' },
];

interface Props {
  stats: MemoryStats | null;
  searchQuery: string;
  activeType: string | null;
  activeScopes: string[];
  sortBy: 'recency' | 'rating' | 'access_count' | 'reviews';
  sortDirection: 'desc' | 'asc';
  onSearchChange: (query: string) => void;
  onSearch: (query: string) => void;
  onTypeChange: (type: string | null) => void;
  onScopeToggle: (scope: string) => void;
  onSortChange: (sortBy: 'recency' | 'rating' | 'access_count' | 'reviews') => void;
  onToggleSortDirection: () => void;
}

export function MemoryFilters({
  stats,
  searchQuery,
  activeType,
  activeScopes,
  sortBy,
  sortDirection,
  onSearchChange,
  onSearch,
  onTypeChange,
  onScopeToggle,
  onSortChange,
  onToggleSortDirection,
}: Props) {
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  const cancelDebounce = useCallback(() => {
    clearTimeout(debounceRef.current);
  }, []);

  const handleSearchInput = useCallback((value: string) => {
    onSearchChange(value);
    cancelDebounce();
    debounceRef.current = setTimeout(() => {
      onSearch(value);
    }, 300);
  }, [onSearchChange, onSearch, cancelDebounce]);

  const handleClear = useCallback(() => {
    cancelDebounce();
    onSearchChange('');
    onSearch('');
  }, [cancelDebounce, onSearchChange, onSearch]);

  useEffect(() => {
    return cancelDebounce;
  }, [cancelDebounce]);

  const allScopes = stats?.all_scopes ?? [];

  return (
    <div className="space-y-3 mb-4">
      {/* Search bar */}
      <div className="relative">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => handleSearchInput(e.target.value)}
          placeholder="Search memories..."
          className="w-full pl-9 pr-8 py-2 rounded-lg border border-border text-sm bg-input-bg text-text placeholder:text-text-muted focus:outline-none focus:border-primary transition-colors"
        />
        {searchQuery && (
          <button
            onClick={handleClear}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text transition-colors"
          >
            <X size={14} />
          </button>
        )}
      </div>

      {/* Scope chips */}
      {allScopes.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {allScopes.map((scope) => {
            const isActive = activeScopes.includes(scope);
            return (
              <button
                key={scope}
                onClick={() => onScopeToggle(scope)}
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

      {/* Type tabs + Sort */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex border-b border-border">
          {typeLabels.map((label) => {
            const value = typeValues[label];
            const isActive = activeType === value;
            return (
              <button
                key={label}
                onClick={() => onTypeChange(value)}
                className={`px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap ${
                  isActive
                    ? 'text-text border-b-2 border-accent -mb-px'
                    : 'text-text-muted hover:text-text'
                }`}
              >
                {label}
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={onToggleSortDirection}
            className="p-1 rounded text-text-muted hover:text-text transition-colors"
            title={sortDirection === 'desc' ? 'Descending (click to toggle)' : 'Ascending (click to toggle)'}
          >
            {sortDirection === 'desc' ? <ArrowDown size={12} /> : <ArrowUp size={12} />}
          </button>
          {sortOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => onSortChange(opt.value)}
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
  );
}
