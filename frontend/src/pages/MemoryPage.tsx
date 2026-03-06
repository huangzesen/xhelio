import { useState, useEffect, useCallback, useMemo } from 'react';
import { Brain, ShieldCheck, Star } from 'lucide-react';
import { useSessionStore } from '../stores/sessionStore';
import { useMemoryStore } from '../stores/memoryStore';
import { MemoryStatsBar } from '../components/memory/MemoryStatsBar';
import { MemoryFilters } from '../components/memory/MemoryFilters';
import { MemoryCardList } from '../components/memory/MemoryCardList';
import { MemoryTimeline } from '../components/memory/MemoryTimeline';
import { ArchiveBrowser } from '../components/memory/ArchiveBrowser';
import { MemoryDashboardSkeleton } from '../components/memory/MemoryDashboardSkeleton';
import { ValidationViewer } from '../components/memory/ValidationViewer';
import { RecentReviews } from '../components/memory/RecentReviews';

export function MemoryPage() {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const [activeTab, setActiveTab] = useState<'memories' | 'reviews' | 'validation'>('memories');
  const {
    memories,
    globalEnabled,
    loading,
    error,
    stats,
    searchQuery,
    searchResults,
    searchLoading,
    activeType,
    activeScopes,
    sortBy,
    sortDirection,
    loadMemories,
    setSearchQuery,
    searchMemories,
    clearSearch,
    setActiveType,
    toggleScope,
    setSortBy,
    ratingTier,
    setRatingTier,
    toggleSortDirection,
  } = useMemoryStore();

  useEffect(() => {
    if (activeSessionId) {
      loadMemories(activeSessionId);
    }
  }, [activeSessionId, loadMemories]);

  // Exclude review-type entries from the displayed count
  const memoryCount = useMemo(
    () => memories.filter((m) => m.type !== 'review').length,
    [memories],
  );

  const reviewCount = useMemo(
    () => memories.filter((m) => m.type === 'review').length,
    [memories],
  );

  const handleSearch = useCallback(
    (query: string) => {
      if (!activeSessionId) return;
      if (!query.trim()) {
        clearSearch();
        return;
      }
      searchMemories(activeSessionId, query);
    },
    [activeSessionId, searchMemories, clearSearch],
  );

  return (
    <div className="flex-1 overflow-y-auto p-4 bg-surface">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-semibold text-text">
            {activeTab === 'memories'
              ? 'Agent Memory'
              : activeTab === 'reviews'
                ? 'Agent Reviews'
                : 'Data Validation'}
          </h1>
          {activeTab === 'memories' && activeSessionId && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-text-muted">
                ({memoryCount} {memoryCount === 1 ? 'entry' : 'entries'})
              </span>
              <span
                className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                  globalEnabled
                    ? 'bg-badge-green-bg text-badge-green-text'
                    : 'bg-badge-red-bg text-badge-red-text'
                }`}
              >
                {globalEnabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          )}
        </div>

        {/* Tab bar */}
        <div className="flex border-b border-border mb-6">
          <button
            onClick={() => setActiveTab('memories')}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === 'memories'
                ? 'text-text border-b-2 border-accent'
                : 'text-text-muted hover:text-text'
            }`}
          >
            <Brain size={14} />
            Memories
          </button>
          <button
            onClick={() => setActiveTab('reviews')}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === 'reviews'
                ? 'text-text border-b-2 border-accent'
                : 'text-text-muted hover:text-text'
            }`}
          >
            <Star size={14} />
            Reviews
            {reviewCount > 0 && (
              <span className="text-[10px] text-text-muted">({reviewCount})</span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('validation')}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === 'validation'
                ? 'text-text border-b-2 border-accent'
                : 'text-text-muted hover:text-text'
            }`}
          >
            <ShieldCheck size={14} />
            Validation
          </button>
        </div>

        {/* Validation tab (session-independent) */}
        {activeTab === 'validation' && <ValidationViewer />}

        {/* Reviews tab */}
        {activeTab === 'reviews' && (
          !activeSessionId ? (
            <div className="flex items-center justify-center py-16 text-text-muted">
              <p>No active session. Start a chat first.</p>
            </div>
          ) : loading ? (
            <MemoryDashboardSkeleton />
          ) : (
            <RecentReviews memories={memories} />
          )
        )}

        {/* Memories tab */}
        {activeTab === 'memories' && (
          <>
            {!activeSessionId ? (
              <div className="flex items-center justify-center py-16 text-text-muted">
                <p>No active session. Start a chat first.</p>
              </div>
            ) : (
              <>
                {/* Error state */}
                {error && (
                  <div className="text-sm text-status-error-text bg-status-error-bg border border-status-error-border rounded-lg p-3 mb-4">
                    {error}
                  </div>
                )}

                {/* Loading state */}
                {loading && <MemoryDashboardSkeleton />}

                {/* Empty state */}
                {!loading && memories.length === 0 && (
                  <div className="text-center py-16 text-text-muted">
                    <Brain size={40} className="mx-auto mb-3 opacity-30" />
                    <p className="text-sm">No memories stored yet.</p>
                    <p className="text-xs mt-1">
                      Memories are extracted from conversations as you interact with the agent.
                    </p>
                  </div>
                )}

                {/* Dashboard content */}
                {!loading && memories.length > 0 && (
                  <>
                    {/* Stats bar */}
                    {stats && (
                      <MemoryStatsBar stats={stats} totalMemories={memoryCount} memories={memories} ratingTier={ratingTier} onRatingTierChange={setRatingTier} />
                    )}

                    {/* Filters */}
                    <MemoryFilters
                      stats={stats}
                      searchQuery={searchQuery}
                      activeType={activeType}
                      activeScopes={activeScopes}
                      sortBy={sortBy}
                      sortDirection={sortDirection}
                      onSearchChange={setSearchQuery}
                      onSearch={handleSearch}
                      onTypeChange={setActiveType}
                      onScopeToggle={toggleScope}
                      onSortChange={setSortBy}
                      onToggleSortDirection={toggleSortDirection}
                    />

                    {/* Search loading indicator */}
                    {searchLoading && (
                      <div className="text-xs text-text-muted mb-2">Searching...</div>
                    )}

                    {/* Card list */}
                    <MemoryCardList
                      memories={memories}
                      searchResults={searchResults}
                      searchQuery={searchQuery}
                      activeType={activeType}
                      activeScopes={activeScopes}
                      sortBy={sortBy}
                      sortDirection={sortDirection}
                      ratingTier={ratingTier}
                      sessionId={activeSessionId}
                    />

                    {/* Timeline */}
                    <MemoryTimeline memories={memories} />

                    {/* Archive */}
                    <ArchiveBrowser sessionId={activeSessionId} />
                  </>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
