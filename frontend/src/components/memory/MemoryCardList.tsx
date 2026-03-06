import { useMemo, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { stagger } from '../common/MotionPresets';
import { MemoryCard } from './MemoryCard';
import type { MemoryEntry } from '../../api/types';
import { buildReviewMap, avgReviewStars } from './reviewUtils';

const PAGE_SIZE = 20;

interface Props {
  memories: MemoryEntry[];
  searchResults: MemoryEntry[] | null;
  searchQuery?: string;
  activeType: string | null;
  activeScopes: string[];
  sortBy: 'recency' | 'rating' | 'access_count' | 'reviews';
  sortDirection: 'desc' | 'asc';
  ratingTier: 'all_time' | 'recent';
  sessionId: string;
}

export function MemoryCardList({
  memories,
  searchResults,
  searchQuery,
  activeType,
  activeScopes,
  sortBy,
  sortDirection,
  ratingTier,
  sessionId,
}: Props) {
  // Build review lookup: target memory ID → list of review MemoryEntries
  const reviewMap = useMemo(() => buildReviewMap(memories), [memories]);

  const filtered = useMemo(() => {
    // If searching, use search results as base
    let items = searchResults !== null ? searchResults : memories;

    // If the search query looks like a memory ID (hex string), try exact/prefix match
    const trimmedQuery = searchQuery?.trim().toLowerCase() ?? '';
    if (trimmedQuery && /^[0-9a-f]+$/.test(trimmedQuery)) {
      const idMatches = memories.filter((m) => m.id.startsWith(trimmedQuery));
      if (idMatches.length > 0) {
        items = idMatches;
      }
    }

    // Exclude review-type entries from the main list (they're shown inline on their targets)
    items = items.filter((m) => m.type !== 'review');

    // Filter by type
    if (activeType) {
      items = items.filter((m) => m.type === activeType);
    }

    // Filter by scopes (memory must have at least one matching scope)
    if (activeScopes.length > 0) {
      items = items.filter((m) =>
        m.scopes.some((s) => activeScopes.includes(s)),
      );
    }

    // Sort (always sort descending first, then flip if asc)
    const sorted = [...items];
    const dir = sortDirection === 'asc' ? -1 : 1;
    switch (sortBy) {
      case 'recency':
        sorted.sort((a, b) => dir * (new Date(b.created_at).getTime() - new Date(a.created_at).getTime()));
        break;
      case 'rating':
        sorted.sort((a, b) => {
          const aStars = a.review_summary?.[ratingTier]?.avg_stars ?? avgReviewStars(reviewMap[a.id] ?? []);
          const bStars = b.review_summary?.[ratingTier]?.avg_stars ?? avgReviewStars(reviewMap[b.id] ?? []);
          return dir * (bStars - aStars);
        });
        break;
      case 'reviews':
        sorted.sort((a, b) => {
          const aCount = a.review_summary?.all_time?.total_count ?? (reviewMap[a.id]?.length ?? 0);
          const bCount = b.review_summary?.all_time?.total_count ?? (reviewMap[b.id]?.length ?? 0);
          return dir * (bCount - aCount);
        });
        break;
      case 'access_count':
        sorted.sort((a, b) => dir * ((b.lineage_access_count ?? b.access_count) - (a.lineage_access_count ?? a.access_count)));
        break;
    }

    return sorted;
  }, [memories, searchResults, searchQuery, activeType, activeScopes, sortBy, sortDirection, ratingTier, reviewMap]);

  // Pagination: show PAGE_SIZE items at a time, reset when filters change
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  // Reset pagination when filters/sort/search change
  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
  }, [activeType, activeScopes, sortBy, sortDirection, searchResults, searchQuery]);

  const visible = filtered.slice(0, visibleCount);
  const hasMore = visibleCount < filtered.length;
  const remaining = filtered.length - visibleCount;

  if (filtered.length === 0) {
    return (
      <div className="text-center py-8 text-text-muted">
        <p className="text-sm">No memories match your filters.</p>
      </div>
    );
  }

  return (
    <div>
      {/* Count indicator */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-text-muted">
          Showing {visible.length} of {filtered.length}
        </span>
      </div>

      <motion.div
        className="space-y-3"
        variants={stagger}
        initial={false}
        animate="visible"
      >
        <AnimatePresence>
          {visible.map((entry) => (
            <MemoryCard key={entry.id} entry={entry} sessionId={sessionId} reviews={reviewMap[entry.id] ?? []} ratingTier={ratingTier} />
          ))}
        </AnimatePresence>
      </motion.div>

      {/* Show more button */}
      {hasMore && (
        <div className="flex justify-center mt-4">
          <button
            onClick={() => setVisibleCount((c) => c + PAGE_SIZE)}
            className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-text-muted hover:text-text bg-surface-elevated border border-border rounded-lg hover:border-border-hover transition-colors"
          >
            <ChevronDown size={14} />
            Show more ({remaining} remaining)
          </button>
        </div>
      )}
    </div>
  );
}
