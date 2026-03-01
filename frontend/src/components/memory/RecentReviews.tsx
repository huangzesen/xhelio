import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { ChevronDown, MessageSquareQuote, Search, X, ArrowUp, ArrowDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { fadeSlideIn, stagger } from '../common/MotionPresets';
import { StarRating } from './ReviewList';
import {
  parseReviewStars,
  parseReviewComment,
  parseReviewAgent,
} from './reviewUtils';
import type { MemoryEntry } from '../../api/types';

const typeColor: Record<string, string> = {
  preference: 'bg-badge-blue-bg text-badge-blue-text',
  pitfall: 'bg-badge-red-bg text-badge-red-text',
  summary: 'bg-badge-green-bg text-badge-green-text',
  reflection: 'bg-badge-purple-bg text-badge-purple-text',
};

const defaultBadge = 'bg-badge-gray-bg text-badge-gray-text';

type SortBy = 'recency' | 'rating';

/** Insert a space before "Agent" and keep bracket portions, e.g. MissionAgent[PSP] → Mission Agent [PSP] */
function formatAgentName(raw: string): string {
  const bracketIdx = raw.indexOf('[');
  const base = bracketIdx >= 0 ? raw.slice(0, bracketIdx) : raw;
  const suffix = bracketIdx >= 0 ? ' ' + raw.slice(bracketIdx) : '';
  const spaced = base.replace(/Agent$/, ' Agent');
  return spaced + suffix;
}

function relativeTime(iso: string): string {
  if (!iso) return '';
  try {
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 30) return `${days}d ago`;
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return '';
  }
}

interface AgentGroup {
  agent: string;
  reviews: MemoryEntry[];
  avgStars: number;
  latestTimestamp: string;
}

const INITIAL_SHOW = 5;

interface Props {
  memories: MemoryEntry[];
}

export function RecentReviews({ memories }: Props) {
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [sortBy, setSortBy] = useState<SortBy>('recency');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [activeScopes, setActiveScopes] = useState<string[]>([]);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  const handleSearchInput = useCallback((value: string) => {
    setSearchQuery(value);
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setDebouncedQuery(value.toLowerCase().trim());
    }, 200);
  }, []);

  const handleClear = useCallback(() => {
    clearTimeout(debounceRef.current);
    setSearchQuery('');
    setDebouncedQuery('');
  }, []);

  useEffect(() => {
    return () => clearTimeout(debounceRef.current);
  }, []);

  const toggleScope = useCallback((scope: string) => {
    setActiveScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope],
    );
  }, []);

  // Build target map: memory id → MemoryEntry (non-review entries)
  const targetMap = useMemo(() => {
    const map: Record<string, MemoryEntry> = {};
    for (const m of memories) {
      if (m.type !== 'review') {
        map[m.id] = m;
      }
    }
    return map;
  }, [memories]);

  // Collect all scopes from target memories of reviews
  const allScopes = useMemo(() => {
    const scopeSet = new Set<string>();
    for (const m of memories) {
      if (m.type !== 'review') continue;
      const target = m.review_of ? targetMap[m.review_of] : undefined;
      if (target) {
        for (const s of target.scopes) scopeSet.add(s);
      }
    }
    return Array.from(scopeSet).sort();
  }, [memories, targetMap]);

  // Filter and group reviews
  const agentGroups = useMemo(() => {
    // 1. Collect all reviews
    let reviews: MemoryEntry[] = [];
    for (const m of memories) {
      if (m.type !== 'review') continue;
      reviews.push(m);
    }

    // 2. Filter by search query (matches comment text or target content)
    if (debouncedQuery) {
      reviews = reviews.filter((r) => {
        const comment = parseReviewComment(r).toLowerCase();
        if (comment.includes(debouncedQuery)) return true;
        const target = r.review_of ? targetMap[r.review_of] : undefined;
        if (target && target.content.toLowerCase().includes(debouncedQuery)) return true;
        const agent = parseReviewAgent(r).toLowerCase();
        if (agent.includes(debouncedQuery)) return true;
        return false;
      });
    }

    // 3. Filter by scope (target memory must have at least one active scope)
    if (activeScopes.length > 0) {
      reviews = reviews.filter((r) => {
        const target = r.review_of ? targetMap[r.review_of] : undefined;
        if (!target) return false;
        return target.scopes.some((s) => activeScopes.includes(s));
      });
    }

    // 4. Group by agent
    const groups: Record<string, MemoryEntry[]> = {};
    for (const r of reviews) {
      const agent = parseReviewAgent(r) || 'Unknown';
      if (!groups[agent]) groups[agent] = [];
      groups[agent].push(r);
    }

    // 5. Build agent groups with sorting applied within each
    const dir = sortDirection === 'desc' ? -1 : 1;
    const result: AgentGroup[] = [];
    for (const [agent, agentReviews] of Object.entries(groups)) {
      // Sort reviews within group
      agentReviews.sort((a, b) => {
        if (sortBy === 'rating') {
          const sa = parseReviewStars(a);
          const sb = parseReviewStars(b);
          if (sa !== sb) return (sa - sb) * dir;
        }
        // Recency (or rating tiebreaker)
        return (new Date(a.created_at).getTime() - new Date(b.created_at).getTime()) * dir;
      });

      let totalStars = 0;
      let starCount = 0;
      for (const r of agentReviews) {
        const s = parseReviewStars(r);
        if (s > 0) { totalStars += s; starCount++; }
      }
      const avgStars = starCount > 0 ? totalStars / starCount : -1;

      result.push({
        agent,
        reviews: agentReviews,
        avgStars,
        latestTimestamp: agentReviews.reduce((latest, r) =>
          new Date(r.created_at).getTime() > new Date(latest).getTime() ? r.created_at : latest,
          agentReviews[0].created_at,
        ),
      });
    }

    // 6. Sort groups by most recent review
    result.sort(
      (a, b) =>
        new Date(b.latestTimestamp).getTime() -
        new Date(a.latestTimestamp).getTime(),
    );

    return result;
  }, [memories, targetMap, debouncedQuery, activeScopes, sortBy, sortDirection]);

  const totalFiltered = agentGroups.reduce((sum, g) => sum + g.reviews.length, 0);
  const totalReviews = memories.filter((m) => m.type === 'review').length;
  const isFiltered = debouncedQuery || activeScopes.length > 0;

  return (
    <div className="space-y-4">
      {/* Toolbar: search + scope chips + sort */}
      <div className="space-y-3">
        {/* Search bar */}
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearchInput(e.target.value)}
            placeholder="Search reviews..."
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
                  onClick={() => toggleScope(scope)}
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
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setSortDirection((d) => (d === 'desc' ? 'asc' : 'desc'))}
              className="p-1 rounded text-text-muted hover:text-text transition-colors"
              title={sortDirection === 'desc' ? 'Newest first (click to reverse)' : 'Oldest first (click to reverse)'}
            >
              {sortDirection === 'desc' ? <ArrowDown size={12} /> : <ArrowUp size={12} />}
            </button>
            {([
              { label: 'Recent', value: 'recency' as SortBy },
              { label: 'Rating', value: 'rating' as SortBy },
            ]).map((opt) => (
              <button
                key={opt.value}
                onClick={() => setSortBy(opt.value)}
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
          {isFiltered && (
            <span className="text-[11px] text-text-muted">
              {totalFiltered} of {totalReviews} reviews
            </span>
          )}
        </div>
      </div>

      {/* Agent groups */}
      {agentGroups.length === 0 ? (
        <div className="text-center py-16 text-text-muted">
          <MessageSquareQuote size={40} className="mx-auto mb-3 opacity-30" />
          {isFiltered ? (
            <>
              <p className="text-sm">No reviews match your filters.</p>
              <button
                onClick={() => { handleClear(); setActiveScopes([]); }}
                className="text-xs text-primary hover:text-primary/80 mt-2 transition-colors"
              >
                Clear filters
              </button>
            </>
          ) : (
            <>
              <p className="text-sm">No reviews yet.</p>
              <p className="text-xs mt-1">
                Reviews are created when agents evaluate memories they used during a session.
              </p>
            </>
          )}
        </div>
      ) : (
        <motion.div
          variants={stagger}
          initial="hidden"
          animate="visible"
          className="space-y-6"
        >
          {agentGroups.map((group) => (
            <AgentSection
              key={group.agent}
              group={group}
              targetMap={targetMap}
            />
          ))}
        </motion.div>
      )}
    </div>
  );
}

function AgentSection({
  group,
  targetMap,
}: {
  group: AgentGroup;
  targetMap: Record<string, MemoryEntry>;
}) {
  const [open, setOpen] = useState(true);
  const [showAll, setShowAll] = useState(false);
  const visibleReviews =
    showAll || group.reviews.length <= INITIAL_SHOW
      ? group.reviews
      : group.reviews.slice(0, INITIAL_SHOW);
  const hiddenCount = group.reviews.length - INITIAL_SHOW;

  return (
    <motion.div variants={fadeSlideIn}>
      {/* Agent header — clickable to collapse */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full mb-3 group cursor-pointer"
      >
        <div className="flex items-center gap-2">
          <ChevronDown
            size={14}
            className={`text-text-muted transition-transform duration-200 ${open ? '' : '-rotate-90'}`}
          />
          <h3 className="text-sm font-semibold text-text group-hover:text-primary transition-colors">
            {formatAgentName(group.agent)}
          </h3>
          {group.avgStars > 0 && (
            <span className="flex items-center gap-1">
              <StarRating stars={Math.round(group.avgStars)} />
              <span className="text-[11px] text-text-muted font-medium tabular-nums">
                {group.avgStars.toFixed(1)}
              </span>
            </span>
          )}
        </div>
        <span className="text-xs text-text-muted">
          {group.reviews.length}{' '}
          {group.reviews.length === 1 ? 'review' : 'reviews'}
        </span>
      </button>

      {/* Collapsible review cards */}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="space-y-2">
              {visibleReviews.map((review) => (
                <ReviewCard
                  key={review.id}
                  review={review}
                  target={review.review_of ? targetMap[review.review_of] : undefined}
                />
              ))}
            </div>

            {/* Show all / collapse toggle */}
            {hiddenCount > 0 && (
              <button
                onClick={() => setShowAll((v) => !v)}
                className="flex items-center gap-1 mt-2 text-xs text-primary hover:text-primary/80 transition-colors"
              >
                <ChevronDown
                  size={12}
                  className={`transition-transform duration-200 ${showAll ? 'rotate-180' : ''}`}
                />
                {showAll
                  ? 'Show less'
                  : `Show all ${group.reviews.length} reviews`}
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function ReviewCard({
  review,
  target,
}: {
  review: MemoryEntry;
  target?: MemoryEntry;
}) {
  const stars = parseReviewStars(review);
  const comment = parseReviewComment(review);
  const badge = target ? typeColor[target.type] ?? defaultBadge : defaultBadge;

  return (
    <motion.div
      layout
      variants={fadeSlideIn}
      initial="hidden"
      animate="visible"
      exit="exit"
      className="bg-surface border border-border rounded-lg px-4 py-3"
    >
      {/* Star rating + review metadata */}
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <StarRating stars={stars} />
          <span className="text-[11px] text-text-muted font-medium tabular-nums">
            {stars > 0 ? `${stars}.0` : '—'}
          </span>
          {review.version > 1 && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-teal-bg text-badge-teal-text">
              v{review.version}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-[11px] text-text-muted">
          <span>{relativeTime(review.created_at)}</span>
          <span
            className="font-mono cursor-pointer hover:text-text transition-colors"
            title="Click to copy review ID"
            onClick={() => navigator.clipboard.writeText(review.id)}
          >
            {review.id}
          </span>
        </div>
      </div>

      {/* Review comment */}
      {comment && (
        <p className="text-sm text-text leading-relaxed mb-2">{comment}</p>
      )}

      {/* Target memory */}
      {target ? (
        <div className="bg-surface-secondary rounded-md px-3 py-2">
          <div className="flex items-center justify-between gap-2 mb-1">
            <div className="flex items-center gap-1.5 flex-wrap">
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${badge}`}
              >
                {target.type.charAt(0).toUpperCase() + target.type.slice(1)}
              </span>
              {target.version > 1 && (
                <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-teal-bg text-badge-teal-text">
                  v{target.version}
                </span>
              )}
              {target.scopes.length > 0 && (
                <span className="text-[10px] text-text-muted">
                  {target.scopes.join(', ')}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-[10px] text-text-muted shrink-0">
              <span>{relativeTime(target.created_at)}</span>
              <span
                className="font-mono cursor-pointer hover:text-text transition-colors"
                title="Click to copy ID"
                onClick={() => navigator.clipboard.writeText(target.id)}
              >
                {target.id}
              </span>
            </div>
          </div>
          <p className="text-xs text-text-muted whitespace-pre-wrap leading-relaxed">
            {target.content}
          </p>
        </div>
      ) : review.review_of ? (
        <div className="bg-surface-secondary rounded-md px-3 py-2">
          <p className="text-xs text-text-muted italic">
            Target memory archived or superseded{' '}
            <span
              className="font-mono not-italic cursor-pointer hover:text-text transition-colors"
              title="Click to copy target ID"
              onClick={() => navigator.clipboard.writeText(review.review_of)}
            >
              ({review.review_of})
            </span>
          </p>
        </div>
      ) : null}
    </motion.div>
  );
}
