import { Brain, Star, Zap } from 'lucide-react';
import type { MemoryEntry, MemoryStats } from '../../api/types';

const TYPES = ['preference', 'pitfall', 'summary', 'reflection'] as const;

const typeStyles: Record<string, { badge: string; bar: string; label: string }> = {
  preference: { badge: 'bg-badge-blue-bg text-badge-blue-text', bar: 'bg-blue-500', label: 'Pref' },
  pitfall:    { badge: 'bg-badge-red-bg text-badge-red-text',   bar: 'bg-red-500',  label: 'Pitf' },
  summary:    { badge: 'bg-badge-green-bg text-badge-green-text', bar: 'bg-green-500', label: 'Summ' },
  reflection: { badge: 'bg-badge-purple-bg text-badge-purple-text', bar: 'bg-purple-500', label: 'Refl' },
};

function gaugeColor(ratio: number): string {
  if (ratio < 0.6) return 'text-green-500';
  if (ratio < 0.85) return 'text-amber-500';
  return 'text-red-500';
}

type RatingTier = 'all_time' | 'recent';

interface Props {
  stats: MemoryStats;
  totalMemories: number;
  memories: MemoryEntry[];
  ratingTier: RatingTier;
  onRatingTierChange: (tier: RatingTier) => void;
}

export function MemoryStatsBar({ stats, totalMemories, memories, ratingTier, onRatingTierChange }: Props) {
  const ratio = stats.token_budget > 0 ? stats.total_tokens / stats.token_budget : 0;
  const pct = Math.min(ratio * 100, 100);

  // Compute review aggregates from review_summary (two-tier) on non-review memories
  let totalReviews = 0;
  let weightedStars = 0;
  for (const m of memories) {
    if (m.type === 'review') continue;
    const tier = m.review_summary?.[ratingTier];
    if (tier && tier.total_count > 0) {
      totalReviews += tier.total_count;
      weightedStars += tier.avg_stars * tier.total_count;
    }
  }
  const avgRating = totalReviews > 0 ? weightedStars / totalReviews : 0;

  // Per-type token segments
  const segments = TYPES.map((type) => {
    const tokens = stats.type_tokens[type] ?? 0;
    return { type, tokens };
  });
  const segmentTotal = segments.reduce((sum, s) => sum + s.tokens, 0);

  return (
    <div className="flex flex-wrap gap-3 mb-4">
      {/* Total memories */}
      <div className="bg-surface-elevated border border-border rounded-lg p-3 flex items-center gap-2.5 min-w-[120px]">
        <Brain size={16} className="text-primary shrink-0" />
        <div>
          <div className="text-lg font-semibold text-text leading-tight">{totalMemories}</div>
          <div className="text-[11px] text-text-muted">Memories</div>
        </div>
      </div>

      {/* Review aggregate with tier toggle */}
      {totalReviews > 0 && (
        <div className="bg-surface-elevated border border-border rounded-lg p-3 flex items-center gap-2.5 min-w-[160px]">
          <Star size={16} className="fill-amber-400 text-amber-400 shrink-0" />
          <div>
            <div className="text-lg font-semibold text-text leading-tight tabular-nums">
              {avgRating.toFixed(1)}
            </div>
            <div className="text-[11px] text-text-muted">
              {totalReviews} {totalReviews === 1 ? 'review' : 'reviews'}
            </div>
          </div>
          {/* Tier toggle */}
          <div className="flex rounded-md overflow-hidden border border-border ml-auto text-[10px] font-medium">
            <button
              onClick={() => onRatingTierChange('recent')}
              className={`px-2 py-0.5 transition-colors ${
                ratingTier === 'recent'
                  ? 'bg-primary/15 text-primary'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              Recent
            </button>
            <button
              onClick={() => onRatingTierChange('all_time')}
              className={`px-2 py-0.5 transition-colors ${
                ratingTier === 'all_time'
                  ? 'bg-primary/15 text-primary'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              All time
            </button>
          </div>
        </div>
      )}

      {/* Token budget gauge — segmented by type */}
      <div className="bg-surface-elevated border border-border rounded-lg p-3 flex-1 min-w-[240px]">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            <Zap size={14} className="text-text-muted shrink-0" />
            <span className="text-xs text-text-muted">Token Budget</span>
          </div>
          <span className={`text-xs font-medium tabular-nums ${gaugeColor(ratio)}`}>
            {stats.total_tokens.toLocaleString()} / {stats.token_budget.toLocaleString()} ({pct.toFixed(0)}%)
          </span>
        </div>

        {/* Type proportion bar — segments fill 100% to show relative distribution */}
        {segmentTotal > 0 && (
          <div className="flex h-4 bg-border/30 rounded-full overflow-hidden mb-2">
            {segments.map(({ type, tokens }) => {
              const typePct = (tokens / segmentTotal) * 100;
              return typePct > 0 ? (
                <div
                  key={type}
                  className={`h-full ${typeStyles[type].bar} first:rounded-l-full last:rounded-r-full transition-all duration-300`}
                  style={{ width: `${typePct}%` }}
                  title={`${type}: ${tokens.toLocaleString()} tokens (${typePct.toFixed(1)}%)`}
                />
              ) : null;
            })}
          </div>
        )}

        {/* Budget usage bar (thin) */}
        <div className="flex h-1.5 bg-border/30 rounded-full overflow-hidden mb-2">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              ratio < 0.6 ? 'bg-green-500' : ratio < 0.85 ? 'bg-amber-500' : 'bg-red-500'
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>

        {/* Per-type legend */}
        <div className="flex items-center gap-3 flex-wrap">
          {segments.map(({ type, tokens }) => {
            const count = stats.type_counts[type] ?? 0;
            if (count === 0 && tokens === 0) return null;
            const typePct = segmentTotal > 0 ? (tokens / segmentTotal) * 100 : 0;
            return (
              <div key={type} className="flex items-center gap-1.5 text-[11px] text-text-muted">
                <span className={`w-2 h-2 rounded-full ${typeStyles[type].bar} shrink-0`} />
                <span>{type.charAt(0).toUpperCase() + type.slice(1)}</span>
                <span className="tabular-nums">{typePct.toFixed(0)}%</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Type count breakdown */}
      <div className="bg-surface-elevated border border-border rounded-lg p-3 flex items-center gap-2 flex-wrap">
        {TYPES.map((type) => {
          const count = stats.type_counts[type] ?? 0;
          return (
            <span
              key={type}
              className={`px-2 py-1 rounded text-xs font-medium ${typeStyles[type].badge}`}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}: {count}
            </span>
          );
        })}
      </div>
    </div>
  );
}
