import { Brain, Star, Zap } from 'lucide-react';
import type { MemoryEntry, MemoryStats } from '../../api/types';
import { parseReviewStars } from './reviewUtils';

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

interface Props {
  stats: MemoryStats;
  totalMemories: number;
  memories: MemoryEntry[];
}

export function MemoryStatsBar({ stats, totalMemories, memories }: Props) {
  const ratio = stats.token_budget > 0 ? stats.total_tokens / stats.token_budget : 0;
  const pct = Math.min(ratio * 100, 100);

  // Compute review aggregates from review-type memory entries
  let totalReviews = 0;
  let totalStars = 0;
  for (const m of memories) {
    if (m.type === 'review') {
      const stars = parseReviewStars(m);
      if (stars > 0) {
        totalReviews += 1;
        totalStars += stars;
      }
    }
  }
  const avgRating = totalReviews > 0 ? totalStars / totalReviews : 0;

  // Per-type token segments (as % of budget)
  const segments = TYPES.map((type) => {
    const tokens = stats.type_tokens[type] ?? 0;
    const segPct = stats.token_budget > 0 ? (tokens / stats.token_budget) * 100 : 0;
    return { type, tokens, pct: Math.min(segPct, 100) };
  });

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

      {/* Review aggregate */}
      {totalReviews > 0 && (
        <div className="bg-surface-elevated border border-border rounded-lg p-3 flex items-center gap-2.5 min-w-[120px]">
          <Star size={16} className="fill-amber-400 text-amber-400 shrink-0" />
          <div>
            <div className="text-lg font-semibold text-text leading-tight tabular-nums">
              {avgRating.toFixed(1)}
            </div>
            <div className="text-[11px] text-text-muted">
              {totalReviews} {totalReviews === 1 ? 'review' : 'reviews'}
            </div>
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
            {pct.toFixed(0)}%
          </span>
        </div>

        {/* Segmented bar */}
        <div className="flex h-2.5 bg-border/30 rounded-full overflow-hidden mb-2">
          {segments.map(({ type, pct: segPct }) =>
            segPct > 0 ? (
              <div
                key={type}
                className={`h-full ${typeStyles[type].bar} first:rounded-l-full last:rounded-r-full`}
                style={{ width: `${segPct}%` }}
                title={`${type}: ${segPct.toFixed(1)}%`}
              />
            ) : null,
          )}
        </div>

        {/* Per-type legend */}
        <div className="flex items-center gap-3 flex-wrap">
          {segments.map(({ type, tokens }) => {
            const count = stats.type_counts[type] ?? 0;
            if (count === 0 && tokens === 0) return null;
            return (
              <div key={type} className="flex items-center gap-1.5 text-[11px] text-text-muted">
                <span className={`w-2 h-2 rounded-full ${typeStyles[type].bar} shrink-0`} />
                <span>{type.charAt(0).toUpperCase() + type.slice(1)}</span>
                <span className="tabular-nums">{tokens.toLocaleString()}</span>
              </div>
            );
          })}
          <span className="ml-auto text-[11px] text-text-muted tabular-nums">
            {stats.total_tokens.toLocaleString()} / {stats.token_budget.toLocaleString()}
          </span>
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
