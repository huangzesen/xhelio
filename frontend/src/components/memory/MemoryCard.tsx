import { useState } from 'react';
import { Activity, History, MessageSquare, MessageCircle, Lightbulb, User, Layers, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { fadeSlideIn } from '../common/MotionPresets';
import { VersionHistory } from './VersionHistory';
import { StarRating } from './ReviewList';
import { parseReviewStars, parseReviewComment, parseReviewAgent, avgReviewStars } from './reviewUtils';
import type { MemoryEntry } from '../../api/types';

const typeColor: Record<string, { border: string; badge: string }> = {
  preference: {
    border: 'border-l-[color:var(--badge-blue-bg,#3b82f6)]',
    badge: 'bg-badge-blue-bg text-badge-blue-text',
  },
  pitfall: {
    border: 'border-l-[color:var(--badge-red-bg,#ef4444)]',
    badge: 'bg-badge-red-bg text-badge-red-text',
  },
  summary: {
    border: 'border-l-[color:var(--badge-green-bg,#22c55e)]',
    badge: 'bg-badge-green-bg text-badge-green-text',
  },
  reflection: {
    border: 'border-l-[color:var(--badge-purple-bg,#a855f7)]',
    badge: 'bg-badge-purple-bg text-badge-purple-text',
  },
};

const defaultColor = {
  border: 'border-l-[color:var(--badge-gray-bg,#6b7280)]',
  badge: 'bg-badge-gray-bg text-badge-gray-text',
};

const sourceConfig: Record<string, { label: string; icon: typeof MessageSquare }> = {
  extracted: { label: 'Learned from chat', icon: MessageSquare },
  reflected: { label: 'Self-reflection', icon: Lightbulb },
  user_explicit: { label: 'User-defined', icon: User },
  consolidated: { label: 'Consolidated', icon: Layers },
};

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
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
    return formatDate(iso);
  } catch {
    return '';
  }
}

interface Props {
  entry: MemoryEntry;
  sessionId: string;
  reviews?: MemoryEntry[];
  ratingTier?: 'all_time' | 'recent';
}

export function MemoryCard({ entry, sessionId, reviews = [], ratingTier = 'recent' }: Props) {
  const colors = typeColor[entry.type] ?? defaultColor;
  const [historyOpen, setHistoryOpen] = useState(false);
  const [reviewOpen, setReviewOpen] = useState(false);
  const hasHistory = entry.supersedes !== '';
  const hasReviews = reviews.length > 0 || (entry.review_summary?.all_time?.total_count ?? 0) > 0;
  // Both rating and count from tier stats (lineage-wide); expanded list shows current-version only
  const tierStats = entry.review_summary?.[ratingTier];
  const displayAvg = tierStats?.avg_stars ?? avgReviewStars(reviews);
  const displayCount = tierStats?.total_count ?? reviews.length;
  const src = sourceConfig[entry.source] ?? { label: entry.source, icon: MessageSquare };
  const SourceIcon = src.icon;

  return (
    <motion.div
      layout
      variants={fadeSlideIn}
      initial="hidden"
      animate="visible"
      exit="exit"
      className={`bg-surface border border-border rounded-lg p-4 border-l-4 ${colors.border}`}
    >
      {/* Top row: type badge + scopes + version + rating */}
      <div className="flex items-center justify-between mb-2 gap-2">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${colors.badge}`}>
            {entry.type.charAt(0).toUpperCase() + entry.type.slice(1)}
          </span>
          {entry.version > 1 && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-teal-bg text-badge-teal-text">
              v{entry.version}
            </span>
          )}
          {entry.scopes.map((scope) => (
            <span
              key={scope}
              className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-badge-gray-bg text-badge-gray-text"
            >
              {scope}
            </span>
          ))}
        </div>
        {/* Compact: avg star rating + review count — clickable to expand */}
        {hasReviews && (
          <button
            onClick={() => setReviewOpen((v) => !v)}
            className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text transition-colors"
          >
            <StarRating stars={Math.round(displayAvg)} />
            <span className="text-[11px] font-medium tabular-nums">{displayAvg.toFixed(1)}</span>
            <span className="inline-flex items-center gap-0.5 text-[11px]">
              <MessageCircle size={11} />
              {displayCount}
            </span>
            <ChevronDown
              size={12}
              className={`transition-transform duration-200 ${reviewOpen ? 'rotate-180' : ''}`}
            />
          </button>
        )}
      </div>

      {/* Content */}
      <div className="text-sm text-text whitespace-pre-wrap leading-relaxed mb-3">
        {entry.content}
      </div>

      {/* Collapsible reviews list */}
      <AnimatePresence>
        {hasReviews && reviewOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="space-y-2 mb-3">
              {reviews.map((rev) => (
                <div key={rev.id} className="bg-surface-secondary rounded-md px-3 py-2 text-xs">
                  <div className="flex items-center gap-2 mb-1">
                    <StarRating stars={parseReviewStars(rev)} />
                    <span className="text-text-muted">
                      {parseReviewAgent(rev)}{rev.version > 1 ? ` · v${rev.version}` : ''} · {relativeTime(rev.created_at)}
                    </span>
                  </div>
                  <div className="text-text-muted whitespace-pre-wrap leading-relaxed">
                    {parseReviewComment(rev)}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Metadata row */}
      <div className="flex items-center flex-wrap gap-x-3 gap-y-1 text-xs text-text-muted">
        <span className="inline-flex items-center gap-1">
          <SourceIcon size={11} />
          {src.label}
        </span>
        <span className="flex items-center gap-1">
          <Activity size={11} />
          {entry.lineage_access_count ?? entry.access_count}
        </span>
        {entry.tags.length > 0 && (
          <span>Tags: {entry.tags.join(', ')}</span>
        )}
        <span>{formatDate(entry.created_at)}</span>
        <span
          className="inline-flex items-center gap-1 cursor-pointer hover:text-text transition-colors"
          title="Click to copy ID"
          onClick={() => navigator.clipboard.writeText(entry.id)}
        >
          ID: <span className="font-mono">{entry.id}</span>
        </span>
      </div>

      {/* Version history toggle */}
      {hasHistory && (
        <div className="flex items-center gap-3 mt-2">
          <button
            onClick={() => setHistoryOpen((v) => !v)}
            className="flex items-center gap-1 text-xs text-primary hover:text-primary/80 transition-colors"
          >
            <History size={12} />
            {historyOpen ? 'Hide history' : 'View history'}
          </button>
        </div>
      )}

      {historyOpen && hasHistory && (
        <VersionHistory memoryId={entry.id} sessionId={sessionId} />
      )}
    </motion.div>
  );
}
