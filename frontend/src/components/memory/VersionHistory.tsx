import { useEffect, useMemo, useState } from 'react';
import { Loader2, ChevronDown, MessageCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { expandCollapse } from '../common/MotionPresets';
import { useMemoryStore } from '../../stores/memoryStore';
import { StarRating } from './ReviewList';
import { buildReviewMap, avgReviewStars, parseReviewStars, parseReviewComment, parseReviewAgent } from './reviewUtils';

interface Props {
  memoryId: string;
  sessionId: string;
}

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

export function VersionHistory({ memoryId, sessionId }: Props) {
  const versionHistories = useMemoryStore((s) => s.versionHistories);
  const versionLoading = useMemoryStore((s) => s.versionLoading);
  const loadVersionHistory = useMemoryStore((s) => s.loadVersionHistory);
  const allMemories = useMemoryStore((s) => s.memories);
  const [openReviews, setOpenReviews] = useState<Set<string>>(new Set());

  const versions = versionHistories[memoryId];
  const isLoading = versionLoading === memoryId;
  const reviewMap = useMemo(() => buildReviewMap(allMemories), [allMemories]);

  const toggleReviews = (versionId: string) => {
    setOpenReviews((prev) => {
      const next = new Set(prev);
      if (next.has(versionId)) next.delete(versionId);
      else next.add(versionId);
      return next;
    });
  };

  useEffect(() => {
    if (!versions) {
      loadVersionHistory(sessionId, memoryId);
    }
  }, [memoryId, sessionId, versions, loadVersionHistory]);

  return (
    <motion.div
      variants={expandCollapse}
      initial="hidden"
      animate="visible"
      className="mt-3 pl-3 border-l-2 border-border"
    >
      {isLoading && (
        <div className="flex items-center gap-2 py-2 text-text-muted text-xs">
          <Loader2 size={12} className="animate-spin" />
          Loading history...
        </div>
      )}

      {versions && versions.length > 0 && (
        <div className="space-y-0">
          {versions.map((v, idx) => {
            const reviews = reviewMap[v.id] ?? [];
            const avg = avgReviewStars(reviews);
            const hasReviews = reviews.length > 0;
            const isReviewOpen = openReviews.has(v.id);
            return (
              <div key={v.id} className="relative pl-4 pb-3">
                {/* Timeline connector */}
                {idx < versions.length - 1 && (
                  <div className="absolute left-[5px] top-3 bottom-0 w-px bg-border" />
                )}
                {/* Dot */}
                <div
                  className={`absolute left-0 top-1.5 w-[11px] h-[11px] rounded-full border-2 ${
                    idx === 0
                      ? 'bg-primary border-primary'
                      : 'bg-surface border-border'
                  }`}
                />
                {/* Version content */}
                <div className="text-xs">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="font-medium text-text">
                      v{v.version}
                    </span>
                    <span className="text-text-muted">
                      {formatDate(v.created_at)}
                    </span>
                    {idx === 0 && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-green-bg text-badge-green-text">
                        Current
                      </span>
                    )}
                    {avg > 0 && (
                      <button
                        onClick={() => toggleReviews(v.id)}
                        className="flex items-center gap-1 text-text-muted hover:text-text transition-colors"
                      >
                        <StarRating stars={Math.round(avg)} />
                        <span className="text-[10px] font-medium tabular-nums">{avg.toFixed(1)}</span>
                        <span className="inline-flex items-center gap-0.5 text-[10px]">
                          <MessageCircle size={10} />
                          {reviews.length}
                        </span>
                        <ChevronDown
                          size={10}
                          className={`transition-transform duration-200 ${isReviewOpen ? 'rotate-180' : ''}`}
                        />
                      </button>
                    )}
                  </div>
                  <div className="text-text-muted whitespace-pre-wrap leading-relaxed">
                    {v.content}
                  </div>

                  {/* Collapsible reviews for this version */}
                  <AnimatePresence>
                    {hasReviews && isReviewOpen && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <div className="space-y-1.5 mt-2">
                          {reviews.map((rev) => (
                            <div key={rev.id} className="bg-surface-secondary rounded-md px-2.5 py-1.5 text-[11px]">
                              <div className="flex items-center gap-1.5 mb-0.5">
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
                </div>
              </div>
            );
          })}
        </div>
      )}

      {versions && versions.length === 0 && (
        <div className="text-xs text-text-muted py-2">No version history available.</div>
      )}
    </motion.div>
  );
}
