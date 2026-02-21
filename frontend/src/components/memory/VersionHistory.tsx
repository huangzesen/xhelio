import { useEffect, useMemo } from 'react';
import { Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { expandCollapse } from '../common/MotionPresets';
import { useMemoryStore } from '../../stores/memoryStore';
import { StarRating } from './ReviewList';
import { buildReviewMap, avgReviewStars } from './reviewUtils';

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

export function VersionHistory({ memoryId, sessionId }: Props) {
  const versionHistories = useMemoryStore((s) => s.versionHistories);
  const versionLoading = useMemoryStore((s) => s.versionLoading);
  const loadVersionHistory = useMemoryStore((s) => s.loadVersionHistory);
  const allMemories = useMemoryStore((s) => s.memories);

  const versions = versionHistories[memoryId];
  const isLoading = versionLoading === memoryId;
  const reviewMap = useMemo(() => buildReviewMap(allMemories), [allMemories]);

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
                      <StarRating stars={Math.round(avg)} />
                    )}
                  </div>
                  <div className="text-text-muted whitespace-pre-wrap leading-relaxed">
                    {v.content}
                  </div>
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
