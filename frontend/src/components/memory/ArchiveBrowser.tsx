import { useEffect, useRef } from 'react';
import { Archive, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { expandCollapse } from '../common/MotionPresets';
import { useMemoryStore } from '../../stores/memoryStore';

const typeColor: Record<string, string> = {
  preference: 'bg-badge-blue-bg text-badge-blue-text',
  pitfall: 'bg-badge-red-bg text-badge-red-text',
  summary: 'bg-badge-green-bg text-badge-green-text',
  reflection: 'bg-badge-purple-bg text-badge-purple-text',
};

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return iso;
  }
}

interface Props {
  sessionId: string;
}

export function ArchiveBrowser({ sessionId }: Props) {
  const {
    archivedMemories,
    archivedLoading,
    archiveExpanded,
    setArchiveExpanded,
    loadArchived,
  } = useMemoryStore();

  // Track which session we've loaded archives for
  const loadedSession = useRef<string | null>(null);

  // Load archived memories when expanded for the first time, or when session changes
  useEffect(() => {
    if (archiveExpanded && loadedSession.current !== sessionId) {
      loadedSession.current = sessionId;
      loadArchived(sessionId);
    }
  }, [archiveExpanded, sessionId, loadArchived]);

  return (
    <div className="mt-6 border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setArchiveExpanded(!archiveExpanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-surface-elevated hover:bg-hover-bg transition-colors"
      >
        <div className="flex items-center gap-2">
          <Archive size={16} className="text-text-muted" />
          <span className="text-sm font-medium text-text">Archived Memories</span>
          {archivedMemories.length > 0 && (
            <span className="text-xs text-text-muted">({archivedMemories.length})</span>
          )}
        </div>
        {archiveExpanded ? (
          <ChevronUp size={16} className="text-text-muted" />
        ) : (
          <ChevronDown size={16} className="text-text-muted" />
        )}
      </button>

      {/* Expandable content */}
      <AnimatePresence>
        {archiveExpanded && (
          <motion.div
            variants={expandCollapse}
            initial="hidden"
            animate="visible"
            exit="hidden"
            className="border-t border-border"
          >
            <div className="p-4">
              {archivedLoading && (
                <div className="flex items-center gap-2 py-4 justify-center text-text-muted text-sm">
                  <Loader2 size={16} className="animate-spin" />
                  Loading archived memories...
                </div>
              )}

              {!archivedLoading && archivedMemories.length === 0 && (
                <div className="text-center py-4 text-text-muted text-sm">
                  No archived memories.
                </div>
              )}

              {!archivedLoading && archivedMemories.length > 0 && (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {archivedMemories.map((m) => (
                    <div
                      key={m.id}
                      className="bg-surface border border-border/50 rounded-lg p-3"
                    >
                      <div className="flex items-center gap-1.5 mb-1.5">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            typeColor[m.type] ?? 'bg-badge-gray-bg text-badge-gray-text'
                          }`}
                        >
                          {m.type}
                        </span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-badge-gray-bg text-badge-gray-text">
                          Archived
                        </span>
                        <span className="text-[11px] text-text-muted ml-auto">
                          {formatDate(m.created_at)}
                        </span>
                      </div>
                      <div className="text-xs text-text-muted whitespace-pre-wrap leading-relaxed">
                        {m.content}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
