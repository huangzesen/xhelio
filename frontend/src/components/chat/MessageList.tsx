import { useEffect, useRef, useMemo, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChatMessage } from './ChatMessage';
import { ToolCallGroup } from './ToolCallGroup';
import { TypingIndicator } from './TypingIndicator';
import { PlotlyFigure } from '../plot/PlotlyFigure';
import { PlotThumbnail } from '../plot/PlotThumbnail';
import { PlotFullscreen } from '../plot/PlotFullscreen';
import { MplFullscreen } from '../plot/MplFullscreen';
import JsxComponent from '../plot/JsxComponent';
import { fadeSlideIn } from '../common/MotionPresets';
import { ArrowDown, ExternalLink, ImageOff, Clock3 } from 'lucide-react';
import type { ChatMessage as ChatMessageType, ToolEvent, CommentaryEvent } from '../../api/types';
import * as api from '../../api/client';
import { useChatStore } from '../../stores/chatStore';
import { groupEventsByRound, type ThinkingEvent } from '../../utils/groupEventsByTurn';
import { friendlyAgentName } from '../../utils/friendlyAgentName';

/** Live commentary lines — shows the last 3 commentary events with agent names. */
function CommentaryLines({ events }: { events: CommentaryEvent[] }) {
  const recent = events.slice(-3);
  return (
    <div className="ml-11 space-y-0.5">
      {recent.map((evt, i) => (
        <div
          key={evt.id}
          className={`flex items-center gap-2 text-xs text-text-muted${
            i < recent.length - 1 ? ' opacity-50' : ''
          }`}
        >
          {i === recent.length - 1 && (
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shrink-0" />
          )}
          {i < recent.length - 1 && (
            <span className="w-1.5 h-1.5 shrink-0" />
          )}
          <span className="shrink-0 font-medium">[{friendlyAgentName(evt.agent)}]</span>
          <span className="truncate">{evt.text}</span>
        </div>
      ))}
    </div>
  );
}

interface Props {
  messages: ChatMessageType[];
  toolEvents: ToolEvent[];
  isStreaming: boolean;
  onRegenerate?: () => void;
}

export function MessageList({ messages, toolEvents, isStreaming, onRegenerate }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);
  const [hasNewContent, setHasNewContent] = useState(false);
  const [viewerIndex, setViewerIndex] = useState<number | null>(null);
  const [mplViewerIndex, setMplViewerIndex] = useState<number | null>(null);
  const [loadingThumbnails, setLoadingThumbnails] = useState<Set<string>>(new Set());
  const roundMarkers = useChatStore((s) => s.roundMarkers);
  const memoryEvents = useChatStore((s) => s.memoryEvents);
  const commentaryText = useChatStore((s) => s.commentaryText);
  const commentaryEvents = useChatStore((s) => s.commentaryEvents);
  const roundTokenUsage = useChatStore((s) => s.roundTokenUsage);

  const handleLoadInteractivePlot = useCallback(async (msgId: string, sessionId: string) => {
    setLoadingThumbnails((prev) => new Set(prev).add(msgId));
    try {
      const { figure, figure_url } = await api.getFigure(sessionId);
      if (figure) {
        useChatStore.setState((s) => ({
          figureJson: figure,
          messages: s.messages.map((m) =>
            m.id === msgId
              ? { ...m, figure, figure_url, thumbnailUrl: undefined, thumbnailSessionId: undefined }
              : m,
          ),
        }));
      } else if (figure_url) {
        useChatStore.setState((s) => ({
          messages: s.messages.map((m) =>
            m.id === msgId
              ? { ...m, figure_url, thumbnailUrl: undefined, thumbnailSessionId: undefined }
              : m,
          ),
        }));
      }
    } catch {
      // Failed to load — remove loading state, keep thumbnail
    }
    setLoadingThumbnails((prev) => {
      const next = new Set(prev);
      next.delete(msgId);
      return next;
    });
  }, []);

  // Track whether user is near the bottom using IntersectionObserver
  useEffect(() => {
    const el = bottomRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsAtBottom(entry.isIntersecting);
        if (entry.isIntersecting) setHasNewContent(false);
      },
      { threshold: 0.1 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Auto-scroll only when user is near bottom
  useEffect(() => {
    if (isAtBottom) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    } else {
      setHasNewContent(true);
    }
  }, [messages, toolEvents, isAtBottom]);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    setHasNewContent(false);
  }, []);

  // Filter out thinking messages from the visible message list —
  // thinking is shown inside ToolCallGroup / Activity panel instead.
  const visibleMessages = useMemo(
    () => messages.filter((m) => m.role !== 'thinking'),
    [messages],
  );

  // Collect all figures from plot messages (preserving order)
  const allFigures = useMemo(
    () => visibleMessages.filter((m) => m.role === 'plot' && m.figure).map((m) => m.figure!),
    [visibleMessages],
  );

  const allMplImages = useMemo(
    () =>
      visibleMessages
        .filter((m) => m.role === 'plot' && !!m.mplImageUrl)
        .map((m) => ({
          imageUrl: m.mplImageUrl!,
          description: m.content || 'Matplotlib plot',
          scriptUrl: m.mplScriptId
            ? `/api/sessions/${m.thumbnailSessionId || ''}/mpl-scripts/${m.mplScriptId}.py`
            : undefined,
        })),
    [visibleMessages],
  );

  // Find the last plot message index to mark it as primary
  const lastPlotIdx = useMemo(
    () => visibleMessages.reduce((acc, m, i) => (m.role === 'plot' ? i : acc), -1),
    [visibleMessages],
  );

  // Build round groups for activity display
  const rounds = useMemo(
    () => groupEventsByRound(messages, toolEvents, memoryEvents, roundMarkers, commentaryEvents),
    [messages, toolEvents, memoryEvents, roundMarkers, commentaryEvents],
  );

  // Build a set of user message IDs that belong to a round (not queued)
  const roundUserMsgIds = useMemo(() => {
    const ids = new Set<string>();
    for (const r of rounds) {
      for (const m of r.userMessages) ids.add(m.id);
    }
    return ids;
  }, [rounds]);

  // Determine the timestamp of the last round_end marker
  const lastRoundEndTs = useMemo(() => {
    for (let i = roundMarkers.length - 1; i >= 0; i--) {
      if (roundMarkers[i].type === 'end') return roundMarkers[i].timestamp;
    }
    return 0;
  }, [roundMarkers]);

  // Identify queued user messages: sent after the last round_end while streaming is active,
  // and not yet assigned to a round (no round_start after them).
  const queuedMsgIds = useMemo(() => {
    const ids = new Set<string>();
    if (!isStreaming) return ids;
    for (const msg of visibleMessages) {
      if (msg.role === 'user' && msg.timestamp > lastRoundEndTs && !roundUserMsgIds.has(msg.id)) {
        ids.add(msg.id);
      }
    }
    return ids;
  }, [isStreaming, visibleMessages, lastRoundEndTs, roundUserMsgIds]);

  // Map each user message timestamp to its round's activity data
  const activityByRound = useMemo(() => {
    const map = new Map<string, {
      tools: ToolEvent[];
      thinking: ThinkingEvent[];
      commentary: CommentaryEvent[];
      isActiveRound: boolean;
      roundStartTs: number;
      roundEndTs?: number;
      roundTokenUsage?: Record<string, number>;
    }>();

    for (const round of rounds) {
      if (round.userMessages.length === 0) continue;
      // Attach activity to the FIRST user message in the round
      const firstUserMsg = round.userMessages[0];
      const isActive = round.endTs === Infinity;
      map.set(firstUserMsg.id, {
        tools: round.toolEvents,
        thinking: round.thinkingEvents,
        commentary: round.commentaryEvents,
        isActiveRound: isActive,
        roundStartTs: round.startTs,
        roundEndTs: isFinite(round.endTs) ? round.endTs : undefined,
        roundTokenUsage: isActive ? (roundTokenUsage ?? undefined) : round.roundTokenUsage,
      });
    }
    return map;
  }, [rounds, roundTokenUsage]);

  // Collect orphan rounds: rounds with events but no user messages
  const orphanRounds = useMemo(
    () => rounds.filter(
      (r) => r.userMessages.length === 0
        && (r.toolEvents.length > 0 || r.thinkingEvents.length > 0),
    ),
    [rounds],
  );

  const showTyping = isStreaming
    && toolEvents.length === 0
    && !visibleMessages.some((m) => m.role === 'agent' && m.timestamp > lastRoundEndTs);

  // Find last agent message index for regenerate button
  const lastAgentIdx = visibleMessages.reduce((acc, m, i) => (m.role === 'agent' ? i : acc), -1);

  // Pre-compute stable figure indices (avoids mutating a counter during render)
  const figureIndices = useMemo(() => {
    const indices = new Map<string, number>();
    let counter = 0;
    for (const msg of visibleMessages) {
      if (msg.role === 'plot' && msg.figure) {
        indices.set(msg.id, counter++);
      }
    }
    return indices;
  }, [visibleMessages]);

  const mplImageIndices = useMemo(() => {
    const indices = new Map<string, number>();
    let counter = 0;
    for (const msg of visibleMessages) {
      if (msg.role === 'plot' && msg.mplImageUrl) {
        indices.set(msg.id, counter++);
      }
    }
    return indices;
  }, [visibleMessages]);

  return (
    <div data-testid="message-list" ref={scrollContainerRef} className="relative flex-1 overflow-y-auto overflow-x-hidden px-4 py-4" role="log" aria-live="polite">
      <div className="max-w-3xl mx-auto space-y-5">
        <AnimatePresence mode="popLayout">
          {visibleMessages.map((msg, i) => {
            const isMplPlot = msg.role === 'plot' && !!msg.mplImageUrl;
            const isJsxComponent = msg.role === 'plot' && !!msg.jsxScriptId;
            const isInlinePlot = msg.role === 'plot' && msg.figure && !isMplPlot && !isJsxComponent;
            const isThumbnailPlot = msg.role === 'plot' && msg.thumbnailUrl && !msg.figure && !isMplPlot && !isJsxComponent;
            const isLargePlot = msg.role === 'plot' && !msg.figure && !msg.thumbnailUrl && msg.figure_url && !isMplPlot && !isJsxComponent;
            const isOrphanedPlot = msg.role === 'plot' && !msg.figure && !msg.thumbnailUrl && !msg.figure_url && !isMplPlot && !isJsxComponent;
            const figIdx = figureIndices.get(msg.id) ?? -1;
            const isQueued = queuedMsgIds.has(msg.id);
            const roundActivity = activityByRound.get(msg.id);

            return (
              <motion.div
                key={msg.id}
                variants={fadeSlideIn}
                initial="hidden"
                animate="visible"
                exit="exit"
                layout
                className="min-w-0"
              >
                {/* Queued indicator for user messages sent during an active round */}
                {isQueued && msg.role === 'user' && (
                  <div className="flex items-center gap-1.5 text-xs text-text-muted mb-1 ml-10 opacity-60">
                    <Clock3 size={11} />
                    <span>Queued</span>
                  </div>
                )}
                {isMplPlot ? (
                  <div
                    className="max-w-full rounded-lg border border-border overflow-hidden bg-panel cursor-pointer transition-all hover:shadow-md"
                    onClick={() => setMplViewerIndex(mplImageIndices.get(msg.id) ?? null)}
                    role="button"
                    tabIndex={0}
                    aria-label={`Matplotlib plot: ${msg.content || 'Untitled'}. Click to expand.`}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ')
                        setMplViewerIndex(mplImageIndices.get(msg.id) ?? null);
                    }}
                  >
                    <div className="flex items-center justify-between px-3 py-1.5 bg-panel border-b border-border">
                      <span className="text-xs text-text-muted font-medium">Matplotlib Plot</span>
                      {msg.mplScriptId && (
                        <a
                          href={`/api/sessions/${msg.thumbnailSessionId || ''}/mpl-scripts/${msg.mplScriptId}.py`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-primary hover:underline"
                          onClick={(e) => e.stopPropagation()}
                        >
                          View Script
                        </a>
                      )}
                    </div>
                    <img
                      src={msg.mplImageUrl}
                      alt={msg.content || 'Matplotlib plot'}
                      className="w-full max-h-[600px] object-contain bg-white"
                      loading="lazy"
                    />
                    {msg.content && (
                      <div className="px-3 py-1.5 border-t border-border">
                        <p className="text-xs text-text-muted">{msg.content}</p>
                      </div>
                    )}
                  </div>
                ) : isJsxComponent ? (
                  <div className="max-w-full rounded-lg border border-border overflow-hidden bg-panel">
                    <div className="flex items-center justify-between px-3 py-1.5 bg-panel border-b border-border">
                      <span className="text-xs text-text-muted font-medium">Recharts Component</span>
                      {msg.jsxScriptId && (
                        <a
                          href={`/api/sessions/${msg.jsxSessionId || ''}/jsx-scripts/${msg.jsxScriptId}.tsx`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-primary hover:underline"
                        >
                          View Source
                        </a>
                      )}
                    </div>
                    <JsxComponent
                      sessionId={msg.jsxSessionId || ''}
                      scriptId={msg.jsxScriptId!}
                    />
                    {msg.content && (
                      <div className="px-3 py-1.5 border-t border-border">
                        <p className="text-xs text-text-muted">{msg.content}</p>
                      </div>
                    )}
                  </div>
                ) : isInlinePlot ? (
                  <PlotlyFigure
                    figure={msg.figure!}
                    isPrimary={i === lastPlotIdx}
                    onOpenFullscreen={() => setViewerIndex(figIdx)}
                  />
                ) : isThumbnailPlot ? (
                  <PlotThumbnail
                    thumbnailUrl={msg.thumbnailUrl!}
                    isLoading={loadingThumbnails.has(msg.id)}
                    onLoadInteractive={() => handleLoadInteractivePlot(msg.id, msg.thumbnailSessionId!)}
                  />
                ) : isLargePlot ? (
                  <div className="rounded-xl border border-border bg-panel p-6 flex items-center justify-between">
                    <div>
                      <h4 className="text-sm font-medium text-text">Large interactive plot</h4>
                      <p className="text-xs text-text-muted mt-1">
                        This figure is too large to display inline. Open it in a new tab for full interactivity.
                      </p>
                    </div>
                    <a
                      href={msg.figure_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 ml-4 flex items-center gap-2 px-4 py-2 rounded-lg
                        bg-primary text-white text-sm font-medium hover:bg-primary-dark transition-colors"
                    >
                      <ExternalLink size={14} />
                      Open in new tab
                    </a>
                  </div>
                ) : isOrphanedPlot ? (
                  <div className="max-w-full rounded-lg border border-border overflow-hidden bg-panel">
                    <div className="flex items-center justify-between px-3 py-1.5 bg-panel border-b border-border">
                      <span className="text-xs text-text-muted font-medium">Plot</span>
                    </div>
                    <div className="flex flex-col items-center justify-center py-8 px-4 gap-3">
                      <ImageOff size={32} className="text-text-muted/50" />
                      <p className="text-sm text-text-muted text-center">Preview unavailable</p>
                      {msg.thumbnailSessionId && (
                        <button
                          onClick={() => handleLoadInteractivePlot(msg.id, msg.thumbnailSessionId!)}
                          disabled={loadingThumbnails.has(msg.id)}
                          className="flex items-center gap-2 px-4 py-2 rounded-lg
                            bg-primary text-white text-sm font-medium hover:bg-primary-dark
                            transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {loadingThumbnails.has(msg.id) ? (
                            <>
                              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                              Loading...
                            </>
                          ) : (
                            'Load interactive plot'
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                ) : (
                  <ChatMessage
                    message={msg}
                    isQueued={isQueued}
                    onRegenerate={i === lastAgentIdx && !isStreaming ? onRegenerate : undefined}
                  />
                )}
                {/* Show activity group after the first user message of each round */}
                {roundActivity && (roundActivity.tools.length > 0 || roundActivity.thinking.length > 0 || roundActivity.commentary.length > 0 || (roundActivity.isActiveRound && isStreaming)) && (
                  <div className="mt-2">
                    <ToolCallGroup
                      events={roundActivity.tools}
                      thinkingEvents={roundActivity.thinking}
                      commentaryEvents={roundActivity.commentary}
                      isStreaming={roundActivity.isActiveRound && isStreaming}
                      streamStartTs={roundActivity.roundStartTs}
                      roundEndTs={roundActivity.roundEndTs}
                      roundTokenUsage={roundActivity.roundTokenUsage}
                    />
                  </div>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>

        {/* Orphan rounds (rounds with events but no user messages) */}
        {orphanRounds.map((round) => (
          <div key={`orphan-${round.roundIndex}`} className="mt-2">
            <ToolCallGroup
              events={round.toolEvents}
              thinkingEvents={round.thinkingEvents}
              commentaryEvents={round.commentaryEvents}
              isStreaming={round.endTs === Infinity && isStreaming}
              streamStartTs={round.startTs}
              roundEndTs={isFinite(round.endTs) ? round.endTs : undefined}
              roundTokenUsage={round.endTs === Infinity ? (roundTokenUsage ?? undefined) : round.roundTokenUsage}
            />
          </div>
        ))}

        {/* Commentary lines or typing indicator */}
        {isStreaming && commentaryEvents.length > 0 && commentaryText && (
          <CommentaryLines events={commentaryEvents} />
        )}
        <AnimatePresence>
          {showTyping && !commentaryText && (
            <motion.div
              variants={fadeSlideIn}
              initial="hidden"
              animate="visible"
              exit="exit"
            >
              <TypingIndicator />
            </motion.div>
          )}
        </AnimatePresence>

        <div ref={bottomRef} />
      </div>

      {/* New messages pill */}
      <AnimatePresence>
        {hasNewContent && !isAtBottom && (
          <motion.button
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            onClick={scrollToBottom}
            className="sticky bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-4 py-2 rounded-full
              bg-primary text-white text-sm font-medium shadow-lg hover:bg-primary-dark transition-colors z-10"
          >
            <ArrowDown size={14} />
            New messages below
          </motion.button>
        )}
      </AnimatePresence>

      {/* Fullscreen figure viewer */}
      {viewerIndex !== null && allFigures.length > 0 && (
        <PlotFullscreen
          figures={allFigures}
          currentIndex={viewerIndex}
          onNavigate={setViewerIndex}
          onClose={() => setViewerIndex(null)}
        />
      )}

      {mplViewerIndex !== null && allMplImages.length > 0 && (
        <MplFullscreen
          images={allMplImages}
          currentIndex={mplViewerIndex}
          onNavigate={setMplViewerIndex}
          onClose={() => setMplViewerIndex(null)}
        />
      )}
    </div>
  );
}
