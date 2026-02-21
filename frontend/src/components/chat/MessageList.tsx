import { useEffect, useRef, useMemo, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChatMessage } from './ChatMessage';
import { ToolCallGroup } from './ToolCallGroup';
import { TypingIndicator } from './TypingIndicator';
import { PlotlyFigure } from '../plot/PlotlyFigure';
import { PlotThumbnail } from '../plot/PlotThumbnail';
import { PlotFullscreen } from '../plot/PlotFullscreen';
import { fadeSlideIn } from '../common/MotionPresets';
import { ArrowDown, ExternalLink } from 'lucide-react';
import type { ChatMessage as ChatMessageType, ToolEvent } from '../../api/types';
import * as api from '../../api/client';
import { useChatStore } from '../../stores/chatStore';
import { extractThinkingEvents, type ThinkingEvent } from '../../utils/groupEventsByTurn';

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
  const [loadingThumbnails, setLoadingThumbnails] = useState<Set<string>>(new Set());

  const handleLoadInteractivePlot = useCallback(async (msgId: string, sessionId: string) => {
    setLoadingThumbnails((prev) => new Set(prev).add(msgId));
    try {
      const { figure, figure_url } = await api.getFigure(sessionId);
      if (figure) {
        // Replace the thumbnail message with a real plot message
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

  // Find the last plot message index to mark it as primary
  const lastPlotIdx = useMemo(
    () => visibleMessages.reduce((acc, m, i) => (m.role === 'plot' ? i : acc), -1),
    [visibleMessages],
  );

  // Group tool events and thinking events by user message timestamp window
  const lastUserIdx = visibleMessages.map((m) => m.role).lastIndexOf('user');
  const thinkingEvents = useMemo(() => extractThinkingEvents(messages), [messages]);
  const activityByUserTs = useMemo(() => {
    const toolMap = new Map<number, ToolEvent[]>();
    const thinkMap = new Map<number, ThinkingEvent[]>();
    const userMsgs = messages.filter((m) => m.role === 'user');
    for (let i = 0; i < userMsgs.length; i++) {
      const startTs = userMsgs[i].timestamp;
      const endTs = i < userMsgs.length - 1 ? userMsgs[i + 1].timestamp : Infinity;
      const tools = toolEvents.filter((e) => e.timestamp >= startTs && e.timestamp < endTs);
      const thinks = thinkingEvents.filter((e) => e.timestamp >= startTs && e.timestamp < endTs);
      if (tools.length > 0) toolMap.set(startTs, tools);
      if (thinks.length > 0) thinkMap.set(startTs, thinks);
    }
    return { toolMap, thinkMap };
  }, [messages, toolEvents, thinkingEvents]);

  const showTyping = isStreaming && !visibleMessages.some((m, i) => m.role === 'agent' && i > lastUserIdx) && toolEvents.length === 0;

  // Find last agent message index for regenerate button
  const lastAgentIdx = visibleMessages.reduce((acc, m, i) => (m.role === 'agent' ? i : acc), -1);

  // Track figure index counter during render
  let figureCounter = 0;

  return (
    <div ref={scrollContainerRef} className="relative flex-1 overflow-y-auto overflow-x-hidden px-4 py-4" role="log" aria-live="polite">
      <div className="max-w-3xl mx-auto space-y-4">
        <AnimatePresence mode="popLayout">
          {visibleMessages.map((msg, i) => {
            const isInlinePlot = msg.role === 'plot' && msg.figure;
            const isThumbnailPlot = msg.role === 'plot' && msg.thumbnailUrl && !msg.figure;
            const isLargePlot = msg.role === 'plot' && !msg.figure && !msg.thumbnailUrl && msg.figure_url;
            const figIdx = isInlinePlot ? figureCounter++ : -1;

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
                {isInlinePlot ? (
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
                ) : (
                  <ChatMessage
                    message={msg}
                    onRegenerate={i === lastAgentIdx && !isStreaming ? onRegenerate : undefined}
                  />
                )}
                {/* Show tool events + thinking for each user message that triggered activity */}
                {msg.role === 'user' && (activityByUserTs.toolMap.has(msg.timestamp) || activityByUserTs.thinkMap.has(msg.timestamp)) && (
                  <div className="mt-2">
                    <ToolCallGroup
                      events={activityByUserTs.toolMap.get(msg.timestamp) ?? []}
                      thinkingEvents={activityByUserTs.thinkMap.get(msg.timestamp)}
                      isStreaming={i === lastUserIdx && isStreaming}
                      streamStartTs={msg.timestamp}
                    />
                  </div>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>

        {/* Typing indicator */}
        <AnimatePresence>
          {showTyping && (
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
    </div>
  );
}
