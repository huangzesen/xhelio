import { useCallback, useEffect, useRef, useState } from 'react';
import { Sidebar } from '../components/layout/Sidebar';
import { ActivityPanel } from '../components/layout/ActivityPanel';
import { ChatContainer } from '../components/chat/ChatContainer';
import { useSessionStore } from '../stores/sessionStore';
import { useChatStore } from '../stores/chatStore';
import type { ChatMessage } from '../api/types';
import * as api from '../api/client';

const MIN_PANEL_WIDTH = 240;
const MAX_PANEL_WIDTH = 550;

interface Props {
  sidebarOpen: boolean;
  activityOpen: boolean;
}

export function ChatPage({ sidebarOpen, activityOpen }: Props) {
  const {
    activeSessionId,
    model,
    resuming,
    savedSessions,
    tokenUsage,
    createSession,
    resumeSession,
    loadSavedSessions,
    renameSession,
  } = useSessionStore();

  const { messages, toolEvents, logLines, memoryEvents, commentaryEvents, roundMarkers, isStreaming, clearChat } = useChatStore();

  const [sidebarWidth, setSidebarWidth] = useState(320);
  const [activityWidth, setActivityWidth] = useState(320);
  const [isDragging, setIsDragging] = useState(false);
  // Track active drag cleanup so we can remove listeners if component unmounts mid-drag
  const dragCleanup = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => { dragCleanup.current?.(); };
  }, []);

  const startResize = useCallback(
    (side: 'sidebar' | 'activity') => (e: React.MouseEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startWidth = side === 'sidebar' ? sidebarWidth : activityWidth;
      const setter = side === 'sidebar' ? setSidebarWidth : setActivityWidth;

      setIsDragging(true);

      let rafId = 0;
      const onMouseMove = (ev: MouseEvent) => {
        if (rafId) return;
        rafId = requestAnimationFrame(() => {
          const delta = side === 'sidebar' ? ev.clientX - startX : startX - ev.clientX;
          const newWidth = Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, startWidth + delta));
          setter(newWidth);
          rafId = 0;
        });
      };

      const cleanup = () => {
        if (rafId) cancelAnimationFrame(rafId);
        setIsDragging(false);
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        dragCleanup.current = null;
      };

      const onMouseUp = () => cleanup();

      dragCleanup.current = cleanup;
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    },
    [sidebarWidth, activityWidth],
  );

  const handleNewChat = useCallback(async () => {
    try {
      const oldId = activeSessionId;
      clearChat();
      await createSession();
      // Clean up old session after new one is created successfully
      if (oldId) {
        api.deleteSession(oldId).catch(() => {});
      }
    } catch (err) {
      console.error('[helion] Failed to create new session:', err);
    }
  }, [activeSessionId, clearChat, createSession]);

  const handleResumeSession = useCallback(async (savedId: string) => {
    try {
      // Resume first — clearChat only after resume succeeds (Bug #2 fix)
      await resumeSession(savedId);
      clearChat();
      const sess = useSessionStore.getState();
      if (sess.activeSessionId) {
        // Prefer structured event_log for full panel rebuild;
        // fall back to display_log for old sessions without events.jsonl
        const eventLog = sess.lastEventLog || [];
        if (eventLog.length > 0) {
          useChatStore.getState().rebuildFromEventLog(eventLog, sess.activeSessionId ?? undefined);
        } else {
          // Legacy path: display_log only has user/agent/thinking messages
          const displayLog = sess.lastDisplayLog || [];
          const restoredMessages: ChatMessage[] = displayLog
            .filter((e) => e.role === 'user' || e.role === 'agent' || e.role === 'thinking')
            .map((e, i) => ({
              id: `restored-${i}-${Date.now()}`,
              role: e.role as 'user' | 'agent' | 'thinking',
              content: e.content,
              timestamp: e.timestamp ? new Date(e.timestamp).getTime() : Date.now(),
            }));
          useChatStore.setState({ messages: restoredMessages });
        }

        // Check if rebuildFromEventLog already created inline plot messages
        // (from per-render thumbnails). Only fall back to single thumbnail
        // if no plots were created (backward compat for old sessions).
        const hasPlotMessages = useChatStore.getState().messages.some((m) => m.role === 'plot');
        if (!hasPlotMessages) {
          // Show a lightweight PNG thumbnail immediately (no data loading);
          // the full interactive Plotly figure is fetched on demand.
          const thumbUrl = api.getFigureThumbnailUrl(sess.activeSessionId);
          let figureShown = false;
          // Probe whether the thumbnail exists with a HEAD request
          try {
            const thumbRes = await fetch(thumbUrl, { method: 'HEAD' });
            if (thumbRes.ok) {
              useChatStore.setState((s) => ({
                messages: [
                  ...s.messages,
                  {
                    id: `plot-thumb-${Date.now()}`,
                    role: 'plot' as const,
                    content: '',
                    timestamp: Date.now(),
                    thumbnailUrl: thumbUrl,
                    thumbnailSessionId: sess.activeSessionId ?? undefined,
                  },
                ],
              }));
              figureShown = true;
            }
          } catch {
            // Network error — thumbnail unavailable
          }

          // Fallback: try full figure if thumbnail wasn't shown (404 or network error)
          if (!figureShown) {
            try {
              const { figure, figure_url } = await api.getFigure(sess.activeSessionId);
              if (figure) {
                useChatStore.setState((s) => ({
                  figureJson: figure,
                  messages: [
                    ...s.messages,
                    {
                      id: `plot-resume-${Date.now()}`,
                      role: 'plot' as const,
                      content: '',
                      timestamp: Date.now(),
                      figure,
                    },
                  ],
                }));
              } else if (figure_url) {
                useChatStore.setState((s) => ({
                  messages: [
                    ...s.messages,
                    {
                      id: `plot-resume-${Date.now()}`,
                      role: 'plot' as const,
                      content: '',
                      timestamp: Date.now(),
                      figure_url,
                    },
                  ],
                }));
              }
            } catch {
              // no figure available
            }
          }
        }
      }
    } catch {
      // handled by store
    }
  }, [clearChat, resumeSession]);

  const handleRenameSession = useCallback(async (id: string, name: string) => {
    try {
      await renameSession(id, name);
    } catch {
      // handled by store
    }
  }, [renameSession]);

  const handleDeleteSession = useCallback(async (id: string) => {
    const isActive = id === activeSessionId;
    // Delete from disk
    await api.deleteSavedSession(id).catch(() => {});
    // Delete from live memory (no-op if not live)
    await api.deleteSession(id).catch(() => {});
    if (isActive) {
      clearChat();
      await createSession();
    }
    await loadSavedSessions();
  }, [activeSessionId, clearChat, createSession, loadSavedSessions]);

  return (
    <div
      className={`flex-1 flex overflow-hidden ${isDragging ? 'select-none cursor-col-resize' : ''}`}
    >
      {/* Sidebar — resizable, slides in/out */}
      <div
        className={`shrink-0 overflow-hidden ${isDragging ? '' : 'transition-[width] duration-200 ease-in-out'}`}
        style={{ width: sidebarOpen ? sidebarWidth : 0 }}
      >
        <div className="w-full h-full" style={{ minWidth: sidebarWidth }}>
          <Sidebar
            savedSessions={savedSessions}
            activeSessionId={activeSessionId}
            resuming={resuming}
            onNewChat={handleNewChat}
            onResumeSession={handleResumeSession}
            onDeleteSession={handleDeleteSession}
            onRenameSession={handleRenameSession}
          />
        </div>
      </div>

      {/* Sidebar resize handle */}
      {sidebarOpen && (
        <div
          className="shrink-0 w-1 cursor-col-resize hover:bg-blue-500/30 active:bg-blue-500/50 transition-colors"
          onMouseDown={startResize('sidebar')}
        />
      )}

      {/* Main chat area — fills remaining space */}
      <div className="flex-1 min-w-0">
        <ChatContainer />
      </div>

      {/* Activity resize handle */}
      {activityOpen && (
        <div
          className="shrink-0 w-1 cursor-col-resize hover:bg-blue-500/30 active:bg-blue-500/50 transition-colors"
          onMouseDown={startResize('activity')}
        />
      )}

      {/* Activity panel — resizable, slides in/out */}
      <div
        className={`shrink-0 overflow-hidden ${isDragging ? '' : 'transition-[width] duration-200 ease-in-out'}`}
        style={{ width: activityOpen ? activityWidth : 0 }}
      >
        <div className="w-full h-full" style={{ minWidth: activityWidth }}>
          <ActivityPanel
            model={model}
            tokenUsage={tokenUsage}
            messages={messages}
            toolEvents={toolEvents}
            logLines={logLines}
            memoryEvents={memoryEvents}
            commentaryEvents={commentaryEvents}
            roundMarkers={roundMarkers}
            isStreaming={isStreaming}
          />
        </div>
      </div>
    </div>
  );
}
