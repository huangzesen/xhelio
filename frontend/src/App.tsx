import { useEffect, useState, useMemo, useCallback } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Header } from './components/layout/Header';
import { ChatPage } from './pages/ChatPage';
import { DataToolsPage } from './pages/DataToolsPage';
import { SettingsPage } from './pages/SettingsPage';
import { PipelinePage } from './pages/PipelinePage';
import { GalleryPage } from './pages/GalleryPage';
import { MemoryPage } from './pages/MemoryPage';
import { AssetsPage } from './pages/AssetsPage';
import { EurekaPage } from './pages/EurekaPage';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { CommandPalette } from './components/common/CommandPalette';
import { KeyboardShortcuts } from './components/common/KeyboardShortcuts';
import { SetupScreen } from './components/common/SetupScreen';
import { useSessionStore } from './stores/sessionStore';
import { useChatStore } from './stores/chatStore';
import { useTheme } from './hooks/useTheme';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import * as api from './api/client';
import { Loader2 } from 'lucide-react';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activityOpen, setActivityOpen] = useState(true);
  const [initializing, setInitializing] = useState(true);
  const [needsSetup, setNeedsSetup] = useState(false);
  const [setupDone, setSetupDone] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();

  const { activeSessionId, createSession, loadSavedSessions } = useSessionStore();

  const handleCommandPalette = useCallback(() => {
    setCommandPaletteOpen((v) => !v);
  }, []);

  const handleShortcutsDialog = useCallback(() => {
    setShortcutsOpen((v) => !v);
  }, []);

  const shortcuts = useMemo(() => ({
    onToggleSidebar: () => setSidebarOpen((v) => !v),
    onToggleActivity: () => setActivityOpen((v) => !v),
    onCommandPalette: handleCommandPalette,
    onShortcutsHelp: handleShortcutsDialog,
    onFocusInput: () => {
      const textarea = document.querySelector('textarea');
      textarea?.focus();
    },
  }), [handleCommandPalette, handleShortcutsDialog]);

  useKeyboardShortcuts(shortcuts);

  // Initialize: check server, resume persisted session or create new one
  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        const status = await api.getStatus();
        if (cancelled) return;

        // If no API key is configured, show setup screen
        if (!status.api_key_configured) {
          setNeedsSetup(true);
          setInitializing(false);
          return;
        }

        // Check if we have a persisted session from localStorage
        const persistedId = useSessionStore.getState().activeSessionId;
        if (persistedId) {
          try {
            const detail = await api.getSession(persistedId);
            // Session still alive — refresh token usage from backend
            useSessionStore.getState().setTokenUsage(detail.token_usage ?? {});
            // If session is still busy from a previous in-flight request
            // (e.g. browser refresh mid-stream), cancel it so the user can
            // send new messages.
            if (detail.busy) {
              await api.cancelChat(persistedId).catch(() => {});
            }
            // Recover panel state from server event log if messages are empty
            // (localStorage was cleared or didn't persist due to quota)
            if (useChatStore.getState().messages.length === 0) {
              try {
                const { events } = await api.getSessionEvents(persistedId);
                if (events.length > 0) {
                  useChatStore.getState().rebuildFromEventLog(events, persistedId);
                  // Check if rebuild already created inline plot messages
                  const hasPlotMessages = useChatStore.getState().messages.some((m) => m.role === 'plot');
                  if (!hasPlotMessages) {
                    // Also recover figure (may be inline JSON or large HTML URL)
                    const { figure, figure_url } = await api.getFigure(persistedId);
                    if (figure) {
                      useChatStore.setState((s) => ({
                        figureJson: figure,
                        messages: [
                          ...s.messages,
                          {
                            id: `plot-recover-${Date.now()}`,
                            role: 'plot' as const,
                            content: '',
                            timestamp: Date.now(),
                            figure,
                          },
                        ],
                      }));
                    } else {
                      // Large figure (figure_url) or no figure at all —
                      // prefer thumbnail for instant visual preview.
                      // Clicking the thumbnail loads the full figure (or figure_url card).
                      const thumbUrl = api.getFigureThumbnailUrl(persistedId);
                      let thumbShown = false;
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
                                thumbnailSessionId: persistedId,
                              },
                            ],
                          }));
                          thumbShown = true;
                        }
                      } catch {
                        // No thumbnail available
                      }
                      // Fallback: show "open in new tab" card if no thumbnail
                      if (!thumbShown && figure_url) {
                        useChatStore.setState((s) => ({
                          messages: [
                            ...s.messages,
                            {
                              id: `plot-recover-${Date.now()}`,
                              role: 'plot' as const,
                              content: '',
                              timestamp: Date.now(),
                              figure_url,
                            },
                          ],
                        }));
                      }
                    }
                  }
                }
              } catch {
                // Event log not available — no recovery needed
              }
            }
          } catch {
            // Session no longer exists in memory — clear stale UI state
            // immediately so no stale data flashes, then try resuming from disk.
            if (cancelled) return;
            useChatStore.getState().clearChat();
            useSessionStore.getState().setActiveSessionId(null);
            useSessionStore.getState().setTokenUsage({});
            try {
              const info = await api.resumeSession(persistedId);
              if (cancelled) return;
              useSessionStore.getState().setActiveSessionId(info.session_id);
              useSessionStore.getState().setTokenUsage({});
              // Rebuild UI from event log
              const eventLog = info.event_log || [];
              if (eventLog.length > 0) {
                useChatStore.getState().rebuildFromEventLog(eventLog, info.session_id);
              }
              // Fetch token usage
              const detail = await api.getSession(info.session_id);
              useSessionStore.getState().setTokenUsage(detail.token_usage ?? {});
              // Check if rebuild already created inline plot messages
              const hasPlotMessages = useChatStore.getState().messages.some((m) => m.role === 'plot');
              if (!hasPlotMessages) {
                // Recover figure: prefer thumbnail, fall back to full figure
                const thumbUrl = api.getFigureThumbnailUrl(info.session_id);
                let figureRecovered = false;
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
                          thumbnailSessionId: info.session_id,
                        },
                      ],
                    }));
                    figureRecovered = true;
                  }
                } catch {
                  // No thumbnail available
                }
                if (!figureRecovered) {
                  try {
                    const { figure, figure_url } = await api.getFigure(info.session_id);
                    if (figure) {
                      useChatStore.setState((s) => ({
                        figureJson: figure,
                        messages: [
                          ...s.messages,
                          {
                            id: `plot-recover-${Date.now()}`,
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
                            id: `plot-recover-${Date.now()}`,
                            role: 'plot' as const,
                            content: '',
                            timestamp: Date.now(),
                            figure_url,
                          },
                        ],
                      }));
                    }
                  } catch {
                    // No figure available
                  }
                }
              }
            } catch {
              // Disk session also gone — create fresh
              if (cancelled) return;
              await createSession();
            }
          }
        } else {
          await createSession();
        }

        if (cancelled) return;
        await loadSavedSessions();
      } catch (err) {
        if (!cancelled) {
          setError(`Cannot connect to server: ${(err as Error).message}`);
        }
      } finally {
        if (!cancelled) setInitializing(false);
      }
    }
    init();
    return () => { cancelled = true; };
  }, [setupDone]); // eslint-disable-line react-hooks/exhaustive-deps

  // Collapse panels on mobile
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 768px)');
    if (mq.matches) {
      setSidebarOpen(false);
      setActivityOpen(false);
    }
  }, []);

  if (initializing) {
    return (
      <div data-testid="app-loading" className="h-full flex items-center justify-center bg-surface">
        <div className="flex flex-col items-center gap-3 text-text-muted">
          <Loader2 size={32} className="animate-spin text-primary" />
          <span className="text-sm">Connecting to server...</span>
        </div>
      </div>
    );
  }

  if (needsSetup) {
    return (
      <SetupScreen onComplete={() => {
        setNeedsSetup(false);
        setInitializing(true);
        setSetupDone((n) => n + 1);
      }} />
    );
  }

  if (error && !activeSessionId) {
    return (
      <div data-testid="app-error" className="h-full flex items-center justify-center bg-surface">
        <div className="text-center max-w-md px-6">
          <div className="text-status-error-text text-lg font-semibold mb-2">Connection Error</div>
          <p className="text-sm text-text-muted mb-4">{error}</p>
          <p className="text-xs text-text-muted">
            Make sure the API server is running:{' '}
            <code className="bg-code-bg text-text-inverted px-1.5 py-0.5 rounded text-xs">python api_server.py</code>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-surface">
      <a href="#main-content" className="skip-to-content">Skip to content</a>
      <Header
        sidebarOpen={sidebarOpen}
        activityOpen={activityOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
        onToggleActivity={() => setActivityOpen((v) => !v)}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <ErrorBoundary>
        <main id="main-content" className="flex-1 flex flex-col overflow-hidden">
        <Routes>
          <Route path="/" element={<ChatPage sidebarOpen={sidebarOpen} activityOpen={activityOpen} />} />
          <Route path="/data" element={<DataToolsPage />} />
          <Route path="/pipeline" element={<PipelinePage />} />
          <Route path="/gallery" element={<GalleryPage />} />
          <Route path="/memory" element={<MemoryPage />} />
          <Route path="/eureka" element={<EurekaPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/settings/assets" element={<AssetsPage />} />
        </Routes>
        </main>
      </ErrorBoundary>

      <CommandPalette
        open={commandPaletteOpen}
        onOpenChange={setCommandPaletteOpen}
        theme={theme}
        onToggleTheme={toggleTheme}
        onOpenShortcuts={() => setShortcutsOpen(true)}
      />

      <KeyboardShortcuts
        open={shortcutsOpen}
        onOpenChange={setShortcutsOpen}
      />
    </div>
  );
}
