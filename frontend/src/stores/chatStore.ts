import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatMessage, ToolEvent, LogLine, MemoryEvent, PlotlyFigure, SessionEventRecord } from '../api/types';
import { chatStream } from '../api/sse';
import * as api from '../api/client';
import { useSessionStore } from './sessionStore';
import { useMemoryStore } from './memoryStore';

interface ChatState {
  messages: ChatMessage[];
  toolEvents: ToolEvent[];
  logLines: LogLine[];
  memoryEvents: MemoryEvent[];
  isStreaming: boolean;
  isCancelling: boolean;
  abortController: AbortController | null;
  figureJson: PlotlyFigure | null;
  storageWarning: boolean;

  sendMessage: (sessionId: string, message: string) => Promise<void>;
  cancelStream: (sessionId: string) => Promise<void>;
  regenerateLastResponse: (sessionId: string) => Promise<void>;
  rebuildFromEventLog: (events: SessionEventRecord[]) => void;
  clearChat: () => void;
  dismissStorageWarning: () => void;
}

let nextId = 0;
function uid() {
  return `msg-${Date.now()}-${nextId++}`;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
  messages: [],
  toolEvents: [],
  logLines: [],
  memoryEvents: [],
  isStreaming: false,
  isCancelling: false,
  abortController: null,
  figureJson: null,
  storageWarning: false,

  sendMessage: async (sessionId: string, message: string) => {
    if (get().isCancelling) return;

    // Intercept slash commands
    if (message.startsWith('/')) {
      const command = message.slice(1).split(/\s+/)[0].toLowerCase();
      const userMsg: ChatMessage = {
        id: uid(),
        role: 'user',
        content: message,
        timestamp: Date.now(),
      };
      set((s) => ({ messages: [...s.messages, userMsg] }));

      try {
        const response = await api.executeCommand(sessionId, command);

        // Special handling for /reset
        if (command === 'reset' && response.data?.session_id) {
          get().clearChat();
          useSessionStore.getState().setActiveSessionId(response.data.session_id as string);
        }

        // Special handling for /branch
        if (command === 'branch' && response.data?.session_id) {
          useSessionStore.getState().setActiveSessionId(response.data.session_id as string);
          useSessionStore.getState().loadSavedSessions();
        }

        set((s) => ({
          messages: [
            ...s.messages,
            {
              id: uid(),
              role: 'system' as const,
              content: response.content,
              timestamp: Date.now(),
            },
          ],
        }));
      } catch (err) {
        set((s) => ({
          messages: [
            ...s.messages,
            {
              id: uid(),
              role: 'system' as const,
              content: `Error: ${(err as Error).message}`,
              timestamp: Date.now(),
            },
          ],
        }));
      }
      return;
    }

    const userMsg: ChatMessage = {
      id: uid(),
      role: 'user',
      content: message,
      timestamp: Date.now(),
    };

    const controller = new AbortController();
    set((s) => ({
      messages: [...s.messages, userMsg],
      isStreaming: true,
      abortController: controller,
    }));

    // Prepare agent message placeholder
    const agentMsgId = uid();
    let agentText = '';

    try {
      for await (const event of chatStream(sessionId, message, controller.signal)) {
        switch (event.type) {
          case 'text_delta':
            agentText += event.text;
            set((s) => {
              const msgs = [...s.messages];
              const existing = msgs.findIndex((m) => m.id === agentMsgId);
              if (existing >= 0) {
                msgs[existing] = { ...msgs[existing], content: agentText };
              } else {
                msgs.push({
                  id: agentMsgId,
                  role: 'agent',
                  content: agentText,
                  timestamp: Date.now(),
                });
              }
              return { messages: msgs };
            });
            break;

          case 'thinking':
            set((s) => ({
              messages: [
                ...s.messages,
                {
                  id: uid(),
                  role: 'thinking' as const,
                  content: event.text,
                  timestamp: Date.now(),
                },
              ],
            }));
            break;

          case 'tool_call':
            set((s) => ({
              toolEvents: [
                ...s.toolEvents,
                {
                  id: uid(),
                  type: 'call',
                  tool_name: event.tool_name,
                  tool_args: event.tool_args,
                  timestamp: Date.now(),
                },
              ],
            }));
            break;

          case 'tool_result':
            set((s) => ({
              toolEvents: [
                ...s.toolEvents,
                {
                  id: uid(),
                  type: 'result',
                  tool_name: event.tool_name,
                  status: event.status,
                  timestamp: Date.now(),
                },
              ],
            }));
            break;

          case 'plot': {
            const { figure, figure_url } = await api.getFigure(sessionId);
            set((s) => ({
              figureJson: figure,  // keep for gallery/pipeline pages (null for large figures)
              messages: [
                ...s.messages,
                {
                  id: uid(),
                  role: 'plot' as const,
                  content: '',
                  timestamp: Date.now(),
                  figure: figure ?? undefined,
                  figure_url,
                },
              ],
            }));
            break;
          }

          case 'log_line':
            set((s) => ({
              logLines: [
                ...s.logLines,
                {
                  id: uid(),
                  text: event.text,
                  level: event.level,
                  timestamp: Date.now(),
                },
              ],
            }));
            break;

          case 'memory_update':
            set((s) => ({
              memoryEvents: [
                ...s.memoryEvents,
                {
                  id: uid(),
                  actions: (event as { actions: Record<string, number> }).actions ?? {},
                  timestamp: Date.now(),
                },
              ],
            }));
            break;

          case 'session_title':
            // Refresh saved sessions to pick up the new title
            useSessionStore.getState().loadSavedSessions();
            break;

          case 'done':
            useSessionStore.getState().setTokenUsage(event.token_usage);
            // Refresh saved sessions so the sidebar picks up newly saved sessions
            useSessionStore.getState().loadSavedSessions();
            // Auto-refresh memories after agent turn completes
            { const sid = useSessionStore.getState().activeSessionId;
              if (sid) useMemoryStore.getState().loadMemories(sid); }
            break;

          case 'error':
            set((s) => {
              const msgs = [...s.messages];
              const existing = msgs.findIndex((m) => m.id === agentMsgId);
              const errorMsg: ChatMessage = {
                id: agentMsgId,
                role: 'agent',
                content: `Error: ${event.message}`,
                timestamp: Date.now(),
              };
              if (existing >= 0) {
                msgs[existing] = errorMsg;
              } else {
                msgs.push(errorMsg);
              }
              return { messages: msgs };
            });
            break;
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        const errorContent = `Error: ${(err as Error).message}`;
        set((s) => {
          const msgs = [...s.messages];
          const existing = msgs.findIndex((m) => m.id === agentMsgId);
          const errorMsg: ChatMessage = {
            id: agentMsgId,
            role: 'agent',
            content: errorContent,
            timestamp: Date.now(),
          };
          if (existing >= 0) {
            msgs[existing] = errorMsg;
          } else {
            msgs.push(errorMsg);
          }
          return { messages: msgs };
        });
      }
    } finally {
      set({ isStreaming: false, abortController: null });
    }
  },

  regenerateLastResponse: async (sessionId: string) => {
    const { messages, isStreaming } = get();
    if (isStreaming) return;

    // Find the last user message
    let lastUserIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserIdx = i;
        break;
      }
    }
    if (lastUserIdx < 0) return;

    const lastUserMessage = messages[lastUserIdx].content;
    // Remove everything after (and including any agent response to) the last user message
    const trimmed = messages.slice(0, lastUserIdx);
    set({ messages: trimmed, toolEvents: [] });

    // Re-send the same message
    await get().sendMessage(sessionId, lastUserMessage);
  },

  cancelStream: async (sessionId: string) => {
    const { abortController } = get();
    if (abortController) {
      abortController.abort();
    }
    set({ isStreaming: false, abortController: null, isCancelling: true });
    try {
      await api.cancelChat(sessionId);
    } catch {
      // ignore
    }
    set({ isCancelling: false });
  },

  rebuildFromEventLog: (events: SessionEventRecord[]) => {
    const messages: ChatMessage[] = [];
    const toolEvents: ToolEvent[] = [];
    const logLines: LogLine[] = [];
    const memoryEvents: MemoryEvent[] = [];

    for (const ev of events) {
      const ts = new Date(ev.ts).getTime();

      // Chat messages
      if (ev.type === 'user_message') {
        messages.push({
          id: uid(),
          role: 'user',
          content: (ev.data?.text as string) || ev.msg,
          timestamp: ts,
        });
      } else if (ev.type === 'agent_response') {
        messages.push({
          id: uid(),
          role: 'agent',
          content: (ev.data?.text as string) || ev.msg,
          timestamp: ts,
        });
      } else if (ev.type === 'thinking' && ev.tags?.includes('display')) {
        messages.push({
          id: uid(),
          role: 'thinking',
          content: (ev.data?.text as string) || ev.msg,
          timestamp: ts,
        });
      }

      // Tool events
      if (ev.type === 'tool_call') {
        toolEvents.push({
          id: uid(),
          type: 'call',
          tool_name: (ev.data?.tool_name as string) || '',
          tool_args: (ev.data?.tool_args as Record<string, unknown>) || {},
          timestamp: ts,
        });
      } else if (ev.type === 'tool_result') {
        toolEvents.push({
          id: uid(),
          type: 'result',
          tool_name: (ev.data?.tool_name as string) || '',
          status: (ev.data?.status as string) || '',
          timestamp: ts,
        });
      }

      // Memory events
      if (ev.type === 'memory_extraction_done') {
        memoryEvents.push({
          id: uid(),
          actions: (ev.data?.actions as Record<string, number>) || {},
          timestamp: ts,
        });
      }

      // Log lines (requires "console" tag — matches SSEEventListener)
      if (ev.tags?.includes('console')) {
        logLines.push({
          id: uid(),
          text: ev.msg,
          level: ev.level,
          timestamp: ts,
        });
      }
    }

    set({ messages, toolEvents, logLines, memoryEvents });
  },

  clearChat: () =>
    set({
      messages: [],
      toolEvents: [],
      logLines: [],
      memoryEvents: [],
      isStreaming: false,
      isCancelling: false,
      abortController: null,
      figureJson: null,
      storageWarning: false,
    }),

  dismissStorageWarning: () => set({ storageWarning: false }),
}),
    {
      name: 'helion-chat',
      partialize: (state) => ({
        messages: state.messages,
        toolEvents: state.toolEvents,
        logLines: state.logLines,
        memoryEvents: state.memoryEvents,
        figureJson: state.figureJson,
      }),
      storage: {
        getItem: (name) => {
          const str = localStorage.getItem(name);
          return str ? JSON.parse(str) : null;
        },
        setItem: (name, value) => {
          try {
            localStorage.setItem(name, JSON.stringify(value));
          } catch {
            // localStorage quota exceeded — prune old data and retry
            console.warn('[helion] localStorage quota exceeded, pruning old log/tool data');
            const MAX_KEPT = 200;
            const state = value?.state;
            if (state) {
              if (Array.isArray(state.logLines) && state.logLines.length > MAX_KEPT) {
                state.logLines = state.logLines.slice(-MAX_KEPT);
              }
              if (Array.isArray(state.toolEvents) && state.toolEvents.length > MAX_KEPT) {
                state.toolEvents = state.toolEvents.slice(-MAX_KEPT);
              }
            }
            try {
              localStorage.setItem(name, JSON.stringify(value));
            } catch {
              // Still too large — clear persisted chat to prevent permanent breakage
              console.warn('[helion] localStorage still full after pruning, clearing chat persistence');
              localStorage.removeItem(name);
            }
            // Set warning flag (outside persist to avoid re-triggering)
            useChatStore.setState({ storageWarning: true });
          }
        },
        removeItem: (name) => localStorage.removeItem(name),
      },
    },
  ),
);
