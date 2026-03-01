import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatMessage, ToolEvent, LogLine, MemoryEvent, CommentaryEvent, PlotlyFigure, SessionEventRecord, SSEEvent } from '../api/types';
import { subscribeToSession } from '../api/sse';
import * as api from '../api/client';
import { useSessionStore } from './sessionStore';
import { useMemoryStore } from './memoryStore';
import { useEurekaStore } from './eurekaStore';

export interface RoundMarker {
  type: 'start' | 'end';
  timestamp: number;
  roundTokenUsage?: Record<string, number>;
}

interface ChatState {
  messages: ChatMessage[];
  toolEvents: ToolEvent[];
  logLines: LogLine[];
  memoryEvents: MemoryEvent[];
  commentaryText: string | null;
  commentaryEvents: CommentaryEvent[];
  isStreaming: boolean;
  isCancelling: boolean;
  figureJson: PlotlyFigure | null;
  storageWarning: boolean;
  roundMarkers: RoundMarker[];
  roundTokenUsage: Record<string, number> | null;

  sendMessage: (sessionId: string, message: string) => Promise<void>;
  cancelStream: (sessionId: string) => Promise<void>;
  regenerateLastResponse: (sessionId: string) => Promise<void>;
  rebuildFromEventLog: (events: SessionEventRecord[], sessionId?: string) => void;
  ensureSubscribed: (sessionId: string) => void;
  closeSubscription: () => void;
  clearChat: () => void;
  dismissStorageWarning: () => void;
}

let nextId = 0;
function uid() {
  return `msg-${Date.now()}-${nextId++}`;
}

// Module-level persistent EventSource (not in Zustand state — not serializable)
let _eventSource: EventSource | null = null;
let _subscribedSessionId: string | null = null;
// Current agent message ID — each text_delta event from a turn appends to this
let _currentAgentMsgId: string | null = null;
let _currentAgentText = '';
let _textBuffer = '';
let _rafHandle: number | null = null;
// Reconnect timer
let _reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function flushTextBuffer(
  set: (fn: (s: ChatState) => Partial<ChatState>) => void,
  agentMsgId: string,
) {
  if (!_textBuffer) {
    _rafHandle = null;
    return;
  }
  const chunk = _textBuffer;
  _textBuffer = '';
  _rafHandle = null;

  _currentAgentText += chunk;
  const text = _currentAgentText;

  set((s) => {
    const msgs = [...s.messages];
    const existing = msgs.findIndex((m) => m.id === agentMsgId);
    if (existing >= 0) {
      msgs[existing] = { ...msgs[existing], content: text };
    } else {
      msgs.push({
        id: agentMsgId,
        role: 'agent',
        content: text,
        timestamp: Date.now(),
      });
    }
    return { messages: msgs, commentaryText: null };
  });
}

/**
 * Handle a single SSE event from the persistent EventSource.
 * Shared by both live subscription and (potentially) replay.
 */
function handleSSEEvent(
  event: SSEEvent,
  set: (fn: (s: ChatState) => Partial<ChatState>) => void,
  get: () => ChatState,
  sessionId: string,
) {
  switch (event.type) {
    case 'text_delta': {
      const ev = event as { text: string; commentary?: boolean; agent?: string };
      if (ev.commentary) {
        // Commentary: update live line (replace previous) and append to events
        const text = ev.text.replace(/\n+$/, '');
        set((s) => ({
          commentaryText: text,
          commentaryEvents: [
            ...s.commentaryEvents,
            {
              id: uid(),
              text,
              agent: ev.agent || 'orchestrator',
              timestamp: Date.now(),
            },
          ],
        }));
        break;
      }
      // Regular text delta — clear commentary line and accumulate into agent message
      // Lazily allocate an agent message ID for this turn
      if (!_currentAgentMsgId) {
        _currentAgentMsgId = uid();
        _currentAgentText = '';
      }
      const agentMsgId = _currentAgentMsgId;
      _textBuffer += ev.text;
      if (!_rafHandle) {
        _rafHandle = requestAnimationFrame(() => flushTextBuffer(set, agentMsgId));
      }
      break;
    }

    case 'thinking':
      set((s) => ({
        messages: [
          ...s.messages,
          {
            id: uid(),
            role: 'thinking' as const,
            content: (event as { text: string }).text,
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
            tool_name: (event as { tool_name: string }).tool_name,
            tool_args: (event as { tool_args: Record<string, unknown> }).tool_args,
            timestamp: Date.now(),
            agent: (event as { agent?: string }).agent,
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
            tool_name: (event as { tool_name: string }).tool_name,
            status: (event as { status: string }).status,
            timestamp: Date.now(),
            agent: (event as { agent?: string }).agent,
          },
        ],
      }));
      break;

    case 'plot': {
      api.getFigure(sessionId).then(({ figure, figure_url }) => {
        set((s) => ({
          figureJson: figure,
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
      });
      break;
    }

    case 'mpl_image': {
      const { script_id, description } = event as { script_id: string; description: string };
      const imageUrl = `/api/sessions/${sessionId}/mpl-outputs/${script_id}.png`;
      set((s) => ({
        messages: [
          ...s.messages,
          {
            id: uid(),
            role: 'plot' as const,
            content: description || 'Matplotlib plot',
            timestamp: Date.now(),
            mplImageUrl: imageUrl,
            mplScriptId: script_id,
            thumbnailSessionId: sessionId,
          },
        ],
      }));
      break;
    }

    case 'jsx_component': {
      const { script_id, description } = event as { script_id: string; description: string };
      set((s) => ({
        messages: [
          ...s.messages,
          {
            id: uid(),
            role: 'plot' as const,
            content: description || 'Recharts component',
            timestamp: Date.now(),
            jsxScriptId: script_id,
            jsxSessionId: sessionId,
          },
        ],
      }));
      break;
    }

    case 'log_line':
      set((s) => {
        const updated = [
          ...s.logLines,
          {
            id: uid(),
            text: (event as { text: string }).text,
            level: (event as { level: string }).level,
            details: (event as { details?: string }).details,
            timestamp: Date.now(),
          },
        ];
        return { logLines: updated.length > 300 ? updated.slice(-300) : updated };
      });
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

    case 'insight_result':
      if ((event as { level: string }).level !== 'debug') {
        set((s) => ({
          messages: [
            ...s.messages,
            {
              id: uid(),
              role: 'insight' as const,
              content: (event as { text: string }).text,
              timestamp: Date.now(),
            },
          ],
        }));
      }
      break;

    case 'insight_feedback':
      if ((event as { level: string }).level !== 'debug') {
        set((s) => ({
          messages: [
            ...s.messages,
            {
              id: uid(),
              role: 'insight_feedback' as const,
              content: (event as { text: string }).text,
              timestamp: Date.now(),
            },
          ],
        }));
      }
      break;

    case 'eureka_finding': {
      const { type: _, ...eureka } = event;
      useEurekaStore.getState().addEureka(eureka);
      break;
    }

    case 'session_title':
      useSessionStore.getState().loadSavedSessions();
      break;

    case 'round_start':
      if (_rafHandle) {
        cancelAnimationFrame(_rafHandle);
        _rafHandle = null;
      }
      if (_textBuffer && _currentAgentMsgId) {
        flushTextBuffer(set, _currentAgentMsgId);
      }
      _currentAgentMsgId = null;
      _currentAgentText = '';
      _textBuffer = '';
      set((s) => ({
        isStreaming: true,
        commentaryText: null,
        roundTokenUsage: null,
        roundMarkers: [...s.roundMarkers, { type: 'start', timestamp: Date.now() }],
      }));
      break;

    case 'round_end':
      if (_rafHandle) {
        cancelAnimationFrame(_rafHandle);
        _rafHandle = null;
      }
      if (_textBuffer && _currentAgentMsgId) {
        flushTextBuffer(set, _currentAgentMsgId);
      }
      useSessionStore.getState().setTokenUsage((event as { token_usage: Record<string, number> }).token_usage);
      useSessionStore.getState().loadSavedSessions();
      { const sid = useSessionStore.getState().activeSessionId;
        if (sid) useMemoryStore.getState().loadMemories(sid); }
      _currentAgentMsgId = null;
      _currentAgentText = '';
      _textBuffer = '';
      set((s) => ({
        isStreaming: false,
        commentaryText: null,
        roundTokenUsage: (event as { round_token_usage: Record<string, number> }).round_token_usage,
        roundMarkers: [...s.roundMarkers, {
          type: 'end',
          timestamp: Date.now(),
          roundTokenUsage: (event as { round_token_usage: Record<string, number> }).round_token_usage,
        }],
      }));
      break;

    case 'error': {
      if (_rafHandle) {
        cancelAnimationFrame(_rafHandle);
        _rafHandle = null;
      }
      if (_textBuffer && _currentAgentMsgId) {
        flushTextBuffer(set, _currentAgentMsgId);
      }
      const errorMsgId = _currentAgentMsgId || uid();
      set((s) => {
        const msgs = [...s.messages];
        const existing = msgs.findIndex((m) => m.id === errorMsgId);
        const errorMsg: ChatMessage = {
          id: errorMsgId,
          role: 'agent',
          content: `Error: ${(event as { message: string }).message}`,
          timestamp: Date.now(),
        };
        if (existing >= 0) {
          msgs[existing] = errorMsg;
        } else {
          msgs.push(errorMsg);
        }
        // Don't set isStreaming false here — the round will end naturally
        // via ROUND_END from the backend.
        return { messages: msgs, commentaryText: null };
      });
      _currentAgentMsgId = null;
      _currentAgentText = '';
      _textBuffer = '';
      break;
    }
  }
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
  messages: [],
  toolEvents: [],
  logLines: [],
  memoryEvents: [],
  commentaryText: null,
  commentaryEvents: [],
  isStreaming: false,
  isCancelling: false,
  figureJson: null,
  storageWarning: false,
  roundMarkers: [],
  roundTokenUsage: null,

  ensureSubscribed: (sessionId: string) => {
    if (_eventSource && _subscribedSessionId === sessionId) return;

    // Close stale subscription
    if (_eventSource) {
      _eventSource.close();
      _eventSource = null;
    }
    if (_reconnectTimer) {
      clearTimeout(_reconnectTimer);
      _reconnectTimer = null;
    }

    _subscribedSessionId = sessionId;
    _eventSource = subscribeToSession(
      sessionId,
      (event) => handleSSEEvent(event, set, get, sessionId),
      () => {
        // readyState CLOSED means a fatal HTTP error (e.g. 404 — session
        // not in server memory). Do NOT reconnect — it will loop forever.
        if (_eventSource?.readyState === EventSource.CLOSED) {
          _eventSource = null;
          _subscribedSessionId = null;
          return;
        }
        // Transient error (network blip) — reconnect after 1s
        if (_subscribedSessionId === sessionId) {
          _reconnectTimer = setTimeout(() => {
            _subscribedSessionId = null; // force re-subscribe
            get().ensureSubscribed(sessionId);
          }, 1000);
        }
      },
    );
  },

  closeSubscription: () => {
    if (_eventSource) {
      _eventSource.close();
      _eventSource = null;
    }
    _subscribedSessionId = null;
    if (_rafHandle) {
      cancelAnimationFrame(_rafHandle);
      _rafHandle = null;
    }
    _currentAgentMsgId = null;
    _currentAgentText = '';
    _textBuffer = '';
    if (_reconnectTimer) {
      clearTimeout(_reconnectTimer);
      _reconnectTimer = null;
    }
  },

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

    set((s) => ({
      messages: [...s.messages, userMsg],
      isStreaming: true,  // optimistic — confirmed by round_start, cleared by round_end
    }));

    // Ensure persistent EventSource is connected
    get().ensureSubscribed(sessionId);

    // Fire-and-forget POST — run_loop() will process it.
    // Multiple messages can be queued while the agent is working.
    try {
      await fetch(`/api/sessions/${sessionId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });
    } catch (err) {
      // POST failed — no round_start/round_end will arrive, so clear streaming directly.
      set((s) => ({
        messages: [
          ...s.messages,
          {
            id: uid(),
            role: 'agent',
            content: `Error: ${(err as Error).message}`,
            timestamp: Date.now(),
          },
        ],
        isStreaming: false,
      }));
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
    set({ isCancelling: true });
    try {
      await api.cancelChat(sessionId);
    } catch {
      // ignore
    }
    // Reset agent message state; EventSource stays open for next turn
    if (_rafHandle) {
      cancelAnimationFrame(_rafHandle);
      _rafHandle = null;
    }
    _currentAgentMsgId = null;
    _currentAgentText = '';
    _textBuffer = '';
    set({ isStreaming: false, isCancelling: false, commentaryText: null });
  },

  rebuildFromEventLog: (events: SessionEventRecord[], sessionId?: string) => {
    const messages: ChatMessage[] = [];
    const toolEvents: ToolEvent[] = [];
    const logLines: LogLine[] = [];
    const memoryEvents: MemoryEvent[] = [];
    const commentaryEvents: CommentaryEvent[] = [];
    const roundMarkers: RoundMarker[] = [];

    for (const ev of events) {
      const rawTs = new Date(ev.ts).getTime();
      const ts = isFinite(rawTs) ? rawTs : 0;

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

      // Insight analysis results
      if (ev.type === 'insight_result' && ev.level !== 'debug') {
        messages.push({
          id: uid(),
          role: 'insight',
          content: (ev.data?.text as string) || ev.msg,
          timestamp: ts,
        });
      }

      // Insight feedback (automatic figure review)
      if (ev.type === 'insight_feedback' && ev.level !== 'debug') {
        messages.push({
          id: uid(),
          role: 'insight_feedback',
          content: (ev.data?.text as string) || ev.msg,
          timestamp: ts,
        });
      }

      // Commentary events (text_delta with commentary flag)
      if (ev.type === 'text_delta' && ev.data?.commentary) {
        const text = ((ev.data?.text as string) || ev.msg).replace(/\n+$/, '');
        commentaryEvents.push({
          id: uid(),
          text,
          agent: ev.agent || 'orchestrator',
          timestamp: ts,
        });
      }

      // Render events → inline plot thumbnails
      if (ev.type === 'render_executed' && ev.data?.op_id && sessionId) {
        const opId = ev.data.op_id as string;
        messages.push({
          id: uid(),
          role: 'plot' as const,
          content: '',
          timestamp: ts,
          thumbnailUrl: api.getRenderThumbnailUrl(sessionId, opId),
          thumbnailSessionId: sessionId,
        });
      }

      // MPL render events → static image
      if (ev.type === 'mpl_render_executed' && ev.data?.script_id && sessionId) {
        const scriptId = ev.data.script_id as string;
        const description = (ev.data.description as string) || 'Matplotlib plot';
        messages.push({
          id: uid(),
          role: 'plot' as const,
          content: description,
          timestamp: ts,
          mplImageUrl: `/api/sessions/${sessionId}/mpl-outputs/${scriptId}.png`,
          mplScriptId: scriptId,
          thumbnailSessionId: sessionId,
        });
      }

      // JSX render events → iframe component
      if (ev.type === 'jsx_render_executed' && ev.data?.script_id && sessionId) {
        const scriptId = ev.data.script_id as string;
        const description = (ev.data.description as string) || 'Recharts component';
        messages.push({
          id: uid(),
          role: 'plot' as const,
          content: description,
          timestamp: ts,
          jsxScriptId: scriptId,
          jsxSessionId: sessionId,
        });
      }

      // Round markers (skip if timestamp is invalid/zero)
      if (ev.type === 'round_start' && ts > 0) {
        roundMarkers.push({ type: 'start', timestamp: ts });
      } else if (ev.type === 'round_end' && ts > 0) {
        roundMarkers.push({
          type: 'end',
          timestamp: ts,
          roundTokenUsage: (ev.data?.round_token_usage as Record<string, number>) || undefined,
        });
      } else if (ev.type === 'turn_done' && ts > 0) {
        // Backward compat: treat each old turn_done as a round boundary
        // (infer round_start from the first event after previous turn_done)
        roundMarkers.push({ type: 'end', timestamp: ts });
      }

      // Tool events
      if (ev.type === 'tool_call' || ev.type === 'tool_started') {
        toolEvents.push({
          id: uid(),
          type: 'call',
          tool_name: (ev.data?.tool_name as string) || '',
          tool_args: (ev.data?.tool_args as Record<string, unknown>) || {},
          timestamp: ts,
          agent: ev.agent || undefined,
        });
      } else if (ev.type === 'tool_result' || ev.type === 'tool_error') {
        toolEvents.push({
          id: uid(),
          type: 'result',
          tool_name: (ev.data?.tool_name as string) || '',
          status: (ev.data?.status as string) || (ev.type === 'tool_error' ? 'error' : ''),
          timestamp: ts,
          agent: ev.agent || undefined,
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

    set({ messages, toolEvents, logLines: logLines.slice(-200), memoryEvents, commentaryEvents, roundMarkers });
  },

  clearChat: () => {
    // Close the EventSource when clearing
    get().closeSubscription();
    set({
      messages: [],
      toolEvents: [],
      logLines: [],
      memoryEvents: [],
      commentaryText: null,
      commentaryEvents: [],
      isStreaming: false,
      isCancelling: false,
      figureJson: null,
      storageWarning: false,
      roundMarkers: [],
      roundTokenUsage: null,
    });
  },

  dismissStorageWarning: () => set({ storageWarning: false }),
}),
    {
      name: 'helion-chat',
      partialize: (state) => ({
        // Strip large Plotly figure objects from messages before persisting.
        // Convert inline-figure plots to thumbnail references so they render
        // as clickable previews after a page refresh instead of blank messages.
        messages: state.messages.map((m) => {
          if (!m.figure) return m;
          const sessionId = m.thumbnailSessionId || useSessionStore.getState().activeSessionId;
          return {
            ...m,
            figure: undefined,
            thumbnailUrl: m.thumbnailUrl || (sessionId ? api.getFigureThumbnailUrl(sessionId) : undefined),
            thumbnailSessionId: sessionId || undefined,
          };
        }),
        toolEvents: state.toolEvents,
        logLines: state.logLines,
        memoryEvents: state.memoryEvents,
        commentaryEvents: state.commentaryEvents,
        roundMarkers: state.roundMarkers,
        // Don't persist figureJson — it can be multiple MB of data arrays
        // figureJson: state.figureJson,
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
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        // Reset streaming flag — can't be streaming after a page refresh
        state.isStreaming = false;
        // Close orphan round markers: a 'start' without a matching 'end'
        const markers = state.roundMarkers;
        if (markers.length > 0 && markers[markers.length - 1].type === 'start') {
          state.roundMarkers = [
            ...markers,
            { type: 'end', timestamp: markers[markers.length - 1].timestamp },
          ];
        }
      },
    },
  ),
);
