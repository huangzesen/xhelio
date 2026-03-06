import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SavedSessionInfo, SessionEventRecord } from '../api/types';
import * as api from '../api/client';

export interface DisplayLogEntry {
  role: string;
  content: string;
  timestamp?: string;
}

interface SessionState {
  activeSessionId: string | null;
  model: string;
  busy: boolean;
  resuming: boolean;
  savedSessions: SavedSessionInfo[];
  tokenUsage: Record<string, number>;
  lastDisplayLog: DisplayLogEntry[];
  lastEventLog: SessionEventRecord[];

  createSession: () => Promise<string>;
  resumeSession: (savedId: string) => Promise<string>;
  deleteSession: (id: string) => Promise<void>;
  loadSavedSessions: () => Promise<void>;
  renameSession: (savedId: string, name: string) => Promise<void>;
  setTokenUsage: (usage: Record<string, number>) => void;
  setActiveSessionId: (id: string | null) => void;
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
  activeSessionId: null,
  model: '',
  busy: false,
  resuming: false,
  savedSessions: [],
  tokenUsage: {},
  lastDisplayLog: [],
  lastEventLog: [],

  createSession: async () => {
    const info = await api.createSession();
    set({
      activeSessionId: info.session_id,
      model: info.model,
      busy: false,
      tokenUsage: {},
    });
    // Refresh sidebar so the new session appears immediately
    await get().loadSavedSessions();
    return info.session_id;
  },

  resumeSession: async (savedId: string) => {
    if (get().resuming) return get().activeSessionId ?? '';
    set({ resuming: true });
    try {
      // Resume first — only delete the old session after resume succeeds
      const current = get().activeSessionId;
      const info = await api.resumeSession(savedId);
      // Resume succeeded — now safe to clean up the old session
      if (current && current !== info.session_id) {
        api.deleteSession(current).catch(() => {});
      }
      set({
        activeSessionId: info.session_id,
        model: info.model,
        busy: false,
        lastDisplayLog: info.display_log || [],
        lastEventLog: info.event_log || [],
      });
      // Fetch session detail to get token_usage from the resumed session
      const detail = await api.getSession(info.session_id);
      set({ tokenUsage: detail.token_usage ?? {} });
      return info.session_id;
    } finally {
      set({ resuming: false });
    }
  },

  deleteSession: async (id: string) => {
    await api.deleteSession(id);
    if (get().activeSessionId === id) {
      set({ activeSessionId: null, model: '', tokenUsage: {} });
    }
  },

  loadSavedSessions: async () => {
    const sessions = await api.getSavedSessions();
    set({ savedSessions: sessions });
  },

  renameSession: async (savedId: string, name: string) => {
    await api.renameSession(savedId, name);
    set((state) => ({
      savedSessions: state.savedSessions.map((s) =>
        s.id === savedId ? { ...s, name } : s,
      ),
    }));
  },

  setTokenUsage: (usage) => set({ tokenUsage: usage }),
  setActiveSessionId: (id) => set({ activeSessionId: id }),
}),
    {
      name: 'helion-session',
      partialize: (state) => ({
        activeSessionId: state.activeSessionId,
        model: state.model,
        tokenUsage: state.tokenUsage,
      }),
    },
  ),
);
