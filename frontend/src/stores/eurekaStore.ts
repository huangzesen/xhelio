import { create } from 'zustand';
import type { EurekaEntry, EurekaChatMessage } from '../api/types';
import * as api from '../api/client';
import { useSessionStore } from './sessionStore';

interface EurekaStore {
  eurekas: EurekaEntry[];
  chatMessages: EurekaChatMessage[];
  loading: boolean;
  chatLoading: boolean;
  chatError: string | null;
  filters: { status?: string; tag?: string };

  fetchEurekas: () => Promise<void>;
  fetchChatHistory: () => Promise<void>;
  addEureka: (eureka: EurekaEntry) => void;
  updateStatus: (id: string, status: string) => Promise<void>;
  setFilter: (key: 'status' | 'tag', value: string | undefined) => void;
  filteredEurekas: () => EurekaEntry[];

  sendChatMessage: (message: string) => Promise<void>;
  clearChatError: () => void;
}

export const useEurekaStore = create<EurekaStore>((set, get) => ({
  eurekas: [],
  chatMessages: [],
  loading: false,
  chatLoading: false,
  chatError: null,
  filters: {},

  fetchEurekas: async () => {
    set({ loading: true });
    try {
      const eurekas = await api.fetchEurekas();
      const sorted = [...eurekas].sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
      set({ eurekas: sorted, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  fetchChatHistory: async () => {
    set({ chatLoading: true });
    try {
      const messages = await api.fetchEurekaChatHistory();
      set({ chatMessages: messages, chatLoading: false });
    } catch {
      set({ chatLoading: false });
    }
  },

  addEureka: (eureka: EurekaEntry) => {
    const existing = get().eurekas;
    if (existing.some((e) => e.id === eureka.id)) return;
    set({ eurekas: [eureka, ...existing] });
  },

  updateStatus: async (id: string, status: string) => {
    const current = get().eurekas;
    set({
      eurekas: current.map((e) =>
        e.id === id ? { ...e, status: status as EurekaEntry['status'] } : e
      ),
    });
    try {
      await api.updateEurekaStatus(id, status);
    } catch {
      set({ eurekas: current });
    }
  },

  setFilter: (key: 'status' | 'tag', value: string | undefined) => {
    set({ filters: { ...get().filters, [key]: value } });
  },

  filteredEurekas: () => {
    const { eurekas, filters } = get();
    return eurekas.filter((e) => {
      if (filters.status && e.status !== filters.status) return false;
      if (filters.tag && !e.tags.includes(filters.tag)) return false;
      return true;
    });
  },

  sendChatMessage: async (message: string) => {
    const sessionId = useSessionStore.getState().activeSessionId;
    if (!sessionId) {
      set({ chatError: 'No active session' });
      return;
    }

    // Add user message immediately
    const userMessage: EurekaChatMessage = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
    };
    set((state) => ({
      chatMessages: [...state.chatMessages, userMessage],
      chatLoading: true,
      chatError: null,
    }));

    try {
      const stream = api.eurekaChatStream(sessionId, message);
      for await (const event of stream) {
        if (event.type === 'eureka_chat_response') {
          const assistantMessage: EurekaChatMessage = {
            role: 'assistant',
            content: event.content,
            timestamp: new Date().toISOString(),
          };
          set((state) => ({
            chatMessages: [...state.chatMessages, assistantMessage],
          }));
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
      if (errorMessage.includes('409')) {
        set({ chatError: 'Eureka is busy. Please wait...' });
      } else {
        set({ chatError: errorMessage });
      }
    } finally {
      set({ chatLoading: false });
    }
  },

  clearChatError: () => {
    set({ chatError: null });
  },
}));
