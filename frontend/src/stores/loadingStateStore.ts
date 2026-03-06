import { create } from 'zustand';
import type { MissionLoadingState } from '../api/types';

interface LoadingStateStore {
  state: MissionLoadingState | null;
  fetchState: () => Promise<void>;
  subscribeSSE: () => void;
  unsubscribeSSE: () => void;
}

let eventSource: EventSource | null = null;

export const useLoadingStateStore = create<LoadingStateStore>((set, get) => ({
  state: null,

  fetchState: async () => {
    try {
      const res = await fetch('/api/catalog/status');
      if (!res.ok) return;
      const data = await res.json();
      if (data.loading) {
        set({ state: data.loading as MissionLoadingState });
        // Auto-subscribe to SSE if loading is in progress
        if (data.loading.is_loading) {
          get().subscribeSSE();
        }
      }
    } catch {
      // Ignore — server may not be ready yet
    }
  },

  subscribeSSE: () => {
    if (eventSource) return; // Already subscribed

    eventSource = new EventSource('/api/catalog/loading-progress');

    eventSource.addEventListener('progress', (e) => {
      try {
        const data = JSON.parse(e.data) as MissionLoadingState;
        set({ state: data });
      } catch {
        // Skip malformed JSON
      }
    });

    eventSource.addEventListener('done', (e) => {
      try {
        const data = JSON.parse(e.data) as MissionLoadingState;
        set({ state: data });
      } catch {
        // Skip malformed JSON
      }
      // Clean up — loading is complete
      get().unsubscribeSSE();
    });

    eventSource.onerror = () => {
      // Connection lost — clean up
      get().unsubscribeSSE();
    };
  },

  unsubscribeSSE: () => {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  },
}));
