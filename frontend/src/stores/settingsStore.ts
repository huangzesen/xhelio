import { create } from 'zustand';
import type { AppConfig } from '../api/types';
import * as api from '../api/client';
import { useChatStore } from './chatStore';
import { useSessionStore } from './sessionStore';

interface SettingsState {
  config: AppConfig;
  loading: boolean;
  saving: boolean;
  error: string | null;
  saved: boolean;
  sessionSwitched: boolean;

  loadConfig: () => Promise<void>;
  updateConfig: (partial: Partial<AppConfig>) => void;
  saveConfig: () => Promise<void>;
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  config: {},
  loading: false,
  saving: false,
  error: null,
  saved: false,
  sessionSwitched: false,

  loadConfig: async () => {
    set({ loading: true, error: null, saved: false, sessionSwitched: false });
    try {
      const config = await api.getConfig();
      set({ config, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  updateConfig: (partial: Partial<AppConfig>) => {
    set((s) => ({ config: { ...s.config, ...partial }, saved: false }));
  },

  saveConfig: async () => {
    set({ saving: true, error: null, sessionSwitched: false });
    try {
      const result = await api.updateConfig(get().config);
      if (result.needs_new_session) {
        useChatStore.getState().clearChat();
        await useSessionStore.getState().createSession();
        set({ saving: false, saved: true, sessionSwitched: true });
      } else {
        set({ saving: false, saved: true });
      }
    } catch (err) {
      set({ error: (err as Error).message, saving: false });
    }
  },
}));
