import { create } from 'zustand';
import type { AssetOverview, CleanupRequest, CleanupResponse } from '../api/types';
import * as api from '../api/client';

interface AssetState {
  overview: AssetOverview | null;
  loading: boolean;
  cleaning: string | null;
  cleanResult: CleanupResponse | null;
  error: string | null;

  loadOverview: () => Promise<void>;
  clean: (category: string, req: CleanupRequest) => Promise<void>;
  clearResult: () => void;
}

export const useAssetStore = create<AssetState>((set) => ({
  overview: null,
  loading: false,
  cleaning: null,
  cleanResult: null,
  error: null,

  loadOverview: async () => {
    set({ loading: true, error: null });
    try {
      const overview = await api.getAssetOverview();
      set({ overview, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  clean: async (category: string, req: CleanupRequest) => {
    set({ cleaning: category, cleanResult: null, error: null });
    try {
      const result = await api.cleanAssets(category, req);
      set({ cleanResult: result, cleaning: null });
    } catch (err) {
      set({ error: (err as Error).message, cleaning: null });
    }
  },

  clearResult: () => set({ cleanResult: null }),
}));
