import { create } from 'zustand';
import type { GalleryItem, ReplayResult } from '../api/types';
import * as api from '../api/client';

interface GalleryState {
  items: GalleryItem[];
  selectedItem: GalleryItem | null;
  replaying: boolean;
  replayResult: ReplayResult | null;
  loading: boolean;
  error: string | null;

  loadItems: () => Promise<void>;
  deleteItem: (id: string) => Promise<void>;
  replayItem: (id: string) => Promise<void>;
  selectItem: (item: GalleryItem | null) => void;
}

export const useGalleryStore = create<GalleryState>((set, get) => ({
  items: [],
  selectedItem: null,
  replaying: false,
  replayResult: null,
  loading: false,
  error: null,

  loadItems: async () => {
    set({ loading: true, error: null });
    try {
      const items = await api.getGalleryItems();
      set({ items, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  deleteItem: async (id: string) => {
    try {
      await api.deleteGalleryItem(id);
      const items = get().items.filter((it) => it.id !== id);
      const selected = get().selectedItem;
      set({
        items,
        selectedItem: selected?.id === id ? null : selected,
        replayResult: selected?.id === id ? null : get().replayResult,
      });
    } catch (err) {
      set({ error: (err as Error).message });
    }
  },

  replayItem: async (id: string) => {
    set({ replaying: true, replayResult: null, error: null });
    try {
      const result = await api.replayGalleryItem(id);
      set({ replayResult: result, replaying: false });
    } catch (err) {
      set({ error: (err as Error).message, replaying: false });
    }
  },

  selectItem: (item) => set({ selectedItem: item, replayResult: null }),
}));
