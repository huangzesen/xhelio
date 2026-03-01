import { create } from 'zustand';
import type { MemoryEntry, MemoryStats } from '../api/types';
import * as api from '../api/client';

interface MemoryState {
  memories: MemoryEntry[];
  globalEnabled: boolean;
  loading: boolean;
  error: string | null;

  // Dashboard state
  stats: MemoryStats | null;
  searchQuery: string;
  searchResults: MemoryEntry[] | null; // null = not searching
  searchLoading: boolean;
  activeType: string | null; // type filter
  activeScopes: string[]; // scope filter (empty = all)
  sortBy: 'recency' | 'rating' | 'access_count' | 'reviews';
  sortDirection: 'desc' | 'asc';
  ratingTier: 'all_time' | 'recent';
  archivedMemories: MemoryEntry[];
  archivedLoading: boolean;
  archiveExpanded: boolean;
  versionHistories: Record<string, MemoryEntry[]>;
  versionLoading: string | null;

  // Actions
  loadMemories: (sessionId: string) => Promise<void>;
  deleteMemory: (sessionId: string, memoryId: string) => Promise<void>;
  toggleGlobal: (sessionId: string, enabled: boolean) => Promise<void>;
  clearAll: (sessionId: string) => Promise<void>;
  refresh: (sessionId: string) => Promise<void>;

  // Dashboard actions
  setSearchQuery: (query: string) => void;
  searchMemories: (sessionId: string, query: string) => Promise<void>;
  clearSearch: () => void;
  setActiveType: (type: string | null) => void;
  toggleScope: (scope: string) => void;
  setSortBy: (sortBy: 'recency' | 'rating' | 'access_count' | 'reviews') => void;
  toggleSortDirection: () => void;
  setRatingTier: (tier: 'all_time' | 'recent') => void;
  loadArchived: (sessionId: string) => Promise<void>;
  setArchiveExpanded: (expanded: boolean) => void;
  loadVersionHistory: (sessionId: string, memoryId: string) => Promise<void>;
}

export const useMemoryStore = create<MemoryState>((set, get) => ({
  memories: [],
  globalEnabled: true,
  loading: false,
  error: null,

  // Dashboard state defaults
  stats: null,
  searchQuery: '',
  searchResults: null,
  searchLoading: false,
  activeType: null,
  activeScopes: [],
  sortBy: 'recency',
  sortDirection: 'desc',
  ratingTier: 'recent',
  archivedMemories: [],
  archivedLoading: false,
  archiveExpanded: false,
  versionHistories: {},
  versionLoading: null,

  loadMemories: async (sessionId: string) => {
    // Clear ALL stale data when loading — prevents showing old memories
    // from a previous session or after a backend reset
    set({
      loading: true,
      error: null,
      memories: [],
      stats: null,
      archivedMemories: [],
      versionHistories: {},
      searchResults: null,
      searchQuery: '',
    });
    try {
      const data = await api.getMemories(sessionId);
      set({
        memories: data.memories,
        globalEnabled: data.global_enabled,
        stats: data.stats,
        loading: false,
      });
    } catch (err) {
      set({ memories: [], stats: null, error: (err as Error).message, loading: false });
    }
  },

  deleteMemory: async (sessionId: string, memoryId: string) => {
    try {
      await api.deleteMemory(sessionId, memoryId);
      set((s) => ({ memories: s.memories.filter((m) => m.id !== memoryId) }));
    } catch (err) {
      set({ error: (err as Error).message });
    }
  },

  toggleGlobal: async (sessionId: string, enabled: boolean) => {
    try {
      await api.toggleGlobalMemory(sessionId, enabled);
      set({ globalEnabled: enabled });
    } catch (err) {
      set({ error: (err as Error).message });
    }
  },

  clearAll: async (sessionId: string) => {
    try {
      await api.clearAllMemories(sessionId);
      set({ memories: [] });
    } catch (err) {
      set({ error: (err as Error).message });
    }
  },

  refresh: async (sessionId: string) => {
    try {
      await api.refreshMemories(sessionId);
      const data = await api.getMemories(sessionId);
      set({
        memories: data.memories,
        globalEnabled: data.global_enabled,
        stats: data.stats,
      });
    } catch (err) {
      set({ error: (err as Error).message });
    }
  },

  // Dashboard actions
  setSearchQuery: (query: string) => {
    set({ searchQuery: query });
  },

  searchMemories: async (sessionId: string, query: string) => {
    if (!query.trim()) {
      set({ searchResults: null, searchLoading: false });
      return;
    }
    set({ searchLoading: true });
    try {
      // Don't pass type/scope filters to the API — CardList applies them
      // client-side. This avoids stale filter state from debounce delay.
      const data = await api.searchMemories(sessionId, query);
      set({ searchResults: data.results, searchLoading: false });
    } catch (err) {
      set({ searchLoading: false, error: (err as Error).message });
    }
  },

  clearSearch: () => {
    set({ searchQuery: '', searchResults: null, searchLoading: false });
  },

  setActiveType: (type: string | null) => {
    set({ activeType: type });
  },

  toggleScope: (scope: string) => {
    set((s) => {
      const idx = s.activeScopes.indexOf(scope);
      if (idx >= 0) {
        return { activeScopes: s.activeScopes.filter((sc) => sc !== scope) };
      }
      return { activeScopes: [...s.activeScopes, scope] };
    });
  },

  setSortBy: (sortBy: 'recency' | 'rating' | 'access_count' | 'reviews') => {
    set((s) => {
      // If clicking the same sort, toggle direction
      if (s.sortBy === sortBy) {
        return { sortDirection: s.sortDirection === 'desc' ? 'asc' : 'desc' };
      }
      return { sortBy, sortDirection: 'desc' };
    });
  },

  toggleSortDirection: () => {
    set((s) => ({ sortDirection: s.sortDirection === 'desc' ? 'asc' : 'desc' }));
  },

  setRatingTier: (tier: 'all_time' | 'recent') => {
    set({ ratingTier: tier });
  },

  loadArchived: async (sessionId: string) => {
    set({ archivedLoading: true });
    try {
      const data = await api.getArchivedMemories(sessionId);
      set({ archivedMemories: data.archived, archivedLoading: false });
    } catch (err) {
      set({ archivedLoading: false, error: (err as Error).message });
    }
  },

  setArchiveExpanded: (expanded: boolean) => {
    set({ archiveExpanded: expanded });
  },

  loadVersionHistory: async (sessionId: string, memoryId: string) => {
    set({ versionLoading: memoryId });
    try {
      const data = await api.getMemoryVersionHistory(sessionId, memoryId);
      set((s) => ({
        versionHistories: { ...s.versionHistories, [memoryId]: data.versions },
        versionLoading: null,
      }));
    } catch (err) {
      set({ versionLoading: null, error: (err as Error).message });
    }
  },
}));
