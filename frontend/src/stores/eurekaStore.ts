import { create } from 'zustand';
import type { EurekaEntry } from '../api/types';
import * as api from '../api/client';

interface EurekaStore {
  eurekas: EurekaEntry[];
  loading: boolean;
  filters: { status?: string; tag?: string };

  fetchEurekas: () => Promise<void>;
  addEureka: (eureka: EurekaEntry) => void;
  updateStatus: (id: string, status: string) => Promise<void>;
  setFilter: (key: 'status' | 'tag', value: string | undefined) => void;
  filteredEurekas: () => EurekaEntry[];
}

export const useEurekaStore = create<EurekaStore>((set, get) => ({
  eurekas: [],
  loading: false,
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
}));
