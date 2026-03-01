import { create } from 'zustand';
import type { MissionInfo, DatasetInfo, ParameterInfo, TimeRange } from '../api/types';
import * as api from '../api/client';

interface CatalogState {
  missions: MissionInfo[];
  datasets: DatasetInfo[];
  parameters: ParameterInfo[];
  timeRange: TimeRange | null;
  selectedMission: string | null;
  selectedDataset: string | null;
  loading: boolean;
  error: string | null;

  loadMissions: () => Promise<void>;
  selectMission: (missionId: string) => Promise<void>;
  selectDataset: (datasetId: string) => Promise<void>;
  reset: () => void;
}

export const useCatalogStore = create<CatalogState>((set) => ({
  missions: [],
  datasets: [],
  parameters: [],
  timeRange: null,
  selectedMission: null,
  selectedDataset: null,
  loading: false,
  error: null,

  loadMissions: async () => {
    set({ loading: true, error: null });
    try {
      const missions = await api.getMissions();
      set({ missions, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  selectMission: async (missionId: string) => {
    set({
      selectedMission: missionId,
      selectedDataset: null,
      datasets: [],
      parameters: [],
      timeRange: null,
      loading: true,
      error: null,
    });
    try {
      const datasets = await api.getDatasets(missionId);
      set({ datasets, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  selectDataset: async (datasetId: string) => {
    set({ selectedDataset: datasetId, parameters: [], timeRange: null, loading: true, error: null });
    try {
      const [params, timeRange] = await Promise.all([
        api.getParameters(datasetId),
        api.getTimeRange(datasetId),
      ]);
      set({ parameters: params, timeRange, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  reset: () => set({
    datasets: [],
    parameters: [],
    timeRange: null,
    selectedMission: null,
    selectedDataset: null,
    error: null,
  }),
}));
