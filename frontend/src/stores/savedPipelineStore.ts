import { create } from 'zustand';
import type {
  SavedPipelineIndexEntry,
  SavedPipelineDetail,
  SavedPipelineStep,
  PlotlyFigure,
  PipelineExecuteResult,
  PipelineRecord,
} from '../api/types';
import * as api from '../api/client';

interface SavedPipelineState {
  // List
  pipelines: SavedPipelineIndexEntry[];
  loading: boolean;
  error: string | null;

  // Detail
  selectedPipelineId: string | null;
  detail: SavedPipelineDetail | null;
  dagFigure: PlotlyFigure | null;
  detailLoading: boolean;
  selectedStep: PipelineRecord | null;

  // Execution
  executing: boolean;
  executeResult: PipelineExecuteResult | null;

  // Actions
  loadPipelines: () => Promise<void>;
  selectPipeline: (id: string) => Promise<void>;
  clearSelection: () => void;
  deletePipeline: (id: string) => Promise<void>;
  executePipeline: (id: string, timeStart: string, timeEnd: string) => Promise<void>;
  selectStep: (record: PipelineRecord | null) => void;
  updatePipeline: (id: string, updates: { name?: string; description?: string }) => Promise<void>;
  addFeedback: (id: string, comment: string) => Promise<void>;
}

/** Adapt a SavedPipelineStep into a PipelineRecord for StepTable / CodeViewer reuse. */
export function stepToRecord(step: SavedPipelineStep): PipelineRecord {
  return {
    id: step.step_id,
    timestamp: '',
    tool: step.tool,
    status: 'success',
    inputs: step.inputs,
    outputs: step.output_label ? [step.output_label] : [],
    args: step.params,
  };
}

export const useSavedPipelineStore = create<SavedPipelineState>((set, get) => ({
  pipelines: [],
  loading: false,
  error: null,

  selectedPipelineId: null,
  detail: null,
  dagFigure: null,
  detailLoading: false,
  selectedStep: null,

  executing: false,
  executeResult: null,

  loadPipelines: async () => {
    set({ loading: true, error: null });
    try {
      const pipelines = await api.listSavedPipelines();
      set({ pipelines, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  selectPipeline: async (id: string) => {
    set({
      selectedPipelineId: id,
      detail: null,
      dagFigure: null,
      detailLoading: true,
      selectedStep: null,
      executeResult: null,
      error: null,
    });
    try {
      const [detail, dagData] = await Promise.all([
        api.getSavedPipeline(id),
        api.getSavedPipelineDAG(id),
      ]);
      set({
        detail,
        dagFigure: dagData.figure,
        detailLoading: false,
      });
    } catch (err) {
      set({ error: (err as Error).message, detailLoading: false });
    }
  },

  clearSelection: () =>
    set({
      selectedPipelineId: null,
      detail: null,
      dagFigure: null,
      selectedStep: null,
      executeResult: null,
      error: null,
    }),

  deletePipeline: async (id: string) => {
    try {
      await api.deleteSavedPipeline(id);
      // Clear selection and reload list
      set({
        selectedPipelineId: null,
        detail: null,
        dagFigure: null,
        selectedStep: null,
        executeResult: null,
        error: null,
      });
      get().loadPipelines();
    } catch (err) {
      set({ error: (err as Error).message });
    }
  },

  executePipeline: async (id: string, timeStart: string, timeEnd: string) => {
    set({ executing: true, executeResult: null, error: null });
    try {
      const result = await api.executeSavedPipeline(id, timeStart, timeEnd);
      set({ executeResult: result, executing: false });
    } catch (err) {
      set({ error: (err as Error).message, executing: false });
    }
  },

  selectStep: (record) => set({ selectedStep: record }),

  updatePipeline: async (id: string, updates: { name?: string; description?: string }) => {
    try {
      const updated = await api.updateSavedPipeline(id, updates);
      set({ detail: updated, error: null });
      // Refresh the list to reflect name/description changes
      get().loadPipelines();
    } catch (err) {
      set({ error: (err as Error).message });
    }
  },

  addFeedback: async (id: string, comment: string) => {
    try {
      const entry = await api.addPipelineFeedback(id, comment);
      const { detail } = get();
      if (detail && detail.id === id) {
        const feedback = [...(detail.feedback || []), entry];
        set({ detail: { ...detail, feedback }, error: null });
      }
    } catch (err) {
      set({ error: (err as Error).message });
    }
  },
}));
