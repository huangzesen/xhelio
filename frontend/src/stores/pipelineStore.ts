import { create } from 'zustand';
import type { SavedSessionWithOps, PipelineRecord, PlotlyFigure, ReplayResult } from '../api/types';
import * as api from '../api/client';
import { RENDER_TOOL_NAMES } from '../constants/toolColors';

function downloadFile(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

interface RenderOption {
  label: string;
  opId: string | null;
}

interface PipelineState {
  sessions: SavedSessionWithOps[];
  selectedSessionId: string | null;
  pipeline: PipelineRecord[];
  allRecords: PipelineRecord[];
  dagFigure: PlotlyFigure | null;
  selectedStep: PipelineRecord | null;
  renderOptions: RenderOption[];
  selectedRenderOpId: string | null;
  replaying: boolean;
  replayResult: ReplayResult | null;
  generatingScript: boolean;
  loading: boolean;
  error: string | null;

  loadSessions: () => Promise<void>;
  selectSession: (sessionId: string) => Promise<void>;
  selectStep: (record: PipelineRecord | null) => void;
  selectRender: (opId: string | null) => Promise<void>;
  replay: (useCache?: boolean) => Promise<void>;
  generateScript: () => Promise<void>;
  reset: () => void;
}

function buildRenderOptions(pipeline: PipelineRecord[]): RenderOption[] {
  const options: RenderOption[] = [{ label: 'All (latest)', opId: null }];
  for (const r of pipeline) {
    if (!RENDER_TOOL_NAMES.has(r.tool) || r.status !== 'success') continue;
    const inputsStr = (Array.isArray(r.inputs) && r.inputs.length > 0) ? r.inputs.join(', ') : 'no inputs';
    const sc = r.state_count ?? 1;
    const si = r.state_index ?? 0;
    let label: string;
    if (sc > 1) {
      label = `plot: ${inputsStr} \u2014 state ${si + 1}/${sc} (${r.id})`;
    } else {
      label = `plot: ${inputsStr} (${r.id})`;
    }
    options.push({ label, opId: r.id });
  }
  return options;
}

export const usePipelineStore = create<PipelineState>((set, get) => ({
  sessions: [],
  selectedSessionId: null,
  pipeline: [],
  allRecords: [],
  dagFigure: null,
  selectedStep: null,
  renderOptions: [],
  selectedRenderOpId: null,
  replaying: false,
  replayResult: null,
  generatingScript: false,
  loading: false,
  error: null,

  loadSessions: async () => {
    set({ loading: true, error: null });
    try {
      const sessions = await api.getSavedSessionsWithOps();
      set({ sessions: Array.isArray(sessions) ? sessions : [], loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  selectSession: async (sessionId: string) => {
    set({
      selectedSessionId: sessionId,
      pipeline: [],
      allRecords: [],
      dagFigure: null,
      selectedStep: null,
      renderOptions: [],
      selectedRenderOpId: null,
      replayResult: null,
      loading: true,
      error: null,
    });
    try {
      const [opsData, dagData] = await Promise.all([
        api.getPipelineOperations(sessionId),
        api.getPipelineDAG(sessionId),
      ]);
      const safePipeline = Array.isArray(opsData?.pipeline) ? opsData.pipeline : [];
      const safeAllRecords = Array.isArray(opsData?.all_records) ? opsData.all_records : [];
      const renderOptions = buildRenderOptions(safePipeline);
      set({
        pipeline: safePipeline,
        allRecords: safeAllRecords,
        dagFigure: dagData.figure,
        renderOptions,
        loading: false,
      });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  selectStep: (record) => set({ selectedStep: record }),

  selectRender: async (opId: string | null) => {
    const sid = get().selectedSessionId;
    if (!sid) return;
    set({ selectedRenderOpId: opId, loading: true, error: null });
    try {
      const dagData = await api.getPipelineDAG(sid, opId ?? undefined);
      set({ dagFigure: dagData.figure, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  replay: async (useCache = true) => {
    const sid = get().selectedSessionId;
    if (!sid) return;
    set({ replaying: true, replayResult: null, error: null });
    try {
      const result = await api.replayPipeline(sid, useCache, get().selectedRenderOpId ?? undefined);
      set({ replayResult: result, replaying: false });
    } catch (err) {
      set({ error: (err as Error).message, replaying: false });
    }
  },

  generateScript: async () => {
    const { selectedSessionId, selectedRenderOpId } = get();
    if (!selectedSessionId) return;
    set({ generatingScript: true, error: null });
    try {
      const result = await api.generateSessionScript(selectedSessionId, selectedRenderOpId ?? undefined);
      for (const [filename, content] of Object.entries(result.files)) {
        downloadFile(filename, content);
      }
      set({ generatingScript: false });
    } catch (err) {
      set({ error: (err as Error).message, generatingScript: false });
    }
  },

  reset: () => set({
    selectedSessionId: null,
    pipeline: [],
    allRecords: [],
    dagFigure: null,
    selectedStep: null,
    renderOptions: [],
    selectedRenderOpId: null,
    replayResult: null,
    error: null,
  }),
}));
