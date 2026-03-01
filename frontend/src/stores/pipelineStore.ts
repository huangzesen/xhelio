import { create } from 'zustand';
import type { SavedSessionWithOps, PipelineRecord, PlotlyFigure, ReplayResult } from '../api/types';
import * as api from '../api/client';

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
  loading: boolean;
  error: string | null;

  loadSessions: () => Promise<void>;
  selectSession: (sessionId: string) => Promise<void>;
  selectStep: (record: PipelineRecord | null) => void;
  selectRender: (opId: string | null) => Promise<void>;
  replay: (useCache?: boolean) => Promise<void>;
  reset: () => void;
}

function buildRenderOptions(pipeline: PipelineRecord[]): RenderOption[] {
  const options: RenderOption[] = [{ label: 'All (latest)', opId: null }];
  for (const r of pipeline) {
    if (r.tool !== 'render_plotly_json' || r.status !== 'success') continue;
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
  loading: false,
  error: null,

  loadSessions: async () => {
    set({ loading: true, error: null });
    try {
      const sessions = await api.getSavedSessionsWithOps();
      set({ sessions, loading: false });
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
      const renderOptions = buildRenderOptions(opsData.pipeline);
      set({
        pipeline: opsData.pipeline,
        allRecords: opsData.all_records,
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
