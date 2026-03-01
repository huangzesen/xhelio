import type {
  SessionInfo,
  SessionDetail,
  SavedSessionInfo,
  SessionEventRecord,
  ServerStatus,
  PlotlyFigure,
  MissionInfo,
  DatasetInfo,
  ParameterInfo,
  TimeRange,
  DataPreview,
  DataEntrySummary,
  AppConfig,
  MemoryEntry,
  MemoryStats,
  SavedSessionWithOps,
  PipelineRecord,
  PipelineFeedback,
  ReplayResult,
  TokenBreakdown,
  CommandResponse,
  GalleryItem,
  ValidationOverview,
  SavedPipelineIndexEntry,
  SavedPipelineDetail,
  PipelineExecuteResult,
  AssetOverview,
  AssetCategory,
  CleanupRequest,
  CleanupResponse,
  EurekaEntry,
} from './types';

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ---- Sessions ----

export function createSession(): Promise<SessionInfo> {
  return request('/sessions', { method: 'POST' });
}

export function listSessions(): Promise<SessionInfo[]> {
  return request('/sessions');
}

export function getSession(id: string): Promise<SessionDetail> {
  return request(`/sessions/${id}`);
}

export function deleteSession(id: string): Promise<void> {
  return request(`/sessions/${id}`, { method: 'DELETE' });
}

// ---- Saved sessions ----

export function getSavedSessions(): Promise<SavedSessionInfo[]> {
  return request('/sessions/saved');
}

export function deleteSavedSession(id: string): Promise<void> {
  return request(`/sessions/saved/${id}`, { method: 'DELETE' });
}

export function resumeSession(savedId: string): Promise<SessionInfo & { resumed_from: string; turn_count: number; round_count: number; display_log: Array<{ role: string; content: string; timestamp?: string }>; event_log: SessionEventRecord[] }> {
  return request('/sessions/resume', {
    method: 'POST',
    body: JSON.stringify({ session_id: savedId }),
  });
}

export function renameSession(savedId: string, name: string): Promise<{ status: string; name: string }> {
  return request(`/sessions/saved/${encodeURIComponent(savedId)}/name`, {
    method: 'PATCH',
    body: JSON.stringify({ name }),
  });
}

// ---- Event log ----

export function getSessionEvents(
  sessionId: string,
  opts?: { limit?: number; offset?: number },
): Promise<{ events: SessionEventRecord[]; total: number }> {
  const params = new URLSearchParams();
  if (opts?.limit != null) params.set('limit', String(opts.limit));
  if (opts?.offset != null) params.set('offset', String(opts.offset));
  const qs = params.toString();
  return request(`/sessions/${sessionId}/event-log${qs ? `?${qs}` : ''}`);
}

// ---- Cancel ----

export function cancelChat(sessionId: string): Promise<{ status: string }> {
  return request(`/sessions/${sessionId}/cancel`, { method: 'POST' });
}

// ---- Commands ----

export function executeCommand(sessionId: string, command: string): Promise<CommandResponse> {
  return request(`/sessions/${sessionId}/command`, {
    method: 'POST',
    body: JSON.stringify({ command }),
  });
}

// ---- Data ----

export function getData(sessionId: string): Promise<DataEntrySummary[]> {
  return request(`/sessions/${sessionId}/data`);
}

export function fetchData(
  sessionId: string,
  datasetId: string,
  parameterId: string,
  timeMin: string,
  timeMax: string,
): Promise<DataEntrySummary> {
  return request(`/sessions/${sessionId}/fetch-data`, {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: datasetId,
      parameter_id: parameterId,
      time_min: timeMin,
      time_max: timeMax,
    }),
  });
}

export function getDataPreview(sessionId: string, label: string): Promise<DataPreview> {
  return request(`/sessions/${sessionId}/data/${encodeURIComponent(label)}/preview`);
}

// ---- Figure ----

export function getFigure(sessionId: string): Promise<{ figure: PlotlyFigure | null; figure_url?: string }> {
  return request(`/sessions/${sessionId}/figure`);
}

export function getFigureThumbnailUrl(sessionId: string): string {
  return `${BASE}/sessions/${sessionId}/figure-thumbnail`;
}

export function getRenderThumbnailUrl(sessionId: string, opId: string): string {
  return `${BASE}/sessions/${sessionId}/thumbnails/${opId}.png`;
}

// ---- Plan ----

export function getPlanStatus(sessionId: string): Promise<{ plan_status: string | null }> {
  return request(`/sessions/${sessionId}/plan`);
}

// ---- Catalog ----

export function getMissions(): Promise<MissionInfo[]> {
  return request('/catalog/missions');
}

export function getDatasets(missionId: string): Promise<DatasetInfo[]> {
  return request(`/catalog/missions/${encodeURIComponent(missionId)}/datasets`);
}

export function getParameters(datasetId: string): Promise<ParameterInfo[]> {
  return request(`/catalog/datasets/${encodeURIComponent(datasetId)}/parameters`);
}

export function getTimeRange(datasetId: string): Promise<TimeRange> {
  return request(`/catalog/datasets/${encodeURIComponent(datasetId)}/time-range`);
}

// ---- Config ----

export function getConfig(): Promise<AppConfig> {
  return request('/config');
}

export function getConfigSchema(): Promise<{ descriptions: Record<string, string> }> {
  return request('/config/schema');
}

export function updateConfig(config: Partial<AppConfig>): Promise<{ status: string; needs_new_session?: boolean }> {
  return request('/config', {
    method: 'PUT',
    body: JSON.stringify({ config }),
  });
}

export interface ModelInfo {
  id: string;
  display_name: string;
  input_token_limit?: number;
}

export function listModels(provider: string): Promise<{ models: ModelInfo[]; error?: string }> {
  return request(`/models?provider=${encodeURIComponent(provider)}`);
}

// ---- Memories ----

export function getMemories(sessionId: string): Promise<{ memories: MemoryEntry[]; global_enabled: boolean; stats: MemoryStats | null }> {
  return request(`/sessions/${sessionId}/memories/list`);
}

export function searchMemories(
  sessionId: string,
  query: string,
  type?: string,
  scope?: string,
  limit?: number,
): Promise<{ results: MemoryEntry[] }> {
  const params = new URLSearchParams({ q: query });
  if (type) params.set('type', type);
  if (scope) params.set('scope', scope);
  if (limit) params.set('limit', String(limit));
  return request(`/sessions/${sessionId}/memories/search?${params}`);
}

export function getArchivedMemories(sessionId: string): Promise<{ archived: MemoryEntry[] }> {
  return request(`/sessions/${sessionId}/memories/archived`);
}

export function getMemoryVersionHistory(sessionId: string, memoryId: string): Promise<{ versions: MemoryEntry[] }> {
  return request(`/sessions/${sessionId}/memories/${memoryId}/history`);
}

export function deleteMemory(sessionId: string, memoryId: string): Promise<{ status: string }> {
  return request(`/sessions/${sessionId}/memories/${memoryId}`, { method: 'DELETE' });
}

export function toggleGlobalMemory(sessionId: string, enabled: boolean): Promise<{ global_enabled: boolean }> {
  return request(`/sessions/${sessionId}/memories/toggle-global`, {
    method: 'POST',
    body: JSON.stringify({ enabled }),
  });
}

export function clearAllMemories(sessionId: string): Promise<{ status: string }> {
  return request(`/sessions/${sessionId}/memories`, { method: 'DELETE' });
}

export function refreshMemories(sessionId: string): Promise<{ status: string }> {
  return request(`/sessions/${sessionId}/memories/refresh`, { method: 'POST' });
}

// ---- Pipeline ----

export function getSavedSessionsWithOps(): Promise<SavedSessionWithOps[]> {
  return request('/sessions/saved-with-ops');
}

export function getPipelineOperations(savedId: string): Promise<{ pipeline: PipelineRecord[]; all_records: PipelineRecord[] }> {
  return request(`/pipeline/${savedId}/operations`);
}

export function getPipelineDAG(savedId: string, renderOpId?: string): Promise<{ figure: PlotlyFigure | null; figure_url?: string }> {
  const params = renderOpId ? `?render_op_id=${encodeURIComponent(renderOpId)}` : '';
  return request(`/pipeline/${savedId}/dag${params}`);
}

export function replayPipeline(
  savedId: string,
  useCache = true,
  renderOpId?: string,
): Promise<ReplayResult> {
  return request(`/pipeline/${savedId}/replay`, {
    method: 'POST',
    body: JSON.stringify({ use_cache: useCache, render_op_id: renderOpId }),
  });
}

// ---- Saved Pipelines ----

export function listSavedPipelines(): Promise<SavedPipelineIndexEntry[]> {
  return request('/pipelines');
}

export function getSavedPipeline(id: string): Promise<SavedPipelineDetail> {
  return request(`/pipelines/${encodeURIComponent(id)}`);
}

export function executeSavedPipeline(
  id: string,
  timeStart: string,
  timeEnd: string,
): Promise<PipelineExecuteResult> {
  return request(`/pipelines/${encodeURIComponent(id)}/execute`, {
    method: 'POST',
    body: JSON.stringify({ time_start: timeStart, time_end: timeEnd }),
  });
}

export function deleteSavedPipeline(id: string): Promise<void> {
  return request(`/pipelines/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export function updateSavedPipeline(
  id: string,
  updates: { name?: string; description?: string; tags?: string[] },
): Promise<SavedPipelineDetail> {
  return request(`/pipelines/${encodeURIComponent(id)}`, {
    method: 'PUT',
    body: JSON.stringify(updates),
  });
}

export function addPipelineFeedback(
  id: string,
  comment: string,
): Promise<PipelineFeedback> {
  return request(`/pipelines/${encodeURIComponent(id)}/feedback`, {
    method: 'POST',
    body: JSON.stringify({ comment }),
  });
}

export function restorePipeline(id: string): Promise<SavedPipelineDetail> {
  return request(`/pipelines/${encodeURIComponent(id)}/restore`, {
    method: 'POST',
  });
}

export function getSavedPipelineDAG(
  id: string,
): Promise<{ figure: PlotlyFigure | null; figure_url?: string }> {
  return request(`/pipelines/${encodeURIComponent(id)}/dag`);
}

// ---- Autocomplete + Input History ----

export function getCompletions(sessionId: string, partial: string): Promise<{ completions: string[] }> {
  return request(`/sessions/${sessionId}/completions`, {
    method: 'POST',
    body: JSON.stringify({ partial }),
  });
}

export function getInputHistory(): Promise<{ history: string[] }> {
  return request('/input-history');
}

export function addInputHistory(text: string): Promise<{ status: string }> {
  return request('/input-history', {
    method: 'POST',
    body: JSON.stringify({ text }),
  });
}

// ---- Token Breakdown ----

export function getTokenBreakdown(sessionId: string): Promise<TokenBreakdown> {
  return request(`/sessions/${sessionId}/token-breakdown`);
}

// ---- Gallery ----

export function getGalleryItems(): Promise<GalleryItem[]> {
  return request('/gallery');
}

export function saveToGallery(name: string, sessionId: string, renderOpId: string): Promise<GalleryItem> {
  return request('/gallery', {
    method: 'POST',
    body: JSON.stringify({ name, session_id: sessionId, render_op_id: renderOpId }),
  });
}

export function deleteGalleryItem(id: string): Promise<void> {
  return request(`/gallery/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export function replayGalleryItem(id: string): Promise<ReplayResult> {
  return request(`/gallery/${encodeURIComponent(id)}/replay`, { method: 'POST' });
}

// ---- Validation ----

export function getValidationOverview(): Promise<ValidationOverview> {
  return request('/validation/overview');
}

// ---- API Key ----

export interface ApiKeyStatus {
  configured: boolean;
  masked: string | null;
}

export interface ApiKeyUpdateResult {
  status: string;
  valid: boolean;
  error: string | null;
  masked: string;
}

export function getApiKeyStatus(provider: string = 'gemini'): Promise<ApiKeyStatus> {
  return request(`/api-key-status?provider=${encodeURIComponent(provider)}`);
}

export function updateApiKey(provider: string, key: string): Promise<ApiKeyUpdateResult> {
  return request('/api-key', {
    method: 'PUT',
    body: JSON.stringify({ provider, key }),
  });
}

// ---- Assets ----

export function getAssetOverview(): Promise<AssetOverview> {
  return request('/assets');
}

export function getAssetDetail(category: string): Promise<AssetCategory> {
  return request(`/assets/${category}`);
}

export function cleanAssets(category: string, req: CleanupRequest): Promise<CleanupResponse> {
  return request(`/assets/${category}/clean`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

// ---- Status ----

export function getStatus(): Promise<ServerStatus> {
  return request('/status');
}

// ---- Eureka ----

export async function fetchEurekas(params?: { session_id?: string; status?: string }): Promise<EurekaEntry[]> {
  const query = new URLSearchParams();
  if (params?.session_id) query.set('session_id', params.session_id);
  if (params?.status) query.set('status', params.status);
  const data = await request<{ eurekas: EurekaEntry[] }>(`/eureka?${query}`);
  return data.eurekas;
}

export function fetchEureka(id: string): Promise<EurekaEntry> {
  return request(`/eureka/${id}`);
}

export function updateEurekaStatus(id: string, status: string): Promise<void> {
  return request(`/eureka/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  });
}
