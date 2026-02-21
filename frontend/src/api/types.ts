// Types mirroring api/models.py

// ---- Session ----

export interface SessionInfo {
  session_id: string;
  model: string;
  created_at: string;
  last_active: string;
  busy: boolean;
}

export interface SessionDetail extends SessionInfo {
  token_usage: Record<string, number>;
  data_entries: number;
  plan_status: string | null;
}

export interface SavedSessionInfo {
  id: string;
  name: string | null;
  model: string | null;
  turn_count: number;
  last_message_preview: string;
  created_at: string | null;
  updated_at: string | null;
  token_usage: Record<string, number>;
}

export interface ServerStatus {
  status: string;
  active_sessions: number;
  max_sessions: number;
  uptime_seconds: number;
}

// ---- SSE Events ----

export interface SSEToolCallEvent {
  type: 'tool_call';
  tool_name: string;
  tool_args: Record<string, unknown>;
}

export interface SSEToolResultEvent {
  type: 'tool_result';
  tool_name: string;
  status: string;
}

export interface SSETextDeltaEvent {
  type: 'text_delta';
  text: string;
}

export interface SSEPlotEvent {
  type: 'plot';
  available: boolean;
}

export interface SSEDoneEvent {
  type: 'done';
  token_usage: Record<string, number>;
}

export interface SSEThinkingEvent {
  type: 'thinking';
  text: string;
}

export interface SSESessionTitleEvent {
  type: 'session_title';
  name: string;
}

export interface SSEErrorEvent {
  type: 'error';
  message: string;
}

export interface SSELogLineEvent {
  type: 'log_line';
  text: string;
  level: string;
}

export interface SSEMemoryUpdateEvent {
  type: 'memory_update';
  text: string;
  level: string;
  actions: Record<string, number>;
}

export type SSEEvent =
  | SSEToolCallEvent
  | SSEToolResultEvent
  | SSETextDeltaEvent
  | SSEThinkingEvent
  | SSEPlotEvent
  | SSESessionTitleEvent
  | SSEDoneEvent
  | SSEErrorEvent
  | SSELogLineEvent
  | SSEMemoryUpdateEvent;

// ---- Catalog SSE Events ----

export interface CatalogProgressEvent {
  type: 'progress';
  phase?: 'cdaweb' | 'ppi' | 'refresh';
  step: string;
  message?: string;
  done?: number;
  total?: number;
  failed?: number;
  pct?: number;
  current?: string;
}

export interface CatalogDoneEvent {
  type: 'done';
  message: string;
}

export interface CatalogErrorEvent {
  type: 'error';
  message: string;
}

export type CatalogSSEEvent = CatalogProgressEvent | CatalogDoneEvent | CatalogErrorEvent;

// ---- Persisted event log (from events.jsonl) ----

export interface SessionEventRecord {
  type: string;
  ts: string;
  agent: string;
  level: string;
  msg: string;
  data: Record<string, unknown>;
  tags: string[];
}

// ---- Frontend-only ----

export interface ChatMessage {
  id: string;
  role: 'user' | 'agent' | 'thinking' | 'plot' | 'system';
  content: string;
  timestamp: number;
  figure?: PlotlyFigure;
  figure_url?: string;
  /** Pre-rendered PNG thumbnail URL (used during session resume for instant preview). */
  thumbnailUrl?: string;
  /** Session ID for loading the full interactive figure from a thumbnail. */
  thumbnailSessionId?: string;
}

export interface CommandResponse {
  command: string;
  content: string;
  data?: Record<string, unknown>;
}

export interface ToolEvent {
  id: string;
  type: 'call' | 'result';
  tool_name: string;
  tool_args?: Record<string, unknown>;
  status?: string;
  timestamp: number;
}

export interface LogLine {
  id: string;
  text: string;
  level: string;
  timestamp: number;
}

export interface MemoryEvent {
  id: string;
  actions: Record<string, number>;
  timestamp: number;
}

// ---- Plotly ----

export interface PlotlyFigure {
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
}

// ---- Catalog ----

export interface MissionInfo {
  id: string;
  name: string;
}

export interface DatasetInfo {
  id: string;
  name: string;
  description?: string;
}

export interface ParameterInfo {
  name: string;
  description: string;
  units: string;
  size: number;
  dataset_id: string;
}

export interface TimeRange {
  start: string | null;
  stop: string | null;
}

// ---- Data ----

export interface DataPreview {
  label: string;
  total_rows: number;
  columns: string[];
  rows: Record<string, unknown>[];
}

export interface DataEntrySummary {
  label: string;
  num_points: number;
  units: string;
  description: string;
  source: string;
  is_timeseries: boolean;
  time_min: string | null;
  time_max: string | null;
  shape: string;
}

// ---- Config ----

export type AppConfig = Record<string, unknown>;

// ---- Memory ----

export interface MemoryEntry {
  id: string;
  type: string;
  scopes: string[];
  content: string;
  created_at: string;
  enabled: boolean;
  source: string;
  tags: string[];
  access_count: number;
  last_accessed: string;
  version: number;
  supersedes: string;
  source_session: string;
  review_of: string;
  review_summary?: {
    total_count: number;
    avg_stars: number;
  };
}

export interface MemoryStats {
  total_tokens: number;
  token_budget: number;
  type_counts: Record<string, number>;
  type_tokens: Record<string, number>;
  all_scopes: string[];
}

// ---- Pipeline ----

export interface SavedSessionWithOps {
  id: string;
  name: string | null;
  model: string | null;
  turn_count: number;
  last_message_preview: string;
  created_at: string | null;
  updated_at: string | null;
  op_count: number;
  has_renders: boolean;
}

export interface PipelineRecord {
  id: string;
  timestamp: string;
  tool: string;
  status: string;
  inputs: string[];
  outputs: string[];
  args: Record<string, unknown>;
  error?: string;
  contributes_to?: string[];
  state_count?: number;
  state_index?: number;
  product_family?: string;
}

export interface ReplayResult {
  steps_completed: number;
  steps_total: number;
  errors: { op_id: string; tool: string; error: string }[];
  figure: PlotlyFigure | null;
  figure_url?: string;
}

// ---- Gallery ----

export interface GalleryItem {
  id: string;
  name: string;
  session_id: string;
  render_op_id: string;
  created_at: string;
  thumbnail: string;
}

// ---- Validation ----

export interface ValidationRecord {
  version: number;
  source_file: string;
  validated_at: string;
  source_url: string;
  discrepancy_count: number;
}

export interface DatasetValidation {
  dataset_id: string;
  validated: boolean;
  validation_count: number;
  phantom_count: number;
  undocumented_count: number;
  phantom_params: string[];
  undocumented_params: string[];
  validations: ValidationRecord[];
}

export interface MissionValidation {
  mission_stem: string;
  display_name: string;
  dataset_count: number;
  validated_count: number;
  issue_count: number;
  total_phantom: number;
  total_undocumented: number;
  datasets: DatasetValidation[];
}

export interface ValidationOverview {
  missions: MissionValidation[];
}

// ---- Token Breakdown ----

export interface AgentUsageRow {
  agent: string;
  input: number;
  output: number;
  thinking: number;
  cached: number;
  calls: number;
}

export interface TokenBreakdown {
  total: Record<string, number>;
  breakdown: AgentUsageRow[];
  memory_bytes: number;
  data_entries: number;
}
