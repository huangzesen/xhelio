/**
 * Canned API responses for mocked tests.
 * Each export matches a specific API endpoint's response shape.
 */

export const statusOk = {
  status: 'ok',
  active_sessions: 0,
  max_sessions: 10,
  uptime_seconds: 120,
  api_key_configured: true,
};

export const statusNoApiKey = {
  ...statusOk,
  api_key_configured: false,
};

export const sessionCreated = {
  session_id: 'test-001',
  model: 'gemini-2.0-flash',
  created_at: '2025-01-01T00:00:00Z',
  last_active: '2025-01-01T00:00:00Z',
  busy: false,
};

export const sessionDetail = {
  ...sessionCreated,
  token_usage: {},
  data_entries: 0,
  plan_status: null,
};

export const savedSessionsList: unknown[] = [];

export const chatQueued = {
  status: 'queued',
};

export const loadingStateReady = {
  phase: 'ready',
  is_ready: true,
  is_loading: false,
  progress_pct: 100,
  progress_message: '',
  error: null,
};

export const inputHistory = {
  history: [],
};

export const completions = {
  completions: [],
};

export const configResponse = {
  llm_provider: 'gemini',
  providers: {
    gemini: {
      model: 'gemini-2.0-flash',
      sub_agent_model: 'gemini-2.0-flash',
      insight_model: 'gemini-2.0-flash',
      inline_model: 'gemini-2.0-flash-lite',
    },
  },
};

export const commandResponse = (command: string, content: string) => ({
  command,
  content,
});

export const savedSession = (id: string, name: string, turns: number) => ({
  id,
  name,
  model: 'gemini-2.0-flash',
  turn_count: turns,
  last_message_preview: `Preview for ${name}`,
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
  token_usage: {},
});
