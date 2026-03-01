import type { SSEEvent, CatalogSSEEvent } from './types';

/**
 * Session-lifetime SSE subscription using native EventSource (GET-based).
 *
 * Opens a persistent connection to `/api/sessions/{sessionId}/events`.
 * All agent events (text_delta, tool_call, done, etc.) arrive here.
 * The caller must call `.close()` on the returned EventSource to disconnect.
 */
export function subscribeToSession(
  sessionId: string,
  onEvent: (data: SSEEvent) => void,
  onError?: (err: Event) => void,
): EventSource {
  const es = new EventSource(`/api/sessions/${sessionId}/events`);
  const types = [
    'text_delta', 'tool_call', 'tool_result', 'plot', 'thinking',
    'round_start', 'round_end', 'error', 'log_line', 'memory_update', 'mpl_image', 'session_title',
    'queued', 'insight_result', 'insight_feedback', 'token_usage',
  ];
  for (const type of types) {
    es.addEventListener(type, (e) => {
      try {
        onEvent(JSON.parse((e as MessageEvent).data));
      } catch {
        /* skip malformed JSON */
      }
    });
  }
  if (onError) es.onerror = onError;
  return es;
}

/**
 * POST-based SSE stream parser.
 * Uses fetch + ReadableStream instead of EventSource (which only supports GET).
 */
export async function* chatStream(
  sessionId: string,
  message: string,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const res = await fetch(`/api/sessions/${sessionId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      // Keep the last incomplete line in the buffer
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const raw = line.slice(5).trim();
          try {
            const data = JSON.parse(raw) as SSEEvent;
            yield data;
          } catch {
            // Skip malformed JSON
          }
        }
        // event:, empty lines, and comments (':') are ignored
        // (event type is already in the JSON data's `type` field)
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * POST-based SSE stream for catalog operations (rebuild/refresh).
 * Same parsing pattern as chatStream but takes a plain endpoint URL.
 */
export async function* catalogStream(
  endpoint: string,
  signal?: AbortSignal,
): AsyncGenerator<CatalogSSEEvent> {
  const res = await fetch(endpoint, {
    method: 'POST',
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const raw = line.slice(5).trim();
          try {
            const data = JSON.parse(raw) as CatalogSSEEvent;
            yield data;
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
