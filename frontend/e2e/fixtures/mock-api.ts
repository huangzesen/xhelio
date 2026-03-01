import type { Page, Route } from '@playwright/test';
import * as responses from '../mocks/responses';

/**
 * Response overrides: callers can provide partial overrides for specific endpoints.
 */
export type ResponseOverrides = {
  status?: unknown;
  sessions?: unknown;
  savedSessions?: unknown;
  loadingState?: unknown;
  config?: unknown;
  /** Extra route handlers keyed by URL pattern */
  routes?: Record<string, (route: Route) => Promise<void>>;
};

/** JSON response helper */
function json(data: unknown, status = 200) {
  return {
    status,
    contentType: 'application/json' as const,
    body: JSON.stringify(data),
  };
}

/** Empty SSE stream response */
const emptySSE = {
  status: 200,
  contentType: 'text/event-stream' as const,
  headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
  body: '',
};

/**
 * Check if a request URL is an actual API endpoint (not a Vite module request).
 * Vite serves source files like /src/api/client.ts which also contain "/api/".
 * We only want to intercept requests to /api/* (no /src/ prefix).
 */
function isApiRequest(url: string): boolean {
  const pathname = new URL(url).pathname;
  return pathname.startsWith('/api/');
}

/**
 * Set up mock API routes on the page.
 * Intercepts all /api/** requests with canned responses.
 *
 * IMPORTANT: Playwright matches routes in LIFO order (most recently registered wins).
 * So we register the catch-all first and specific routes last.
 *
 * NOTE: We use URL-based matching functions (not glob patterns) to avoid
 * intercepting Vite module requests like /src/api/client.ts.
 */
export async function setupMockAPI(page: Page, overrides: ResponseOverrides = {}) {
  const currentSessionId = responses.sessionCreated.session_id;

  // --- Catch-all for any unmatched API routes (registered FIRST = lowest priority) ---
  await page.route((url) => isApiRequest(url.toString()), (route) => {
    const url = route.request().url();
    if (!isApiRequest(url)) return route.fallback();
    return route.fulfill(json({}));
  });

  // --- Session wildcard (matches /api/sessions/{id}) ---
  await page.route((url) => {
    const p = new URL(url).pathname;
    return /^\/api\/sessions\/[^/]+$/.test(p);
  }, (route) => {
    const pathname = new URL(route.request().url()).pathname;
    // Skip /api/sessions/saved — let the more specific route handle it
    if (pathname === '/api/sessions/saved') return route.fallback();
    if (route.request().method() === 'GET') {
      return route.fulfill(json({
        ...responses.sessionDetail,
        session_id: currentSessionId,
      }));
    }
    if (route.request().method() === 'DELETE') {
      return route.fulfill({ status: 204, body: '' });
    }
    return route.fulfill(json({ ok: true }));
  });

  // --- POST /api/sessions (create) and GET /api/sessions (list) ---
  await page.route((url) => new URL(url).pathname === '/api/sessions', (route) => {
    if (route.request().method() === 'POST') {
      return route.fulfill(json(overrides.sessions ?? responses.sessionCreated));
    }
    return route.fulfill(json([]));
  });

  // --- Top-level endpoints ---

  await page.route((url) => new URL(url).pathname === '/api/status', (route) =>
    route.fulfill(json(overrides.status ?? responses.statusOk)),
  );

  await page.route((url) => new URL(url).pathname === '/api/loading-state/events', (route) =>
    route.fulfill(emptySSE),
  );

  await page.route((url) => new URL(url).pathname === '/api/loading-state', (route) =>
    route.fulfill(json(overrides.loadingState ?? responses.loadingStateReady)),
  );

  await page.route((url) => new URL(url).pathname === '/api/config', (route) =>
    route.fulfill(json(overrides.config ?? responses.configResponse)),
  );

  await page.route((url) => new URL(url).pathname === '/api/input-history', (route) => {
    if (route.request().method() === 'GET') {
      return route.fulfill(json(responses.inputHistory));
    }
    return route.fulfill(json({ ok: true }));
  });

  await page.route((url) => new URL(url).pathname === '/api/catalog/status', (route) =>
    route.fulfill(json({
      mission_count: 0,
      mission_names: [],
      total_datasets: 0,
      oldest_date: null,
    })),
  );

  // --- Session list endpoint ---

  await page.route((url) => new URL(url).pathname === '/api/sessions/saved', (route) =>
    route.fulfill(json(overrides.savedSessions ?? responses.savedSessionsList)),
  );

  // --- Session-specific sub-routes ---

  await page.route((url) => /^\/api\/sessions\/[^/]+\/events$/.test(new URL(url).pathname), (route) =>
    route.fulfill(emptySSE),
  );

  await page.route((url) => /^\/api\/sessions\/[^/]+\/events-log$/.test(new URL(url).pathname), (route) =>
    route.fulfill(json({ events: [] })),
  );

  await page.route((url) => /^\/api\/sessions\/[^/]+\/chat$/.test(new URL(url).pathname), (route) =>
    route.fulfill(json(responses.chatQueued)),
  );

  await page.route((url) => /^\/api\/sessions\/[^/]+\/completions$/.test(new URL(url).pathname), (route) =>
    route.fulfill(json(responses.completions)),
  );

  await page.route((url) => /^\/api\/sessions\/[^/]+\/command$/.test(new URL(url).pathname), (route) => {
    const body = JSON.parse(route.request().postData() || '{}');
    const cmd = body.command || '';
    return route.fulfill(json(responses.commandResponse(cmd, `Response for ${cmd}`)));
  });

  await page.route((url) => /^\/api\/sessions\/[^/]+\/plan$/.test(new URL(url).pathname), (route) =>
    route.fulfill(json({ plan_status: null })),
  );

  await page.route((url) => /^\/api\/sessions\/[^/]+\/figure\/thumbnail$/.test(new URL(url).pathname), (route) =>
    route.fulfill({ status: 404, body: '' }),
  );

  await page.route((url) => /^\/api\/sessions\/[^/]+\/figure$/.test(new URL(url).pathname), (route) =>
    route.fulfill(json({ figure: null, figure_url: null })),
  );

  await page.route((url) => /^\/api\/sessions\/[^/]+\/resume$/.test(new URL(url).pathname), (route) =>
    route.fulfill(json({ session_id: currentSessionId, event_log: [] })),
  );

  await page.route((url) => /^\/api\/sessions\/[^/]+\/rename$/.test(new URL(url).pathname), (route) =>
    route.fulfill(json({ ok: true })),
  );

  await page.route((url) => /^\/api\/sessions\/[^/]+\/token-breakdown$/.test(new URL(url).pathname), (route) =>
    route.fulfill(json({
      total: {},
      breakdown: [
        {
          agent: 'Orchestrator',
          input: 0,
          output: 0,
          thinking: 0,
          cached: 0,
          calls: 0,
          ctx_system: 0,
          ctx_tools: 0,
          ctx_history: 0,
          ctx_total: 0,
        },
      ],
      memory_bytes: 0,
      data_entries: 0,
      context_limits: {},
    })),
  );

  // Apply extra custom route handlers (highest priority since registered last)
  if (overrides.routes) {
    for (const [pattern, handler] of Object.entries(overrides.routes)) {
      await page.route(pattern, handler);
    }
  }

  return { currentSessionId };
}

/**
 * Inject a mock EventSource into the page.
 * Replaces window.EventSource so SSE tests can emit events via page.evaluate().
 */
export async function setupMockSSE(page: Page) {
  await page.addInitScript(() => {
    type Listener = (event: MessageEvent) => void;

    class MockEventSource {
      url: string;
      readyState = 1; // OPEN
      private listeners: Record<string, Listener[]> = {};

      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSED = 2;

      CONNECTING = 0;
      OPEN = 1;
      CLOSED = 2;

      onopen: ((ev: Event) => void) | null = null;
      onmessage: ((ev: MessageEvent) => void) | null = null;
      onerror: ((ev: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        (window as any).__mockSSEInstances = (window as any).__mockSSEInstances || [];
        (window as any).__mockSSEInstances.push(this);

        // Simulate async open
        setTimeout(() => {
          if (this.onopen) this.onopen(new Event('open'));
        }, 0);
      }

      addEventListener(type: string, listener: Listener) {
        if (!this.listeners[type]) this.listeners[type] = [];
        this.listeners[type].push(listener);
      }

      removeEventListener(type: string, listener: Listener) {
        const list = this.listeners[type];
        if (list) {
          this.listeners[type] = list.filter((l) => l !== listener);
        }
      }

      dispatchEvent(event: Event): boolean {
        const type = event.type;
        const listeners = this.listeners[type] || [];
        for (const listener of listeners) {
          listener(event as MessageEvent);
        }
        return true;
      }

      close() {
        this.readyState = 2; // CLOSED
      }

      _emit(type: string, data: unknown) {
        const event = new MessageEvent(type, {
          data: JSON.stringify(data),
        });
        this.dispatchEvent(event);
      }
    }

    (window as any).EventSource = MockEventSource;

    (window as any).__mockSSE = {
      emit(type: string, data: unknown) {
        const instances = (window as any).__mockSSEInstances || [];
        for (const instance of instances) {
          if (instance.readyState === 1) {
            instance._emit(type, data);
          }
        }
      },
      getInstances() {
        return (window as any).__mockSSEInstances || [];
      },
    };
  });
}
