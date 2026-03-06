/**
 * Sandboxed iframe renderer for compiled JSX/Recharts components.
 *
 * Fetches the compiled bundle (.js) and data (.data.json) from the API,
 * builds a self-contained HTML page via iframeTemplate, and renders it
 * in a sandboxed iframe. Auto-resizes via postMessage from the iframe.
 */

import { useEffect, useRef, useState } from 'react';
import { buildIframeHtml } from './iframeTemplate';

interface JsxComponentProps {
  /** Session ID for API requests */
  sessionId: string;
  /** Script ID identifying the compiled component */
  scriptId: string;
}

export default function JsxComponent({ sessionId, scriptId }: JsxComponentProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [html, setHtml] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [height, setHeight] = useState(400); // default height

  // Fetch bundle + data in parallel, then build HTML
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [bundleRes, dataRes] = await Promise.all([
          fetch(`/api/sessions/${sessionId}/jsx-outputs/${scriptId}.js`),
          fetch(`/api/sessions/${sessionId}/jsx-outputs/${scriptId}.data.json`),
        ]);

        if (cancelled) return;

        if (!bundleRes.ok) {
          setError(`Failed to load bundle: ${bundleRes.status} ${bundleRes.statusText}`);
          setLoading(false);
          return;
        }

        const bundleCode = await bundleRes.text();
        const dataJson = dataRes.ok ? await dataRes.text() : '{}';

        if (cancelled) return;

        const builtHtml = buildIframeHtml(bundleCode, dataJson);
        setHtml(builtHtml);
        setLoading(false);
      } catch (err) {
        if (!cancelled) {
          setError(String(err));
          setLoading(false);
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, [sessionId, scriptId]);

  // Listen for resize messages from the iframe
  useEffect(() => {
    function handleMessage(event: MessageEvent) {
      if (event.data?.type === 'xhelio-resize' && typeof event.data.height === 'number') {
        setHeight(Math.max(100, Math.min(event.data.height + 16, 2000)));
      }
    }

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8 bg-panel rounded-lg border border-border">
        <div className="flex items-center gap-2 text-sm text-text-muted">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span>Loading component...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
        <p className="text-sm text-red-700 dark:text-red-300 font-medium">Failed to load component</p>
        <p className="text-xs text-red-600 dark:text-red-400 mt-1">{error}</p>
      </div>
    );
  }

  if (!html) return null;

  return (
    <iframe
      ref={iframeRef}
      sandbox="allow-scripts"
      srcDoc={html}
      style={{
        width: '100%',
        height: `${height}px`,
        border: 'none',
        borderRadius: '8px',
        background: 'white',
      }}
      title="Recharts visualization"
    />
  );
}
