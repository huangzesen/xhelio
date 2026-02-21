import { useState, useEffect, useRef, useCallback } from 'react';
import { Brain, ChevronDown, ChevronRight, Wrench, Check, X, Clock } from 'lucide-react';
import type { ToolEvent } from '../../api/types';
import type { ThinkingEvent } from '../../utils/groupEventsByTurn';

interface Props {
  events: ToolEvent[];
  thinkingEvents?: ThinkingEvent[];
  isStreaming?: boolean;
  /** Timestamp (ms) when the user message was sent — used as start time for the clock. */
  streamStartTs?: number;
}

const toolCategory: Record<string, string> = {
  fetch_data: 'data',
  custom_operation: 'compute',
  render_plotly_json: 'viz',
  manage_plot: 'viz',
  list_fetched_data: 'data',
  store_dataframe: 'data',
  get_spacecraft_position: 'spice',
  get_spacecraft_trajectory: 'spice',
  compute_distance: 'spice',
  search_datasets: 'catalog',
  discover_datasets: 'catalog',
};

const categoryColor: Record<string, string> = {
  data: 'text-badge-blue-text',
  compute: 'text-badge-orange-text',
  viz: 'text-badge-red-text',
  spice: 'text-badge-purple-text',
  catalog: 'text-badge-teal-text',
};

function truncateJson(obj: Record<string, unknown>, maxLen = 160): string {
  const s = JSON.stringify(obj);
  return s.length > maxLen ? s.slice(0, maxLen) + '...' : s;
}

function formatElapsed(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  if (totalSec < 60) return `${totalSec}s`;
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}m ${sec.toString().padStart(2, '0')}s`;
}

function ThinkingItem({ evt }: { evt: ThinkingEvent }) {
  const [expanded, setExpanded] = useState(false);
  const firstLine = evt.content.split('\n')[0].replace(/^[#*\s]+/, '').trim();
  const summary = firstLine.length > 80 ? firstLine.slice(0, 80) + '...' : firstLine;
  return (
    <div className="text-xs text-text-muted">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1.5 text-badge-purple-text hover:opacity-80 transition-colors"
      >
        {expanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <Brain size={11} />
        <span className="text-text-muted truncate">{summary}</span>
      </button>
      {expanded && (
        <div className="ml-5 mt-0.5 text-[10px] text-text-muted whitespace-pre-wrap break-words opacity-80 max-h-32 overflow-y-auto">
          {evt.content}
        </div>
      )}
    </div>
  );
}

/**
 * Robust elapsed-time hook.
 *
 * Uses a ref-based interval that survives HMR remounts and transient
 * isStreaming flickers.  The interval ticks `now` every second while
 * streaming is active.  When streaming stops, we record the final
 * timestamp once so the displayed time stays correct.
 */
function useElapsedTime(isStreaming: boolean, startTs: number, lastEventTs: number) {
  const [now, setNow] = useState(Date.now);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frozenEndRef = useRef<number | null>(null);

  const startTicking = useCallback(() => {
    if (intervalRef.current) return;                // already running
    setNow(Date.now());                             // sync immediately
    intervalRef.current = setInterval(() => setNow(Date.now()), 1000);
  }, []);

  const stopTicking = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (isStreaming) {
      frozenEndRef.current = null;                  // clear frozen end
      startTicking();
    } else {
      stopTicking();
      frozenEndRef.current = Date.now();            // snapshot the end time once
    }
    return stopTicking;
  }, [isStreaming, startTicking, stopTicking]);

  const endTs = isStreaming
    ? now
    : frozenEndRef.current ?? lastEventTs;

  return Math.max(0, endTs - startTs);
}

export function ToolCallGroup({ events, thinkingEvents = [], isStreaming = false, streamStartTs }: Props) {
  const steps = events.filter((e) => e.type === 'call').length;

  // Merge tool events and thinking events into a single timeline
  const merged = [
    ...events.map((e) => ({ kind: 'tool' as const, ts: e.timestamp, item: e })),
    ...thinkingEvents.map((e) => ({ kind: 'thinking' as const, ts: e.timestamp, item: e })),
  ].sort((a, b) => a.ts - b.ts);

  const lastTs = merged.length > 0 ? merged[merged.length - 1].ts : 0;

  // Compute elapsed time — start from when the user sent the message (includes LLM wait)
  const firstTs = streamStartTs ?? (merged.length > 0 ? merged[0].ts : Date.now());
  const elapsed = useElapsedTime(isStreaming, firstTs, lastTs);

  // Don't render if there's nothing to show (placed after hooks to avoid conditional hook calls)
  if (events.length === 0 && thinkingEvents.length === 0) return null;

  return (
    <details className="group mx-10 my-1">
      <summary className="flex items-center gap-2 text-xs text-text-muted cursor-pointer select-none hover:text-text transition-colors">
        <ChevronDown
          size={14}
          className="transition-transform group-open:rotate-0 -rotate-90"
        />
        <Wrench size={12} />
        Agent activity ({steps} {steps === 1 ? 'step' : 'steps'})
        <span className="flex items-center gap-1 ml-1 opacity-70">
          <Clock size={10} />
          {formatElapsed(elapsed)}
        </span>
      </summary>
      <div className="ml-5 mt-1 space-y-0.5 border-l-2 border-border pl-3">
        {merged.map((entry) => {
          if (entry.kind === 'thinking') {
            const te = entry.item as ThinkingEvent;
            return <ThinkingItem key={te.id} evt={te} />;
          }
          const evt = entry.item as ToolEvent;
          const cat = toolCategory[evt.tool_name] ?? 'default';
          const color = categoryColor[cat] ?? 'text-text-muted';
          return (
            <div key={evt.id} className="text-xs text-text-muted">
              <div className="flex items-center gap-1.5">
                {evt.type === 'result' ? (
                  evt.status === 'success' ? (
                    <Check size={11} className="text-status-success-text" />
                  ) : (
                    <X size={11} className="text-status-error-text" />
                  )
                ) : (
                  <span className={`w-[11px] h-[11px] flex items-center justify-center ${color}`}>
                    &bull;
                  </span>
                )}
                <span className={`font-mono ${color}`}>{evt.tool_name}</span>
              </div>
              {evt.type === 'call' && evt.tool_args && Object.keys(evt.tool_args).length > 0 && (
                <div className="ml-5 mt-0.5 text-[10px] text-text-muted font-mono truncate opacity-70">
                  {truncateJson(evt.tool_args)}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </details>
  );
}
