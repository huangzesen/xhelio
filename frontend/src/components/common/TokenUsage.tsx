import { useState, useEffect } from 'react';
import type { TokenBreakdown } from '../../api/types';
import * as api from '../../api/client';
import { useSessionStore } from '../../stores/sessionStore';
import { useChatStore } from '../../stores/chatStore';
import { ChevronDown } from 'lucide-react';

interface Props {
  usage: Record<string, number>;
}

export function TokenUsage({ usage }: Props) {
  const input = usage.input_tokens ?? usage.prompt_tokens ?? 0;
  const output = usage.output_tokens ?? usage.completion_tokens ?? 0;
  const total = usage.total_tokens ?? input + output;
  const thinking = usage.thinking_tokens ?? 0;
  const cached = usage.cached_tokens ?? 0;
  const apiCalls = usage.api_calls ?? 0;

  const [breakdown, setBreakdown] = useState<TokenBreakdown | null>(null);
  const [expanded, setExpanded] = useState(false);
  const { activeSessionId } = useSessionStore();
  const isStreaming = useChatStore((s) => s.isStreaming);

  // Poll token usage from the session detail while streaming
  useEffect(() => {
    if (!isStreaming || !activeSessionId) return;
    const interval = setInterval(async () => {
      try {
        const detail = await api.getSession(activeSessionId);
        if (detail.token_usage) {
          useSessionStore.getState().setTokenUsage(detail.token_usage);
        }
      } catch {
        // ignore
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [isStreaming, activeSessionId]);

  // Fetch breakdown when expanded
  useEffect(() => {
    if (!expanded || !activeSessionId) return;
    api.getTokenBreakdown(activeSessionId)
      .then(setBreakdown)
      .catch(() => {});
  }, [expanded, activeSessionId, total]);

  return (
    <div className="text-xs text-text-muted space-y-1">
      <div
        className="font-medium text-text mb-1 cursor-pointer flex items-center gap-1"
        onClick={() => setExpanded((v) => !v)}
      >
        Token Usage
        <ChevronDown size={12} className={`transition-transform ${expanded ? '' : '-rotate-90'}`} />
      </div>
      {total === 0 ? (
        <div className="text-text-muted italic">No API calls yet</div>
      ) : (
        <>
          <div className="flex justify-between">
            <span>Input</span>
            <span className="font-mono">{input.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span>Output</span>
            <span className="font-mono">{output.toLocaleString()}</span>
          </div>
          {thinking > 0 && (
            <div className="flex justify-between">
              <span>Thinking</span>
              <span className="font-mono">{thinking.toLocaleString()}</span>
            </div>
          )}
          {cached > 0 && (
            <div className="flex justify-between">
              <span>Cached</span>
              <span className="font-mono">{cached.toLocaleString()}</span>
            </div>
          )}
          <div className="flex justify-between border-t border-border pt-1">
            <span className="font-medium">Total</span>
            <span className="font-mono font-medium">{total.toLocaleString()}</span>
          </div>
          {apiCalls > 0 && (
            <div className="flex justify-between">
              <span>API Calls</span>
              <span className="font-mono">{apiCalls.toLocaleString()}</span>
            </div>
          )}
        </>
      )}

      {expanded && breakdown && (
        <div className="mt-2 pt-2 border-t border-border space-y-2">
          {/* Per-agent breakdown */}
          {breakdown.breakdown.length > 0 && (
            <div className="space-y-1">
              <div className="font-medium text-text text-[10px] uppercase tracking-wide">By Agent</div>
              {breakdown.breakdown.map((row) => {
                const rowTotal = row.input + row.output + row.thinking;
                return (
                  <div key={row.agent} className="flex justify-between">
                    <span>{row.agent}</span>
                    <span className="font-mono">{rowTotal.toLocaleString()}</span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Data in RAM */}
          <div className="space-y-1">
            <div className="font-medium text-text text-[10px] uppercase tracking-wide">Data Store</div>
            <div className="flex justify-between">
              <span>Entries</span>
              <span className="font-mono">{breakdown.data_entries}</span>
            </div>
            <div className="flex justify-between">
              <span>Memory</span>
              <span className="font-mono">
                {breakdown.memory_bytes < 1024 * 1024
                  ? `${(breakdown.memory_bytes / 1024).toFixed(1)} KB`
                  : `${(breakdown.memory_bytes / (1024 * 1024)).toFixed(1)} MB`}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
