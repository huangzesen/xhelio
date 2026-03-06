import { useState, useEffect } from 'react';
import type { TokenBreakdown, AgentUsageRow } from '../../api/types';
import * as api from '../../api/client';
import { useSessionStore } from '../../stores/sessionStore';
import { useChatStore } from '../../stores/chatStore';
import { ArrowDown, ArrowUp, ChevronRight, ChevronDown } from 'lucide-react';
import { friendlyAgentName } from '../../utils/friendlyAgentName';
import { agentModelTier } from '../../utils/agentModelTier';

interface Props {
  usage: Record<string, number>;
}

function AgentRow(props: { row: AgentUsageRow, limits: Record<string, number> }) {
  const [expanded, setExpanded] = useState(false);
  const rowTotal = props.row.input + props.row.output + (props.row.thinking || 0);
  const canExpand = props.row.ctx_total !== undefined && props.row.ctx_total > 0;
  
  const tier = agentModelTier(props.row.agent);
  const limit = props.limits[tier] || 0;
  const ctxTotal = props.row.ctx_total || 0;
  const pct = limit > 0 ? (ctxTotal / limit) * 100 : 0;
  
  const system = props.row.ctx_system || 0;
  const tools = props.row.ctx_tools || 0;
  const history = props.row.ctx_history || 0;

  // Bar segments
  const systemW = limit > 0 ? (system / limit) * 100 : 0;
  const toolsW = limit > 0 ? (tools / limit) * 100 : 0;
  const historyW = limit > 0 ? (history / limit) * 100 : 0;

  const pctColor = pct > 95 ? 'text-red-500' : pct > 80 ? 'text-amber-500' : 'text-text-muted';

  return (
    <div className="space-y-1">
      <div 
        className={`flex justify-between items-center ${canExpand ? 'cursor-pointer hover:bg-white/5' : ''} -mx-1 px-1 rounded transition-colors`}
        onClick={() => canExpand && setExpanded(!expanded)}
      >
        <div className="flex items-center gap-1 min-w-0">
          {canExpand && (
            <span className="text-text-muted">
              {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
            </span>
          )}
          {!canExpand && <span className="w-[10px]" />}
          <span className="truncate">{friendlyAgentName(props.row.agent)}</span>
        </div>
        <span className="font-mono">{rowTotal.toLocaleString()}</span>
      </div>

      {expanded && canExpand && (
        <div className="pl-3.5 pr-1 py-1 space-y-1.5 border-l border-border/50 ml-1.5 mt-0.5">
          {/* Stacked Bar */}
          <div className="space-y-1">
            <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden flex">
              <div style={{ width: `${systemW}%` }} className="h-full bg-blue-500/80" title={`System: ${system.toLocaleString()}`} />
              <div style={{ width: `${toolsW}%` }} className="h-full bg-purple-500/80" title={`Tools: ${tools.toLocaleString()}`} />
              <div style={{ width: `${historyW}%` }} className="h-full bg-green-500/80" title={`History: ${history.toLocaleString()}`} />
            </div>
            <div className={`flex justify-between text-[9px] ${pctColor}`}>
               <span>{ctxTotal.toLocaleString()} / {limit.toLocaleString()}</span>
               <span>{pct.toFixed(1)}%</span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-y-0.5 text-[10px]">
            <div className="flex justify-between text-text-muted">
              <span>System</span>
              <span className="font-mono text-blue-400/80">{system.toLocaleString()}</span>
            </div>
            <div className="flex justify-between text-text-muted">
              <span>Tools</span>
              <span className="font-mono text-purple-400/80">{tools.toLocaleString()}</span>
            </div>
            <div className="flex justify-between text-text-muted">
              <span>History</span>
              <span className="font-mono text-green-400/80">{history.toLocaleString()}</span>
            </div>
            <div className="flex justify-between text-text-muted border-t border-border/30 pt-0.5 mt-0.5">
              <span>Limit</span>
              <span className="font-mono">{limit.toLocaleString()}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function TokenUsage({ usage }: Props) {
  const input = usage.input_tokens ?? usage.prompt_tokens ?? 0;
  const output = usage.output_tokens ?? usage.completion_tokens ?? 0;
  const total = usage.total_tokens ?? input + output;
  const thinking = usage.thinking_tokens ?? 0;
  const cached = usage.cached_tokens ?? 0;
  const apiCalls = usage.api_calls ?? 0;

  const [breakdown, setBreakdown] = useState<TokenBreakdown | null>(null);
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

  // Fetch breakdown on mount and when total changes
  useEffect(() => {
    if (!activeSessionId) return;
    api.getTokenBreakdown(activeSessionId)
      .then(setBreakdown)
      .catch(() => {});
  }, [activeSessionId, total]);

  return (
    <div className="text-xs text-text-muted space-y-1">
      {total === 0 ? (
        <div className="text-text-muted italic">No API calls yet</div>
      ) : (
        <>
          <div className="flex justify-between">
            <span className="font-medium">Total</span>
            <span className="font-mono font-medium flex items-center gap-1">
              <span className="flex items-center gap-0.5 text-text-muted font-normal text-[10px]">
                (<ArrowUp size={9} className="inline" />{input.toLocaleString()}
                <ArrowDown size={9} className="inline" />{output.toLocaleString()})
              </span>
              {total.toLocaleString()}
            </span>
          </div>
          {apiCalls > 0 && (
            <div className="flex justify-between">
              <span>API Calls</span>
              <span className="font-mono">{apiCalls.toLocaleString()}</span>
            </div>
          )}
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

          {/* Per-agent breakdown */}
          {breakdown && breakdown.breakdown.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border space-y-1">
              <div className="font-medium text-text text-[10px] uppercase tracking-wide">By Agent</div>
              {breakdown.breakdown.map((row) => (
                <AgentRow key={row.agent} row={row} limits={breakdown.context_limits} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
