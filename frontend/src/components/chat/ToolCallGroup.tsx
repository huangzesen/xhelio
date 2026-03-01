import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { AlertTriangle, Brain, ChevronDown, ChevronRight, Check, Clock, MessageSquareText, X, Loader2 } from 'lucide-react';
import type { ToolEvent, CommentaryEvent } from '../../api/types';
import type { ThinkingEvent } from '../../utils/groupEventsByTurn';
import { friendlyAgentName } from '../../utils/friendlyAgentName';

// ---- Agent grouping types ----

interface AgentToolCall {
  callEvent: ToolEvent;
  resultEvent?: ToolEvent;
}

interface AgentGroup {
  rawId: string;
  displayName: string;
  toolCalls: AgentToolCall[];
  status: 'active' | 'complete' | 'partial' | 'error';
  commentary: CommentaryEvent[];
  thinking: ThinkingEvent[];
}

// ---- Commentary grouping helper ----

interface CommentaryGroup {
  agentId: string;
  displayName: string;
  events: CommentaryEvent[];
}

function groupCommentaryByAgent(events: CommentaryEvent[]): CommentaryGroup[] {
  const map = new Map<string, CommentaryEvent[]>();
  for (const evt of events) {
    const agentId = evt.agent || 'orchestrator';
    let list = map.get(agentId);
    if (!list) {
      list = [];
      map.set(agentId, list);
    }
    list.push(evt);
  }
  // Sort: orchestrator first, then alphabetical
  const sorted = [...map.entries()].sort((a, b) => {
    if (a[0] === 'orchestrator') return -1;
    if (b[0] === 'orchestrator') return 1;
    return a[0].localeCompare(b[0]);
  });
  return sorted.map(([id, evts]) => ({
    agentId: id,
    displayName: friendlyAgentName(id),
    events: evts,
  }));
}

interface Props {
  events: ToolEvent[];
  thinkingEvents?: ThinkingEvent[];
  commentaryEvents?: CommentaryEvent[];
  isStreaming?: boolean;
  /** Timestamp (ms) when the round started — used as start time for the clock. */
  streamStartTs?: number;
  /** Timestamp (ms) when the round ended — used for accurate completed-round clocks. */
  roundEndTs?: number;
  /** Per-round token delta (set on round_end). */
  roundTokenUsage?: Record<string, number> | null;
}

// ---- Helpers ----

function formatElapsed(ms: number): string {
  if (!isFinite(ms) || ms < 0) return '0s';
  const totalSec = Math.floor(ms / 1000);
  if (totalSec < 60) return `${totalSec}s`;
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  if (min >= 60) {
    const hr = Math.floor(min / 60);
    const rm = min % 60;
    return `${hr}h ${rm.toString().padStart(2, '0')}m`;
  }
  return `${min}m ${sec.toString().padStart(2, '0')}s`;
}

function formatTokenK(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function truncateResult(result: string | undefined, maxLength: number = 120): string {
  if (!result) return '';
  const trimmed = result.trim();
  if (trimmed.length <= maxLength) return trimmed;
  return trimmed.slice(0, maxLength) + '...';
}

// ---- Elapsed time hook ----

function useElapsedTime(isStreaming: boolean, startTs: number, lastEventTs: number, completedEndTs?: number) {
  const [now, setNow] = useState(Date.now());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startTicking = useCallback(() => {
    if (intervalRef.current) return;
    setNow(Date.now());
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
      startTicking();
    } else {
      stopTicking();
    }
    return stopTicking;
  }, [isStreaming, startTicking, stopTicking]);

  const endTs = isStreaming
    ? now
    : completedEndTs ?? lastEventTs;

  const raw = endTs - startTs;
  const MAX_REASONABLE_MS = 24 * 60 * 60 * 1000;
  if (raw < 0 || raw > MAX_REASONABLE_MS || !isFinite(raw)) return 0;
  return raw;
}

// ---- Group tool calls by agent ----

function groupEventsByAgent(
  events: ToolEvent[],
  commentaryEvents: CommentaryEvent[],
  thinkingEvents: ThinkingEvent[]
): AgentGroup[] {
  const agentMap = new Map<string, {
    toolCalls: AgentToolCall[];
    commentary: CommentaryEvent[];
    thinking: ThinkingEvent[];
  }>();

  // Get all unique agent IDs
  const agentIds = new Set<string>();
  for (const evt of events) {
    agentIds.add(evt.agent || 'orchestrator');
  }
  for (const evt of commentaryEvents) {
    agentIds.add(evt.agent || 'orchestrator');
  }
  for (const _ of thinkingEvents) {
    // Thinking events may not have agent, default to orchestrator
    agentIds.add('orchestrator');
  }

  // Initialize agent groups
  for (const id of agentIds) {
    agentMap.set(id, { toolCalls: [], commentary: [], thinking: [] });
  }

  // Separate calls and results per agent (preserving order)
  const agentCalls = new Map<string, ToolEvent[]>();
  const agentResults = new Map<string, ToolEvent[]>();

  for (const evt of events) {
    const agentId = evt.agent || 'orchestrator';
    if (evt.type === 'call') {
      let list = agentCalls.get(agentId);
      if (!list) { list = []; agentCalls.set(agentId, list); }
      list.push(evt);
    } else if (evt.type === 'result') {
      let list = agentResults.get(agentId);
      if (!list) { list = []; agentResults.set(agentId, list); }
      list.push(evt);
    }
  }

  // Pair sequentially: 1st call pairs with 1st result, etc.
  for (const [agentId, calls] of agentCalls.entries()) {
    const group = agentMap.get(agentId);
    if (!group) continue;
    const results = agentResults.get(agentId) || [];
    for (let i = 0; i < calls.length; i++) {
      group.toolCalls.push({
        callEvent: calls[i],
        resultEvent: i < results.length ? results[i] : undefined,
      });
    }
    agentResults.delete(agentId);
  }

  // Handle any orphaned results (results for agents that had no calls — defensive fallback)
  for (const [agentId, results] of agentResults.entries()) {
    const group = agentMap.get(agentId);
    if (!group) continue;
    for (const evt of results) {
      group.toolCalls.push({
        callEvent: {
          id: evt.id,
          type: 'call',
          tool_name: evt.tool_name,
          timestamp: evt.timestamp,
          agent: agentId,
        },
        resultEvent: evt,
      });
    }
  }

  // Group commentary by agent
  for (const evt of commentaryEvents) {
    const agentId = evt.agent || 'orchestrator';
    const group = agentMap.get(agentId);
    if (group) {
      group.commentary.push(evt);
    }
  }

  // Group thinking by agent (attribute all to orchestrator for now)
  for (const evt of thinkingEvents) {
    const group = agentMap.get('orchestrator');
    if (group) {
      group.thinking.push(evt);
    }
  }

  // Determine status for each agent
  const result: AgentGroup[] = [];
  for (const [rawId, data] of agentMap.entries()) {
    const completedResults = data.toolCalls.filter(tc => tc.resultEvent !== undefined);
    const errorCount = completedResults.filter(tc => tc.resultEvent!.status === 'error').length;
    const successCount = completedResults.length - errorCount;
    const allDone = completedResults.length === data.toolCalls.length;

    let status: 'active' | 'complete' | 'partial' | 'error' = 'complete';
    if (!allDone) {
      status = 'active';
    } else if (errorCount > 0 && successCount > 0) {
      status = 'partial';
    } else if (errorCount > 0 && successCount === 0) {
      status = 'error';
    }

    result.push({
      rawId,
      displayName: friendlyAgentName(rawId),
      toolCalls: data.toolCalls,
      status,
      commentary: data.commentary,
      thinking: data.thinking,
    });
  }

  // Sort: orchestrator first, then alphabetical
  return result.sort((a, b) => {
    if (a.rawId === 'orchestrator') return -1;
    if (b.rawId === 'orchestrator') return 1;
    return a.rawId.localeCompare(b.rawId);
  });
}

// ---- Animated spinner component ----

function Spinner({ size = 12 }: { size?: number }) {
  return (
    <Loader2
      size={size}
      className="animate-spin text-primary"
    />
  );
}

// ---- Progress bar component ----

function ProgressBar({ completed, total }: { completed: number; total: number }) {
  const ratio = total > 0 ? (completed / total) * 100 : 0;
  return (
    <div className="w-16 h-1 bg-border rounded-full overflow-hidden">
      <div
        className="h-full bg-primary transition-all duration-300"
        style={{ width: `${ratio}%` }}
      />
    </div>
  );
}

// ---- Thinking sub-component ----

function ThinkingItem({ evt }: { evt: ThinkingEvent }) {
  const [expanded, setExpanded] = useState(false);
  const firstLine = evt.content.split('\n')[0].replace(/^[#*\s]+/, '').trim();
  const summary = firstLine.length > 80 ? firstLine.slice(0, 80) + '...' : firstLine;
  return (
    <div className="text-xs">
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

// ---- Commentary per-agent group sub-component ----

function CommentaryAgentGroup({ group }: { group: CommentaryGroup }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="text-xs">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1.5 cursor-pointer select-none hover:text-text transition-colors"
      >
        {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        <span className="font-mono">{group.displayName}</span>
        <span className="opacity-60">({group.events.length})</span>
      </button>
      {expanded && (
        <div className="ml-4 mt-0.5 space-y-0.5 max-h-40 overflow-y-auto">
          {group.events.map((ce) => (
            <div key={ce.id} className="text-[10px] text-text-muted italic opacity-80 break-words">
              {ce.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Per-agent tool call detail ----

function AgentToolDetail({ toolCall }: { toolCall: AgentToolCall }) {
  const [expanded, setExpanded] = useState(false);
  const call = toolCall.callEvent;
  const result = toolCall.resultEvent;

  const duration = result && call
    ? Math.round((result.timestamp - call.timestamp) / 1000)
    : null;

  return (
    <div className="ml-4 py-1 border-l border-border pl-2">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-2 text-xs w-full hover:bg-hover-bg rounded px-1 py-0.5 transition-colors"
      >
        {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        <span className="font-mono text-primary-dark">{call.tool_name}</span>
        {duration !== null && (
          <span className="opacity-60 text-[10px]">{duration}s</span>
        )}
        {result && (
          <span className={`ml-auto text-[10px] px-1 rounded ${
            result.status === 'error'
              ? 'bg-badge-red-bg text-badge-red-text'
              : 'bg-badge-green-bg text-badge-green-text'
          }`}>
            {result.status === 'error' ? 'error' : 'done'}
          </span>
        )}
      </button>
      {expanded && result && (
        <div className="ml-5 mt-1 text-[10px] text-text-muted font-mono bg-code-bg text-code-text p-2 rounded max-h-32 overflow-auto">
          {truncateResult(result.tool_name ? '' : JSON.stringify(result), 500)}
          {result.tool_name && result.status !== 'error' && (
            <span className="opacity-70">Result received</span>
          )}
        </div>
      )}
    </div>
  );
}

// ---- Agent group detail ----

function AgentGroupDetail({ group, isLast }: { group: AgentGroup; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const completedTools = group.toolCalls.filter(tc => tc.resultEvent).length;
  const totalTools = group.toolCalls.length;

  return (
    <div className={`${!isLast ? 'border-b border-border/30' : ''} py-1.5`}>
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-2 text-xs w-full hover:bg-hover-bg rounded px-1 py-0.5 transition-colors"
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        
        {/* Status icon */}
        {group.status === 'active' ? (
          <Spinner size={11} />
        ) : group.status === 'error' ? (
          <X size={11} className="text-status-error-text" />
        ) : group.status === 'partial' ? (
          <AlertTriangle size={11} className="text-status-warning-text" />
        ) : (
          <Check size={11} className="text-status-success-text" />
        )}

        {/* Agent name */}
        <span className="font-mono font-medium">{group.displayName}</span>

        {/* Tool count */}
        <span className="opacity-60">
          {completedTools}/{totalTools} tools
        </span>

        {/* Progress bar when active */}
        {group.status === 'active' && (
          <ProgressBar completed={completedTools} total={totalTools} />
        )}
      </button>

      {/* Tool calls list */}
      {expanded && group.toolCalls.length > 0 && (
        <div className="ml-2 mt-1">
          {group.toolCalls.map((tc, idx) => (
            <AgentToolDetail key={tc.callEvent.id || idx} toolCall={tc} />
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Main Component ----

export function ToolCallGroup({
  events,
  thinkingEvents = [],
  commentaryEvents = [],
  isStreaming = false,
  streamStartTs,
  roundEndTs,
  roundTokenUsage,
}: Props) {
  // Auto-expand/collapse state
  const [userToggled, setUserToggled] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [commentaryExpanded, setCommentaryExpanded] = useState(false);

  // Determine initial expanded state based on streaming/completion
  useEffect(() => {
    if (!userToggled) {
      // Auto-expand when streaming, auto-collapse when completed
      setExpanded(isStreaming);
    }
  }, [isStreaming, userToggled]);

  // Merge timestamps for elapsed time computation
  const allTs = useMemo(() => [
    ...events.map((e) => e.timestamp),
    ...thinkingEvents.map((e) => e.timestamp),
  ], [events, thinkingEvents]);

  const lastTs = allTs.length > 0 ? Math.max(...allTs) : 0;
  const firstTs = streamStartTs ?? (allTs.length > 0 ? Math.min(...allTs) : Date.now());
  const elapsed = useElapsedTime(isStreaming, firstTs, lastTs, roundEndTs);

  // Group events by agent
  const agentGroups = useMemo(
    () => groupEventsByAgent(events, commentaryEvents, thinkingEvents),
    [events, commentaryEvents, thinkingEvents]
  );

  // Calculate total stats
  const totalCalls = events.filter((e) => e.type === 'call').length;
  const completedCalls = events.filter((e) => e.type === 'result').length;
  const tokenIn = roundTokenUsage?.input_tokens || 0;
  const tokenOut = roundTokenUsage?.output_tokens || 0;
  const showTokens = tokenIn > 0 || tokenOut > 0;

  // Get latest commentary for collapsed header
  const latestCommentary = commentaryEvents.length > 0
    ? commentaryEvents[commentaryEvents.length - 1].text
    : null;

  // Determine overall status
  const allError = agentGroups.length > 0 && agentGroups.every(g => g.status === 'error');
  const anyErrorOrPartial = agentGroups.some(g => g.status === 'error' || g.status === 'partial');

  // Don't render if nothing to show and not streaming
  if (events.length === 0 && thinkingEvents.length === 0 && !isStreaming) return null;

  // Processing indicator when streaming but no events yet
  if (events.length === 0 && thinkingEvents.length === 0 && isStreaming) {
    return (
      <div className="mx-10 my-1 flex items-center gap-2 text-xs text-text-muted">
        <Spinner size={12} />
        <span className="font-medium">Processing...</span>
        <span className="flex items-center gap-1 ml-1 opacity-70">
          <Clock size={10} />
          {formatElapsed(elapsed)}
        </span>
      </div>
    );
  }

  const handleToggle = () => {
    setUserToggled(true);
    setExpanded(v => !v);
  };

  return (
    <div className="mx-10 my-1">
      {/* Collapsed header — always visible */}
      <button
        onClick={handleToggle}
        className="flex items-center gap-2 text-xs text-text-muted cursor-pointer select-none hover:text-text transition-colors w-full text-left"
      >
        {/* Status icon */}
        {isStreaming ? (
          <Spinner size={14} />
        ) : allError ? (
          <X size={14} className="text-status-error-text" />
        ) : anyErrorOrPartial ? (
          <AlertTriangle size={14} className="text-status-warning-text" />
        ) : (
          <Check size={14} className="text-status-success-text" />
        )}

        {/* Chevron */}
        <ChevronDown
          size={14}
          className={`transition-transform ${expanded ? 'rotate-0' : '-rotate-90'}`}
        />

        {/* Status text */}
        <span className={`font-medium ${
          isStreaming
            ? 'text-primary'
            : allError
              ? 'text-status-error-text'
              : anyErrorOrPartial
                ? 'text-status-warning-text'
                : 'text-status-success-text'
        }`}>
          {isStreaming
            ? 'Working'
            : allError
              ? 'Failed'
              : anyErrorOrPartial
                ? 'Completed with errors'
                : 'Completed'}
        </span>

        {/* Agent count */}
        {agentGroups.length > 0 && (
          <span className="opacity-70">
            {agentGroups.length} {agentGroups.length === 1 ? 'agent' : 'agents'}
          </span>
        )}

        {/* Elapsed time */}
        <span className="flex items-center gap-1 opacity-70">
          <Clock size={10} />
          {formatElapsed(elapsed)}
        </span>

        {/* Tokens */}
        {showTokens && (
          <span className="flex items-center gap-1.5 opacity-60 font-mono">
            <span title="Input tokens">↑{formatTokenK(tokenIn)}</span>
            <span title="Output tokens">↓{formatTokenK(tokenOut)}</span>
          </span>
        )}

        {/* Tools count (when no tokens) */}
        {!showTokens && totalCalls > 0 && (
          <span className="opacity-60">
            {completedCalls}/{totalCalls} tools
          </span>
        )}

        {/* Progress bar when streaming */}
        {isStreaming && totalCalls > 0 && (
          <ProgressBar completed={completedCalls} total={totalCalls} />
        )}
      </button>

      {/* Latest commentary inline (collapsed state) */}
      {!expanded && latestCommentary && (
        <div className="ml-8 mt-0.5 text-xs text-text-muted italic opacity-80 truncate max-w-md">
          {latestCommentary}
        </div>
      )}

      {/* Expanded content */}
      {expanded && (
        <div className="ml-5 mt-1 space-y-0.5 border-l-2 border-border pl-3">
          {/* Agent groups */}
          {agentGroups.map((group, idx) => (
            <AgentGroupDetail
              key={group.rawId}
              group={group}
              isLast={idx === agentGroups.length - 1 && commentaryEvents.length === 0 && thinkingEvents.length === 0}
            />
          ))}

          {/* Commentary section */}
          {commentaryEvents.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border/50">
              <button
                onClick={() => setCommentaryExpanded((v) => !v)}
                className="flex items-center gap-1.5 text-xs text-text-muted cursor-pointer select-none hover:text-text transition-colors"
              >
                {commentaryExpanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                <MessageSquareText size={11} className="opacity-70" />
                <span>Commentary</span>
                <span className="opacity-60">({commentaryEvents.length})</span>
              </button>
              {commentaryExpanded && (
                <div className="ml-4 mt-1 space-y-1">
                  {groupCommentaryByAgent(commentaryEvents).map((group) => (
                    <CommentaryAgentGroup key={group.agentId} group={group} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Thinking section */}
          {thinkingEvents.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border/50">
              {thinkingEvents.map((te) => (
                <ThinkingItem key={te.id} evt={te} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
