import { useEffect, useRef, useState, useMemo } from 'react';
import { Activity, Brain, Cpu, Terminal, Search, X, ChevronDown, ChevronRight, Layers, Copy, Check, ArrowDown, Zap, Database, ClipboardList } from 'lucide-react';
import type { ChatMessage, ToolEvent, LogLine, MemoryEvent, CommentaryEvent } from '../../api/types';
import { groupEventsByRound, type ThinkingEvent } from '../../utils/groupEventsByTurn';
import { useChatStore, type RoundMarker } from '../../stores/chatStore';
import { TokenUsage } from '../common/TokenUsage';
import { DataStore } from '../common/DataStore';
import { PlanStatus } from '../chat/PlanStatus';
import { useSessionStore } from '../../stores/sessionStore';
import * as api from '../../api/client';

interface ModelTiers {
  smart: string;
  subAgent: string;
  insight: string;
  inline: string;
}

interface Props {
  model: string;
  tokenUsage: Record<string, number>;
  messages: ChatMessage[];
  toolEvents: ToolEvent[];
  logLines: LogLine[];
  memoryEvents: MemoryEvent[];
  commentaryEvents: CommentaryEvent[];
  roundMarkers: RoundMarker[];
  isStreaming: boolean;
}

/** Map tool names to color classes for the Activity tab. */
function toolColor(name: string): string {
  // Delegation / routing — purple
  if (name.startsWith('delegate_to'))
    return 'text-badge-purple-text';
  // Data fetching — teal
  if (name === 'fetch_data' || name === 'list_fetched_data' || name === 'preview_data')
    return 'text-badge-teal-text';
  // Computation / custom ops — orange
  if (name === 'custom_operation' || name === 'store_dataframe')
    return 'text-badge-orange-text';
  // Rendering / visualization — pink
  if (name === 'render_plotly_json' || name === 'manage_plot')
    return 'text-badge-pink-text';
  // Catalog / discovery — blue
  if (name === 'search_datasets' || name === 'browse_datasets' || name === 'list_parameters'
    || name === 'search_full_catalog' || name === 'get_data_availability' || name === 'get_dataset_docs')
    return 'text-badge-blue-text';
  // SPICE / ephemeris — green
  if (name.startsWith('get_spacecraft') || name.startsWith('compute_distance')
    || name === 'transform_coordinates' || name.startsWith('list_spice') || name === 'list_coordinate_frames')
    return 'text-badge-green-text';
  // Default — muted
  return 'text-text-muted';
}

/** Badge background for delegation tool calls. */
function delegationBadge(name: string): string | null {
  if (name === 'delegate_to_mission') return 'Mission';
  if (name === 'delegate_to_viz_plotly') return 'Viz [Plotly]';
  if (name === 'delegate_to_viz_mpl') return 'Viz [Mpl]';
  if (name === 'delegate_to_data_ops') return 'DataOps';
  if (name === 'delegate_to_data_extraction') return 'Extract';
  return null;
}

// Regex: match leading [BracketedTag] in log text
const LOG_TAG_RE = /^\[([^\]]+)\]\s*/;

/** Map log tag prefixes to colors for the Console tab. */
function logTagColor(tag: string): string {
  const t = tag.toLowerCase();
  if (t === 'catalog' || t === 'cdf' || t === 'search') return 'text-badge-blue-text';
  if (t === 'dataops' || t === 'export') return 'text-badge-orange-text';
  if (t === 'session' || t === 'sessiontitle') return 'text-badge-teal-text';
  if (t === 'tool' || t.startsWith('tool:')) return 'text-text-muted';
  if (t === 'planner' || t === 'plan') return 'text-badge-purple-text';
  if (t === 'memory' || t === 'memoryagent') return 'text-badge-green-text';
  if (t === 'followup' || t === 'inlinecomplete') return 'text-badge-gray-text';
  if (t === 'warning' || t === 'error' || t === 'critical') return 'text-status-error-text';
  return 'text-text-muted';
}

/** Level indicator dot color. */
function levelDotColor(level: string): string {
  if (level === 'error' || level === 'critical') return 'bg-status-error-text';
  if (level === 'warning') return 'bg-status-warning-text';
  if (level === 'info') return 'bg-status-info-text';
  return '';
}

/** Hook for managing collapsible section state in localStorage. */
function useCollapsibleState(sectionKey: string, defaultOpen: boolean): [boolean, (open: boolean) => void] {
  const [isOpen, setIsOpen] = useState<boolean>(() => {
    if (typeof window === 'undefined') return defaultOpen;
    const stored = localStorage.getItem(`xhelio-panel-${sectionKey}-open`);
    if (stored === null) return defaultOpen;
    return stored === 'true';
  });

  const setOpen = (open: boolean) => {
    setIsOpen(open);
    localStorage.setItem(`xhelio-panel-${sectionKey}-open`, String(open));
  };

  return [isOpen, setOpen];
}

/** Highlight search matches in text. */
function HighlightedText({ text, search }: { text: string; search: string }) {
  if (!search.trim()) return <>{text}</>;
  
  const parts: { text: string; highlight: boolean }[] = [];
  const lowerText = text.toLowerCase();
  const lowerSearch = search.toLowerCase();
  let lastIndex = 0;
  let idx = lowerText.indexOf(lowerSearch);
  
  while (idx !== -1) {
    if (idx > lastIndex) {
      parts.push({ text: text.slice(lastIndex, idx), highlight: false });
    }
    parts.push({ text: text.slice(idx, idx + search.length), highlight: true });
    lastIndex = idx + search.length;
    idx = lowerText.indexOf(lowerSearch, lastIndex);
  }
  
  if (lastIndex < text.length) {
    parts.push({ text: text.slice(lastIndex), highlight: false });
  }
  
  if (parts.length === 0) return <>{text}</>;
  
  return (
    <>
      {parts.map((part, i) => (
        part.highlight ? (
          <span key={i} className="bg-accent/30 text-accent rounded px-0.5">{part.text}</span>
        ) : (
          <span key={i}>{part.text}</span>
        )
      ))}
    </>
  );
}

/** Render a single tool/memory event row (shared by grouped and ungrouped views). */
function EventRow({ evt }: { evt: ToolEvent }) {
  return (
    <div className="text-xs font-mono bg-surface-elevated text-text-muted rounded px-2 py-1.5 min-w-0 overflow-hidden">
      {evt.type === 'call' ? (
        <div className="min-w-0">
          <span className="text-badge-blue-text">CALL</span>{' '}
          {delegationBadge(evt.tool_name) && (
            <span className="text-badge-purple-text opacity-75 mr-1">
              [{delegationBadge(evt.tool_name)}]
            </span>
          )}
          <span className={toolColor(evt.tool_name)}>{evt.tool_name}</span>
          {evt.tool_args && Object.keys(evt.tool_args).length > 0 && (
            <div className="text-text-muted mt-0.5 truncate opacity-70">
              {JSON.stringify(evt.tool_args)}
            </div>
          )}
        </div>
      ) : (
        <span>
          <span className={evt.status === 'success' ? 'text-status-success-text' : 'text-status-error-text'}>
            {evt.status === 'success' ? 'OK' : 'ERR'}
          </span>{' '}
          <span className={toolColor(evt.tool_name)}>{evt.tool_name}</span>
        </span>
      )}
    </div>
  );
}

function MemoryRow({ evt }: { evt: MemoryEvent }) {
  const summary = Object.entries(evt.actions)
    .map(([k, v]) => `+${v} ${k}`)
    .join(', ') || 'no changes';
  return (
    <div className="text-xs font-mono bg-surface-elevated rounded px-2 py-1.5">
      <span className="text-badge-green-text">MEMORY</span>{' '}
      <span className="text-text-muted">{summary}</span>
    </div>
  );
}

function ThinkingRow({ evt }: { evt: ThinkingEvent }) {
  const [expanded, setExpanded] = useState(false);
  // Extract first line or first ~80 chars as summary
  const firstLine = evt.content.split('\n')[0].replace(/^[#*\s]+/, '').trim();
  const summary = firstLine.length > 80 ? firstLine.slice(0, 80) + '...' : firstLine;
  return (
    <div className="text-xs font-mono bg-surface-elevated rounded px-2 py-1.5">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1 text-badge-purple-text hover:opacity-80 transition-colors w-full text-left"
      >
        {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        <Brain size={10} />
        <span className="font-medium">THINK</span>
        <span className="text-text-muted truncate ml-1">{summary}</span>
      </button>
      {expanded && (
        <div className="mt-1 ml-4 text-text-muted whitespace-pre-wrap break-words opacity-80 max-h-40 overflow-y-auto">
          {evt.content}
        </div>
      )}
    </div>
  );
}

function CommentaryRow({ evt }: { evt: CommentaryEvent }) {
  return (
    <div className="text-xs font-mono bg-surface-elevated rounded px-2 py-1.5 italic">
      <span className="text-text-muted opacity-70">{evt.agent}</span>{' '}
      <span className="text-text-muted">{evt.text}</span>
    </div>
  );
}

/** Truncate user message text for display as a group header. */
function truncateText(text: string, maxLen = 60): string {
  const cleaned = text.replace(/\s+/g, ' ').trim();
  return cleaned.length > maxLen ? cleaned.slice(0, maxLen) + '...' : cleaned;
}

function formatTokenK(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function formatDuration(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

function TimelineBar({ duration, maxDuration }: { duration: number; maxDuration: number }) {
  const widthPercent = maxDuration > 0 ? Math.max(5, (duration / maxDuration) * 100) : 0;
  
  return (
    <div className="relative h-1.5 bg-surface-elevated rounded-full overflow-hidden w-full mt-1">
      <div
        className="absolute left-0 top-0 h-full bg-gradient-to-r from-accent to-accent/60 rounded-full transition-all duration-300"
        style={{ width: `${widthPercent}%` }}
      />
    </div>
  );
}

/** Activity tab content — events grouped by orchestrator round. */
function ActivityTab({
  messages,
  toolEvents,
  memoryEvents,
  commentaryEvents,
  roundMarkers,
  roundTokenUsage,
}: {
  messages: ChatMessage[];
  toolEvents: ToolEvent[];
  memoryEvents: MemoryEvent[];
  commentaryEvents: CommentaryEvent[];
  roundMarkers: RoundMarker[];
  roundTokenUsage: Record<string, number> | null;
}) {
  const roundGroups = useMemo(
    () => groupEventsByRound(messages, toolEvents, memoryEvents, roundMarkers, commentaryEvents),
    [messages, toolEvents, memoryEvents, roundMarkers, commentaryEvents],
  );

  // Filter to only rounds that have events
  const nonEmptyGroups = useMemo(
    () => roundGroups.filter(
      (g) => g.toolEvents.length > 0 || g.memoryEvents.length > 0 || g.thinkingEvents.length > 0 || g.commentaryEvents.length > 0,
    ),
    [roundGroups],
  );

  // Calculate max duration for proportional bars
  const maxDuration = useMemo(() => {
    let max = 0;
    for (const group of nonEmptyGroups) {
      if (group.endTs !== Infinity) {
        const duration = group.endTs - group.startTs;
        if (duration > max) max = duration;
      }
    }
    return max;
  }, [nonEmptyGroups]);

  if (nonEmptyGroups.length === 0) {
    return (
      <div className="text-xs text-text-muted text-center py-4">
        No activity yet
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {nonEmptyGroups.map((group, gi) => {
        const stepCount = group.toolEvents.filter((e) => e.type === 'call').length;
        const isLast = gi === nonEmptyGroups.length - 1;
        const isActiveRound = group.endTs === Infinity;
        const duration = isActiveRound ? 0 : group.endTs - group.startTs;
        
        // For the active round, use live token usage from the store
        const tokens = isActiveRound ? (roundTokenUsage ?? undefined) : group.roundTokenUsage;
        // Use first user message as header, or fallback to round index
        const headerText = group.userMessages.length > 0
          ? truncateText(group.userMessages[0].content)
          : `Round ${group.roundIndex + 1}`;
        const hasMultipleUsers = group.userMessages.length > 1;

        // Merge and sort events chronologically within this round
        const merged = [
          ...group.toolEvents.map((e) => ({ kind: 'tool' as const, ts: e.timestamp, evt: e })),
          ...group.memoryEvents.map((e) => ({ kind: 'memory' as const, ts: e.timestamp, evt: e })),
          ...group.thinkingEvents.map((e) => ({ kind: 'thinking' as const, ts: e.timestamp, evt: e })),
          ...group.commentaryEvents.map((e) => ({ kind: 'commentary' as const, ts: e.timestamp, evt: e })),
        ].sort((a, b) => a.ts - b.ts);

        return (
          <details
            key={`round-${group.roundIndex}`}
            open={isLast}
            className="group/turn"
          >
            <summary className="flex items-center gap-1.5 text-xs cursor-pointer select-none hover:text-text transition-colors text-text-muted">
              <ChevronDown
                size={12}
                className="shrink-0 transition-transform group-open/turn:rotate-0 -rotate-90"
              />
              <span className="font-medium text-text truncate">
                {headerText}
              </span>
              {hasMultipleUsers && (
                <span className="shrink-0 text-badge-blue-text opacity-70">
                  +{group.userMessages.length - 1} msg
                </span>
              )}
              <span className="shrink-0 opacity-70">
                ({stepCount} {stepCount === 1 ? 'step' : 'steps'})
              </span>
              {tokens && (tokens.input_tokens > 0 || tokens.output_tokens > 0) && (
                <span className="shrink-0 opacity-50 font-mono">
                  ↑{formatTokenK(tokens.input_tokens || 0)} ↓{formatTokenK(tokens.output_tokens || 0)}
                </span>
              )}
            </summary>
            <div className="ml-3 mt-1 space-y-1 border-l-2 border-border pl-2">
              {/* Timeline bar */}
              {!isActiveRound && duration > 0 && (
                <div 
                  className="text-[10px] text-text-muted opacity-50 hover:opacity-100 transition-opacity cursor-default"
                  title={`Duration: ${formatDuration(duration)}`}
                >
                  <TimelineBar duration={duration} maxDuration={maxDuration} />
                </div>
              )}
              {/* Show additional user messages if batched */}
              {hasMultipleUsers && group.userMessages.slice(1).map((um) => (
                <div key={um.id} className="text-xs font-mono bg-surface-elevated rounded px-2 py-1.5 text-text-muted">
                  <span className="text-badge-blue-text">USER</span>{' '}
                  <span className="truncate">{truncateText(um.content, 80)}</span>
                </div>
              ))}
              {merged.map((item) =>
                item.kind === 'thinking' ? (
                  <ThinkingRow key={(item.evt as ThinkingEvent).id} evt={item.evt as ThinkingEvent} />
                ) : item.kind === 'memory' ? (
                  <MemoryRow key={(item.evt as MemoryEvent).id} evt={item.evt as MemoryEvent} />
                ) : item.kind === 'commentary' ? (
                  <CommentaryRow key={(item.evt as CommentaryEvent).id} evt={item.evt as CommentaryEvent} />
                ) : (
                  <EventRow key={(item.evt as ToolEvent).id} evt={item.evt as ToolEvent} />
                ),
              )}
            </div>
          </details>
        );
      })}
    </div>
  );
}

export function ActivityPanel({ model, tokenUsage, messages, toolEvents, logLines, memoryEvents, commentaryEvents, roundMarkers, isStreaming }: Props) {
  const { activeSessionId } = useSessionStore();
  const roundTokenUsageLive = useChatStore((s) => s.roundTokenUsage);
  const [tiers, setTiers] = useState<ModelTiers | null>(null);
  const [provider, setProvider] = useState<string | null>(null);
  const [vizBackend, setVizBackend] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'activity' | 'console'>('activity');
  const [consoleFilter, setConsoleFilter] = useState('');
  const [levelFilter, setLevelFilter] = useState<'all' | 'error' | 'warning' | 'info'>('all');
  const [expandedLogIds, setExpandedLogIds] = useState<Set<string>>(new Set());
  const [autoScroll, setAutoScroll] = useState(true);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const consoleEndRef = useRef<HTMLDivElement>(null);
  const consoleContainerRef = useRef<HTMLDivElement>(null);

  const [modelsOpen, setModelsOpen] = useCollapsibleState('models', false);
  const [tokenUsageOpen, setTokenUsageOpen] = useCollapsibleState('token-usage', true);
  const [dataStoreOpen, setDataStoreOpen] = useCollapsibleState('data-store', true);
  const [planStatusOpen, setPlanStatusOpen] = useCollapsibleState('plan-status', false);

  useEffect(() => {
    api.getConfig().then((cfg) => {
      const prov = (cfg.llm_provider as string) || 'gemini';
      setProvider(prov);
      setVizBackend((cfg.prefer_viz_backend as string) || 'matplotlib');
      const providers = (cfg.providers ?? {}) as Record<string, Record<string, unknown>>;
      const p = providers[prov] ?? {};
      setTiers({
        smart: (p.model as string) || '',
        subAgent: (p.sub_agent_model as string) || '',
        insight: (p.insight_model as string) || '',
        inline: (p.inline_model as string) || '',
      });
    }).catch(() => {});
  }, []);

  const filteredLogLines = useMemo(() => {
    let lines = logLines;
    if (levelFilter !== 'all') {
      lines = lines.filter((l) => levelFilter === 'error' ? (l.level === 'error' || l.level === 'critical') : l.level === levelFilter);
    }
    if (consoleFilter.trim()) {
      const needle = consoleFilter.toLowerCase();
      lines = lines.filter((l) => l.text.toLowerCase().includes(needle));
    }
    return lines;
  }, [logLines, consoleFilter, levelFilter]);

  const matchCount = useMemo(() => {
    if (!consoleFilter.trim()) return 0;
    return filteredLogLines.length;
  }, [filteredLogLines, consoleFilter]);

  const activityCount = useMemo(() => {
    return toolEvents.length + memoryEvents.length + commentaryEvents.length;
  }, [toolEvents, memoryEvents, commentaryEvents]);

  const consoleScrollToBottom = () => {
    setAutoScroll(true);
    consoleEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleConsoleScroll = () => {
    if (!consoleContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = consoleContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    if (isAtBottom && !autoScroll) {
      setAutoScroll(true);
    } else if (!isAtBottom && autoScroll) {
      setAutoScroll(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'console' && autoScroll && consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logLines.length, activeTab, autoScroll]);

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const getLogText = (line: LogLine): string => {
    return line.details ? `${line.text}\n${line.details}` : line.text;
  };

  return (
    <div data-testid="activity-panel" className="relative flex flex-col h-full bg-panel border-l border-border">
      {/* Session info */}
      <div className="p-3 border-b border-border max-h-[67vh] overflow-y-auto">
        <div className="flex items-center gap-2 text-sm font-medium text-text mb-2">
          <Cpu size={14} />
          Session Info
        </div>
        {activeSessionId && (
          <div className="flex justify-between text-xs text-text-muted gap-2">
            <span className="shrink-0">Session</span>
            <span className="font-mono text-text break-all text-right">
              {activeSessionId}
            </span>
          </div>
        )}
        {provider && (
          <div className="flex justify-between text-xs text-text-muted mt-0.5">
            <span>Provider</span>
            <span className="font-mono text-text capitalize">{provider}</span>
          </div>
        )}
        {vizBackend && (
          <div className="flex justify-between text-xs text-text-muted mt-0.5">
            <span>Viz Backend</span>
            <span className="font-mono text-text capitalize">{vizBackend}</span>
          </div>
        )}

        {/* Models collapsible section */}
        {(model || tiers) && (
          <details open={modelsOpen} className="mt-1">
            <summary 
              className="text-xs text-text-muted cursor-pointer flex items-center gap-1 list-none hover:text-text"
              onClick={(e) => { e.preventDefault(); setModelsOpen(!modelsOpen); }}
            >
              <ChevronDown size={12} className={`transition-transform ${modelsOpen ? '' : '-rotate-90'}`} />
              <Layers size={12} className="text-text-muted" />
              <span className="font-medium">Models</span>
            </summary>
            <div className="ml-4 space-y-0.5 mt-1">
              <div className="flex justify-between">
                <span>Main</span>
                <span className="font-mono text-text">{tiers?.smart || model}</span>
              </div>
              {tiers?.subAgent && tiers.subAgent !== tiers.smart && (
                <div className="flex justify-between">
                  <span>Sub-agent</span>
                  <span className="font-mono text-text">{tiers.subAgent}</span>
                </div>
              )}
              {tiers?.insight && tiers.insight !== tiers.subAgent && (
                <div className="flex justify-between">
                  <span>Insight</span>
                  <span className="font-mono text-text">{tiers.insight}</span>
                </div>
              )}
              {tiers?.inline && (
                <div className="flex justify-between">
                  <span>Inline</span>
                  <span className="font-mono text-text">{tiers.inline}</span>
                </div>
              )}
            </div>
          </details>
        )}

        {/* Token Usage collapsible section */}
        <details open={tokenUsageOpen} className="mt-2">
          <summary
            className="text-xs text-text-muted cursor-pointer flex items-center gap-1 list-none hover:text-text"
            onClick={(e) => { e.preventDefault(); setTokenUsageOpen(!tokenUsageOpen); }}
          >
            <ChevronDown size={12} className={`transition-transform ${tokenUsageOpen ? '' : '-rotate-90'}`} />
            <Zap size={12} />
            <span className="font-medium">Token Usage</span>
          </summary>
          <div className="ml-4 mt-1">
            <TokenUsage usage={tokenUsage} />
          </div>
        </details>

        {/* Data Store collapsible section */}
        {activeSessionId && (
          <details open={dataStoreOpen} className="mt-2">
            <summary
              className="text-xs text-text-muted cursor-pointer flex items-center gap-1 list-none hover:text-text"
              onClick={(e) => { e.preventDefault(); setDataStoreOpen(!dataStoreOpen); }}
            >
              <ChevronDown size={12} className={`transition-transform ${dataStoreOpen ? '' : '-rotate-90'}`} />
              <Database size={12} />
              <span className="font-medium">Data Store</span>
            </summary>
            <div className="ml-4 mt-1">
              <DataStore sessionId={activeSessionId} />
            </div>
          </details>
        )}

        {/* Plan Status collapsible section */}
        {activeSessionId && (
          <details open={planStatusOpen} className="mt-2">
            <summary
              className="text-xs text-text-muted cursor-pointer flex items-center gap-1 list-none hover:text-text"
              onClick={(e) => { e.preventDefault(); setPlanStatusOpen(!planStatusOpen); }}
            >
              <ChevronDown size={12} className={`transition-transform ${planStatusOpen ? '' : '-rotate-90'}`} />
              <ClipboardList size={12} />
              <span className="font-medium">Plan Status</span>
            </summary>
            <div className="ml-4 mt-1">
              <PlanStatus sessionId={activeSessionId} isStreaming={isStreaming} />
            </div>
          </details>
        )}
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border">
        <button
          onClick={() => setActiveTab('activity')}
          className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors ${
            activeTab === 'activity'
              ? 'text-text border-b-2 border-accent'
              : 'text-text-muted hover:text-text'
          }`}
        >
          <Activity size={12} />
          Activity
          <span className="ml-0.5 px-1.5 py-0.5 rounded-full text-[10px] bg-surface-elevated text-text-muted">
            {activityCount}
          </span>
          {isStreaming && activeTab === 'activity' && (
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          )}
        </button>
        <button
          onClick={() => setActiveTab('console')}
          className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors ${
            activeTab === 'console'
              ? 'text-text border-b-2 border-accent'
              : 'text-text-muted hover:text-text'
          }`}
        >
          <Terminal size={12} />
          Console
          <span className="ml-0.5 px-1.5 py-0.5 rounded-full text-[10px] bg-surface-elevated text-text-muted">
            {logLines.length}
          </span>
          {isStreaming && activeTab === 'console' && (
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          )}
        </button>
      </div>

      {/* Tab content */}
      <div 
        ref={consoleContainerRef}
        className="flex-1 overflow-y-auto p-3"
        onScroll={activeTab === 'console' ? handleConsoleScroll : undefined}
      >
        {activeTab === 'activity' ? (
          <ActivityTab
            messages={messages}
            toolEvents={toolEvents}
            memoryEvents={memoryEvents}
            commentaryEvents={commentaryEvents}
            roundMarkers={roundMarkers}
            roundTokenUsage={roundTokenUsageLive}
          />
        ) : (
          <>
            {/* Console search and filter */}
            <div className="relative mb-2 space-y-2">
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-text-muted" />
                  <input
                    type="text"
                    value={consoleFilter}
                    onChange={(e) => setConsoleFilter(e.target.value)}
                    placeholder="Filter logs..."
                    className="w-full pl-7 pr-7 py-1 rounded border border-border text-xs bg-input-bg text-text placeholder:text-text-muted focus:outline-none focus:border-primary"
                  />
                  {consoleFilter && (
                    <button
                      onClick={() => setConsoleFilter('')}
                      className="absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text"
                    >
                      <X size={12} />
                    </button>
                  )}
                </div>
                <select
                  value={levelFilter}
                  onChange={(e) => setLevelFilter(e.target.value as 'all' | 'error' | 'warning' | 'info')}
                  className="px-2 py-1 rounded border border-border text-xs bg-input-bg text-text focus:outline-none focus:border-primary"
                >
                  <option value="all">All</option>
                  <option value="error">Error</option>
                  <option value="warning">Warning</option>
                  <option value="info">Info</option>
                </select>
              </div>
              {consoleFilter && matchCount > 0 && (
                <div className="text-[10px] text-text-muted">
                  {matchCount} {matchCount === 1 ? 'match' : 'matches'}
                </div>
              )}
            </div>
            {filteredLogLines.length === 0 && (
              <div className="text-xs text-text-muted text-center py-4">
                {logLines.length === 0 ? 'No console output yet' : 'No matching logs'}
              </div>
            )}
            <div className="space-y-0.5">
              {filteredLogLines.map((line) => {
                const match = LOG_TAG_RE.exec(line.text);
                const tag = match?.[1] ?? null;
                const rest = match ? line.text.slice(match[0].length) : line.text;
                const dot = levelDotColor(line.level);
                const hasDetails = !!line.details && line.details !== line.text;
                const isExpanded = expandedLogIds.has(line.id);

                return (
                  <div key={line.id} className="group relative">
                    <div
                      className={`text-xs font-mono px-1.5 py-0.5 break-all flex items-start gap-1.5${hasDetails ? ' cursor-pointer hover:bg-bg-surface-secondary rounded' : ''}`}
                      onClick={hasDetails ? () => setExpandedLogIds((prev) => {
                        const next = new Set(prev);
                        if (next.has(line.id)) next.delete(line.id);
                        else next.add(line.id);
                        return next;
                      }) : undefined}
                    >
                      {hasDetails && (
                        isExpanded
                          ? <ChevronDown size={12} className="shrink-0 mt-0.5 text-text-muted" />
                          : <ChevronRight size={12} className="shrink-0 mt-0.5 text-text-muted" />
                      )}
                      {dot && (
                        <span className={`shrink-0 w-1.5 h-1.5 rounded-full mt-1 ${dot}`} />
                      )}
                      <span className={
                        line.level === 'error' || line.level === 'critical'
                          ? 'text-status-error-text'
                          : line.level === 'warning'
                            ? 'text-status-warning-text'
                            : 'text-badge-gray-text'
                      }>
                        {tag && (
                          <span className={`${logTagColor(tag)} opacity-80`}>[{tag}] </span>
                        )}
                        <HighlightedText text={rest} search={consoleFilter} />
                      </span>
                    </div>
                    {hasDetails && isExpanded && (
                      <pre className="text-xs font-mono text-text-muted px-6 py-1 whitespace-pre-wrap break-all">
                        {line.details}
                      </pre>
                    )}
                    {/* Copy button */}
                    <button
                      onClick={() => copyToClipboard(getLogText(line), line.id)}
                      className="absolute right-1 top-1 opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-bg-surface-secondary rounded"
                      title="Copy to clipboard"
                    >
                      {copiedId === line.id ? (
                        <Check size={12} className="text-status-success-text" />
                      ) : (
                        <Copy size={12} className="text-text-muted" />
                      )}
                    </button>
                  </div>
                );
              })}
              <div ref={consoleEndRef} />
            </div>
          </>
        )}
      </div>

      {/* Scroll to bottom button */}
      {activeTab === 'console' && !autoScroll && logLines.length > 0 && (
        <div className="absolute bottom-20 right-8">
          <button
            onClick={consoleScrollToBottom}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-surface-elevated border border-border text-xs text-text-muted hover:text-text hover:border-primary transition-colors shadow-lg"
          >
            <ArrowDown size={12} />
            Scroll to bottom
          </button>
        </div>
      )}
    </div>
  );
}
