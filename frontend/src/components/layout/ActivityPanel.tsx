import { useEffect, useRef, useState, useMemo } from 'react';
import { Activity, Brain, Cpu, Terminal, Search, X, ChevronDown, ChevronRight } from 'lucide-react';
import type { ChatMessage, ToolEvent, LogLine, MemoryEvent } from '../../api/types';
import { groupEventsByTurn, type ThinkingEvent } from '../../utils/groupEventsByTurn';
import { TokenUsage } from '../common/TokenUsage';
import { PlanStatus } from '../chat/PlanStatus';
import { useSessionStore } from '../../stores/sessionStore';
import * as api from '../../api/client';

interface ModelTiers {
  smart: string;
  subAgent: string;
  inline: string;
}

interface Props {
  model: string;
  tokenUsage: Record<string, number>;
  messages: ChatMessage[];
  toolEvents: ToolEvent[];
  logLines: LogLine[];
  memoryEvents: MemoryEvent[];
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
  // Rendering / visualization — red (warm)
  if (name === 'render_plotly_json' || name === 'manage_plot')
    return 'text-badge-red-text';
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
  if (name === 'delegate_to_visualization') return 'Viz';
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

/** Truncate user message text for display as a group header. */
function truncateText(text: string, maxLen = 60): string {
  const cleaned = text.replace(/\s+/g, ' ').trim();
  return cleaned.length > maxLen ? cleaned.slice(0, maxLen) + '...' : cleaned;
}

/** Activity tab content — events grouped by interaction turn. */
function ActivityTab({
  messages,
  toolEvents,
  memoryEvents,
}: {
  messages: ChatMessage[];
  toolEvents: ToolEvent[];
  memoryEvents: MemoryEvent[];
}) {
  const turnGroups = useMemo(
    () => groupEventsByTurn(messages, toolEvents, memoryEvents),
    [messages, toolEvents, memoryEvents],
  );

  // Filter to only turns that have events
  const nonEmptyGroups = turnGroups.filter(
    (g) => g.toolEvents.length > 0 || g.memoryEvents.length > 0 || g.thinkingEvents.length > 0,
  );

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

        // Merge and sort events chronologically within this turn
        const merged = [
          ...group.toolEvents.map((e) => ({ kind: 'tool' as const, ts: e.timestamp, evt: e })),
          ...group.memoryEvents.map((e) => ({ kind: 'memory' as const, ts: e.timestamp, evt: e })),
          ...group.thinkingEvents.map((e) => ({ kind: 'thinking' as const, ts: e.timestamp, evt: e })),
        ].sort((a, b) => a.ts - b.ts);

        return (
          <details
            key={group.userMessage.id}
            open={isLast}
            className="group/turn"
          >
            <summary className="flex items-center gap-1.5 text-xs cursor-pointer select-none hover:text-text transition-colors text-text-muted">
              <ChevronDown
                size={12}
                className="shrink-0 transition-transform group-open/turn:rotate-0 -rotate-90"
              />
              <span className="font-medium text-text truncate">
                {truncateText(group.userMessage.content)}
              </span>
              <span className="shrink-0 opacity-70">
                ({stepCount} {stepCount === 1 ? 'step' : 'steps'})
              </span>
            </summary>
            <div className="ml-3 mt-1 space-y-1 border-l-2 border-border pl-2">
              {merged.map((item) =>
                item.kind === 'thinking' ? (
                  <ThinkingRow key={(item.evt as ThinkingEvent).id} evt={item.evt as ThinkingEvent} />
                ) : item.kind === 'memory' ? (
                  <MemoryRow key={(item.evt as MemoryEvent).id} evt={item.evt as MemoryEvent} />
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

export function ActivityPanel({ model, tokenUsage, messages, toolEvents, logLines, memoryEvents, isStreaming }: Props) {
  const { activeSessionId } = useSessionStore();
  const [tiers, setTiers] = useState<ModelTiers | null>(null);
  const [activeTab, setActiveTab] = useState<'activity' | 'console'>('activity');
  const [consoleFilter, setConsoleFilter] = useState('');
  const consoleEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.getConfig().then((cfg) => {
      const provider = (cfg.llm_provider as string) || 'gemini';
      const providers = (cfg.providers ?? {}) as Record<string, Record<string, unknown>>;
      const p = providers[provider] ?? {};
      setTiers({
        smart: (p.model as string) || '',
        subAgent: (p.sub_agent_model as string) || '',
        inline: (p.inline_model as string) || '',
      });
    }).catch(() => {});
  }, []);

  const filteredLogLines = useMemo(() => {
    if (!consoleFilter.trim()) return logLines;
    const needle = consoleFilter.toLowerCase();
    return logLines.filter((l) => l.text.toLowerCase().includes(needle));
  }, [logLines, consoleFilter]);

  // Auto-scroll console to bottom when new log lines arrive
  useEffect(() => {
    if (activeTab === 'console' && consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logLines.length, activeTab]);

  return (
    <div className="flex flex-col h-full bg-panel border-l border-border">
      {/* Session info */}
      <div className="p-3 border-b border-border">
        <div className="flex items-center gap-2 text-sm font-medium text-text mb-2">
          <Cpu size={14} />
          Session Info
        </div>
        {(model || tiers) && (
          <div className="text-xs text-text-muted space-y-0.5">
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
            {tiers?.inline && (
              <div className="flex justify-between">
                <span>Inline</span>
                <span className="font-mono text-text">{tiers.inline}</span>
              </div>
            )}
          </div>
        )}
        <div className="mt-2">
          <TokenUsage usage={tokenUsage} />
        </div>

        {/* Plan status */}
        {activeSessionId && (
          <div className="mt-2">
            <PlanStatus sessionId={activeSessionId} isStreaming={isStreaming} />
          </div>
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
          {isStreaming && activeTab === 'console' && (
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          )}
        </button>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-3">
        {activeTab === 'activity' ? (
          <ActivityTab
            messages={messages}
            toolEvents={toolEvents}
            memoryEvents={memoryEvents}
          />
        ) : (
          <>
            {/* Console search */}
            <div className="relative mb-2">
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

                return (
                  <div
                    key={line.id}
                    className="text-xs font-mono px-1.5 py-0.5 break-all flex items-start gap-1.5"
                  >
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
                      {rest}
                    </span>
                  </div>
                );
              })}
              <div ref={consoleEndRef} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
