import { useEffect, useState } from 'react';
import * as api from '../../api/client';
import type { PlanData } from '../../api/client';
import { useChatStore } from '../../stores/chatStore';
import { ClipboardList, ChevronRight, ChevronDown } from 'lucide-react';

interface Props {
  sessionId: string;
  isStreaming: boolean;
}

const STATUS_ICON: Record<string, string> = {
  pending: '○',
  in_progress: '◉',
  completed: '✓',
  failed: '✗',
  skipped: '–',
};

const STATUS_COLOR: Record<string, string> = {
  pending: 'opacity-50',
  in_progress: 'text-blue-400',
  completed: 'text-green-400',
  failed: 'text-red-400',
  skipped: 'opacity-40',
};

function StepRow({ step, index }: { step: api.PlanStep; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const icon = STATUS_ICON[step.status] || '~';
  const color = STATUS_COLOR[step.status] || '';
  const missionTag = step.mission ? `[${step.mission}] ` : '';

  return (
    <div className="text-[11px]">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-start gap-1.5 w-full text-left hover:opacity-80 transition-opacity"
      >
        <span className="shrink-0 w-3 text-center pt-px">
          {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        </span>
        <span className={`shrink-0 ${color}`}>{icon}</span>
        <span className="text-status-info-text">
          <span className="opacity-50">{index + 1}. </span>
          {missionTag && <span className="font-medium">{missionTag}</span>}
          {step.title}
        </span>
      </button>

      {expanded && (
        <div className="ml-7 mt-1 mb-1 pl-2 border-l border-status-info-border/40 text-[10px] text-status-info-text opacity-70 space-y-1">
          <div className="whitespace-pre-wrap">{step.details}</div>
          {step.candidate_datasets && step.candidate_datasets.length > 0 && (
            <div className="opacity-60">
              Datasets: {step.candidate_datasets.join(', ')}
            </div>
          )}
          {step.error && (
            <div className="text-red-400">Error: {step.error}</div>
          )}
        </div>
      )}
    </div>
  );
}

export function PlanStatus({ sessionId, isStreaming }: Props) {
  // SSE-driven plan data (instant)
  const ssePlan = useChatStore((s) => s.planData);
  // Polling fallback (for session resume / initial load)
  const [polledPlan, setPolledPlan] = useState<PlanData | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const { plan: data } = await api.getPlanStatus(sessionId);
        if (!cancelled) setPolledPlan(data);
      } catch {
        // ignore
      }
    };
    poll();
    // Poll less frequently when SSE is active — it's just a fallback
    const interval = setInterval(poll, isStreaming ? 15000 : 5000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [sessionId, isStreaming]);

  // SSE data takes priority over polled data
  const plan = ssePlan ?? polledPlan;

  if (!plan) return null;

  // Group steps by round
  const rounds = new Map<number, { step: api.PlanStep; globalIndex: number }[]>();
  plan.steps.forEach((step, i) => {
    const r = step.round || 0;
    if (!rounds.has(r)) rounds.set(r, []);
    rounds.get(r)!.push({ step, globalIndex: i });
  });
  const hasRounds = rounds.size > 1 || (rounds.size === 1 && !rounds.has(0));

  return (
    <div className="bg-status-info-bg border border-status-info-border rounded-lg px-3 py-2 text-xs space-y-1">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left text-status-info-text font-medium hover:opacity-80 transition-opacity"
      >
        <ClipboardList size={14} className="shrink-0" />
        <span className="truncate">Plan Active</span>
        <span className="ml-auto text-[10px] opacity-70 shrink-0">{plan.progress}</span>
        {expanded
          ? <ChevronDown size={14} className="shrink-0" />
          : <ChevronRight size={14} className="shrink-0" />
        }
      </button>

      {/* Summary */}
      <div className="text-status-info-text opacity-80 text-[11px]">
        Plan: {plan.total_steps} steps
      </div>

      {/* Steps */}
      {expanded && (
        <div className="border-t border-status-info-border/40 pt-1 mt-1 space-y-1">
          {hasRounds
            ? [...rounds.entries()].sort(([a], [b]) => a - b).map(([roundNum, entries]) => (
                <div key={roundNum}>
                  {roundNum > 0 && (
                    <div className="text-[10px] text-status-info-text opacity-50 font-medium mt-1">
                      Round {roundNum}
                    </div>
                  )}
                  {entries.map(({ step, globalIndex }) => (
                    <StepRow key={globalIndex} step={step} index={globalIndex} />
                  ))}
                </div>
              ))
            : plan.steps.map((step, i) => (
                <StepRow key={i} step={step} index={i} />
              ))
          }
        </div>
      )}
    </div>
  );
}
