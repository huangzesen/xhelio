import { useEffect, useState } from 'react';
import * as api from '../../api/client';
import { ClipboardList, ChevronRight, ChevronDown } from 'lucide-react';

interface Props {
  sessionId: string;
  isStreaming: boolean;
}

export function PlanStatus({ sessionId, isStreaming }: Props) {
  const [planStatus, setPlanStatus] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const { plan_status } = await api.getPlanStatus(sessionId);
        if (!cancelled) setPlanStatus(plan_status);
      } catch {
        // ignore
      }
    };
    poll();
    const interval = setInterval(poll, isStreaming ? 2000 : 5000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [sessionId, isStreaming]);

  if (!planStatus) return null;

  // Split into summary line (e.g. "Plan: 7 steps") and the rest
  const lines = planStatus.split('\n');
  const summaryLine = lines[0] || 'Plan';
  const detailLines = lines.slice(1).join('\n').trim();

  // Extract progress line (e.g. "Progress: 5/7 completed") from the end
  const progressMatch = planStatus.match(/Progress:\s*\S+/);
  const progressText = progressMatch ? progressMatch[0] : null;

  return (
    <div className="bg-status-info-bg border border-status-info-border rounded-lg px-3 py-2 text-xs space-y-1">
      {/* Header row: icon + title + summary + chevron */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left text-status-info-text font-medium hover:opacity-80 transition-opacity"
      >
        <ClipboardList size={14} className="shrink-0" />
        <span className="truncate">Plan Active</span>
        {progressText && !expanded && (
          <span className="ml-auto text-[10px] opacity-70 shrink-0">{progressText}</span>
        )}
        {expanded
          ? <ChevronDown size={14} className="shrink-0 ml-auto" />
          : <ChevronRight size={14} className="shrink-0 ml-auto" />
        }
      </button>

      {/* Summary line always visible */}
      <div className="text-status-info-text opacity-80 text-[11px]">{summaryLine}</div>

      {/* Collapsible detail */}
      {expanded && detailLines && (
        <div className="text-status-info-text opacity-80 whitespace-pre-wrap border-t border-status-info-border/40 pt-1 mt-1">
          {detailLines}
        </div>
      )}

    </div>
  );
}
