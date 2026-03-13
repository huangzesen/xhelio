import { useState } from 'react';
import { CheckCircle2, Circle, Clock, ChevronDown, ChevronRight, ListChecks } from 'lucide-react';
import type { PlanData, PlanStep } from '../../api/types';

interface PlanStatusProps {
  plan: PlanData;
}

const STATUS_ICONS: Record<string, typeof Circle> = {
  pending: Circle,
  in_progress: Clock,
  completed: CheckCircle2,
};

const STATUS_COLORS: Record<string, string> = {
  pending: 'text-text-muted',
  in_progress: 'text-yellow-400',
  completed: 'text-green-400',
  skipped: 'text-text-muted/50',
  failed: 'text-red-400',
};

function StepRow({ step, index }: { step: PlanStep; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const Icon = STATUS_ICONS[step.status] || Circle;
  const color = STATUS_COLORS[step.status] || 'text-text-muted';
  const hasDetails = !!(step.details || step.note || step.mission);

  return (
    <div className="group">
      <div
        className={`flex items-start gap-2 py-1 px-1 rounded text-xs ${hasDetails ? 'cursor-pointer hover:bg-surface-elevated/50' : ''}`}
        onClick={() => hasDetails && setExpanded(!expanded)}
      >
        <Icon size={14} className={`mt-0.5 shrink-0 ${color}`} />
        <span className={`flex-1 ${step.status === 'completed' ? 'line-through text-text-muted' : 'text-text'}`}>
          <span className="text-text-muted mr-1">{index + 1}.</span>
          {step.title}
        </span>
        {hasDetails && (
          expanded
            ? <ChevronDown size={12} className="mt-0.5 text-text-muted shrink-0" />
            : <ChevronRight size={12} className="mt-0.5 text-text-muted shrink-0 opacity-0 group-hover:opacity-100" />
        )}
      </div>
      {expanded && hasDetails && (
        <div className="ml-6 pl-2 border-l border-border text-xs text-text-muted space-y-0.5 pb-1">
          {step.details && <div>{step.details}</div>}
          {step.mission && <div className="text-accent/80">Mission: {step.mission}</div>}
          {step.note && <div className="italic">Note: {step.note}</div>}
        </div>
      )}
    </div>
  );
}

export function PlanStatus({ plan }: PlanStatusProps) {
  const [collapsed, setCollapsed] = useState(false);

  const done = plan.steps.filter((s) => s.status === 'completed').length;
  const total = plan.total_steps;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;

  return (
    <div className="rounded-lg border border-border bg-surface p-2 text-xs">
      {/* Header */}
      <div
        className="flex items-center gap-2 cursor-pointer select-none"
        onClick={() => setCollapsed(!collapsed)}
      >
        <ListChecks size={14} className="text-accent shrink-0" />
        <span className="font-medium text-text flex-1 truncate">
          {plan.summary || 'Plan'}
        </span>
        <span className="text-text-muted shrink-0">{plan.progress}</span>
        {collapsed
          ? <ChevronRight size={14} className="text-text-muted shrink-0" />
          : <ChevronDown size={14} className="text-text-muted shrink-0" />}
      </div>

      {/* Progress bar */}
      <div className="mt-1.5 h-1 rounded-full bg-surface-elevated overflow-hidden">
        <div
          className="h-full rounded-full bg-accent transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Steps list */}
      {!collapsed && (
        <div className="mt-2 space-y-0">
          {plan.steps.map((step, i) => (
            <StepRow key={i} step={step} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
