import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { MemoryEntry } from '../../api/types';

const typeColorMap: Record<string, string> = {
  preference: '#3b82f6',
  pitfall: '#ef4444',
  summary: '#22c55e',
  reflection: '#a855f7',
};

interface Props {
  memories: MemoryEntry[];
}

export function MemoryTimeline({ memories }: Props) {
  const isDark = document.documentElement.classList.contains('dark');

  const { data, layout } = useMemo(() => {
    if (memories.length < 3) return { data: [], layout: {} };

    // Group by type for separate traces
    const byType: Record<string, MemoryEntry[]> = {};
    for (const m of memories) {
      if (!byType[m.type]) byType[m.type] = [];
      byType[m.type].push(m);
    }

    const traces = Object.entries(byType).map(([type, entries]) => ({
      x: entries.map((m) => m.created_at),
      y: entries.map(() => type.charAt(0).toUpperCase() + type.slice(1)),
      text: entries.map((m) => m.content.slice(0, 80) + (m.content.length > 80 ? '...' : '')),
      mode: 'markers' as const,
      type: 'scatter' as const,
      name: type.charAt(0).toUpperCase() + type.slice(1),
      marker: {
        color: typeColorMap[type] ?? '#6b7280',
        size: entries.map((m) => Math.min(Math.max(m.access_count * 2 + 6, 6), 20)),
        opacity: 0.8,
      },
      hovertemplate: '<b>%{y}</b><br>%{x|%b %d, %Y}<br>%{text}<extra></extra>',
    }));

    const plotLayout: Partial<Plotly.Layout> = {
      height: 180,
      margin: { l: 80, r: 20, t: 10, b: 30 },
      showlegend: false,
      yaxis: {
        categoryorder: 'array' as const,
        categoryarray: ['Reflection', 'Summary', 'Pitfall', 'Preference'],
        tickfont: { size: 11 },
      },
      xaxis: {
        type: 'date',
        tickfont: { size: 10 },
      },
      ...(isDark
        ? {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#94a3b8' },
            yaxis: {
              categoryorder: 'array' as const,
              categoryarray: ['Reflection', 'Summary', 'Pitfall', 'Preference'],
              tickfont: { size: 11 },
              gridcolor: '#1e293b',
            },
            xaxis: {
              type: 'date' as const,
              tickfont: { size: 10 },
              gridcolor: '#1e293b',
            },
          }
        : {}),
    };

    return { data: traces, layout: plotLayout };
  }, [memories, isDark]);

  if (memories.length < 3) {
    return (
      <div className="border border-border rounded-lg p-4 text-center text-sm text-text-muted mt-4">
        Not enough data for timeline (need at least 3 memories).
      </div>
    );
  }

  return (
    <div className="border border-border rounded-lg overflow-hidden mt-4">
      <div className="px-3 py-1.5 bg-surface-elevated border-b border-border">
        <span className="text-xs text-text-muted font-medium">Memory Timeline</span>
      </div>
      <div style={{ pointerEvents: 'auto' }}>
        <Plot
          data={data}
          layout={layout}
          config={{
            responsive: true,
            displayModeBar: false,
          }}
          useResizeHandler
          className="w-full"
        />
      </div>
    </div>
  );
}
