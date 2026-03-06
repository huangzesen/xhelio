import Plot from 'react-plotly.js';
import type { PlotlyFigure } from '../../api/types';

interface Props {
  figure: PlotlyFigure;
  onNodeClick?: (opId: string) => void;
  height?: number;
}

export function PipelineDAG({ figure, onNodeClick, height: heightProp }: Props) {
  // Use server-computed height from figure.layout, with prop override and fallback
  const figHeight = heightProp ?? (figure.layout as Record<string, unknown>)?.height as number | undefined ?? 450;

  const handleClick = (event: Plotly.PlotMouseEvent) => {
    if (!onNodeClick || !event.points || event.points.length === 0) return;
    const point = event.points[0];
    const customdata = (point as Record<string, unknown>).customdata;
    if (typeof customdata === 'string') {
      // customdata format: "op_001 | tool | ..." — extract op_id
      const opId = customdata.split('|')[0]?.trim();
      if (opId) onNodeClick(opId);
    }
  };

  return (
    <div className="w-full overflow-hidden rounded-lg" style={{ minHeight: figHeight }}>
      <Plot
        data={figure.data}
        layout={{
          ...figure.layout,
          width: undefined,
          // Always use white background — the DAG has hardcoded light-theme
          // colors (node text, edge labels, arrows) that are unreadable on dark.
          paper_bgcolor: '#ffffff',
          plot_bgcolor: '#ffffff',
          autosize: true,
          height: figHeight,
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          displaylogo: false,
        }}
        useResizeHandler
        className="w-full"
        onClick={handleClick}
      />
    </div>
  );
}
