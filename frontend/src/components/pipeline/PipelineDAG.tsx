import Plot from 'react-plotly.js';
import type { PlotlyFigure } from '../../api/types';

interface Props {
  figure: PlotlyFigure;
  onNodeClick?: (opId: string) => void;
  height?: number;
}

export function PipelineDAG({ figure, onNodeClick, height: heightProp }: Props) {
  const nodeCount = figure.data?.length ?? 0;
  const computedHeight = heightProp ?? (nodeCount <= 4 ? 400 : Math.min(400 + (nodeCount - 4) * 40, 600));

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
    <div className="w-full overflow-hidden rounded-lg" style={{ minHeight: computedHeight }}>
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
          height: computedHeight,
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
