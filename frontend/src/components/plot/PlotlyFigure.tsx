import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { applyDarkTheme, getPlotTitle, countPanels } from './plotUtils';
import type { PlotlyFigure as PlotlyFigureType } from '../../api/types';

interface Props {
  figure: PlotlyFigureType;
  isPrimary?: boolean;
  onOpenFullscreen: () => void;
}

export function PlotlyFigure({ figure, isPrimary = false, onOpenFullscreen }: Props) {
  const isDark = document.documentElement.classList.contains('dark');
  const title = getPlotTitle(figure);
  const panelCount = countPanels(figure.layout);

  const height = useMemo(() => {
    if (panelCount <= 1) return 300;
    if (panelCount === 2) return 450;
    return Math.min(200 * panelCount, 700);
  }, [panelCount]);

  return (
    <div
      className={`max-w-full rounded-lg border overflow-hidden cursor-pointer transition-all hover:shadow-md
        ${isPrimary
          ? 'border-l-3 border-l-primary border-border'
          : 'border-border opacity-85 hover:opacity-100'}`}
      onClick={onOpenFullscreen}
      role="button"
      tabIndex={0}
      aria-label={`Plot: ${title || 'Untitled'}. Click to expand.`}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onOpenFullscreen(); }}
    >
      {/* Title bar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-panel">
        <span className="text-xs text-text-muted font-medium truncate">
          {title || 'Plot'}
        </span>
        {isPrimary && (
          <span className="text-[10px] font-medium text-primary bg-primary/10 px-1.5 py-0.5 rounded">
            Latest
          </span>
        )}
      </div>

      {/* Plot */}
      <div style={{ pointerEvents: 'none' }}>
        <Plot
          data={figure.data}
          layout={{
            ...figure.layout,
            width: undefined,
            ...(isDark ? applyDarkTheme(figure.layout) : {}),
            autosize: true,
            height,
          }}
          config={{
            responsive: true,
            displayModeBar: false,
            staticPlot: true,
          }}
          useResizeHandler
          className="w-full"
        />
      </div>
    </div>
  );
}
