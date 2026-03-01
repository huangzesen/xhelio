import { useMemo, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { applyDarkTheme, getPlotTitle, countPanels } from './plotUtils';
import type { PlotlyFigure as PlotlyFigureType } from '../../api/types';
import { motion } from 'framer-motion';
import { Download, Maximize2 } from 'lucide-react';

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

  const handleDownloadPng = useCallback(() => {
    const plotly = (window as unknown as { Plotly?: typeof import('plotly.js-dist-min') }).Plotly;
    if (plotly) {
      const div = document.getElementById(`plot-${figure.data[0]?.uid || 'default'}`);
      if (div) {
        plotly.downloadImage(div, { format: 'png', width: 1200, height: 800, filename: 'plot' });
      }
    }
  }, [figure.data]);

  const handleDownloadSvg = useCallback(() => {
    const plotly = (window as unknown as { Plotly?: typeof import('plotly.js-dist-min') }).Plotly;
    if (plotly) {
      const div = document.getElementById(`plot-${figure.data[0]?.uid || 'default'}`);
      if (div) {
        plotly.downloadImage(div, { format: 'svg', width: 1200, height: 800, filename: 'plot' });
      }
    }
  }, [figure.data]);

  const plotId = `plot-${figure.data[0]?.uid || 'default'}`;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.25 }}
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
          <span className="text-[9px] font-medium text-primary bg-primary/10 px-1.5 py-0.5 rounded-full">
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
          divId={plotId}
        />
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-end gap-1 px-2 py-1.5 bg-panel border-t border-border">
        <button
          onClick={(e) => { e.stopPropagation(); handleDownloadPng(); }}
          className="flex items-center gap-1 px-2 py-1 text-[10px] text-text-muted hover:text-text hover:bg-hover-bg rounded transition-colors"
          title="Download PNG"
        >
          <Download size={12} />
          PNG
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); handleDownloadSvg(); }}
          className="flex items-center gap-1 px-2 py-1 text-[10px] text-text-muted hover:text-text hover:bg-hover-bg rounded transition-colors"
          title="Download SVG"
        >
          <Download size={12} />
          SVG
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onOpenFullscreen(); }}
          className="flex items-center gap-1 px-2 py-1 text-[10px] text-text-muted hover:text-text hover:bg-hover-bg rounded transition-colors"
          title="Fullscreen"
        >
          <Maximize2 size={12} />
          Full
        </button>
      </div>
    </motion.div>
  );
}
