import { useEffect, useState, useCallback } from 'react';
import { X, Download, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Plot from 'react-plotly.js';
import { applyDarkTheme, getPlotTitle } from './plotUtils';
import type { PlotlyFigure } from '../../api/types';

interface Props {
  figures: PlotlyFigure[];
  currentIndex: number;
  onNavigate: (index: number) => void;
  onClose: () => void;
}

const TOP_BAR_HEIGHT = 48;

export function PlotFullscreen({ figures, currentIndex, onNavigate, onClose }: Props) {
  const [plotHeight, setPlotHeight] = useState(window.innerHeight - TOP_BAR_HEIGHT);

  const figure = figures[currentIndex];
  const isDark = document.documentElement.classList.contains('dark');
  const title = getPlotTitle(figure);
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex < figures.length - 1;
  const showNav = figures.length > 1;

  const updateHeight = useCallback(() => {
    setPlotHeight(window.innerHeight - TOP_BAR_HEIGHT);
  }, []);

  useEffect(() => {
    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, [updateHeight]);

  // Keyboard: Escape to close, ArrowLeft/Right to navigate
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowLeft' && hasPrev) onNavigate(currentIndex - 1);
      else if (e.key === 'ArrowRight' && hasNext) onNavigate(currentIndex + 1);
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose, onNavigate, currentIndex, hasPrev, hasNext]);

  // Lock body scroll
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = ''; };
  }, []);

  const handleDownload = async (format: 'png' | 'svg') => {
    const plotEl = document.querySelector('.plot-fullscreen .js-plotly-plot') as HTMLElement | null;
    if (!plotEl) return;
    try {
      const Plotly = (window as unknown as Record<string, unknown>).Plotly as {
        toImage: (el: HTMLElement, opts: Record<string, unknown>) => Promise<string>;
      } | undefined;
      if (!Plotly) return;
      const result = await Plotly.toImage(plotEl, {
        format,
        width: 1600,
        height: 900,
      });
      const link = document.createElement('a');
      link.href = result;
      link.download = `xhelio-plot.${format}`;
      link.click();
    } catch {
      // ignore
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-surface/95 backdrop-blur-sm">
      {/* Top bar */}
      <div
        className="flex items-center justify-between px-4 border-b border-border shrink-0"
        style={{ height: TOP_BAR_HEIGHT }}
      >
        {/* Left: title + counter */}
        <div className="flex items-center gap-3 min-w-0 mr-4">
          <span className="text-sm font-medium text-text truncate">
            {title || 'Plot'}
          </span>
          {showNav && (
            <span className="text-xs text-text-muted whitespace-nowrap">
              {currentIndex + 1} / {figures.length}
            </span>
          )}
        </div>

        {/* Right: download + close */}
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={() => handleDownload('png')}>
            <Download size={14} />
            PNG
          </Button>
          <Button variant="ghost" size="sm" onClick={() => handleDownload('svg')}>
            <Download size={14} />
            SVG
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close fullscreen">
            <X size={18} />
          </Button>
        </div>
      </div>

      {/* Plot area with navigation arrows */}
      <div className="relative flex-1 min-h-0">
        {/* Left arrow */}
        {showNav && hasPrev && (
          <button
            onClick={() => onNavigate(currentIndex - 1)}
            className="absolute left-3 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full
              bg-surface/80 border border-border text-text-muted hover:text-text
              hover:bg-hover-bg transition-colors shadow-md"
            aria-label="Previous figure"
          >
            <ChevronLeft size={24} />
          </button>
        )}

        {/* Right arrow */}
        {showNav && hasNext && (
          <button
            onClick={() => onNavigate(currentIndex + 1)}
            className="absolute right-3 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full
              bg-surface/80 border border-border text-text-muted hover:text-text
              hover:bg-hover-bg transition-colors shadow-md"
            aria-label="Next figure"
          >
            <ChevronRight size={24} />
          </button>
        )}

        {/* Plot container — fills available space */}
        <div className="plot-fullscreen h-full w-full">
          <Plot
            data={figure.data}
            layout={{
              ...figure.layout,
              width: undefined,
              ...(isDark ? applyDarkTheme(figure.layout) : {}),
              autosize: true,
              height: plotHeight,
            }}
            config={{
              responsive: true,
              displayModeBar: true,
              displaylogo: false,
            }}
            useResizeHandler
            className="w-full h-full"
          />
        </div>
      </div>
    </div>
  );
}
