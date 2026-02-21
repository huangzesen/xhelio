import { useState, useCallback } from 'react';

interface Props {
  thumbnailUrl: string;
  onLoadInteractive: () => void;
  isLoading?: boolean;
}

/**
 * Shows a pre-rendered PNG thumbnail of a plot with an overlay button
 * to load the full interactive Plotly figure. Used during session resume
 * to provide an instant preview while the heavy figure data loads lazily.
 */
export function PlotThumbnail({ thumbnailUrl, onLoadInteractive, isLoading = false }: Props) {
  const [imgError, setImgError] = useState(false);

  const handleClick = useCallback(() => {
    if (!isLoading) onLoadInteractive();
  }, [isLoading, onLoadInteractive]);

  if (imgError) return null;

  return (
    <div
      className="max-w-full rounded-lg border border-border overflow-hidden cursor-pointer transition-all hover:shadow-md relative group"
      onClick={handleClick}
      role="button"
      tabIndex={0}
      aria-label="Plot preview. Click to load interactive plot."
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleClick(); }}
    >
      {/* Title bar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-panel">
        <span className="text-xs text-text-muted font-medium">Plot Preview</span>
        <span className="text-[10px] font-medium text-amber-600 bg-amber-500/10 px-1.5 py-0.5 rounded">
          Thumbnail
        </span>
      </div>

      {/* Thumbnail image */}
      <div className="relative">
        <img
          src={thumbnailUrl}
          alt="Plot thumbnail"
          className="w-full h-auto"
          loading="eager"
          onError={() => setImgError(true)}
        />

        {/* Overlay */}
        <div className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/20 transition-colors">
          {isLoading ? (
            <div className="bg-panel/90 backdrop-blur-sm rounded-lg px-4 py-2 flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-text-secondary">Loading interactive plot...</span>
            </div>
          ) : (
            <div className="bg-panel/90 backdrop-blur-sm rounded-lg px-4 py-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <span className="text-sm text-text-secondary">Click to load interactive plot</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
