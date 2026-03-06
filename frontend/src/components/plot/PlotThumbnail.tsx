import { useState, useCallback } from 'react';
import { ImageOff } from 'lucide-react';

interface Props {
  thumbnailUrl: string;
  onLoadInteractive: () => void;
  isLoading?: boolean;
}

/**
 * Shows a pre-rendered PNG thumbnail of a plot with an overlay button
 * to load the full interactive Plotly figure. Used during session resume
 * to provide an instant preview while the heavy figure data loads lazily.
 *
 * While the thumbnail image is loading (backend may be regenerating it),
 * a shimmer skeleton is shown. On permanent failure, a fallback card
 * with a "Load interactive plot" button is displayed.
 */
export function PlotThumbnail({ thumbnailUrl, onLoadInteractive, isLoading = false }: Props) {
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgError, setImgError] = useState(false);

  const handleClick = useCallback(() => {
    if (!isLoading) onLoadInteractive();
  }, [isLoading, onLoadInteractive]);

  // Error fallback — thumbnail generation failed (data missing, kaleido unavailable, etc.)
  if (imgError) {
    return (
      <div className="max-w-full rounded-lg border border-border overflow-hidden bg-panel">
        {/* Title bar */}
        <div className="flex items-center justify-between px-3 py-1.5 bg-panel border-b border-border">
          <span className="text-xs text-text-muted font-medium">Plot Preview</span>
          <span className="text-[10px] font-medium text-text-muted bg-surface px-1.5 py-0.5 rounded">
            Unavailable
          </span>
        </div>

        <div className="flex flex-col items-center justify-center py-8 px-4 gap-3">
          <ImageOff size={32} className="text-text-muted/50" />
          <p className="text-sm text-text-muted text-center">
            Preview unavailable
          </p>
          <button
            onClick={handleClick}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg
              bg-primary text-white text-sm font-medium hover:bg-primary-dark
              transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Loading...
              </>
            ) : (
              'Load interactive plot'
            )}
          </button>
        </div>
      </div>
    );
  }

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

      {/* Thumbnail image + skeleton */}
      <div className="relative">
        {/* Shimmer skeleton — visible until image loads */}
        {!imgLoaded && (
          <div className="w-full aspect-[2/1] bg-surface animate-pulse flex items-center justify-center">
            <div className="flex items-center gap-2 text-text-muted/50">
              <div className="w-4 h-4 border-2 border-text-muted/30 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm">Loading preview...</span>
            </div>
          </div>
        )}

        <img
          src={thumbnailUrl}
          alt="Plot thumbnail"
          className={`w-full h-auto ${imgLoaded ? '' : 'absolute top-0 left-0 opacity-0'}`}
          loading="eager"
          onLoad={() => setImgLoaded(true)}
          onError={() => setImgError(true)}
        />

        {/* Overlay — only shown when image is loaded */}
        {imgLoaded && (
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
        )}
      </div>
    </div>
  );
}
