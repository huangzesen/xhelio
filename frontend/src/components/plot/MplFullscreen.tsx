import { useEffect } from 'react';
import { X, Download, ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface MplImage {
  imageUrl: string;
  description: string;
  scriptUrl?: string;
}

interface Props {
  images: MplImage[];
  currentIndex: number;
  onNavigate: (index: number) => void;
  onClose: () => void;
}

const TOP_BAR_HEIGHT = 48;

export function MplFullscreen({ images, currentIndex, onNavigate, onClose }: Props) {
  const image = images[currentIndex];
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex < images.length - 1;
  const showNav = images.length > 1;

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
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, []);

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
            {image.description || 'Matplotlib Plot'}
          </span>
          {showNav && (
            <span className="text-xs text-text-muted whitespace-nowrap">
              {currentIndex + 1} / {images.length}
            </span>
          )}
        </div>

        {/* Right: actions + close */}
        <div className="flex items-center gap-1">
          {image.scriptUrl && (
            <Button variant="ghost" size="sm" asChild>
              <a href={image.scriptUrl} target="_blank" rel="noopener noreferrer">
                <ExternalLink size={14} className="mr-2" />
                View Script
              </a>
            </Button>
          )}
          <Button variant="ghost" size="sm" asChild>
            <a href={image.imageUrl} download={`plot-${currentIndex + 1}.png`}>
              <Download size={14} className="mr-2" />
              PNG
            </a>
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close fullscreen">
            <X size={18} />
          </Button>
        </div>
      </div>

      {/* Image area with navigation arrows */}
      <div className="relative flex-1 min-h-0 flex items-center justify-center p-4">
        {/* Left arrow */}
        {showNav && hasPrev && (
          <button
            onClick={() => onNavigate(currentIndex - 1)}
            className="absolute left-6 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full
              bg-surface/80 border border-border text-text-muted hover:text-text
              hover:bg-hover-bg transition-colors shadow-md"
            aria-label="Previous image"
          >
            <ChevronLeft size={24} />
          </button>
        )}

        {/* Right arrow */}
        {showNav && hasNext && (
          <button
            onClick={() => onNavigate(currentIndex + 1)}
            className="absolute right-6 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full
              bg-surface/80 border border-border text-text-muted hover:text-text
              hover:bg-hover-bg transition-colors shadow-md"
            aria-label="Next image"
          >
            <ChevronRight size={24} />
          </button>
        )}

        {/* Full-resolution image */}
        <img
          src={image.imageUrl}
          alt={image.description}
          className="max-w-full max-h-full object-contain shadow-2xl bg-white"
        />
      </div>
    </div>
  );
}
