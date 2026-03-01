import { Copy, Check, RefreshCw, RotateCcw } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';

interface Props {
  content: string;
  isError?: boolean;
  onRegenerate?: () => void;
}

export function MessageActions({ content, isError, onRegenerate }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore
    }
  };

  return (
    <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
      <Button variant="ghost" size="icon-sm" onClick={handleCopy} aria-label="Copy message">
        {copied ? <Check size={14} className="text-status-success-text" /> : <Copy size={14} />}
      </Button>
      {onRegenerate && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onRegenerate}
          aria-label={isError ? 'Retry' : 'Regenerate response'}
        >
          {isError ? (
            <RotateCcw size={14} className="text-status-error-text" />
          ) : (
            <RefreshCw size={14} />
          )}
        </Button>
      )}
    </div>
  );
}
