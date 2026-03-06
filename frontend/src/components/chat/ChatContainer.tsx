import { useEffect, useRef, useState } from 'react';
import { Download, Loader2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { ExamplePrompts } from './ExamplePrompts';
import { useChatStore } from '../../stores/chatStore';
import { useSessionStore } from '../../stores/sessionStore';
import { useLoadingStateStore } from '../../stores/loadingStateStore';
import { exportSessionAsMarkdown, type ExportFormat } from '../../utils/exportSession';
import { useExportSettings } from '../../hooks/useExportSettings';

export function ChatContainer() {
  const { messages, toolEvents, isStreaming, isCancelling, storageWarning, sendMessage, cancelStream, regenerateLastResponse, dismissStorageWarning } =
    useChatStore();
  const { activeSessionId, savedSessions, tokenUsage } = useSessionStore();
  const loadingState = useLoadingStateStore((s) => s.state);
  const fetchLoadingState = useLoadingStateStore((s) => s.fetchState);
  const [exporting, setExporting] = useState(false);
  useExportSettings(); // persist settings to localStorage
  const [showExportMenu, setShowExportMenu] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  // Close export menu on click outside
  useEffect(() => {
    if (!showExportMenu) return;
    const handler = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showExportMenu]);

  // Fetch loading state on mount
  useEffect(() => {
    fetchLoadingState();
    return () => useLoadingStateStore.getState().unsubscribeSSE();
  }, [fetchLoadingState]);

  const handleSend = (message: string) => {
    if (!activeSessionId) return;
    sendMessage(activeSessionId, message);
  };

  const handleCancel = () => {
    if (!activeSessionId) return;
    cancelStream(activeSessionId);
  };

  const handleRegenerate = () => {
    if (!activeSessionId) return;
    regenerateLastResponse(activeSessionId);
  };

  const handleExport = async (format: ExportFormat) => {
    if (!activeSessionId || exporting) return;
    const session = savedSessions.find((s) => s.id === activeSessionId);
    setExporting(true);
    setShowExportMenu(false);
    try {
      await exportSessionAsMarkdown(messages, activeSessionId, {
        sessionName: session?.name,
        tokenUsage,
        format,
      });
    } finally {
      setExporting(false);
    }
  };

  const hasMessages = messages.length > 0;

  return (
    <div data-testid="chat-container" className="flex flex-col h-full bg-surface relative">
      {/* Storage warning banner */}
      {storageWarning && (
        <div className="flex items-center gap-2 px-4 py-2 bg-status-warning-bg text-status-warning-text text-xs border-b border-border">
          <span>Browser storage is nearly full. Old activity logs were pruned to save space.</span>
          <button onClick={dismissStorageWarning} className="ml-auto p-0.5 rounded hover:bg-black/10" aria-label="Dismiss">
            <X size={14} />
          </button>
        </div>
      )}

      {/* Mission data loading banner */}
      {loadingState?.is_loading && (
        <div className="flex items-center gap-2 px-4 py-2 bg-status-warning-bg text-status-warning-text text-xs border-b border-border">
          <Loader2 size={14} className="animate-spin shrink-0" />
          <span>
            Mission data is loading in the background
            {loadingState.progress_pct > 0 ? ` (${Math.round(loadingState.progress_pct)}%)` : ''}.
            {' '}Some features may be limited.
          </span>
        </div>
      )}

      {/* Export button */}
      {hasMessages && activeSessionId && (
        <div className="absolute top-2 right-2 z-10">
          <div className="relative" ref={exportMenuRef}>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setShowExportMenu(!showExportMenu)}
              disabled={exporting}
              aria-label="Export session"
            >
              {exporting ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
            </Button>
            {showExportMenu && (
              <div className="absolute right-0 top-full mt-1 bg-background border border-border rounded-md shadow-lg py-1 min-w-[160px] z-50">
                <button
                  className="w-full px-3 py-1.5 text-left text-sm hover:bg-accent flex items-center gap-2"
                  onClick={() => handleExport('base64')}
                  disabled={exporting}
                >
                  <span>Embed as Base64</span>
                </button>
                <button
                  className="w-full px-3 py-1.5 text-left text-sm hover:bg-accent flex items-center gap-2"
                  onClick={() => handleExport('local')}
                  disabled={exporting}
                >
                  <span>Save to Local Folder</span>
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Messages or example prompts */}
      {hasMessages ? (
        <MessageList
          messages={messages}
          toolEvents={toolEvents}
          isStreaming={isStreaming}
          onRegenerate={handleRegenerate}
        />
      ) : (
        <ExamplePrompts onSelect={handleSend} />
      )}

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        onCancel={handleCancel}
        isStreaming={isStreaming}
        isCancelling={isCancelling}
        disabled={!activeSessionId}
      />
    </div>
  );
}
