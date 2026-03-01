import { useEffect, useState } from 'react';
import { Download, Loader2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { ExamplePrompts } from './ExamplePrompts';
import { useChatStore } from '../../stores/chatStore';
import { useSessionStore } from '../../stores/sessionStore';
import { useLoadingStateStore } from '../../stores/loadingStateStore';
import { exportSessionAsMarkdown } from '../../utils/exportSession';

export function ChatContainer() {
  const { messages, toolEvents, isStreaming, isCancelling, storageWarning, sendMessage, cancelStream, regenerateLastResponse, dismissStorageWarning } =
    useChatStore();
  const { activeSessionId, savedSessions, tokenUsage } = useSessionStore();
  const loadingState = useLoadingStateStore((s) => s.state);
  const fetchLoadingState = useLoadingStateStore((s) => s.fetchState);
  const [exporting, setExporting] = useState(false);

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

  const handleExport = async () => {
    if (!activeSessionId || exporting) return;
    const session = savedSessions.find((s) => s.id === activeSessionId);
    setExporting(true);
    try {
      await exportSessionAsMarkdown(messages, activeSessionId, {
        sessionName: session?.name,
        tokenUsage,
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
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={handleExport}
            disabled={exporting}
            aria-label="Export session as Markdown"
          >
            {exporting ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
          </Button>
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
