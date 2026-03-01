import { useState, useRef, useCallback, useEffect } from 'react';
import { Send, Square, Loader2 } from 'lucide-react';
import * as api from '../../api/client';
import { useSessionStore } from '../../stores/sessionStore';

const SLASH_COMMANDS = [
  { name: '/branch', description: 'Fork into a new session branch' },
  { name: '/help', description: 'Show available commands' },
  { name: '/status', description: 'Session info: model, tokens, data' },
  { name: '/data', description: 'List data entries in memory' },
  { name: '/figure', description: 'Figure availability status' },
  { name: '/reset', description: 'Reset session' },
  { name: '/sessions', description: 'List saved sessions' },
  { name: '/retry', description: 'Retry failed plan task' },
  { name: '/cancel', description: 'Cancel current plan' },
  { name: '/errors', description: 'Show recent error logs' },
];

interface Props {
  onSend: (message: string) => void;
  onCancel: () => void;
  isStreaming: boolean;
  isCancelling: boolean;
  disabled: boolean;
}

export function ChatInput({ onSend, onCancel, isStreaming, isCancelling, disabled }: Props) {
  const [text, setText] = useState('');
  const [ghostText, setGhostText] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [draftText, setDraftText] = useState('');
  const [commandMatches, setCommandMatches] = useState<typeof SLASH_COMMANDS>([]);
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const completionSeq = useRef(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const cachedCompletions = useRef<string[]>([]);
  const hasSentMessage = useRef(false);
  const { activeSessionId } = useSessionStore();

  // Load input history on mount
  useEffect(() => {
    api.getInputHistory()
      .then(({ history: h }) => setHistory(h))
      .catch(() => {});
  }, []);

  // Try to match current text against cached completions
  const matchCached = useCallback((input: string): string | null => {
    for (const c of cachedCompletions.current) {
      if (c.startsWith(input) && c.length > input.length) {
        return c.slice(input.length);
      }
    }
    return null;
  }, []);

  // Fetch completions from backend and cache them
  const fetchCompletions = useCallback((input: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (isStreaming || !activeSessionId) return;
    // Allow empty input (default suggestions) or 5+ chars
    const len = input.trim().length;
    if (len > 0 && len < 5) return;

    const seq = ++completionSeq.current;
    debounceRef.current = setTimeout(async () => {
      try {
        const { completions } = await api.getCompletions(activeSessionId, input);
        if (seq !== completionSeq.current) return;
        cachedCompletions.current = completions;
        // Compare against the current text — if user typed more, use
        // the live value so we still surface a matching suggestion.
        const currentText = textareaRef.current?.value ?? input;
        for (const c of completions) {
          if (c.startsWith(currentText) && c.length > currentText.length) {
            setGhostText(c.slice(currentText.length));
            return;
          }
        }
        // No match for live text — clear stale ghost text
        setGhostText('');
      } catch {
        // ignore
      }
    }, 1000);
  }, [isStreaming, activeSessionId]);

  // Clear ghost text and cache when streaming starts or session changes
  useEffect(() => {
    setGhostText('');
    cachedCompletions.current = [];
    setCommandMatches([]);
    setSelectedCommandIndex(0);
  }, [isStreaming, activeSessionId]);

  // Fetch default suggestions when input is empty, session is ready,
  // and user has sent at least one message (skip first interaction)
  useEffect(() => {
    if (!text.trim() && !isStreaming && activeSessionId && hasSentMessage.current) {
      setGhostText('');
      cachedCompletions.current = [];
      fetchCompletions('');
    }
  }, [text, isStreaming, activeSessionId, fetchCompletions]);

  const acceptCommand = useCallback((commandName: string) => {
    setText(commandName);
    setCommandMatches([]);
    setSelectedCommandIndex(0);
    setGhostText('');
    textareaRef.current?.focus();
  }, []);

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;

    // If command dropdown is open, accept the selected command instead of submitting
    if (commandMatches.length > 0) {
      acceptCommand(commandMatches[selectedCommandIndex].name);
      return;
    }

    onSend(trimmed);
    hasSentMessage.current = true;
    // Save to history
    api.addInputHistory(trimmed).catch(() => {});
    setHistory((h) => [...h, trimmed]);
    setHistoryIndex(-1);
    setDraftText('');
    setText('');
    setGhostText('');
    setCommandMatches([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [text, disabled, onSend, commandMatches, selectedCommandIndex, acceptCommand]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Command dropdown navigation
    if (commandMatches.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedCommandIndex((i) => (i + 1) % commandMatches.length);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedCommandIndex((i) => (i - 1 + commandMatches.length) % commandMatches.length);
        return;
      }
      if (e.key === 'Tab') {
        e.preventDefault();
        acceptCommand(commandMatches[selectedCommandIndex].name);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        setCommandMatches([]);
        setSelectedCommandIndex(0);
        return;
      }
    }

    // Tab accepts ghost text
    if (e.key === 'Tab' && ghostText) {
      e.preventDefault();
      const accepted = text + ghostText;
      setText(accepted);
      setGhostText('');
      cachedCompletions.current = [];
      fetchCompletions(accepted);
      return;
    }

    // Escape dismisses ghost text
    if (e.key === 'Escape' && ghostText) {
      setGhostText('');
      return;
    }

    // Enter to submit
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
      return;
    }

    // Arrow up/down for history navigation
    if (e.key === 'ArrowUp' && !e.shiftKey) {
      const textarea = textareaRef.current;
      // Only navigate history if cursor is at start
      if (textarea && textarea.selectionStart === 0 && textarea.selectionEnd === 0) {
        e.preventDefault();
        if (history.length === 0) return;
        if (historyIndex === -1) {
          setDraftText(text);
          const newIdx = history.length - 1;
          setHistoryIndex(newIdx);
          setText(history[newIdx]);
        } else if (historyIndex > 0) {
          const newIdx = historyIndex - 1;
          setHistoryIndex(newIdx);
          setText(history[newIdx]);
        }
        setGhostText('');
      }
    }

    if (e.key === 'ArrowDown' && !e.shiftKey) {
      const textarea = textareaRef.current;
      if (textarea && textarea.selectionStart === textarea.value.length) {
        e.preventDefault();
        if (historyIndex === -1) return;
        if (historyIndex < history.length - 1) {
          const newIdx = historyIndex + 1;
          setHistoryIndex(newIdx);
          setText(history[newIdx]);
        } else {
          setHistoryIndex(-1);
          setText(draftText);
        }
        setGhostText('');
      }
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newText = e.target.value;
    setText(newText);
    setHistoryIndex(-1);

    // Slash command mode: show dropdown, disable LLM completions
    if (newText.startsWith('/')) {
      setGhostText('');
      cachedCompletions.current = [];
      if (debounceRef.current) clearTimeout(debounceRef.current);

      const matches = SLASH_COMMANDS.filter(
        (c) => c.name.startsWith(newText.toLowerCase()),
      );
      setCommandMatches(matches);
      setSelectedCommandIndex(0);

      // Auto-grow
      const el = e.target;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 200) + 'px';
      return;
    }

    // Normal mode: clear command dropdown
    setCommandMatches([]);

    // Check if input still matches a cached completion
    if (newText.trim()) {
      const match = matchCached(newText);
      if (match) {
        setGhostText(match);
      } else {
        setGhostText('');
        fetchCompletions(newText);
      }
    } else {
      setGhostText('');
      if (hasSentMessage.current) fetchCompletions('');
    }

    // Auto-grow
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  };

  return (
    <div className="border-t border-border bg-panel px-4 py-3">
      <div className="flex items-end gap-2 max-w-3xl mx-auto">
        <div className="flex-1 relative">
          {/* Command dropdown */}
          {commandMatches.length > 0 && (
            <div className="absolute bottom-full left-0 w-full mb-1 rounded-lg border border-border bg-panel shadow-lg overflow-hidden z-10">
              {commandMatches.map((cmd, i) => (
                <button
                  key={cmd.name}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    acceptCommand(cmd.name);
                  }}
                  className={`w-full flex items-center gap-3 px-3 py-2 text-sm text-left transition-colors
                    ${i === selectedCommandIndex ? 'bg-surface-elevated text-text' : 'text-text-muted hover:bg-surface-elevated/50'}`}
                >
                  <code className="text-primary font-medium shrink-0">{cmd.name}</code>
                  <span className="text-text-muted text-xs truncate">{cmd.description}</span>
                </button>
              ))}
            </div>
          )}
          <textarea
            data-testid="chat-input"
            ref={textareaRef}
            value={text}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={ghostText ? '' : 'Ask about spacecraft data, solar events, or missions...'}
            rows={1}
            disabled={disabled || isCancelling}
            className="w-full resize-none rounded-xl border border-border px-4 py-2.5 text-sm
              focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary
              disabled:opacity-50 bg-input-bg text-text placeholder:text-text-muted"
          />
          {/* Ghost text overlay */}
          {ghostText && (
            <div
              className="absolute top-0 left-0 px-4 py-2.5 text-sm pointer-events-none
                whitespace-nowrap overflow-hidden text-transparent border border-transparent"
              style={{
                width: '100%',
                fontFamily: 'inherit',
                lineHeight: 'inherit',
              }}
            >
              {text}
              <span className="text-text-muted opacity-60">{ghostText}</span>
            </div>
          )}
        </div>
        {isCancelling ? (
          <button
            disabled
            className="p-2.5 rounded-xl bg-status-error-text text-white opacity-70
              cursor-not-allowed shrink-0"
            aria-label="Stopping..."
          >
            <Loader2 size={18} className="animate-spin" />
          </button>
        ) : (
          <div className="flex items-end gap-1.5">
            {isStreaming && (
              <button
                data-testid="chat-stop-btn"
                onClick={onCancel}
                className="p-2.5 rounded-xl bg-status-error-text text-white hover:opacity-90
                  transition-colors shrink-0"
                aria-label="Stop generating"
              >
                <Square size={18} />
              </button>
            )}
            <button
              data-testid="chat-send-btn"
              onClick={handleSubmit}
              disabled={!text.trim() || disabled}
              className="p-2.5 rounded-xl bg-primary text-white hover:bg-primary-dark
                transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
              aria-label="Send message"
            >
              <Send size={18} />
            </button>
          </div>
        )}
      </div>
      {commandMatches.length > 0 && (
        <div className="max-w-3xl mx-auto mt-1">
          <span className="text-[10px] text-text-muted">↑↓ navigate · Tab/Enter accept · Esc dismiss</span>
        </div>
      )}
      {ghostText && commandMatches.length === 0 && (
        <div className="max-w-3xl mx-auto mt-1">
          <span className="text-[10px] text-text-muted">Tab to accept, Esc to dismiss</span>
        </div>
      )}
    </div>
  );
}
