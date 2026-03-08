import { useState, useRef, useCallback, useEffect } from 'react';
import { Send, Square, Loader2, Paperclip, X } from 'lucide-react';
import * as api from '../../api/client';
import type { SlashCommandInfo } from '../../api/client';
import { useSessionStore } from '../../stores/sessionStore';

interface Props {
  onSend: (message: string, files?: File[]) => void;
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
  const [slashCommands, setSlashCommands] = useState<SlashCommandInfo[]>([]);
  const [commandMatches, setCommandMatches] = useState<SlashCommandInfo[]>([]);
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const completionSeq = useRef(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const cachedCompletions = useRef<string[]>([]);
  const hasSentMessage = useRef(false);
  const isAutofilled = useRef(false);
  const { activeSessionId } = useSessionStore();

  // Load input history and slash commands on mount
  useEffect(() => {
    api.getInputHistory()
      .then(({ history: h }) => setHistory(h))
      .catch(() => {});
    api.getCommands().then(setSlashCommands).catch(() => {});
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
    isAutofilled.current = false;
    setCommandMatches([]);
    setSelectedCommandIndex(0);
  }, [isStreaming, activeSessionId]);

  // Autofill a suggestion when input is empty, session is ready,
  // and user has sent at least one message (skip first interaction)
  useEffect(() => {
    if (!text.trim() && !isStreaming && activeSessionId && hasSentMessage.current && !isAutofilled.current) {
      setGhostText('');
      cachedCompletions.current = [];
      const seq = ++completionSeq.current;
      const timer = setTimeout(async () => {
        try {
          const { completions } = await api.getCompletions(activeSessionId, '');
          if (seq !== completionSeq.current || !completions.length) return;
          cachedCompletions.current = completions;
          const words = completions[0].split(/\s+/);
          const truncated = words.length > 10
            ? words.slice(0, 10).join(' ') + '...'
            : completions[0];
          setText(truncated);
          isAutofilled.current = true;
          requestAnimationFrame(() => {
            textareaRef.current?.select();
          });
        } catch {
          // ignore
        }
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [text, isStreaming, activeSessionId]);

  // Global "/" shortcut to focus chat input
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.key === '/' &&
        !e.ctrlKey && !e.metaKey && !e.altKey &&
        document.activeElement?.tagName !== 'INPUT' &&
        document.activeElement?.tagName !== 'TEXTAREA' &&
        document.activeElement?.tagName !== 'SELECT'
      ) {
        e.preventDefault();
        textareaRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const acceptCommand = useCallback((commandName: string) => {
    setText(commandName);
    setCommandMatches([]);
    setSelectedCommandIndex(0);
    setGhostText('');
    textareaRef.current?.focus();
  }, []);

  const dragCounter = useRef(0);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current++;
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) setIsDragging(false);
  }, []);

  const ALLOWED_EXTENSIONS = new Set([
    '.csv','.tsv','.json','.parquet','.xlsx','.xls',
    '.pdf','.png','.jpg','.jpeg','.gif','.webp','.bmp','.tiff',
  ]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    setIsDragging(false);
    const dropped = Array.from(e.dataTransfer.files).filter((f) => {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase();
      return ALLOWED_EXTENSIONS.has(ext);
    });
    if (dropped.length > 0) {
      setAttachedFiles((prev) => [...prev, ...dropped]);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files;
    if (!selected) return;
    setAttachedFiles((prev) => [...prev, ...Array.from(selected)]);
    // Reset so the same file can be re-selected
    e.target.value = '';
  }, []);

  const removeFile = useCallback((index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim();
    if ((!trimmed && attachedFiles.length === 0) || disabled) return;

    // If command dropdown is open, accept the selected command instead of submitting
    if (commandMatches.length > 0) {
      acceptCommand(`/${commandMatches[selectedCommandIndex].name}`);
      return;
    }

    onSend(trimmed, attachedFiles.length > 0 ? attachedFiles : undefined);
    hasSentMessage.current = true;
    // Save to history
    api.addInputHistory(trimmed).catch(() => {});
    setHistory((h) => [...h, trimmed]);
    setHistoryIndex(-1);
    setDraftText('');
    setText('');
    setAttachedFiles([]);
    setGhostText('');
    isAutofilled.current = false;
    setCommandMatches([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [text, disabled, onSend, attachedFiles, commandMatches, selectedCommandIndex, acceptCommand]);

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
        acceptCommand(`/${commandMatches[selectedCommandIndex].name}`);
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

    // Ctrl+Enter (or Cmd+Enter on Mac) to submit
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
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
    isAutofilled.current = false;
    setText(newText);
    setHistoryIndex(-1);

    // Slash command mode: show dropdown, disable LLM completions
    if (newText.startsWith('/')) {
      setGhostText('');
      cachedCompletions.current = [];
      if (debounceRef.current) clearTimeout(debounceRef.current);

      const matches = slashCommands.filter(
        (c) => `/${c.name}`.startsWith(newText.toLowerCase()),
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
    <div
      className={`border-t border-border bg-panel px-4 py-3 transition-colors ${isDragging ? 'bg-primary/5 border-primary' : ''}`}
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".csv,.tsv,.json,.parquet,.xlsx,.xls,.pdf,.png,.jpg,.jpeg,.gif,.webp,.bmp,.tiff"
        className="hidden"
        onChange={handleFileSelect}
      />
      <div className="flex items-end gap-2 max-w-3xl mx-auto">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || isCancelling || isStreaming}
          className="p-2.5 rounded-xl text-text-muted hover:text-text hover:bg-surface-elevated
            transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
          aria-label="Attach file"
        >
          <Paperclip size={18} />
        </button>
        <div className="flex-1 relative">
          {/* Attached files badges */}
          {attachedFiles.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-1.5">
              {attachedFiles.map((f, i) => (
                <span
                  key={`${f.name}-${i}`}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md
                    bg-surface-elevated text-xs text-text-muted border border-border"
                >
                  <Paperclip size={12} />
                  <span className="max-w-[150px] truncate">{f.name}</span>
                  <span className="text-[10px] opacity-60">
                    {f.size < 1024 ? `${f.size} B` : f.size < 1048576 ? `${(f.size / 1024).toFixed(0)} KB` : `${(f.size / 1048576).toFixed(1)} MB`}
                  </span>
                  <button
                    onClick={() => removeFile(i)}
                    className="ml-0.5 p-0.5 rounded hover:bg-border transition-colors"
                    aria-label={`Remove ${f.name}`}
                  >
                    <X size={10} />
                  </button>
                </span>
              ))}
            </div>
          )}
          {/* Command dropdown */}
          {commandMatches.length > 0 && (
            <div data-testid="command-dropdown" className="absolute bottom-full left-0 w-full mb-1 rounded-lg border border-border bg-panel shadow-lg overflow-hidden z-10">
              {commandMatches.map((cmd, i) => (
                <button
                  key={cmd.name}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    acceptCommand(`/${cmd.name}`);
                  }}
                  className={`w-full flex items-center gap-3 px-3 py-2 text-sm text-left transition-colors
                    ${i === selectedCommandIndex ? 'bg-surface-elevated text-text' : 'text-text-muted hover:bg-surface-elevated/50'}`}
                >
                  <code className="text-primary font-medium shrink-0">/{cmd.name}</code>
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
            placeholder=""
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
              disabled={(!text.trim() && attachedFiles.length === 0) || disabled}
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
