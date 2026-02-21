import { useState, useRef, useEffect, useMemo } from 'react';
import { Plus, MessageSquare, Trash2, Pencil } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { SavedSessionInfo } from '../../api/types';

interface Props {
  savedSessions: SavedSessionInfo[];
  activeSessionId: string | null;
  resuming?: boolean;
  onNewChat: () => void;
  onResumeSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, name: string) => void;
}

export function Sidebar({
  savedSessions,
  activeSessionId,
  resuming = false,
  onNewChat,
  onResumeSession,
  onDeleteSession,
  onRenameSession,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingId]);

  // If the active session isn't in the saved list yet (e.g. first turn
  // hasn't completed auto-save), synthesize a placeholder entry so the
  // sidebar always shows the current conversation.
  const sessions = useMemo(() => {
    if (!activeSessionId) return savedSessions;
    if (savedSessions.some((s) => s.id === activeSessionId)) return savedSessions;
    const placeholder: SavedSessionInfo = {
      id: activeSessionId,
      name: null,
      model: null,
      turn_count: 0,
      last_message_preview: '',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      token_usage: {},
    };
    return [placeholder, ...savedSessions];
  }, [savedSessions, activeSessionId]);

  function startEditing(s: SavedSessionInfo, e: React.MouseEvent) {
    e.stopPropagation();
    setEditingId(s.id);
    setEditValue(s.name || s.last_message_preview || '');
  }

  function commitRename() {
    if (editingId && editValue.trim()) {
      onRenameSession(editingId, editValue.trim());
    }
    setEditingId(null);
  }

  function cancelEditing() {
    setEditingId(null);
  }

  return (
    <div className="flex flex-col h-full bg-panel border-r border-border">
      <div className="p-3">
        <Button onClick={onNewChat} className="w-full" disabled={resuming}>
          <Plus size={16} />
          New Chat
        </Button>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 pb-2" aria-label="Session history">
        <div className="text-xs font-medium text-text-muted px-2 py-1.5 uppercase tracking-wide">
          Sessions
        </div>
        {sessions.length === 0 && (
          <div className="text-sm text-text-muted px-2 py-4 text-center">
            No saved sessions
          </div>
        )}
        {sessions.map((s) => {
          const isActive = activeSessionId === s.id;
          return (
          <button
            key={s.id}
            onClick={() => { if (!isActive && !resuming) onResumeSession(s.id); }}
            disabled={resuming}
            className={`w-full text-left px-3 py-2 rounded-lg mb-1 text-sm transition-colors
              group flex items-start gap-2 hover:bg-hover-bg
              ${isActive ? 'border-l-3 border-primary bg-primary/10' : ''}
              ${resuming ? 'opacity-60 cursor-not-allowed' : ''}`}
          >
            <MessageSquare size={14} className="text-text-muted mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              {editingId === s.id ? (
                <input
                  ref={inputRef}
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitRename();
                    if (e.key === 'Escape') cancelEditing();
                  }}
                  onBlur={commitRename}
                  onClick={(e) => e.stopPropagation()}
                  className="w-full bg-transparent border border-primary rounded px-1 py-0 text-sm text-text outline-none"
                  maxLength={100}
                />
              ) : (
                <div className="truncate text-text">
                  {s.name || s.last_message_preview || (isActive ? 'Current session' : 'Empty session')}
                </div>
              )}
              <div className="text-xs text-text-muted mt-0.5">
                {s.turn_count} {s.turn_count === 1 ? 'turn' : 'turns'}
                {s.updated_at && ` · ${formatDate(s.updated_at)}`}
              </div>
            </div>
            <div className="flex items-center gap-0.5 shrink-0">
              <button
                onClick={(e) => startEditing(s, e)}
                className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-hover-bg text-text-muted hover:text-text transition-all"
                aria-label="Rename session"
              >
                <Pencil size={13} />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(s.id);
                }}
                className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-hover-danger-bg text-text-muted hover:text-status-error-text transition-all"
                aria-label="Delete session"
              >
                <Trash2 size={13} />
              </button>
            </div>
          </button>
          );
        })}
      </nav>
    </div>
  );
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const days = Math.floor(diff / 86400000);
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}
