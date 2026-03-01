import { useState, useRef, useEffect } from 'react';
import { MessageSquare, Trash2, Pencil, Pin, FolderOpen, MoreVertical } from 'lucide-react';
import type { SavedSessionInfo } from '../../api/types';
import type { Project } from '../../stores/projectStore';
import { cn } from '@/lib/utils';
import { formatSessionDate } from '../../utils/dateUtils';

interface SessionItemProps {
  session: SavedSessionInfo;
  isActive: boolean;
  resuming: boolean;
  project?: Project;
  isPinned: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, name: string) => void;
  onMoveToProject: (id: string, projectId: string | null) => void;
  onTogglePin: (id: string) => void;
  projects: Project[];
}

export function SessionItem({
  session,
  isActive,
  resuming,
  project,
  isPinned,
  onSelect,
  onDelete,
  onRename,
  onMoveToProject,
  onTogglePin,
  projects,
}: SessionItemProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const [showMoveMenu, setShowMoveMenu] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingId]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowMoveMenu(false);
      }
    }
    if (showMoveMenu) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showMoveMenu]);

  function startEditing(e: React.MouseEvent) {
    e.stopPropagation();
    setEditingId(session.id);
    setEditValue(session.name || session.last_message_preview || '');
  }

  function commitRename() {
    if (editingId && editValue.trim()) {
      onRename(editingId, editValue.trim());
    }
    setEditingId(null);
  }

  function cancelEditing() {
    setEditingId(null);
  }

  const dateGroup = formatSessionDate(session.updated_at);

  return (
    <div className="relative mb-1 group">
      <button
        onClick={() => { if (!isActive && !resuming) onSelect(session.id); }}
        disabled={resuming}
        className={cn(
          "w-full text-left px-3 py-2 rounded-lg text-sm transition-colors",
          "flex items-start gap-2 hover:bg-hover-bg",
          isActive ? "border-l-3 border-primary bg-primary/10" : "",
          resuming ? "opacity-60 cursor-not-allowed" : ""
        )}
      >
        <div className="flex items-center mt-1 shrink-0 gap-1.5">
          {project ? (
            <div 
              className="w-2 h-2 rounded-full" 
              style={{ backgroundColor: project.color }} 
              title={project.name}
            />
          ) : (
            <MessageSquare size={14} className="text-text-muted" />
          )}
        </div>

        <div className="flex-1 min-w-0">
          {editingId === session.id ? (
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
            <div className="truncate text-text font-medium">
              {session.name || session.last_message_preview || (isActive ? 'Current session' : 'Empty session')}
            </div>
          )}
          <div className="text-xs text-text-muted mt-0.5 flex items-center gap-1">
            <span>{session.round_count} {session.round_count === 1 ? 'round' : 'rounds'}</span>
            <span>·</span>
            <span>{dateGroup}</span>
          </div>
        </div>

        <div className="flex items-center shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onTogglePin(session.id);
            }}
            className={cn(
              "p-1 rounded hover:bg-hover-bg transition-colors",
              isPinned ? "text-primary" : "text-text-muted"
            )}
            title={isPinned ? "Unpin session" : "Pin session"}
          >
            <Pin size={13} fill={isPinned ? "currentColor" : "none"} />
          </button>
          <button
            onClick={startEditing}
            className="p-1 rounded hover:bg-hover-bg text-text-muted hover:text-text transition-colors"
            title="Rename session"
          >
            <Pencil size={13} />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowMoveMenu(!showMoveMenu);
            }}
            className="p-1 rounded hover:bg-hover-bg text-text-muted hover:text-text transition-colors"
            title="Move to project"
          >
            <FolderOpen size={13} />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (confirm('Delete this session?')) {
                onDelete(session.id);
              }
            }}
            className="p-1 rounded hover:bg-hover-danger-bg text-text-muted hover:text-status-error-text transition-colors"
            title="Delete session"
          >
            <Trash2 size={13} />
          </button>
        </div>
      </button>

      {showMoveMenu && (
        <div 
          ref={menuRef}
          className="absolute right-0 top-10 z-50 w-48 bg-panel border border-border rounded-md shadow-lg py-1 text-sm overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="px-3 py-1.5 text-xs font-semibold text-text-muted border-b border-border mb-1">
            Move to Project
          </div>
          <button
            onClick={() => {
              onMoveToProject(session.id, null);
              setShowMoveMenu(false);
            }}
            className={cn(
              "w-full text-left px-3 py-1.5 hover:bg-hover-bg flex items-center gap-2",
              !project ? "bg-primary/5 text-primary" : "text-text"
            )}
          >
            <div className="w-2 h-2 rounded-full border border-text-muted" />
            <span>Ungrouped</span>
          </button>
          {projects.map((p) => (
            <button
              key={p.id}
              onClick={() => {
                onMoveToProject(session.id, p.id);
                setShowMoveMenu(false);
              }}
              className={cn(
                "w-full text-left px-3 py-1.5 hover:bg-hover-bg flex items-center gap-2",
                project?.id === p.id ? "bg-primary/5 text-primary" : "text-text"
              )}
            >
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
              <span className="truncate">{p.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
