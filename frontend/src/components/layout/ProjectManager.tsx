import { useState } from 'react';
import { Plus, Trash2, X, Check, Pencil } from 'lucide-react';
import { useProjectStore, PRESET_COLORS } from '../../stores/projectStore';
import type { Project } from '../../stores/projectStore';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

interface ProjectManagerProps {
  sessionCounts: Record<string, number>;
}

export function ProjectManager({ sessionCounts }: ProjectManagerProps) {
  const { projects, createProject, renameProject, deleteProject, setProjectColor } = useProjectStore();
  const [newProjectName, setNewProjectName] = useState('');
  const [selectedColor, setSelectedColor] = useState(PRESET_COLORS[0]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');

  const handleCreate = () => {
    if (newProjectName.trim()) {
      createProject(newProjectName.trim(), selectedColor);
      setNewProjectName('');
    }
  };

  const startEditing = (p: Project) => {
    setEditingId(p.id);
    setEditValue(p.name);
  };

  const handleRename = (id: string) => {
    if (editValue.trim()) {
      renameProject(id, editValue.trim());
    }
    setEditingId(null);
  };

  const handleDelete = (p: Project) => {
    const count = sessionCounts[p.id] || 0;
    const message = count > 0 
      ? `This project has ${count} sessions that will become ungrouped. Delete anyway?`
      : 'Delete this project?';
    
    if (confirm(message)) {
      deleteProject(p.id);
    }
  };

  return (
    <div className="flex flex-col gap-4 p-2">
      <div className="flex flex-col gap-2 p-3 bg-hover-bg/50 rounded-lg border border-border">
        <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">
          Create New Project
        </div>
        <div className="flex gap-2">
          <input
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            placeholder="Project name..."
            className="flex-1 bg-panel border border-border rounded px-2 py-1 text-sm outline-none focus:border-primary"
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
          />
          <Button size="sm" onClick={handleCreate} disabled={!newProjectName.trim()}>
            <Plus size={16} />
          </Button>
        </div>
        <div className="flex gap-2 mt-1">
          {PRESET_COLORS.map((color) => (
            <button
              key={color}
              onClick={() => setSelectedColor(color)}
              className={cn(
                "w-5 h-5 rounded-full transition-transform hover:scale-110",
                selectedColor === color ? "ring-2 ring-primary ring-offset-2 ring-offset-panel" : ""
              )}
              style={{ backgroundColor: color }}
            />
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-1 overflow-y-auto max-h-[300px]">
        {projects.length === 0 && (
          <div className="text-sm text-text-muted text-center py-4">
            No projects yet
          </div>
        )}
        {projects.map((p) => (
          <div 
            key={p.id} 
            className="flex items-center gap-2 p-2 rounded-md hover:bg-hover-bg group border border-transparent hover:border-border"
          >
            <div 
              className="w-3 h-3 rounded-full shrink-0 cursor-pointer" 
              style={{ backgroundColor: p.color }}
              onClick={() => {
                const currentIndex = PRESET_COLORS.indexOf(p.color);
                const nextIndex = (currentIndex + 1) % PRESET_COLORS.length;
                setProjectColor(p.id, PRESET_COLORS[nextIndex]);
              }}
              title="Click to change color"
            />
            
            <div className="flex-1 min-w-0">
              {editingId === p.id ? (
                <div className="flex items-center gap-1">
                  <input
                    autoFocus
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleRename(p.id);
                      if (e.key === 'Escape') setEditingId(null);
                    }}
                    className="flex-1 bg-panel border border-primary rounded px-1 py-0 text-sm outline-none"
                  />
                  <button onClick={() => handleRename(p.id)} className="text-status-success-text">
                    <Check size={14} />
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm font-medium">{p.name}</span>
                  <span className="text-[10px] bg-hover-bg px-1.5 py-0.5 rounded text-text-muted">
                    {sessionCounts[p.id] || 0}
                  </span>
                </div>
              )}
            </div>

            <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={() => startEditing(p)}
                className="p-1 rounded hover:bg-hover-bg text-text-muted hover:text-text"
              >
                <Pencil size={13} />
              </button>
              <button
                onClick={() => handleDelete(p)}
                className="p-1 rounded hover:bg-hover-danger-bg text-text-muted hover:text-status-error-text"
              >
                <Trash2 size={13} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
