import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Folder, Settings, LayoutGrid, Check } from 'lucide-react';
import { useProjectStore } from '../../stores/projectStore';
import { cn } from '@/lib/utils';
import { ProjectManager } from './ProjectManager';

interface ProjectSelectorProps {
  sessionCounts: Record<string, number>;
  ungroupedCount: number;
  totalCount: number;
}

export function ProjectSelector({ sessionCounts, ungroupedCount, totalCount }: ProjectSelectorProps) {
  const { projects, activeProjectId, setActiveProject } = useProjectStore();
  const [isOpen, setIsOpen] = useState(false);
  const [showManager, setShowManager] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setShowManager(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const activeProject = projects.find(p => p.id === activeProjectId);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2 bg-panel border border-border rounded-lg text-sm hover:border-primary transition-colors group"
      >
        <div className="flex items-center gap-2 overflow-hidden">
          {activeProjectId === null ? (
            <>
              <LayoutGrid size={14} className="text-text-muted" />
              <span className="truncate font-medium">All Projects</span>
            </>
          ) : activeProjectId === 'ungrouped' ? (
            <>
              <Folder size={14} className="text-text-muted" />
              <span className="truncate font-medium">Ungrouped</span>
            </>
          ) : (
            <>
              <div 
                className="w-2.5 h-2.5 rounded-full shrink-0" 
                style={{ backgroundColor: activeProject?.color || '#ccc' }} 
              />
              <span className="truncate font-medium">{activeProject?.name || 'Unknown Project'}</span>
            </>
          )}
        </div>
        <ChevronDown 
          size={14} 
          className={cn("text-text-muted transition-transform", isOpen ? "rotate-180" : "")} 
        />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 bg-panel border border-border rounded-lg shadow-xl overflow-hidden flex flex-col max-h-[400px]">
          {!showManager ? (
            <>
              <div className="p-1 overflow-y-auto">
                <button
                  onClick={() => {
                    setActiveProject(null);
                    setIsOpen(false);
                  }}
                  className={cn(
                    "w-full flex items-center justify-between px-3 py-2 rounded-md text-sm hover:bg-hover-bg transition-colors",
                    activeProjectId === null ? "bg-primary/5 text-primary" : "text-text"
                  )}
                >
                  <div className="flex items-center gap-2">
                    <LayoutGrid size={14} className={activeProjectId === null ? "text-primary" : "text-text-muted"} />
                    <span>All Projects</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-text-muted">{totalCount}</span>
                    {activeProjectId === null && <Check size={14} />}
                  </div>
                </button>

                <button
                  onClick={() => {
                    setActiveProject('ungrouped');
                    setIsOpen(false);
                  }}
                  className={cn(
                    "w-full flex items-center justify-between px-3 py-2 rounded-md text-sm hover:bg-hover-bg transition-colors",
                    activeProjectId === 'ungrouped' ? "bg-primary/5 text-primary" : "text-text"
                  )}
                >
                  <div className="flex items-center gap-2">
                    <Folder size={14} className={activeProjectId === 'ungrouped' ? "text-primary" : "text-text-muted"} />
                    <span>Ungrouped</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-text-muted">{ungroupedCount}</span>
                    {activeProjectId === 'ungrouped' && <Check size={14} />}
                  </div>
                </button>

                {projects.length > 0 && <div className="h-px bg-border my-1" />}

                {projects.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => {
                      setActiveProject(p.id);
                      setIsOpen(false);
                    }}
                    className={cn(
                      "w-full flex items-center justify-between px-3 py-2 rounded-md text-sm hover:bg-hover-bg transition-colors",
                      activeProjectId === p.id ? "bg-primary/5 text-primary" : "text-text"
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: p.color }} />
                      <span className="truncate">{p.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-text-muted">{sessionCounts[p.id] || 0}</span>
                      {activeProjectId === p.id && <Check size={14} />}
                    </div>
                  </button>
                ))}
              </div>

              <button
                onClick={() => setShowManager(true)}
                className="flex items-center gap-2 px-3 py-2.5 text-xs font-medium text-text-muted hover:text-text hover:bg-hover-bg border-t border-border transition-colors mt-auto"
              >
                <Settings size={14} />
                Manage Projects
              </button>
            </>
          ) : (
            <div className="flex flex-col h-full overflow-hidden">
              <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-hover-bg/30">
                <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">Manage Projects</span>
                <button 
                  onClick={() => setShowManager(false)}
                  className="p-1 hover:bg-hover-bg rounded-md"
                >
                  <ChevronDown size={14} className="rotate-90" />
                </button>
              </div>
              <div className="overflow-y-auto flex-1">
                <ProjectManager sessionCounts={sessionCounts} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
