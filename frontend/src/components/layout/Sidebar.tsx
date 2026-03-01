import { useState, useMemo, useEffect } from 'react';
import { Plus, Search, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { SavedSessionInfo } from '../../api/types';
import { useProjectStore } from '../../stores/projectStore';
import { useSessionStore } from '../../stores/sessionStore';
import { SessionItem } from './SessionItem';
import { ProjectSelector } from './ProjectSelector';
import { formatSessionDate, getGroupOrder } from '../../utils/dateUtils';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';
import { fadeSlideInStagger, staggerFast } from '@/components/common/MotionPresets';

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
  const [searchQuery, setSearchQuery] = useState('');
  const [animationKey, setAnimationKey] = useState(0);
  const { 
    projects, 
    activeProjectId, 
    sessionProjectMap, 
    pinnedSessionIds,
    assignSession,
    toggleSessionPin,
    pruneOrphanedSessions 
  } = useProjectStore();
  const liveTokenUsage = useSessionStore((s) => s.tokenUsage);

  // Prune orphaned entries from sessionProjectMap on load
  useEffect(() => {
    if (savedSessions.length > 0) {
      const allValidIds = savedSessions.map(s => s.id);
      if (activeSessionId) allValidIds.push(activeSessionId);
      pruneOrphanedSessions(allValidIds);
    }
  }, [savedSessions.length, activeSessionId, pruneOrphanedSessions]);

  // Auto-assign new session to active project
  useEffect(() => {
    if (activeSessionId && activeProjectId && activeProjectId !== 'ungrouped') {
      if (!sessionProjectMap[activeSessionId]) {
        assignSession(activeSessionId, activeProjectId);
      }
    }
  }, [activeSessionId, activeProjectId, assignSession, sessionProjectMap]);

  // Combine saved sessions with active placeholder
  const sessions = useMemo(() => {
    let list = [...savedSessions];
    if (activeSessionId && !list.some((s) => s.id === activeSessionId)) {
      const placeholder: SavedSessionInfo = {
        id: activeSessionId,
        name: null,
        model: null,
        turn_count: 0,
        round_count: 0,
        last_message_preview: '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        token_usage: {},
      };
      list = [placeholder, ...list];
    }
    return list.map(s => s.id === activeSessionId ? { ...s, token_usage: liveTokenUsage } : s);
  }, [savedSessions, activeSessionId, liveTokenUsage]);

  // Filter sessions
  const filteredSessions = useMemo(() => {
    return sessions.filter(s => {
      // Project filter
      const projectId = sessionProjectMap[s.id];
      if (activeProjectId === 'ungrouped') {
        if (projectId) return false;
      } else if (activeProjectId !== null) {
        if (projectId !== activeProjectId) return false;
      }
      
      // Search filter
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        const name = (s.name || '').toLowerCase();
        const preview = (s.last_message_preview || '').toLowerCase();
        const project = projects.find(p => p.id === projectId);
        const projectName = project?.name.toLowerCase() || '';
        
        if (!name.includes(query) && !preview.includes(query) && !projectName.includes(query)) {
          return false;
        }
      }
      
      return true;
    });
  }, [sessions, activeProjectId, sessionProjectMap, searchQuery, projects]);

  // Group sessions
  const groupedSessions = useMemo(() => {
    const groups: Record<string, SavedSessionInfo[]> = {};
    
    filteredSessions.forEach(s => {
      let group = '';
      if (pinnedSessionIds.includes(s.id)) {
        group = 'Pinned';
      } else {
        group = formatSessionDate(s.updated_at);
      }
      
      if (!groups[group]) groups[group] = [];
      groups[group].push(s);
    });
    
    // Sort within groups by updated_at (descending)
    Object.keys(groups).forEach(key => {
      groups[key].sort((a, b) => {
        const dateA = new Date(a.updated_at || 0).getTime();
        const dateB = new Date(b.updated_at || 0).getTime();
        return dateB - dateA;
      });
    });
    
    return groups;
  }, [filteredSessions, pinnedSessionIds]);

  const sortedGroupNames = useMemo(() => {
    return Object.keys(groupedSessions).sort((a, b) => getGroupOrder(a) - getGroupOrder(b));
  }, [groupedSessions]);

  // Stats for ProjectSelector
  const sessionCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    sessions.forEach(s => {
      const pid = sessionProjectMap[s.id];
      if (pid) {
        counts[pid] = (counts[pid] || 0) + 1;
      }
    });
    return counts;
  }, [sessions, sessionProjectMap]);

  const ungroupedCount = useMemo(() => {
    return sessions.filter(s => !sessionProjectMap[s.id]).length;
  }, [sessions, sessionProjectMap]);

  // Trigger staggered animation when filtered sessions change
  useEffect(() => {
    setAnimationKey(k => k + 1);
  }, [filteredSessions.length, activeProjectId, searchQuery]);

  return (
    <div className="flex flex-col h-full bg-panel border-r border-border">
      <div className="p-3 flex flex-col gap-3">
        <Button data-testid="new-chat-btn" onClick={onNewChat} className="w-full" disabled={resuming}>
          <Plus size={16} />
          New Chat
        </Button>

        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search sessions..."
            className="w-full bg-surface border border-border rounded-lg pl-9 pr-8 py-1.5 text-sm outline-none focus:border-primary transition-colors"
          />
          {searchQuery && (
            <button 
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-text-muted hover:text-text"
            >
              <X size={14} />
            </button>
          )}
        </div>

        <ProjectSelector 
          sessionCounts={sessionCounts} 
          ungroupedCount={ungroupedCount}
          totalCount={sessions.length}
        />
      </div>

      <nav data-testid="session-list" className="flex-1 overflow-y-auto px-2 pb-4 scrollbar-thin" aria-label="Session history">
        {sortedGroupNames.length === 0 ? (
          <div className="text-sm text-text-muted px-2 py-8 text-center italic">
            {searchQuery ? 'No sessions match your search' : 'No saved sessions'}
          </div>
        ) : (
          <AnimatePresence mode="sync">
            <motion.div
              key={animationKey}
              variants={staggerFast}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="space-y-0.5"
            >
              {sortedGroupNames.map(groupName => (
                <div key={groupName} className="mb-4">
                  <div className="text-[10px] font-bold text-text-muted px-3 py-1.5 uppercase tracking-widest mb-1">
                    {groupName}
                  </div>
                  <div className="space-y-0.5">
                    {groupedSessions[groupName].map(session => (
                      <motion.div
                        key={session.id}
                        variants={fadeSlideInStagger}
                      >
                        <SessionItem
                          session={session}
                          isActive={activeSessionId === session.id}
                          resuming={resuming}
                          project={projects.find(p => p.id === sessionProjectMap[session.id])}
                          isPinned={pinnedSessionIds.includes(session.id)}
                          onSelect={onResumeSession}
                          onDelete={onDeleteSession}
                          onRename={onRenameSession}
                          onMoveToProject={(sid, pid) => assignSession(sid, pid)}
                          onTogglePin={toggleSessionPin}
                          projects={projects}
                        />
                      </motion.div>
                    ))}
                  </div>
                </div>
              ))}
            </motion.div>
          </AnimatePresence>
        )}
      </nav>
    </div>
  );
}
