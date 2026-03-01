import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Project {
  id: string;
  name: string;
  color: string;
  pinned: boolean;
  created_at: string;
  updated_at: string;
}

interface ProjectState {
  projects: Project[];
  activeProjectId: string | null;
  sessionProjectMap: Record<string, string>;
  pinnedSessionIds: string[];

  createProject: (name: string, color?: string) => Project;
  renameProject: (id: string, name: string) => void;
  deleteProject: (id: string) => void;
  setProjectColor: (id: string, color: string) => void;
  togglePin: (id: string) => void;
  toggleSessionPin: (sessionId: string) => void;
  assignSession: (sessionId: string, projectId: string | null) => void;
  setActiveProject: (id: string | null) => void;
  pruneOrphanedSessions: (validSessionIds: string[]) => void;
}

export const PRESET_COLORS = [
  '#EAB308', // Amber
  '#F97316', // Orange
  '#EF4444', // Red
  '#8B5CF6', // Violet
  '#3B82F6', // Blue
  '#10B981', // Emerald
];

export const useProjectStore = create<ProjectState>()(
  persist(
    (set) => ({
      projects: [],
      activeProjectId: null,
      sessionProjectMap: {},
      pinnedSessionIds: [],

      createProject: (name, color) => {
        const newProject: Project = {
          id: crypto.randomUUID(),
          name,
          color: color || PRESET_COLORS[0],
          pinned: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
        set((state) => ({
          projects: [...state.projects, newProject],
        }));
        return newProject;
      },

      renameProject: (id, name) => {
        set((state) => ({
          projects: state.projects.map((p) =>
            p.id === id ? { ...p, name, updated_at: new Date().toISOString() } : p
          ),
        }));
      },

      deleteProject: (id) => {
        set((state) => ({
          projects: state.projects.filter((p) => p.id !== id),
          // Unassign sessions from the deleted project
          sessionProjectMap: Object.entries(state.sessionProjectMap).reduce(
            (acc, [sessionId, projectId]) => {
              if (projectId !== id) {
                acc[sessionId] = projectId;
              }
              return acc;
            },
            {} as Record<string, string>
          ),
          activeProjectId: state.activeProjectId === id ? null : state.activeProjectId,
        }));
      },

      setProjectColor: (id, color) => {
        set((state) => ({
          projects: state.projects.map((p) =>
            p.id === id ? { ...p, color, updated_at: new Date().toISOString() } : p
          ),
        }));
      },

      togglePin: (id) => {
        set((state) => ({
          projects: state.projects.map((p) =>
            p.id === id ? { ...p, pinned: !p.pinned, updated_at: new Date().toISOString() } : p
          ),
        }));
      },

      toggleSessionPin: (sessionId) => {
        set((state) => ({
          pinnedSessionIds: state.pinnedSessionIds.includes(sessionId)
            ? state.pinnedSessionIds.filter((id) => id !== sessionId)
            : [...state.pinnedSessionIds, sessionId],
        }));
      },

      assignSession: (sessionId, projectId) => {
        set((state) => {
          const newMap = { ...state.sessionProjectMap };
          if (projectId) {
            newMap[sessionId] = projectId;
          } else {
            delete newMap[sessionId];
          }
          return { sessionProjectMap: newMap };
        });
      },

      setActiveProject: (id) => {
        set({ activeProjectId: id });
      },

      pruneOrphanedSessions: (validSessionIds) => {
        set((state) => {
          const newMap = { ...state.sessionProjectMap };
          const validSet = new Set(validSessionIds);
          let changed = false;
          for (const sessionId in newMap) {
            if (!validSet.has(sessionId)) {
              delete newMap[sessionId];
              changed = true;
            }
          }
          const newPinned = state.pinnedSessionIds.filter(id => validSet.has(id));
          if (newPinned.length !== state.pinnedSessionIds.length) {
            changed = true;
          }
          return changed ? { sessionProjectMap: newMap, pinnedSessionIds: newPinned } : state;
        });
      },
    }),
    {
      name: 'xhelio-projects',
    }
  )
);
