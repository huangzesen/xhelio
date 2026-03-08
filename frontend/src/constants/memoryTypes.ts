export const MEMORY_TYPES = ['preference', 'pitfall', 'summary', 'reflection'] as const;
export type MemoryType = (typeof MEMORY_TYPES)[number];

export const MEMORY_TYPE_COLORS: Record<string, { border: string; badge: string; hex: string }> = {
  preference: {
    border: 'border-l-[color:var(--badge-blue-bg,#3b82f6)]',
    badge: 'bg-badge-blue-bg text-badge-blue-text',
    hex: '#3b82f6',
  },
  pitfall: {
    border: 'border-l-[color:var(--badge-red-bg,#ef4444)]',
    badge: 'bg-badge-red-bg text-badge-red-text',
    hex: '#ef4444',
  },
  summary: {
    border: 'border-l-[color:var(--badge-green-bg,#22c55e)]',
    badge: 'bg-badge-green-bg text-badge-green-text',
    hex: '#22c55e',
  },
  reflection: {
    border: 'border-l-[color:var(--badge-purple-bg,#a855f7)]',
    badge: 'bg-badge-purple-bg text-badge-purple-text',
    hex: '#a855f7',
  },
};
