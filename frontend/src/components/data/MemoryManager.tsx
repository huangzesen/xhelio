import { useEffect, useState } from 'react';
import { useMemoryStore } from '../../stores/memoryStore';
import { Brain, Trash2, RefreshCw } from 'lucide-react';

interface Props {
  sessionId: string;
}

export function MemoryManager({ sessionId }: Props) {
  const { memories, globalEnabled, error, loadMemories, deleteMemory, toggleGlobal, clearAll, refresh } =
    useMemoryStore();

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadMemories(sessionId);
  }, [sessionId, loadMemories]);

  const handleToggle = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleDeleteSelected = async () => {
    for (const id of selectedIds) {
      await deleteMemory(sessionId, id);
    }
    setSelectedIds(new Set());
  };

  const typeColor: Record<string, string> = {
    preference: 'bg-badge-blue-bg text-badge-blue-text',
    pitfall: 'bg-badge-red-bg text-badge-red-text',
    summary: 'bg-badge-green-bg text-badge-green-text',
    reflection: 'bg-badge-purple-bg text-badge-purple-text',
  };

  return (
    <div className="bg-panel rounded-xl border border-border p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Brain size={16} className="text-primary" />
          <h2 className="font-medium text-text">Memories</h2>
          <span className="text-xs text-text-muted">({memories.length})</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => refresh(sessionId)}
            className="p-1 rounded hover:bg-hover-bg transition-colors"
            title="Refresh"
          >
            <RefreshCw size={14} className="text-text-muted" />
          </button>
        </div>
      </div>

      {/* Global toggle */}
      <label className="flex items-center gap-2 text-xs text-text-muted mb-3 cursor-pointer">
        <input
          type="checkbox"
          checked={globalEnabled}
          onChange={(e) => toggleGlobal(sessionId, e.target.checked)}
          className="rounded"
        />
        Memory system enabled
      </label>

      {/* Memory list */}
      {memories.length === 0 ? (
        <div className="text-xs text-text-muted text-center py-4">No memories stored</div>
      ) : (
        <div className="space-y-1.5 max-h-64 overflow-y-auto">
          {memories.map((m) => (
            <div key={m.id} className="flex items-start gap-2 text-xs border border-border/50 rounded-lg p-2">
              <input
                type="checkbox"
                checked={selectedIds.has(m.id)}
                onChange={() => handleToggle(m.id)}
                className="mt-0.5 rounded"
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 mb-0.5">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${typeColor[m.type] ?? 'bg-badge-gray-bg text-badge-gray-text'}`}>
                    {m.type}
                  </span>
                  <span className="text-text-muted">{m.scopes.join(', ')}</span>
                </div>
                <div className="text-text truncate">{m.content}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Actions */}
      {memories.length > 0 && (
        <div className="flex gap-2 mt-3">
          <button
            onClick={handleDeleteSelected}
            disabled={selectedIds.size === 0}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs border border-border
              hover:bg-hover-bg transition-colors disabled:opacity-40"
          >
            <Trash2 size={12} />
            Delete ({selectedIds.size})
          </button>
          <button
            onClick={() => { clearAll(sessionId); setSelectedIds(new Set()); }}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs border border-status-error-border
              text-status-error-text hover:bg-status-error-bg transition-colors"
          >
            Clear All
          </button>
        </div>
      )}

      {error && <div className="text-xs text-status-error-text mt-2">{error}</div>}
    </div>
  );
}
