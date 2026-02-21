import { CatalogBrowser } from '../components/data/CatalogBrowser';
import { DataTable } from '../components/data/DataTable';
import { DataPreview } from '../components/data/DataPreview';
import { MemoryManager } from '../components/data/MemoryManager';
import { useSessionStore } from '../stores/sessionStore';

export function DataToolsPage() {
  const { activeSessionId } = useSessionStore();

  if (!activeSessionId) {
    return (
      <div className="flex-1 flex items-center justify-center text-text-muted bg-surface">
        <p>No active session. Start a chat first.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 bg-surface">
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left column */}
        <div className="space-y-4">
          <CatalogBrowser sessionId={activeSessionId} />
          <MemoryManager sessionId={activeSessionId} />
        </div>

        {/* Right column */}
        <div className="space-y-4">
          <DataTable sessionId={activeSessionId} />
          <DataPreview sessionId={activeSessionId} />
        </div>
      </div>
    </div>
  );
}
