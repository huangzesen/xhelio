import { useState } from 'react';
import { RotateCcw, ListTree } from 'lucide-react';
import { ReplayTab } from '../components/pipeline/ReplayTab';
import { SavedPipelinesTab } from '../components/pipeline/SavedPipelinesTab';

type Tab = 'replay' | 'pipelines';

export function PipelinePage() {
  const [activeTab, setActiveTab] = useState<Tab>('replay');

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header: title + tab bar */}
      <div className="shrink-0 px-4 pt-4 bg-surface">
        <h1 className="text-xl font-semibold text-text mb-2">Pipeline</h1>
        <div className="flex border-b border-border">
          <button
            onClick={() => setActiveTab('replay')}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === 'replay'
                ? 'text-text border-b-2 border-accent'
                : 'text-text-muted hover:text-text'
            }`}
          >
            <RotateCcw size={14} />
            Replay
          </button>
          <button
            onClick={() => setActiveTab('pipelines')}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === 'pipelines'
                ? 'text-text border-b-2 border-accent'
                : 'text-text-muted hover:text-text'
            }`}
          >
            <ListTree size={14} />
            Pipelines
          </button>
        </div>
      </div>

      {/* Tab content — fills remaining height */}
      {activeTab === 'replay' && <ReplayTab />}
      {activeTab === 'pipelines' && <SavedPipelinesTab />}
    </div>
  );
}
