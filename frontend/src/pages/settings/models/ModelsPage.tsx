import { useState } from 'react';
import { OverviewTab } from './OverviewTab';
import { CustomizeTab } from './CustomizeTab';

type Tab = 'overview' | 'customize';

export function ModelsPage() {
  const [activeTab, setActiveTab] = useState<Tab>('overview');

  const tabs: { id: Tab; label: string }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'customize', label: 'Customize' },
  ];

  return (
    <div className="space-y-4 pt-4 max-w-4xl">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-text">Models</h2>
        <p className="text-sm text-text-muted mt-1">
          Configure your AI model setup — pick a provider preset or fine-tune individual agents.
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
              activeTab === tab.id
                ? 'border-primary text-primary'
                : 'border-transparent text-text-muted hover:text-text'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'overview' ? (
        <OverviewTab
          onSwitchToCustomize={() => setActiveTab('customize')}
        />
      ) : (
        <CustomizeTab />
      )}
    </div>
  );
}
