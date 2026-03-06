import { useEffect, useMemo, useState, useRef } from 'react';
import { useEurekaStore } from '../stores/eurekaStore';
import { useSessionStore } from '../stores/sessionStore';
import { Loader2, Send, AlertCircle, X, Sparkles } from 'lucide-react';

const STATUS_OPTIONS = ['proposed', 'reviewed', 'confirmed', 'rejected'] as const;

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    value < 0.4 ? 'bg-gray-400' : value < 0.7 ? 'bg-blue-500' : 'bg-green-500';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-text-muted w-10">{value.toFixed(1)}</span>
    </div>
  );
}

function EurekaCard({
  eureka,
  onStatusChange,
}: {
  eureka: import('../api/types').EurekaEntry;
  onStatusChange: (id: string, status: string) => void;
}) {
  const statusColor = {
    proposed: 'bg-yellow-500/20 text-yellow-400',
    reviewed: 'bg-blue-500/20 text-blue-400',
    confirmed: 'bg-green-500/20 text-green-400',
    rejected: 'bg-red-500/20 text-red-400',
  };

  return (
    <div className="bg-panel rounded-xl border border-border p-4 space-y-3">
      <h3 className="text-base font-medium text-text">{eureka.title}</h3>

      <div className="space-y-2 text-sm">
        <div>
          <span className="text-text-muted">Observation: </span>
          <span className="text-text">{eureka.observation}</span>
        </div>
        <div>
          <span className="text-text-muted">Hypothesis: </span>
          <span className="text-text">{eureka.hypothesis}</span>
        </div>
        {eureka.evidence.length > 0 && (
          <div>
            <span className="text-text-muted">Evidence:</span>
            <ul className="mt-1 ml-4 list-disc text-text">
              {eureka.evidence.map((ev, i) => (
                <li key={i}>{ev}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <ConfidenceBar value={eureka.confidence} />

      <div className="flex items-center gap-2 flex-wrap">
        {eureka.tags.map((tag) => (
          <span
            key={tag}
            className="px-2 py-0.5 text-xs bg-surface rounded-full text-text-muted"
          >
            {tag}
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between text-xs text-text-muted">
        <span>
          Session:{' '}
          {new Date(eureka.timestamp).toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
        <select
          value={eureka.status}
          onChange={(e) => onStatusChange(eureka.id, e.target.value)}
          className={`px-2 py-1 rounded text-xs border border-border bg-surface cursor-pointer ${statusColor[eureka.status]}`}
        >
          {STATUS_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

function ChatPanel() {
  const { chatMessages, chatLoading, chatError, sendChatMessage, clearChatError, fetchChatHistory } = useEurekaStore();
  const { sessionId } = useSessionStore();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchChatHistory();
  }, [fetchChatHistory]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || chatLoading || !sessionId) return;
    const message = input.trim();
    setInput('');
    await sendChatMessage(message);
  };

  if (!sessionId) {
    return (
      <div className="flex-1 flex items-center justify-center text-text-muted text-sm">
        Start a session to chat with Eureka
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chatMessages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-2">
            <Sparkles size={32} className="text-text-muted" />
            <p className="text-text-muted text-sm">
              Chat with Eureka about your data analysis session
            </p>
            <p className="text-text-muted text-xs">
              Ask about patterns, anomalies, or request insights
            </p>
          </div>
        ) : (
          chatMessages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-xl px-4 py-2 ${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-panel border border-border text-text'
                }`}
              >
                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))
        )}
        {chatLoading && (
          <div className="flex justify-start">
            <div className="bg-panel border border-border rounded-xl px-4 py-2">
              <Loader2 size={16} className="animate-spin text-text-muted" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {chatError && (
        <div className="mx-4 mb-2 p-2 bg-red-500/20 border border-red-500/50 rounded-lg flex items-center gap-2">
          <AlertCircle size={14} className="text-red-400" />
          <span className="text-xs text-red-400 flex-1">{chatError}</span>
          <button onClick={clearChatError} className="text-red-400 hover:text-red-300">
            <X size={14} />
          </button>
        </div>
      )}

      <form onSubmit={handleSubmit} className="p-4 border-t border-border">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask Eureka..."
            disabled={chatLoading || !sessionId}
            className="flex-1 bg-panel border border-border rounded-lg px-4 py-2 text-sm text-text placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || chatLoading || !sessionId}
            className="p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-surface disabled:text-text-muted rounded-lg transition-colors"
          >
            <Send size={18} />
          </button>
        </div>
      </form>
    </div>
  );
}

export function EurekaPage() {
  const { loading, filters, fetchEurekas, updateStatus, setFilter, filteredEurekas } =
    useEurekaStore();
  const { sessionId } = useSessionStore();
  const [activeTab, setActiveTab] = useState<'chat' | 'findings'>('chat');

  const displayed = useMemo(() => filteredEurekas(), [filteredEurekas]);

  useEffect(() => {
    fetchEurekas();
  }, [fetchEurekas]);

  const allTags = useMemo(() => {
    const store = useEurekaStore.getState();
    const tags = new Set<string>();
    store.eurekas.forEach((e) => e.tags.forEach((t) => tags.add(t)));
    return Array.from(tags).sort();
  }, []);

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-surface">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <h1 className="text-xl font-semibold text-text">Eureka</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-3 py-1.5 rounded-lg text-sm ${
              activeTab === 'chat'
                ? 'bg-blue-600 text-white'
                : 'bg-panel text-text-muted hover:text-text'
            }`}
          >
            Chat
          </button>
          <button
            onClick={() => setActiveTab('findings')}
            className={`px-3 py-1.5 rounded-lg text-sm ${
              activeTab === 'findings'
                ? 'bg-blue-600 text-white'
                : 'bg-panel text-text-muted hover:text-text'
            }`}
          >
            Findings ({displayed.length})
          </button>
        </div>
      </div>

      {activeTab === 'chat' && (
        <div className="flex-1 flex flex-col border-r border-border">
          <div className="p-4 border-b border-border">
            <h2 className="text-lg font-semibold text-text flex items-center gap-2">
              <Sparkles size={20} className="text-blue-400" />
              Assistant
            </h2>
            <p className="text-xs text-text-muted mt-1">
              Chat with Eureka about your data analysis
            </p>
          </div>
          <ChatPanel />
        </div>
      )}

      {activeTab === 'findings' && (
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <select
              value={filters.status || ''}
              onChange={(e) => setFilter('status', e.target.value || undefined)}
              className="px-3 py-1.5 rounded-lg border border-border bg-panel text-sm text-text"
            >
              <option value="">All Status</option>
              {STATUS_OPTIONS.map((s) => (
                <option key={s} value={s}>
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </option>
              ))}
            </select>
            <select
              value={filters.tag || ''}
              onChange={(e) => setFilter('tag', e.target.value || undefined)}
              className="px-3 py-1.5 rounded-lg border border-border bg-panel text-sm text-text"
            >
              <option value="">All Tags</option>
              {allTags.map((t: string) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 size={20} className="animate-spin text-text-muted" />
              </div>
            ) : displayed.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <p className="text-text-muted text-sm">
                  No eurekas yet. Start a session and explore data to generate discoveries.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {displayed.map((eureka) => (
                  <EurekaCard
                    key={eureka.id}
                    eureka={eureka}
                    onStatusChange={updateStatus}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
