import { useState } from 'react';
import { User, Bot, Brain, Terminal, ChevronRight, ChevronDown } from 'lucide-react';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import { MessageActions } from './MessageActions';
import type { ChatMessage as ChatMessageType } from '../../api/types';

interface Props {
  message: ChatMessageType;
  onRegenerate?: () => void;
}

export function ChatMessage({ message, onRegenerate }: Props) {
  const isUser = message.role === 'user';
  const isThinking = message.role === 'thinking';
  const [expanded, setExpanded] = useState(false);

  if (isThinking) {
    return (
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-badge-purple-bg text-badge-purple-text">
          <Brain size={16} />
        </div>
        <div className="max-w-[80%] min-w-0">
          <button
            onClick={() => setExpanded((v) => !v)}
            className="flex items-center gap-1 text-xs text-badge-purple-text hover:opacity-80 transition-colors py-1"
          >
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            Agent thinking
          </button>
          {expanded && (
            <div className="rounded-lg bg-badge-purple-bg border border-border px-3 py-2 text-xs text-badge-purple-text mt-1 min-w-0 overflow-hidden">
              <MarkdownRenderer content={message.content} />
            </div>
          )}
        </div>
      </div>
    );
  }

  if (message.role === 'system') {
    return (
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-surface-elevated text-text-muted">
          <Terminal size={16} />
        </div>
        <div className="max-w-[80%] min-w-0">
          <div className="rounded-xl px-4 py-2.5 text-sm bg-surface-elevated border border-border text-text-muted rounded-bl-sm overflow-hidden">
            <MarkdownRenderer content={message.content} />
          </div>
        </div>
      </div>
    );
  }

  const isError = !isUser && message.content.startsWith('Error:');

  return (
    <div className={`group flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0
          ${isUser ? 'bg-primary text-white' : 'bg-surface-elevated text-text-muted'}`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Content */}
      <div className="max-w-[80%] min-w-0 flex flex-col">
        <div
          className={`rounded-xl px-4 py-2.5 text-sm overflow-hidden
            ${isUser
              ? 'bg-primary text-white rounded-br-sm'
              : 'bg-panel border border-border text-text rounded-bl-sm'}`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <MarkdownRenderer content={message.content} />
          )}
        </div>
        {/* Actions */}
        {!isUser && message.content && (
          <div className="mt-1 ml-1">
            <MessageActions
              content={message.content}
              isError={isError}
              onRegenerate={onRegenerate}
            />
          </div>
        )}
      </div>
    </div>
  );
}
