import { useState, useEffect } from 'react';
import { User, Bot, Brain, Terminal, ChevronRight, ChevronDown, Eye, CheckCircle, AlertTriangle } from 'lucide-react';
import { MarkdownRenderer } from '../common/MarkdownRenderer';
import { MessageActions } from './MessageActions';
import type { ChatMessage as ChatMessageType } from '../../api/types';

interface Props {
  message: ChatMessageType;
  isQueued?: boolean;
  onRegenerate?: () => void;
}

const relativeTimeFormatter = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });

function getRelativeTimeString(timestamp: number): string {
  const now = Date.now();
  const diffInSeconds = Math.floor((timestamp - now) / 1000);
  const absDiff = Math.abs(diffInSeconds);

  if (absDiff < 60) return 'just now';
  
  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (Math.abs(diffInMinutes) < 60) return relativeTimeFormatter.format(diffInMinutes, 'minute');
  
  const diffInHours = Math.floor(diffInMinutes / 60);
  if (Math.abs(diffInHours) < 24) return relativeTimeFormatter.format(diffInHours, 'hour');
  
  const diffInDays = Math.floor(diffInHours / 24);
  return relativeTimeFormatter.format(diffInDays, 'day');
}

function formatAbsoluteTime(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();
  
  return new Intl.DateTimeFormat('en', {
    month: isToday ? undefined : 'short',
    day: isToday ? undefined : 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);
}

export function ChatMessage({ message, isQueued, onRegenerate }: Props) {
  const isUser = message.role === 'user';
  const isThinking = message.role === 'thinking';
  const [expanded, setExpanded] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [relativeTime, setRelativeTime] = useState(getRelativeTimeString(message.timestamp));

  useEffect(() => {
    if (!isHovered) return;
    const interval = setInterval(() => {
      setRelativeTime(getRelativeTimeString(message.timestamp));
    }, 30000);
    return () => clearInterval(interval);
  }, [isHovered, message.timestamp]);

  const handleMouseEnter = () => {
    setIsHovered(true);
    setRelativeTime(getRelativeTimeString(message.timestamp));
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
  };

  const timestampEl = (
    <div 
      className={`text-[10px] text-text-muted transition-opacity duration-200 select-none pointer-events-none mb-0.5
        ${isHovered ? 'opacity-100' : 'opacity-0'} 
        ${isUser ? 'text-left' : 'text-right'}`}
    >
      {relativeTime} • {formatAbsoluteTime(message.timestamp)}
    </div>
  );

  if (isThinking) {
    return (
      <div 
        className="flex gap-3"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <div className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 bg-badge-purple-bg text-badge-purple-text">
          <Brain size={14} />
        </div>
        <div className="max-w-[80%] min-w-0 flex flex-col">
          {timestampEl}
          <button
            onClick={() => setExpanded((v) => !v)}
            className="flex items-center gap-1 text-xs text-badge-purple-text hover:opacity-80 transition-colors py-1"
          >
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            Agent thinking
          </button>
          {expanded && (
            <div className="rounded-lg bg-badge-purple-bg border border-border px-3 py-2 text-xs text-badge-purple-text mt-1 min-w-0 overflow-hidden wrap-anywhere">
              <MarkdownRenderer content={message.content} />
            </div>
          )}
        </div>
      </div>
    );
  }

  if (message.role === 'system') {
    return (
      <div 
        data-testid="message-system" 
        className="flex gap-3"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <div className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 bg-surface-elevated text-text-muted">
          <Terminal size={14} />
        </div>
        <div className="max-w-[80%] min-w-0 flex flex-col">
          {timestampEl}
          <div className="rounded-xl px-4 py-2.5 text-sm bg-surface-elevated border border-border text-text-muted rounded-bl-sm overflow-hidden wrap-anywhere">
            <MarkdownRenderer content={message.content} />
          </div>
        </div>
      </div>
    );
  }

  if (message.role === 'insight_feedback') {
    const passed = message.content.includes('VERDICT: PASS');
    return (
      <div 
        className="flex gap-3"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 ${passed ? 'bg-green-500/10 text-green-600 dark:text-green-400' : 'bg-amber-500/10 text-amber-600 dark:text-amber-400'}`}>
          {passed ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
        </div>
        <div className="max-w-[80%] min-w-0 flex flex-col">
          {timestampEl}
          <div className={`text-xs font-medium mb-1 ${passed ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'}`}>
            Figure Review — {passed ? 'Pass' : 'Needs Improvement'}
          </div>
          <div className={`rounded-xl px-4 py-2.5 text-sm bg-panel border text-text rounded-bl-sm overflow-hidden wrap-anywhere ${passed ? 'border-green-500/20' : 'border-amber-500/20'}`}>
            <MarkdownRenderer content={message.content} />
          </div>
          {message.content && (
            <div className="mt-1 ml-1">
              <MessageActions content={message.content} />
            </div>
          )}
        </div>
      </div>
    );
  }

  if (message.role === 'insight') {
    return (
      <div 
        className="flex gap-3"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <div className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 bg-badge-purple-bg text-badge-purple-text">
          <Eye size={14} />
        </div>
        <div className="max-w-[80%] min-w-0 flex flex-col">
          {timestampEl}
          <div className="text-xs font-medium text-badge-purple-text mb-1">Plot Analysis</div>
          <div className="rounded-xl px-4 py-2.5 text-sm bg-panel border border-badge-purple-text/20 text-text rounded-bl-sm overflow-hidden wrap-anywhere">
            <MarkdownRenderer content={message.content} />
          </div>
          {message.content && (
            <div className="mt-1 ml-1">
              <MessageActions content={message.content} />
            </div>
          )}
        </div>
      </div>
    );
  }

  const isError = !isUser && message.content.startsWith('Error:');

  return (
    <div 
      data-testid={`message-${message.role}`} 
      className={`group flex gap-3 ${isUser ? 'flex-row-reverse' : ''} ${isQueued ? 'opacity-60' : ''}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Avatar */}
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0
          ${isUser ? 'bg-primary text-white' : 'bg-surface-elevated text-text-muted'}`}
      >
        {isUser ? <User size={14} /> : <Bot size={14} />}
      </div>

      {/* Content */}
      <div className="max-w-[80%] min-w-0 flex flex-col">
        {timestampEl}
        <div
          className={`rounded-xl px-4 py-2.5 text-sm overflow-hidden wrap-anywhere border-l-2
            ${isUser
              ? 'bg-primary text-white rounded-br-sm border-l-primary'
              : 'bg-panel border border-border text-text rounded-bl-sm border-l-primary/40'}`}
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

