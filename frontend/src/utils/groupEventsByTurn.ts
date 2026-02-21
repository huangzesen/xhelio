import type { ChatMessage, ToolEvent, MemoryEvent } from '../api/types';

/** Lightweight thinking event extracted from messages with role='thinking'. */
export interface ThinkingEvent {
  id: string;
  content: string;
  timestamp: number;
}

export interface TurnGroup {
  turnIndex: number;
  userMessage: ChatMessage;
  toolEvents: ToolEvent[];
  memoryEvents: MemoryEvent[];
  thinkingEvents: ThinkingEvent[];
  startTs: number;
  endTs: number; // next user msg timestamp, or Infinity for last turn
}

/**
 * Extract thinking events from the messages array.
 * Thinking messages have role='thinking' with content and timestamp.
 */
export function extractThinkingEvents(messages: ChatMessage[]): ThinkingEvent[] {
  return messages
    .filter((m) => m.role === 'thinking')
    .map((m) => ({ id: m.id, content: m.content, timestamp: m.timestamp }));
}

/**
 * Group tool events, memory events, and thinking events by interaction turn.
 * Each turn spans from one user message timestamp to the next.
 */
export function groupEventsByTurn(
  messages: ChatMessage[],
  toolEvents: ToolEvent[],
  memoryEvents: MemoryEvent[],
): TurnGroup[] {
  const userMessages = messages.filter((m) => m.role === 'user');
  if (userMessages.length === 0) return [];

  const thinkingEvents = extractThinkingEvents(messages);

  const groups: TurnGroup[] = userMessages.map((msg, i) => ({
    turnIndex: i,
    userMessage: msg,
    toolEvents: [],
    memoryEvents: [],
    thinkingEvents: [],
    startTs: msg.timestamp,
    endTs: i < userMessages.length - 1 ? userMessages[i + 1].timestamp : Infinity,
  }));

  for (const evt of toolEvents) {
    for (let i = groups.length - 1; i >= 0; i--) {
      if (evt.timestamp >= groups[i].startTs) {
        groups[i].toolEvents.push(evt);
        break;
      }
    }
  }

  for (const evt of memoryEvents) {
    for (let i = groups.length - 1; i >= 0; i--) {
      if (evt.timestamp >= groups[i].startTs) {
        groups[i].memoryEvents.push(evt);
        break;
      }
    }
  }

  for (const evt of thinkingEvents) {
    for (let i = groups.length - 1; i >= 0; i--) {
      if (evt.timestamp >= groups[i].startTs) {
        groups[i].thinkingEvents.push(evt);
        break;
      }
    }
  }

  return groups;
}
