import type { ChatMessage, ToolEvent, MemoryEvent, CommentaryEvent } from '../api/types';
import type { RoundMarker } from '../stores/chatStore';

/** Lightweight thinking event extracted from messages with role='thinking'. */
export interface ThinkingEvent {
  id: string;
  content: string;
  timestamp: number;
}

export interface RoundGroup {
  roundIndex: number;
  userMessages: ChatMessage[];       // all user messages consumed in this round
  toolEvents: ToolEvent[];
  memoryEvents: MemoryEvent[];
  thinkingEvents: ThinkingEvent[];
  commentaryEvents: CommentaryEvent[];
  startTs: number;                   // round_start timestamp (or first user msg)
  endTs: number;                     // round_end timestamp (or Infinity if active)
  roundTokenUsage?: Record<string, number>;  // per-round delta from round_end
}

/** @deprecated Use RoundGroup instead */
export type TurnGroup = RoundGroup & { turnIndex: number; userMessage: ChatMessage };

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
 * Group events by orchestrator round boundaries (round_start/round_end markers).
 *
 * Each round spans from a round_start marker to the corresponding round_end.
 * A round can contain multiple user messages (if they were batched/queued).
 *
 * Falls back to user-message-based grouping when no round markers are present
 * (backward compat with old sessions that used turn_done).
 */
export function groupEventsByRound(
  messages: ChatMessage[],
  toolEvents: ToolEvent[],
  memoryEvents: MemoryEvent[],
  roundMarkers: RoundMarker[] = [],
  commentaryEvents: CommentaryEvent[] = [],
): RoundGroup[] {
  const thinkingEvents = extractThinkingEvents(messages);
  const userMessages = messages.filter((m) => m.role === 'user');

  // If we have round markers, use them to define boundaries
  if (roundMarkers.length > 0) {
    return _groupByMarkers(userMessages, toolEvents, memoryEvents, thinkingEvents, roundMarkers, commentaryEvents);
  }

  // Fallback: group by user message timestamps (old behavior)
  return _groupByUserMessages(userMessages, toolEvents, memoryEvents, thinkingEvents, commentaryEvents);
}

/**
 * @deprecated Use groupEventsByRound instead.
 * Kept for backward compat — delegates to groupEventsByRound with empty markers,
 * which falls back to user-message-based grouping.
 */
export function groupEventsByTurn(
  messages: ChatMessage[],
  toolEvents: ToolEvent[],
  memoryEvents: MemoryEvent[],
): RoundGroup[] {
  return groupEventsByRound(messages, toolEvents, memoryEvents, []);
}

// ---- Internal helpers ----

function _groupByMarkers(
  userMessages: ChatMessage[],
  toolEvents: ToolEvent[],
  memoryEvents: MemoryEvent[],
  thinkingEvents: ThinkingEvent[],
  roundMarkers: RoundMarker[],
  commentaryEvents: CommentaryEvent[] = [],
): RoundGroup[] {
  // Build round intervals from start/end marker pairs
  const rounds: RoundGroup[] = [];
  let roundIdx = 0;
  let startIdx = 0;

  // For old sessions with only 'end' markers (from turn_done backward compat),
  // synthesize start markers: infer round_start as the timestamp of the first
  // event after the previous round_end.
  const hasStarts = roundMarkers.some((m) => m.type === 'start');

  if (hasStarts) {
    // Normal path: pair start/end markers
    while (startIdx < roundMarkers.length) {
      const startMarker = roundMarkers[startIdx];
      if (startMarker.type !== 'start') {
        startIdx++;
        continue;
      }
      // Find matching end, but stop if we hit another start first
      let endMarker: RoundMarker | null = null;
      let nextStartMarker: RoundMarker | null = null;
      for (let j = startIdx + 1; j < roundMarkers.length; j++) {
        if (roundMarkers[j].type === 'end') {
          endMarker = roundMarkers[j];
          startIdx = j + 1;
          break;
        }
        if (roundMarkers[j].type === 'start') {
          // Next round started before this one ended
          nextStartMarker = roundMarkers[j];
          startIdx = j; // will be picked up as next iteration's start
          break;
        }
      }

      let endTs: number;
      let roundTokenUsage: Record<string, number> | undefined;
      if (endMarker) {
        endTs = endMarker.timestamp;
        roundTokenUsage = endMarker.roundTokenUsage;
      } else if (nextStartMarker) {
        // Round ended implicitly when the next round started
        endTs = nextStartMarker.timestamp;
      } else {
        // Last round with no end yet — active round
        endTs = Infinity;
        startIdx = roundMarkers.length; // exit loop
      }

      rounds.push({
        roundIndex: roundIdx++,
        userMessages: [],
        toolEvents: [],
        memoryEvents: [],
        thinkingEvents: [],
        commentaryEvents: [],
        startTs: startMarker.timestamp,
        endTs,
        roundTokenUsage,
      });
    }
  } else {
    // Backward compat: only 'end' markers (from old turn_done events).
    // Infer the first round's start from the earliest event/message timestamp
    // instead of 0, which would produce an elapsed time of ~56 years.
    const earliestEventTs = Math.min(
      ...[
        ...userMessages.map((m) => m.timestamp),
        ...toolEvents.map((e) => e.timestamp),
        ...thinkingEvents.map((e) => e.timestamp),
      ].filter((t) => t > 0),
    );
    let prevEnd = isFinite(earliestEventTs) ? earliestEventTs : undefined;
    for (const marker of roundMarkers) {
      if (marker.type === 'end') {
        rounds.push({
          roundIndex: roundIdx++,
          userMessages: [],
          toolEvents: [],
          memoryEvents: [],
          thinkingEvents: [],
          commentaryEvents: [],
          startTs: prevEnd ?? marker.timestamp,
          endTs: marker.timestamp,
          roundTokenUsage: marker.roundTokenUsage,
        });
        prevEnd = marker.timestamp;
      }
    }
  }

  // If there are events after the last round_end (or no rounds at all),
  // but there are user messages, create an implicit active round
  const lastEndTs = rounds.length > 0 ? rounds[rounds.length - 1].endTs : 0;
  const hasEventsAfter = toolEvents.some((e) => e.timestamp > lastEndTs)
    || thinkingEvents.some((e) => e.timestamp > lastEndTs)
    || userMessages.some((m) => m.timestamp > lastEndTs);

  if (hasEventsAfter && (rounds.length === 0 || rounds[rounds.length - 1].endTs !== Infinity)) {
    // Use the earliest post-lastEnd user message timestamp as the round start
    // so the timer counts from when the user actually sent their message,
    // not from the previous round's end.
    const postEndUserTs = userMessages
      .filter((m) => m.timestamp > lastEndTs)
      .map((m) => m.timestamp);
    const implicitStartTs = postEndUserTs.length > 0
      ? Math.min(...postEndUserTs)
      : (lastEndTs || Date.now());

    rounds.push({
      roundIndex: roundIdx,
      userMessages: [],
      toolEvents: [],
      memoryEvents: [],
      thinkingEvents: [],
      commentaryEvents: [],
      startTs: implicitStartTs,
      endTs: Infinity,
    });
  }

  // Assign events to rounds by timestamp
  _assignToRounds(rounds, userMessages, toolEvents, memoryEvents, thinkingEvents, commentaryEvents);

  return rounds;
}

function _groupByUserMessages(
  userMessages: ChatMessage[],
  toolEvents: ToolEvent[],
  memoryEvents: MemoryEvent[],
  thinkingEvents: ThinkingEvent[],
  commentaryEvents: CommentaryEvent[] = [],
): RoundGroup[] {
  if (userMessages.length === 0) return [];

  const rounds: RoundGroup[] = userMessages.map((msg, i) => ({
    roundIndex: i,
    userMessages: [msg],
    toolEvents: [],
    memoryEvents: [],
    thinkingEvents: [],
    commentaryEvents: [],
    startTs: msg.timestamp,
    endTs: i < userMessages.length - 1 ? userMessages[i + 1].timestamp : Infinity,
  }));

  // Assign tool/memory/thinking/commentary events
  for (const evt of toolEvents) {
    for (let i = rounds.length - 1; i >= 0; i--) {
      if (evt.timestamp >= rounds[i].startTs) {
        rounds[i].toolEvents.push(evt);
        break;
      }
    }
  }
  for (const evt of memoryEvents) {
    for (let i = rounds.length - 1; i >= 0; i--) {
      if (evt.timestamp >= rounds[i].startTs) {
        rounds[i].memoryEvents.push(evt);
        break;
      }
    }
  }
  for (const evt of thinkingEvents) {
    for (let i = rounds.length - 1; i >= 0; i--) {
      if (evt.timestamp >= rounds[i].startTs) {
        rounds[i].thinkingEvents.push(evt);
        break;
      }
    }
  }
  for (const evt of commentaryEvents) {
    for (let i = rounds.length - 1; i >= 0; i--) {
      if (evt.timestamp >= rounds[i].startTs) {
        rounds[i].commentaryEvents.push(evt);
        break;
      }
    }
  }

  return rounds;
}

/** Assign user messages and events to rounds by timestamp range.
 *  If an event precedes all rounds, assign it to the first round. */
function _assignToRounds(
  rounds: RoundGroup[],
  userMessages: ChatMessage[],
  toolEvents: ToolEvent[],
  memoryEvents: MemoryEvent[],
  thinkingEvents: ThinkingEvent[],
  commentaryEvents: CommentaryEvent[] = [],
): void {
  if (rounds.length === 0) return;
  for (const msg of userMessages) {
    let assigned = false;
    for (let i = rounds.length - 1; i >= 0; i--) {
      if (msg.timestamp >= rounds[i].startTs) {
        rounds[i].userMessages.push(msg);
        assigned = true;
        break;
      }
    }
    if (!assigned) rounds[0].userMessages.push(msg);
  }
  for (const evt of toolEvents) {
    let assigned = false;
    for (let i = rounds.length - 1; i >= 0; i--) {
      if (evt.timestamp >= rounds[i].startTs) {
        rounds[i].toolEvents.push(evt);
        assigned = true;
        break;
      }
    }
    if (!assigned) rounds[0].toolEvents.push(evt);
  }
  for (const evt of memoryEvents) {
    let assigned = false;
    for (let i = rounds.length - 1; i >= 0; i--) {
      if (evt.timestamp >= rounds[i].startTs) {
        rounds[i].memoryEvents.push(evt);
        assigned = true;
        break;
      }
    }
    if (!assigned) rounds[0].memoryEvents.push(evt);
  }
  for (const evt of thinkingEvents) {
    let assigned = false;
    for (let i = rounds.length - 1; i >= 0; i--) {
      if (evt.timestamp >= rounds[i].startTs) {
        rounds[i].thinkingEvents.push(evt);
        assigned = true;
        break;
      }
    }
    if (!assigned) rounds[0].thinkingEvents.push(evt);
  }
  for (const evt of commentaryEvents) {
    let assigned = false;
    for (let i = rounds.length - 1; i >= 0; i--) {
      if (evt.timestamp >= rounds[i].startTs) {
        rounds[i].commentaryEvents.push(evt);
        assigned = true;
        break;
      }
    }
    if (!assigned) rounds[0].commentaryEvents.push(evt);
  }
}
