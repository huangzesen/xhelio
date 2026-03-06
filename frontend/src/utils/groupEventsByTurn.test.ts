import { describe, it, expect } from 'vitest';
import { groupEventsByTurn, groupEventsByRound, extractThinkingEvents } from './groupEventsByTurn';
import type { ChatMessage, ToolEvent, MemoryEvent } from '../api/types';
import type { RoundMarker } from '../stores/chatStore';

// ---- Helpers ----

let _id = 0;
function msg(role: ChatMessage['role'], ts: number, content = ''): ChatMessage {
  return { id: `m${_id++}`, role, content, timestamp: ts };
}

function toolEvt(type: 'call' | 'result', name: string, ts: number): ToolEvent {
  return { id: `t${_id++}`, type, tool_name: name, timestamp: ts, ...(type === 'result' ? { status: 'success' } : {}) };
}

function memEvt(ts: number): MemoryEvent {
  return { id: `e${_id++}`, actions: { pitfall: 1 }, timestamp: ts };
}

// ---- Tests ----

describe('groupEventsByTurn', () => {
  it('returns empty array when no user messages', () => {
    const result = groupEventsByTurn(
      [msg('agent', 100)],
      [toolEvt('call', 'fetch_data', 100)],
      [],
    );
    expect(result).toEqual([]);
  });

  it('groups tool events into the correct turn by timestamp', () => {
    const messages = [
      msg('user', 100, 'first question'),
      msg('agent', 200),
      msg('user', 300, 'second question'),
      msg('agent', 400),
    ];
    const tools = [
      toolEvt('call', 'fetch_data', 150),
      toolEvt('result', 'fetch_data', 180),
      toolEvt('call', 'custom_operation', 350),
      toolEvt('result', 'custom_operation', 380),
    ];

    const groups = groupEventsByTurn(messages, tools, []);
    expect(groups).toHaveLength(2);

    expect(groups[0].userMessages[0].content).toBe('first question');
    expect(groups[0].toolEvents).toHaveLength(2);
    expect(groups[0].toolEvents[0].tool_name).toBe('fetch_data');

    expect(groups[1].userMessages[0].content).toBe('second question');
    expect(groups[1].toolEvents).toHaveLength(2);
    expect(groups[1].toolEvents[0].tool_name).toBe('custom_operation');
  });

  it('assigns memory events to the correct turn', () => {
    const messages = [
      msg('user', 100),
      msg('user', 300),
    ];
    const mem = [memEvt(200), memEvt(350)];

    const groups = groupEventsByTurn(messages, [], mem);
    expect(groups[0].memoryEvents).toHaveLength(1);
    expect(groups[1].memoryEvents).toHaveLength(1);
  });

  it('assigns events at exactly the turn boundary to that turn', () => {
    const messages = [
      msg('user', 100),
      msg('user', 200),
    ];
    const tools = [toolEvt('call', 'fetch_data', 200)];

    const groups = groupEventsByTurn(messages, tools, []);
    // Event at ts=200 should belong to turn 1 (startTs=200), not turn 0
    expect(groups[0].toolEvents).toHaveLength(0);
    expect(groups[1].toolEvents).toHaveLength(1);
  });

  it('last turn window extends to infinity', () => {
    const messages = [msg('user', 100)];
    const tools = [toolEvt('call', 'fetch_data', 999999)];

    const groups = groupEventsByTurn(messages, tools, []);
    expect(groups[0].toolEvents).toHaveLength(1);
    expect(groups[0].endTs).toBe(Infinity);
  });

  it('groups thinking events extracted from messages', () => {
    const messages = [
      msg('user', 100, 'question'),
      msg('thinking', 150, 'Let me think about this...'),
      msg('agent', 200, 'answer'),
      msg('user', 300, 'follow-up'),
      msg('thinking', 350, 'Considering the follow-up...'),
      msg('agent', 400, 'response'),
    ];
    const tools = [
      toolEvt('call', 'fetch_data', 160),
      toolEvt('call', 'render_plotly_json', 360),
    ];

    const groups = groupEventsByTurn(messages, tools, []);
    expect(groups).toHaveLength(2);

    expect(groups[0].thinkingEvents).toHaveLength(1);
    expect(groups[0].thinkingEvents[0].content).toBe('Let me think about this...');
    expect(groups[0].toolEvents).toHaveLength(1);

    expect(groups[1].thinkingEvents).toHaveLength(1);
    expect(groups[1].thinkingEvents[0].content).toBe('Considering the follow-up...');
    expect(groups[1].toolEvents).toHaveLength(1);
  });

  it('handles turns with thinking but no tool events', () => {
    const messages = [
      msg('user', 100, 'simple question'),
      msg('thinking', 150, 'thinking...'),
      msg('agent', 200, 'direct answer'),
    ];

    const groups = groupEventsByTurn(messages, [], []);
    expect(groups).toHaveLength(1);
    expect(groups[0].thinkingEvents).toHaveLength(1);
    expect(groups[0].toolEvents).toHaveLength(0);
  });
});

// ---- Round-marker-based grouping tests ----

function marker(type: 'start' | 'end', ts: number, tokenUsage?: Record<string, number>): RoundMarker {
  return { type, timestamp: ts, ...(tokenUsage ? { roundTokenUsage: tokenUsage } : {}) };
}

describe('groupEventsByRound (with round markers)', () => {
  it('pairs start/end markers into separate rounds', () => {
    const markers: RoundMarker[] = [
      marker('start', 100),
      marker('end', 200, { input: 10, output: 20 }),
      marker('start', 300),
      marker('end', 400, { input: 15, output: 25 }),
    ];
    const messages = [msg('user', 100, 'q1'), msg('user', 300, 'q2')];
    const tools = [toolEvt('call', 'fetch_data', 150), toolEvt('call', 'render', 350)];

    const groups = groupEventsByRound(messages, tools, [], markers);
    expect(groups).toHaveLength(2);

    expect(groups[0].roundIndex).toBe(0);
    expect(groups[0].startTs).toBe(100);
    expect(groups[0].endTs).toBe(200);
    expect(groups[0].roundTokenUsage).toEqual({ input: 10, output: 20 });
    expect(groups[0].toolEvents).toHaveLength(1);
    expect(groups[0].toolEvents[0].tool_name).toBe('fetch_data');

    expect(groups[1].roundIndex).toBe(1);
    expect(groups[1].startTs).toBe(300);
    expect(groups[1].endTs).toBe(400);
    expect(groups[1].roundTokenUsage).toEqual({ input: 15, output: 25 });
    expect(groups[1].toolEvents).toHaveLength(1);
    expect(groups[1].toolEvents[0].tool_name).toBe('render');
  });

  it('handles single active round (start without end)', () => {
    const markers: RoundMarker[] = [marker('start', 100)];
    const messages = [msg('user', 100, 'hello')];
    const tools = [toolEvt('call', 'fetch_data', 150)];

    const groups = groupEventsByRound(messages, tools, [], markers);
    expect(groups).toHaveLength(1);
    expect(groups[0].startTs).toBe(100);
    expect(groups[0].endTs).toBe(Infinity);
    expect(groups[0].roundTokenUsage).toBeUndefined();
    expect(groups[0].toolEvents).toHaveLength(1);
  });

  it('creates separate groups for consecutive starts without ends', () => {
    // This is the bug scenario: two round_starts, zero round_ends
    const markers: RoundMarker[] = [
      marker('start', 100),
      marker('start', 300),
    ];
    const messages = [msg('user', 90, 'q1'), msg('user', 250, 'q2')];
    const tools = [
      toolEvt('call', 'fetch_data', 150),
      toolEvt('call', 'render', 350),
    ];

    const groups = groupEventsByRound(messages, tools, [], markers);
    expect(groups).toHaveLength(2);

    // Round 0: ts 100-300 (implicitly ended when round 1 started)
    expect(groups[0].roundIndex).toBe(0);
    expect(groups[0].startTs).toBe(100);
    expect(groups[0].endTs).toBe(300);
    expect(groups[0].roundTokenUsage).toBeUndefined();
    expect(groups[0].toolEvents).toHaveLength(1);
    expect(groups[0].toolEvents[0].tool_name).toBe('fetch_data');

    // Round 1: ts 300-Infinity (last round, still active)
    expect(groups[1].roundIndex).toBe(1);
    expect(groups[1].startTs).toBe(300);
    expect(groups[1].endTs).toBe(Infinity);
    expect(groups[1].toolEvents).toHaveLength(1);
    expect(groups[1].toolEvents[0].tool_name).toBe('render');
  });

  it('handles three consecutive starts without ends', () => {
    const markers: RoundMarker[] = [
      marker('start', 100),
      marker('start', 200),
      marker('start', 300),
    ];
    const tools = [
      toolEvt('call', 'a', 120),
      toolEvt('call', 'b', 220),
      toolEvt('call', 'c', 320),
    ];

    const groups = groupEventsByRound([], tools, [], markers);
    expect(groups).toHaveLength(3);

    expect(groups[0].startTs).toBe(100);
    expect(groups[0].endTs).toBe(200);
    expect(groups[0].toolEvents).toHaveLength(1);
    expect(groups[0].toolEvents[0].tool_name).toBe('a');

    expect(groups[1].startTs).toBe(200);
    expect(groups[1].endTs).toBe(300);
    expect(groups[1].toolEvents).toHaveLength(1);
    expect(groups[1].toolEvents[0].tool_name).toBe('b');

    expect(groups[2].startTs).toBe(300);
    expect(groups[2].endTs).toBe(Infinity);
    expect(groups[2].toolEvents).toHaveLength(1);
    expect(groups[2].toolEvents[0].tool_name).toBe('c');
  });

  it('handles mixed: start, end, start, start (second round has no end)', () => {
    const markers: RoundMarker[] = [
      marker('start', 100),
      marker('end', 200, { input: 5, output: 10 }),
      marker('start', 300),
      marker('start', 400),
    ];

    const groups = groupEventsByRound([], [], [], markers);
    expect(groups).toHaveLength(3);

    expect(groups[0].startTs).toBe(100);
    expect(groups[0].endTs).toBe(200);
    expect(groups[0].roundTokenUsage).toEqual({ input: 5, output: 10 });

    expect(groups[1].startTs).toBe(300);
    expect(groups[1].endTs).toBe(400);
    expect(groups[1].roundTokenUsage).toBeUndefined();

    expect(groups[2].startTs).toBe(400);
    expect(groups[2].endTs).toBe(Infinity);
    expect(groups[2].roundTokenUsage).toBeUndefined();
  });

  it('creates implicit active round for events after last round_end', () => {
    const markers: RoundMarker[] = [
      marker('start', 100),
      marker('end', 200),
    ];
    // Tool event after the last round ended but no new round_start yet
    const tools = [toolEvt('call', 'late_call', 250)];
    const messages = [msg('user', 90, 'q1'), msg('user', 220, 'q2')];

    const groups = groupEventsByRound(messages, tools, [], markers);
    expect(groups).toHaveLength(2);

    expect(groups[0].startTs).toBe(100);
    expect(groups[0].endTs).toBe(200);

    // Implicit active round for events after the last end
    expect(groups[1].endTs).toBe(Infinity);
    expect(groups[1].toolEvents).toHaveLength(1);
    expect(groups[1].toolEvents[0].tool_name).toBe('late_call');
  });
});

describe('extractThinkingEvents', () => {
  it('extracts only thinking messages', () => {
    const messages = [
      msg('user', 100, 'hi'),
      msg('thinking', 150, 'thought 1'),
      msg('agent', 200, 'response'),
      msg('thinking', 250, 'thought 2'),
    ];

    const events = extractThinkingEvents(messages);
    expect(events).toHaveLength(2);
    expect(events[0].content).toBe('thought 1');
    expect(events[1].content).toBe('thought 2');
  });

  it('returns empty array when no thinking messages', () => {
    const messages = [msg('user', 100), msg('agent', 200)];
    expect(extractThinkingEvents(messages)).toEqual([]);
  });

  it('preserves id and timestamp from source message', () => {
    const m = msg('thinking', 12345, 'deep thought');
    const events = extractThinkingEvents([m]);
    expect(events[0].id).toBe(m.id);
    expect(events[0].timestamp).toBe(12345);
  });
});

// ---- groupEventsByRound with markers ----

describe('groupEventsByRound with markers', () => {
  it('end-only markers infer startTs from earliest event (not 0)', () => {
    // Simulates old sessions rebuilt with turn_done → end-only markers.
    // The bug was that the first round got startTs=0, producing ~56y elapsed.
    const messages = [
      msg('user', 1000, 'hello'),
      msg('agent', 1500, 'hi'),
    ];
    const tools = [
      toolEvt('call', 'fetch_data', 1200),
      toolEvt('result', 'fetch_data', 1400),
    ];
    const markers: RoundMarker[] = [
      marker('end', 2000),
    ];

    const groups = groupEventsByRound(messages, tools, [], markers);
    expect(groups).toHaveLength(1);
    // startTs should be the earliest event timestamp (user msg at 1000), NOT 0
    expect(groups[0].startTs).toBe(1000);
    expect(groups[0].endTs).toBe(2000);
    // Elapsed = 2000 - 1000 = 1000ms, not ~56 years
    expect(groups[0].endTs - groups[0].startTs).toBe(1000);
  });

  it('end-only markers: second round starts at previous end timestamp', () => {
    const messages = [
      msg('user', 1000, 'first'),
      msg('user', 3000, 'second'),
    ];
    const tools = [
      toolEvt('call', 'fetch_data', 1500),
      toolEvt('call', 'custom_operation', 3500),
    ];
    const markers: RoundMarker[] = [
      marker('end', 2000),
      marker('end', 4000),
    ];

    const groups = groupEventsByRound(messages, tools, [], markers);
    expect(groups).toHaveLength(2);
    expect(groups[0].startTs).toBe(1000); // inferred from earliest event
    expect(groups[0].endTs).toBe(2000);
    expect(groups[1].startTs).toBe(2000); // previous round's endTs
    expect(groups[1].endTs).toBe(4000);
  });

  it('end-only markers with no events: startTs falls back to marker timestamp', () => {
    // Edge case: end marker but no user messages or tool events
    const markers: RoundMarker[] = [marker('end', 5000)];

    const groups = groupEventsByRound([], [], [], markers);
    expect(groups).toHaveLength(1);
    // With no events, earliestEventTs is Infinity → prevEnd is undefined → falls back to marker.timestamp
    expect(groups[0].startTs).toBe(5000);
    expect(groups[0].endTs).toBe(5000);
  });

  it('paired start/end markers produce correct intervals', () => {
    const messages = [
      msg('user', 1000, 'q1'),
      msg('user', 3000, 'q2'),
    ];
    const tools = [
      toolEvt('call', 'fetch_data', 1500),
      toolEvt('call', 'custom_operation', 3500),
    ];
    const markers: RoundMarker[] = [
      marker('start', 900),
      marker('end', 2000, { input_tokens: 100, output_tokens: 50 }),
      marker('start', 2900),
      marker('end', 4000, { input_tokens: 200, output_tokens: 100 }),
    ];

    const groups = groupEventsByRound(messages, tools, [], markers);
    expect(groups).toHaveLength(2);

    expect(groups[0].startTs).toBe(900);
    expect(groups[0].endTs).toBe(2000);
    expect(groups[0].roundTokenUsage).toEqual({ input_tokens: 100, output_tokens: 50 });

    expect(groups[1].startTs).toBe(2900);
    expect(groups[1].endTs).toBe(4000);
    expect(groups[1].roundTokenUsage).toEqual({ input_tokens: 200, output_tokens: 100 });
  });

  it('orphan start marker (active round) has endTs=Infinity', () => {
    const messages = [msg('user', 1000, 'hello')];
    const tools = [toolEvt('call', 'fetch_data', 1500)];
    const markers: RoundMarker[] = [
      marker('start', 900),
      marker('end', 2000),
      marker('start', 2500), // active round, no matching end
    ];

    const groups = groupEventsByRound(messages, tools, [], markers);
    // Should have 2 rounds: one completed, one active
    expect(groups.length).toBeGreaterThanOrEqual(2);
    const lastRound = groups[groups.length - 1];
    expect(lastRound.startTs).toBe(2500);
    expect(lastRound.endTs).toBe(Infinity);
  });

  it('assigns events to the correct round with paired markers', () => {
    const messages = [
      msg('user', 1000, 'first'),
      msg('user', 3000, 'second'),
    ];
    const tools = [
      toolEvt('call', 'fetch_data', 1200),
      toolEvt('result', 'fetch_data', 1800),
      toolEvt('call', 'custom_operation', 3200),
      toolEvt('result', 'custom_operation', 3800),
    ];
    const markers: RoundMarker[] = [
      marker('start', 900),
      marker('end', 2000),
      marker('start', 2900),
      marker('end', 4000),
    ];

    const groups = groupEventsByRound(messages, tools, [], markers);
    expect(groups[0].toolEvents).toHaveLength(2);
    expect(groups[0].toolEvents[0].tool_name).toBe('fetch_data');
    expect(groups[1].toolEvents).toHaveLength(2);
    expect(groups[1].toolEvents[0].tool_name).toBe('custom_operation');
  });
});
