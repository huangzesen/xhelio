import { describe, it, expect } from 'vitest';
import { groupEventsByTurn, extractThinkingEvents } from './groupEventsByTurn';
import type { ChatMessage, ToolEvent, MemoryEvent } from '../api/types';

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

    expect(groups[0].userMessage.content).toBe('first question');
    expect(groups[0].toolEvents).toHaveLength(2);
    expect(groups[0].toolEvents[0].tool_name).toBe('fetch_data');

    expect(groups[1].userMessage.content).toBe('second question');
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
