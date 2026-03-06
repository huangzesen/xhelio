/**
 * Pre-built SSE event sequences for testing streaming behavior.
 * Each scenario is an array of events that can be emitted in order.
 */

export interface MockSSEEvent {
  type: string;
  data: Record<string, unknown>;
}

/** Simple text reply: round_start → 3 text deltas → round_end */
export const textReply: MockSSEEvent[] = [
  { type: 'round_start', data: { type: 'round_start' } },
  { type: 'text_delta', data: { type: 'text_delta', text: 'Hello! ' } },
  { type: 'text_delta', data: { type: 'text_delta', text: 'I can help you ' } },
  { type: 'text_delta', data: { type: 'text_delta', text: 'with spacecraft data.' } },
  {
    type: 'round_end',
    data: {
      type: 'round_end',
      token_usage: { input_tokens: 100, output_tokens: 20 },
      round_token_usage: { input_tokens: 100, output_tokens: 20 },
    },
  },
];

/** Error during streaming */
export const errorReply: MockSSEEvent[] = [
  { type: 'round_start', data: { type: 'round_start' } },
  { type: 'error', data: { type: 'error', message: 'Internal server error: model timeout' } },
];

/** Tool call + result cycle */
export const toolCallReply: MockSSEEvent[] = [
  { type: 'round_start', data: { type: 'round_start' } },
  {
    type: 'tool_call',
    data: {
      type: 'tool_call',
      tool_name: 'search_datasets',
      tool_args: { query: 'ACE magnetic field' },
      agent: 'orchestrator',
    },
  },
  {
    type: 'tool_result',
    data: {
      type: 'tool_result',
      tool_name: 'search_datasets',
      status: 'success',
      agent: 'orchestrator',
    },
  },
  { type: 'text_delta', data: { type: 'text_delta', text: 'I found ACE magnetic field datasets.' } },
  {
    type: 'round_end',
    data: {
      type: 'round_end',
      token_usage: { input_tokens: 200, output_tokens: 50 },
      round_token_usage: { input_tokens: 200, output_tokens: 50 },
    },
  },
];

/** Thinking event */
export const thinkingReply: MockSSEEvent[] = [
  { type: 'round_start', data: { type: 'round_start' } },
  {
    type: 'thinking',
    data: {
      type: 'thinking',
      text: 'Let me analyze the user request and determine the best approach...',
    },
  },
  { type: 'text_delta', data: { type: 'text_delta', text: 'Here is my analysis.' } },
  {
    type: 'round_end',
    data: {
      type: 'round_end',
      token_usage: { input_tokens: 150, output_tokens: 30 },
      round_token_usage: { input_tokens: 150, output_tokens: 30 },
    },
  },
];
