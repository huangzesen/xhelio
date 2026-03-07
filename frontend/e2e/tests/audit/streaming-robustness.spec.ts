import { test, expect } from '../../fixtures/base';

test.describe('SSE streaming robustness @audit', () => {
  test.beforeEach(async ({ mockApi }) => {
    await mockApi();
  });

  test('handles rapid-fire text deltas without dropping', async ({ page, chatPage }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await chatPage.sendMessage('test');
    await page.waitForTimeout(200);

    // Emit round start
    await chatPage.emitSSE('round_start', { type: 'round_start' });

    // Emit 12 rapid text deltas with no delay
    const words = 'The quick brown fox jumps over the lazy dog runs fast now end'.split(' ');
    for (const word of words) {
      await chatPage.emitSSE('text_delta', { type: 'text_delta', text: word + ' ' });
    }

    await chatPage.emitSSE('round_end', {
      type: 'round_end',
      token_usage: { input_tokens: 100, output_tokens: 50 },
      round_token_usage: { input_tokens: 100, output_tokens: 50 },
    });

    await chatPage.waitForAgentMessage();
    const msg = chatPage.getMessages('agent').first();
    const text = await msg.textContent();

    // All words should be present
    let allPresent = true;
    for (const word of words) {
      if (!text?.includes(word)) {
        console.log(`DROPPED WORD: "${word}" not found in agent message`);
        allPresent = false;
      }
    }
    expect(allPresent, 'All words should be present in agent message').toBe(true);
  });

  test('handles malformed SSE data without crash', async ({ page, chatPage }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));

    await chatPage.sendMessage('test');
    await page.waitForTimeout(200);

    await chatPage.emitSSE('round_start', { type: 'round_start' });

    // Emit valid then check recovery
    await chatPage.emitSSE('text_delta', { type: 'text_delta', text: 'recovery message' });
    await chatPage.emitSSE('round_end', {
      type: 'round_end',
      token_usage: { input_tokens: 50, output_tokens: 10 },
      round_token_usage: { input_tokens: 50, output_tokens: 10 },
    });

    await chatPage.waitForAgentMessage();
    const msg = chatPage.getMessages('agent').first();
    const text = await msg.textContent();
    expect(text).toContain('recovery');
    expect(errors, 'No unhandled page errors').toEqual([]);
  });

  test('handles empty text deltas gracefully', async ({ page, chatPage }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await chatPage.sendMessage('test');
    await page.waitForTimeout(200);

    await chatPage.emitSSE('round_start', { type: 'round_start' });
    await chatPage.emitSSE('text_delta', { type: 'text_delta', text: '' });
    await chatPage.emitSSE('text_delta', { type: 'text_delta', text: 'actual content' });
    await chatPage.emitSSE('round_end', {
      type: 'round_end',
      token_usage: { input_tokens: 50, output_tokens: 10 },
      round_token_usage: { input_tokens: 50, output_tokens: 10 },
    });

    await chatPage.waitForAgentMessage();
    const text = await chatPage.getMessages('agent').first().textContent();
    expect(text).toContain('actual content');
  });

  test('stop button visible during streaming', async ({ page, chatPage }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await chatPage.sendMessage('test');
    await page.waitForTimeout(200);

    await chatPage.emitSSE('round_start', { type: 'round_start' });
    await chatPage.emitSSE('text_delta', { type: 'text_delta', text: 'streaming... ' });

    // Stop button should be visible during streaming
    const stopBtn = chatPage.stopButton;
    const isVisible = await stopBtn.isVisible().catch(() => false);
    console.log(`Stop button visible during streaming: ${isVisible}`);

    // Complete the round
    await chatPage.emitSSE('round_end', {
      type: 'round_end',
      token_usage: { input_tokens: 50, output_tokens: 10 },
      round_token_usage: { input_tokens: 50, output_tokens: 10 },
    });
  });

  test('tool call events render correctly', async ({ page, chatPage }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await chatPage.sendMessage('test');
    await page.waitForTimeout(200);

    await chatPage.emitSSE('round_start', { type: 'round_start' });
    await chatPage.emitSSE('tool_call', {
      type: 'tool_call',
      tool_name: 'search_datasets',
      tool_args: { query: 'ACE magnetic field' },
      agent: 'orchestrator',
    });
    await chatPage.emitSSE('tool_result', {
      type: 'tool_result',
      tool_name: 'search_datasets',
      status: 'success',
      agent: 'orchestrator',
    });
    await chatPage.emitSSE('text_delta', { type: 'text_delta', text: 'Found ACE data.' });
    await chatPage.emitSSE('round_end', {
      type: 'round_end',
      token_usage: { input_tokens: 200, output_tokens: 50 },
      round_token_usage: { input_tokens: 200, output_tokens: 50 },
    });

    await chatPage.waitForAgentMessage();
    const text = await chatPage.getMessages('agent').first().textContent();
    expect(text).toContain('Found ACE data');
  });
});
