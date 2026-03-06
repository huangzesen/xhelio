import { test, expect } from '@playwright/test';

/**
 * Live E2E test — requires real backend running on port 8000.
 * Skipped by default; run with: XHELIO_E2E_REAL=1 npx playwright test --grep @e2e
 */
test.describe('@e2e Live chat', () => {
  test.skip(!process.env.XHELIO_E2E_REAL, 'Requires XHELIO_E2E_REAL=1 and real backend on port 8000');

  test('sends greeting and receives agent response', async ({ page }) => {
    await page.goto('/');

    // Wait for app to initialize (real server, may take a moment)
    const header = page.getByTestId('app-header');
    await expect(header).toBeVisible({ timeout: 15000 });

    // Wait for chat container
    const chatContainer = page.getByTestId('chat-container');
    await expect(chatContainer).toBeVisible();

    // Type and send a message
    const chatInput = page.getByTestId('chat-input');
    await chatInput.fill('Hello');
    await page.getByTestId('chat-send-btn').click();

    // Wait for user message
    await expect(page.getByTestId('message-user').first()).toBeVisible();

    // Wait for agent response (extended timeout for real LLM call)
    await expect(page.getByTestId('message-agent').first()).toBeVisible({ timeout: 30000 });

    // Verify agent message has content
    const agentMessage = page.getByTestId('message-agent').first();
    const text = await agentMessage.textContent();
    expect(text?.length).toBeGreaterThan(0);
  });
});
