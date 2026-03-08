import { Page, expect, Locator } from '@playwright/test';

/** Maximum wait for LLM to respond */
export const LLM_TIMEOUT = 90_000;
/** Maximum wait for data fetch + LLM processing */
export const DATA_TIMEOUT = 150_000;
/** Maximum wait for multi-step workflows */
export const WORKFLOW_TIMEOUT = 300_000;

/** Collected issues found during test run */
export interface Issue {
  feature: string;
  severity: 'critical' | 'major' | 'minor' | 'observation';
  title: string;
  detail: string;
  screenshot?: string;
}

export const issues: Issue[] = [];

export function recordIssue(issue: Issue) {
  issues.push(issue);
  console.log(`[ISSUE] [${issue.severity}] ${issue.feature}: ${issue.title}`);
  console.log(`  Detail: ${issue.detail}`);
}

/**
 * Clean up all active sessions via API to free session slots.
 */
export async function cleanupAllSessions(page: Page): Promise<void> {
  try {
    const response = await page.request.get('http://localhost:8000/api/sessions');
    if (response.ok()) {
      const sessions = await response.json();
      for (const s of sessions) {
        await page.request.delete(`http://localhost:8000/api/sessions/${s.session_id}`).catch(() => {});
      }
    }
  } catch {
    // Server may not be ready yet
  }
}

/**
 * Navigate to app, wait for init, ensure a session exists.
 * Cleans up stale sessions first to avoid hitting the 10-session limit.
 * Returns once chat container is visible and ready.
 */
export async function initApp(page: Page): Promise<void> {
  // Clean up any leftover sessions from previous tests
  await cleanupAllSessions(page);

  await page.goto('/');
  const header = page.getByTestId('app-header');
  await expect(header).toBeVisible({ timeout: 20_000 });
  const chatContainer = page.getByTestId('chat-container');
  await expect(chatContainer).toBeVisible({ timeout: 10_000 });
  // Wait for session to be fully initialized
  await page.waitForTimeout(2_000);
}

/**
 * Send a chat message and wait for the next agent response.
 * Returns the agent message locator.
 */
export async function sendMessage(
  page: Page,
  message: string,
  timeout = LLM_TIMEOUT,
): Promise<Locator> {
  const agentMessages = page.getByTestId('message-agent');
  const countBefore = await agentMessages.count();

  const chatInput = page.getByTestId('chat-input');
  await chatInput.fill(message);
  await page.getByTestId('chat-send-btn').click();

  // Wait for user message to appear
  const userMessages = page.getByTestId('message-user');
  const userCountBefore = countBefore; // approximate
  await expect(userMessages.last()).toBeVisible({ timeout: 5_000 });

  // Wait for next agent response
  await expect(agentMessages.nth(countBefore)).toBeVisible({ timeout });
  return agentMessages.nth(countBefore);
}

/**
 * Send message and return full text content of agent response.
 * Waits extra time for streaming to complete.
 */
export async function sendAndGetText(
  page: Page,
  message: string,
  timeout = LLM_TIMEOUT,
): Promise<string> {
  const msg = await sendMessage(page, message, timeout);
  // Wait for streaming to finish
  await waitForStreamingDone(page, timeout);
  // Give extra buffer for final text
  await page.waitForTimeout(3_000);
  return (await msg.textContent()) ?? '';
}

/**
 * Wait for streaming to finish (stop button disappears).
 */
export async function waitForStreamingDone(page: Page, timeout = LLM_TIMEOUT): Promise<void> {
  const stopBtn = page.getByTestId('chat-stop-btn');
  try {
    // If stop button visible, wait for it to disappear
    await stopBtn.waitFor({ state: 'visible', timeout: 5_000 });
    await expect(stopBtn).toBeHidden({ timeout });
  } catch {
    // Stop button never appeared or already gone — streaming was fast
  }
}

/**
 * Get all text from the activity panel.
 */
export async function getActivityPanelText(page: Page): Promise<string> {
  const panel = page.getByTestId('activity-panel');
  try {
    if (await panel.isVisible()) {
      return (await panel.textContent()) ?? '';
    }
  } catch {
    // Panel not visible
  }
  return '';
}

/**
 * Check if a Plotly chart is visible on the page.
 */
export async function hasPlotlyChart(page: Page): Promise<boolean> {
  const plotlyPlots = page.locator('.js-plotly-plot');
  const plotMessages = page.getByTestId('message-plot');
  return (await plotlyPlots.count()) > 0 || (await plotMessages.count()) > 0;
}

/**
 * Wait for a Plotly chart to appear.
 */
export async function waitForPlot(page: Page, timeout = DATA_TIMEOUT): Promise<void> {
  await expect(
    page.locator('.js-plotly-plot, [data-testid="message-plot"]').first()
  ).toBeVisible({ timeout });
}

/**
 * Collect console errors during a test.
 * Call at start of test; returned array fills as errors occur.
 */
export function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });
  return errors;
}

/**
 * Take a screenshot and return the path.
 */
export async function screenshotOnFailure(
  page: Page,
  testName: string,
): Promise<string> {
  const safeName = testName.replace(/[^a-zA-Z0-9_-]/g, '-').substring(0, 80);
  const path = `e2e/screenshots/${safeName}-${Date.now()}.png`;
  await page.screenshot({ path: `frontend/${path}`, fullPage: true }).catch(() => {});
  return path;
}

/**
 * Create a new session by clicking "New Chat" button.
 */
export async function newSession(page: Page): Promise<void> {
  const newChatBtn = page.getByTestId('new-chat-btn');
  await newChatBtn.click();
  await page.waitForTimeout(2_000);
  await expect(page.getByTestId('example-prompts')).toBeVisible({ timeout: 10_000 });
}

/**
 * Check the API directly for session data.
 */
export async function apiGet(page: Page, path: string): Promise<any> {
  const response = await page.request.get(`http://localhost:8000/api${path}`);
  if (!response.ok()) {
    throw new Error(`API GET ${path} failed: ${response.status()}`);
  }
  return response.json();
}

/**
 * Check the API directly via POST.
 */
export async function apiPost(page: Page, path: string, body?: any): Promise<any> {
  const response = await page.request.post(`http://localhost:8000/api${path}`, {
    data: body,
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok()) {
    throw new Error(`API POST ${path} failed: ${response.status()}`);
  }
  return response.json();
}
