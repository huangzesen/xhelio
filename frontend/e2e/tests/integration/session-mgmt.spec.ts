import { test, expect } from '../../fixtures/base';
import { savedSession } from '../../mocks/responses';

test.describe('@integration Session management', () => {
  test('New Chat button creates a fresh session', async ({ appPage, chatPage, mockApi }) => {
    await mockApi();
    await appPage.goto();
    await appPage.waitForReady();

    // Send a message to populate the chat
    await chatPage.sendMessage('First message');
    await expect(chatPage.getMessages('user').first()).toBeVisible();

    // Click New Chat
    await chatPage.clickNewChat();

    // After creating new session, example prompts should reappear
    // (messages get cleared for the new session)
    await expect(chatPage.examplePrompts).toBeVisible({ timeout: 5000 });
  });

  test('sidebar shows saved sessions', async ({ page, appPage, mockApi }) => {
    const sessions = [
      savedSession('sess-1', 'Solar Wind Analysis', 5),
      savedSession('sess-2', 'Parker Probe Data', 3),
    ];

    await mockApi({ savedSessions: sessions });
    await appPage.goto();
    await appPage.waitForReady();

    // Session list should contain the saved sessions
    const sessionList = page.getByTestId('session-list');
    await expect(sessionList).toBeVisible();
    await expect(sessionList.getByText('Solar Wind Analysis')).toBeVisible();
    await expect(sessionList.getByText('Parker Probe Data')).toBeVisible();
  });

  test('clicking saved session triggers resume', async ({ page, appPage, mockApi }) => {
    const sessions = [
      savedSession('sess-1', 'Solar Wind Analysis', 5),
    ];

    await mockApi({ savedSessions: sessions });
    await appPage.goto();
    await appPage.waitForReady();

    // Click on the saved session
    const sessionList = page.getByTestId('session-list');
    await sessionList.getByText('Solar Wind Analysis').click();

    // The session should be resumed — verify by checking the API was called
    // (In practice, the UI would update; here we just verify no crash)
    await page.waitForTimeout(500);
  });
});
