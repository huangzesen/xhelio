import { test, expect } from '../../fixtures/base';

test.describe('@smoke Chat basics', () => {
  test.beforeEach(async ({ appPage, mockApi }) => {
    await mockApi();
    await appPage.goto();
    await appPage.waitForReady();
  });

  test('shows example prompts when chat is empty', async ({ chatPage }) => {
    await expect(chatPage.examplePrompts).toBeVisible();
    const count = await chatPage.getExamplePromptCount();
    expect(count).toBe(6);
  });

  test('clicking example prompt sends user message', async ({ chatPage }) => {
    await chatPage.clickExamplePrompt(0);

    // After clicking, the example prompts should disappear and a user message should appear
    await expect(chatPage.getMessages('user').first()).toBeVisible({ timeout: 5000 });
  });

  test('typing text and pressing Enter sends a message', async ({ chatPage }) => {
    await chatPage.chatInput.fill('Hello world');

    // Send button should be enabled when text is present
    await expect(chatPage.sendButton).toBeEnabled();

    // Press Enter to send
    await chatPage.chatInput.press('Enter');

    // User message should appear
    await expect(chatPage.getMessages('user').first()).toBeVisible({ timeout: 5000 });
  });

  test('send button is disabled when textarea is empty', async ({ chatPage }) => {
    await expect(chatPage.sendButton).toBeDisabled();
  });
});
