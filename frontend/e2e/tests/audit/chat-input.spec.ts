import { test, expect } from '../../fixtures/base';

test.describe('Chat input behavior @audit', () => {
  test.beforeEach(async ({ appPage, mockApi }) => {
    await mockApi();
    await appPage.goto();
    await appPage.waitForReady();
  });

  test('Enter inserts newline, does not send', async ({ chatPage }) => {
    await expect(chatPage.chatInput).toBeVisible({ timeout: 5000 });

    await chatPage.chatInput.click();
    await chatPage.chatInput.fill('line one');
    await chatPage.chatInput.press('Enter');
    await chatPage.chatInput.page().waitForTimeout(200);

    // Should NOT have sent — no user message should appear
    const userMsgs = chatPage.getMessages('user');
    expect(await userMsgs.count()).toBe(0);

    // Textarea should still have text (with newline)
    const value = await chatPage.chatInput.inputValue();
    expect(value).toContain('line one');
  });

  test('Ctrl+Enter sends message', async ({ chatPage }) => {
    await expect(chatPage.chatInput).toBeVisible({ timeout: 5000 });

    await chatPage.chatInput.fill('hello world');
    await chatPage.chatInput.press('Control+Enter');

    // User message should appear
    await expect(chatPage.getMessages('user').first()).toBeVisible({ timeout: 3000 });
  });

  test('send button disabled when empty', async ({ chatPage }) => {
    await expect(chatPage.sendButton).toBeVisible({ timeout: 5000 });
    await expect(chatPage.sendButton).toBeDisabled();
  });

  test('whitespace-only cannot be sent', async ({ chatPage }) => {
    await expect(chatPage.chatInput).toBeVisible({ timeout: 5000 });

    await chatPage.chatInput.fill('   ');
    await expect(chatPage.sendButton).toBeDisabled();
  });

  test('very long message sends without crash', async ({ page, chatPage }) => {
    await expect(chatPage.chatInput).toBeVisible({ timeout: 5000 });

    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));

    const longMsg = 'A'.repeat(5000);
    await chatPage.chatInput.fill(longMsg);
    await chatPage.sendButton.click();

    await expect(chatPage.getMessages('user').first()).toBeVisible({ timeout: 5000 });
    expect(errors).toEqual([]);
  });

  test('special characters render as text not HTML', async ({ chatPage }) => {
    await expect(chatPage.chatInput).toBeVisible({ timeout: 5000 });

    const special = '<script>alert("xss")</script>';
    await chatPage.chatInput.fill(special);
    await chatPage.sendButton.click();

    const msg = chatPage.getMessages('user').first();
    await expect(msg).toBeVisible({ timeout: 3000 });
    // Should render as text, not execute
    const text = await msg.textContent();
    expect(text).toContain('<script>');
  });

  test('slash command dropdown appears on /', async ({ page, chatPage }) => {
    await expect(chatPage.chatInput).toBeVisible({ timeout: 5000 });

    await chatPage.chatInput.click();
    // Type / character by character to trigger onChange
    await chatPage.chatInput.pressSequentially('/');

    // Dropdown should appear
    const dropdown = page.locator('.absolute.bottom-full');
    await expect(dropdown).toBeVisible({ timeout: 2000 });

    // Should have slash command items
    const items = dropdown.locator('button');
    const count = await items.count();
    expect(count).toBeGreaterThan(0);
    console.log(`Slash command count: ${count}`);
  });

  test('Escape dismisses command dropdown', async ({ page, chatPage }) => {
    await expect(chatPage.chatInput).toBeVisible({ timeout: 5000 });

    await chatPage.chatInput.click();
    await chatPage.chatInput.pressSequentially('/');
    const dropdown = page.locator('.absolute.bottom-full');
    await expect(dropdown).toBeVisible({ timeout: 2000 });

    await chatPage.chatInput.press('Escape');
    await expect(dropdown).not.toBeVisible({ timeout: 1000 });
  });
});
