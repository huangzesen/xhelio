import { test, expect } from '../../fixtures/base';

test.describe('@integration Slash commands', () => {
  test.beforeEach(async ({ appPage, mockApi }) => {
    await mockApi();
    await appPage.goto();
    await appPage.waitForReady();
  });

  test('typing / shows command dropdown with available commands', async ({ chatPage, page }) => {
    await chatPage.chatInput.fill('/');

    // The command dropdown should appear
    const dropdown = page.locator('.absolute.bottom-full');
    await expect(dropdown).toBeVisible();

    // Should show all 10 slash commands
    const items = chatPage.getCommandDropdownItems();
    await expect(items).toHaveCount(10);
  });

  test('selecting /status command sends it and shows system response', async ({ chatPage, page }) => {
    await chatPage.chatInput.fill('/status');

    // Wait for dropdown to appear with filtered results
    const dropdown = page.locator('.absolute.bottom-full');
    await expect(dropdown).toBeVisible();

    // Press Enter to accept the command
    await chatPage.chatInput.press('Enter');

    // Now press Enter again to send the command
    await chatPage.chatInput.press('Enter');

    // A system message should appear with the command response
    await expect(chatPage.getMessages('system').first()).toBeVisible({ timeout: 5000 });
  });
});
