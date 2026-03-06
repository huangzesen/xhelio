import { test, expect } from '../../fixtures/base';

test.describe('@smoke App initialization', () => {
  test('loads the main app when API key is configured', async ({ page, appPage, mockApi }) => {
    await mockApi();
    await appPage.goto();
    await appPage.waitForReady();

    await expect(appPage.header).toBeVisible();
    await expect(page.getByTestId('chat-container')).toBeVisible();
  });

  test('shows setup screen when API key is missing', async ({ appPage, mockApi }) => {
    await mockApi({
      status: {
        status: 'ok',
        active_sessions: 0,
        max_sessions: 10,
        uptime_seconds: 0,
        api_key_configured: false,
      },
    });
    await appPage.goto();

    await expect(appPage.setupScreen).toBeVisible({ timeout: 10000 });
    await expect(appPage.page.getByTestId('setup-api-key-input')).toBeVisible();
    await expect(appPage.page.getByTestId('setup-submit')).toBeVisible();
  });

  test('shows error screen when server is unreachable', async ({ page, appPage, mockApi }) => {
    // Set up all mocks first
    await mockApi();
    // Then override /api/status to fail (LIFO = this takes priority)
    await page.route((url) => new URL(url).pathname === '/api/status', (route) => route.abort('connectionrefused'));
    await appPage.goto();

    await expect(appPage.errorContainer).toBeVisible({ timeout: 10000 });
    const errorText = await appPage.getErrorText();
    expect(errorText).toContain('Connection Error');
  });
});
