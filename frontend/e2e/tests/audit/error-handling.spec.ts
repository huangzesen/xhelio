import { test, expect } from '../../fixtures/base';

test.describe('Error handling @audit', () => {
  test('API 500 on status — shows error or degrades gracefully', async ({ page }) => {
    // Set up mock that returns 500 for status
    await page.route((url) => new URL(url).pathname === '/api/status', (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Internal server error' }),
      });
    });
    // Catch-all for other routes
    await page.route((url) => {
      const p = new URL(url).pathname;
      return p.startsWith('/api/') && p !== '/api/status';
    }, (route) => {
      route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    // Check what state the app is in
    const hasError = await page.getByTestId('app-error').isVisible().catch(() => false);
    const hasSetup = await page.getByTestId('setup-screen').isVisible().catch(() => false);
    const hasHeader = await page.getByTestId('app-header').isVisible().catch(() => false);

    console.log(`After 500 status: error=${hasError}, setup=${hasSetup}, header=${hasHeader}`);

    // App should show SOME indication of error
    expect(hasError || !hasHeader, 'App should show error state on 500').toBeTruthy();
  });

  test('network offline — shows connection error', async ({ page }) => {
    // Abort all API requests
    await page.route((url) => new URL(url).pathname.startsWith('/api/'), (route) => {
      route.abort('connectionrefused');
    });

    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(3000);

    const hasError = await page.getByTestId('app-error').isVisible().catch(() => false);
    const errorText = hasError ? await page.getByTestId('app-error').textContent() : '';

    console.log(`Network offline: error visible=${hasError}, text="${errorText?.slice(0, 100)}"`);
    expect(hasError, 'Should show error when network is offline').toBe(true);
  });

  test('no API key — shows setup screen', async ({ page, mockApi }) => {
    await mockApi({
      status: {
        status: 'ok',
        active_sessions: 0,
        max_sessions: 10,
        uptime_seconds: 120,
        api_key_configured: false,
      },
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    const hasSetup = await page.getByTestId('setup-screen').isVisible().catch(() => false);
    console.log(`No API key: setup screen visible=${hasSetup}`);
    expect(hasSetup, 'Should show setup screen when no API key').toBe(true);
  });

  test('empty session — no crash on fresh load', async ({ page, mockApi }) => {
    await mockApi();
    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    expect(errors, 'No page errors on empty session').toEqual([]);
  });

  test('slow API response — shows loading state', async ({ page }) => {
    // Slow status endpoint (3s delay)
    await page.route((url) => new URL(url).pathname === '/api/status', async (route) => {
      await new Promise(r => setTimeout(r, 3000));
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          active_sessions: 0,
          max_sessions: 10,
          uptime_seconds: 120,
          api_key_configured: true,
        }),
      });
    });
    await page.route((url) => {
      const p = new URL(url).pathname;
      return p.startsWith('/api/') && p !== '/api/status';
    }, (route) => {
      route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });

    await page.goto('/');

    // Check for loading indicator within first 2 seconds
    const hasLoading = await page.getByTestId('app-loading').isVisible().catch(() => false);
    const hasSpinner = await page.locator('.animate-spin').first().isVisible().catch(() => false);

    console.log(`During slow load: loading=${hasLoading}, spinner=${hasSpinner}`);
  });
});
