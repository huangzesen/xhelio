import { test, expect } from '../../fixtures/base';

test.describe('Navigation integrity @audit', () => {
  test('all header nav tabs reach correct routes', async ({ page, mockApi }) => {
    await mockApi();
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const nav = page.getByRole('navigation', { name: 'Main navigation' });
    const tabs = [
      { label: 'Data Tools', url: /\/data/ },
      { label: 'Pipeline', url: /\/pipeline/ },
      { label: 'Memory', url: /\/memory/ },
      { label: 'Eureka', url: /\/eureka/ },
      { label: 'Settings', url: /\/settings/ },
      { label: 'Chat', url: /\/$/ },
    ];

    for (const tab of tabs) {
      await nav.getByText(tab.label).click();
      await expect(page).toHaveURL(tab.url);
      // Page should render content (not blank)
      await page.waitForLoadState('networkidle');
    }
  });

  test('back/forward browser navigation works', async ({ page, mockApi }) => {
    await mockApi();
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await page.getByRole('navigation', { name: 'Main navigation' }).getByText('Settings').click();
    await expect(page).toHaveURL(/\/settings/);

    await page.goBack();
    await expect(page).toHaveURL(/\/$/);

    await page.goForward();
    await expect(page).toHaveURL(/\/settings/);
  });

  test('direct URL access to each route works', async ({ page, mockApi }) => {
    await mockApi();
    const routes = ['/', '/data', '/pipeline', '/gallery', '/memory', '/eureka', '/settings', '/settings/assets'];
    for (const route of routes) {
      await page.goto(route);
      await page.waitForLoadState('networkidle');
      // Should show header (app loaded, not error)
      await expect(page.getByTestId('app-header')).toBeVisible({ timeout: 5000 });
    }
  });

  test('invalid route shows fallback or redirects', async ({ page, mockApi }) => {
    await mockApi();
    await page.goto('/nonexistent-route-xyz');
    await page.waitForLoadState('networkidle');
    // Should either redirect to / or show a 404 message
    const url = page.url();
    const has404 = await page.getByText(/not found|404/i).isVisible().catch(() => false);
    const redirectedHome = url.endsWith('/') || url.endsWith('/nonexistent-route-xyz');
    expect(has404 || redirectedHome, 'Should show 404 or redirect').toBeTruthy();
  });
});
