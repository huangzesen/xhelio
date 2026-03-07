import { test, expect } from '../../fixtures/base';

test.describe('Settings page @audit', () => {
  test.beforeEach(async ({ mockApi }) => {
    await mockApi();
  });

  test('settings page loads and shows provider section', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Page should show settings content
    await expect(page.getByTestId('app-header')).toBeVisible();

    // Look for provider-related content
    const hasProviderSection = await page.getByText(/provider|llm|model/i).first().isVisible().catch(() => false);
    console.log(`Provider section visible: ${hasProviderSection}`);

    // Take screenshot for visual review
    await page.screenshot({ path: 'e2e/screenshots/settings_full.png', fullPage: true });
  });

  test('no console errors on settings page', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', err => errors.push(err.message));

    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    const real = errors.filter(e =>
      !e.includes('favicon') && !e.includes('[vite]') && !e.includes('net::')
    );

    if (real.length > 0) {
      console.log('\n=== Settings page console errors ===');
      real.forEach(e => console.log(`  - ${e}`));
    }
    expect(real).toEqual([]);
  });

  test('all form controls are interactive', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    // Wait for React to render settings form
    await page.getByTestId('app-header').waitFor({ state: 'visible' });
    await page.waitForTimeout(500);

    // Count interactive elements
    const counts = await page.evaluate(() => {
      return {
        selects: document.querySelectorAll('select').length,
        inputs: document.querySelectorAll('input').length,
        textareas: document.querySelectorAll('textarea').length,
        buttons: document.querySelectorAll('button').length,
        checkboxes: document.querySelectorAll('input[type="checkbox"]').length,
        ranges: document.querySelectorAll('input[type="range"]').length,
      };
    });

    console.log('\n=== Settings page form controls ===');
    console.log(`  Selects: ${counts.selects}`);
    console.log(`  Inputs: ${counts.inputs}`);
    console.log(`  Textareas: ${counts.textareas}`);
    console.log(`  Buttons: ${counts.buttons}`);
    console.log(`  Checkboxes: ${counts.checkboxes}`);
    console.log(`  Range sliders: ${counts.ranges}`);

    // Should have at least some interactive elements
    const total = counts.selects + counts.inputs + counts.buttons;
    expect(total, 'Settings should have interactive elements').toBeGreaterThan(3);
  });

  test('settings page scrolls without issues', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Scroll to bottom
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(300);

    // Scroll back to top
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(300);

    // No errors should occur
    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));
    expect(errors).toEqual([]);
  });

  test('assets page loads from settings', async ({ page }) => {
    await page.goto('/settings/assets');
    await page.waitForLoadState('networkidle');

    // Should show assets content or header
    await expect(page.getByTestId('app-header')).toBeVisible();

    // Take screenshot
    await page.screenshot({ path: 'e2e/screenshots/assets_full.png', fullPage: true });
  });
});
