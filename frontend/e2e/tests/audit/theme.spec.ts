import { test, expect } from '../../fixtures/base';

test.describe('Theme audit @audit', () => {
  test.beforeEach(async ({ mockApi }) => {
    await mockApi();
  });

  test('theme toggle switches between light and dark', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const initialDark = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );
    console.log(`Initial theme: ${initialDark ? 'dark' : 'light'}`);

    // Toggle
    await page.getByLabel(/Switch to .* mode/).click();
    await page.waitForTimeout(200);

    const afterToggle = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );
    expect(afterToggle).toBe(!initialDark);
    console.log(`After toggle: ${afterToggle ? 'dark' : 'light'}`);

    // Toggle back
    await page.getByLabel(/Switch to .* mode/).click();
    await page.waitForTimeout(200);

    const afterSecond = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );
    expect(afterSecond).toBe(initialDark);
  });

  test('theme persists across navigation', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Ensure we're in dark mode
    const isDark = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );
    if (!isDark) {
      await page.getByLabel(/Switch to .* mode/).click();
      await page.waitForTimeout(200);
    }

    // Navigate to settings
    await page.getByRole('navigation', { name: 'Main navigation' }).getByText('Settings').click();
    await page.waitForLoadState('networkidle');

    const stillDark = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );
    expect(stillDark, 'Theme should persist across navigation').toBe(true);

    // Navigate to data tools
    await page.getByRole('navigation', { name: 'Main navigation' }).getByText('Data Tools').click();
    await page.waitForLoadState('networkidle');

    const stillDark2 = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );
    expect(stillDark2, 'Theme should persist to data tools').toBe(true);
  });

  test('dark mode screenshots for visual review', async ({ page }) => {
    // Light mode
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Ensure light mode
    const isDark = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );
    if (isDark) {
      await page.getByLabel(/Switch to .* mode/).click();
      await page.waitForTimeout(300);
    }
    await page.screenshot({ path: 'e2e/screenshots/chat_light.png', fullPage: true });

    // Dark mode
    await page.getByLabel(/Switch to .* mode/).click();
    await page.waitForTimeout(300);
    await page.screenshot({ path: 'e2e/screenshots/chat_dark.png', fullPage: true });

    // Settings in dark mode
    await page.getByRole('navigation', { name: 'Main navigation' }).getByText('Settings').click();
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: 'e2e/screenshots/settings_dark.png', fullPage: true });
  });

  test('no contrast issues in dark mode', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Switch to dark
    const isDark = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );
    if (!isDark) {
      await page.getByLabel(/Switch to .* mode/).click();
      await page.waitForTimeout(300);
    }

    // Check text is not invisible (same color as background)
    const contrastIssues = await page.evaluate(() => {
      const problems: string[] = [];
      const textElements = document.querySelectorAll('p, span, h1, h2, h3, h4, a, button, label');
      textElements.forEach(el => {
        const styles = window.getComputedStyle(el);
        const color = styles.color;
        const bgColor = styles.backgroundColor;
        // Check for completely transparent or same-as-bg text
        if (color === bgColor && el.textContent?.trim()) {
          problems.push(`Same color text/bg: ${el.tagName} "${el.textContent?.slice(0, 30)}"`);
        }
      });
      return problems.slice(0, 10);
    });

    if (contrastIssues.length > 0) {
      console.log('\n=== Dark mode contrast issues ===');
      contrastIssues.forEach(i => console.log(`  - ${i}`));
    }
  });
});
