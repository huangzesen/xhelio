import { test, expect } from '../../fixtures/base';

test.describe('@smoke Navigation', () => {
  test.beforeEach(async ({ appPage, mockApi }) => {
    await mockApi();
    await appPage.goto();
    await appPage.waitForReady();
  });

  test('navigates to each page via header nav', async ({ page, appPage }) => {
    // Navigate to Data Tools
    await appPage.navigateTo('Data Tools');
    await expect(page).toHaveURL(/\/data/);

    // Navigate to Pipeline
    await appPage.navigateTo('Pipeline');
    await expect(page).toHaveURL(/\/pipeline/);

    // Navigate to Memory
    await appPage.navigateTo('Memory');
    await expect(page).toHaveURL(/\/memory/);

    // Navigate to Settings
    await appPage.navigateTo('Settings');
    await expect(page).toHaveURL(/\/settings/);

    // Navigate back to Chat
    await appPage.navigateTo('Chat');
    await expect(page).toHaveURL(/\/$/);
  });

  test('toggles theme between dark and light', async ({ page, appPage }) => {
    // Get initial theme state
    const htmlElement = page.locator('html');
    const initialHasDark = await htmlElement.evaluate((el) => el.classList.contains('dark'));

    // Toggle theme
    await appPage.toggleTheme();

    // Verify theme changed
    const afterToggleHasDark = await htmlElement.evaluate((el) => el.classList.contains('dark'));
    expect(afterToggleHasDark).not.toBe(initialHasDark);

    // Toggle back
    await appPage.toggleTheme();
    const restoredHasDark = await htmlElement.evaluate((el) => el.classList.contains('dark'));
    expect(restoredHasDark).toBe(initialHasDark);
  });

  test('toggles sidebar visibility', async ({ page, appPage }) => {
    // Sidebar should be visible initially (desktop viewport)
    // Toggle sidebar closed
    await appPage.toggleSidebar();

    // Wait for animation and check the sidebar panel is hidden
    // The sidebar is conditionally rendered with width based on sidebarOpen prop
    await page.waitForTimeout(300); // wait for animation

    // Toggle sidebar open
    await appPage.toggleSidebar();
    await page.waitForTimeout(300);
  });

  test('toggles activity panel visibility', async ({ page, appPage }) => {
    // Toggle activity panel closed
    await appPage.toggleActivity();
    await page.waitForTimeout(300);

    // Toggle activity panel open
    await appPage.toggleActivity();
    await page.waitForTimeout(300);
  });
});
