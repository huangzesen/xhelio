import { test, expect } from '../../fixtures/base';

test.describe('Data pages audit @audit', () => {
  test.beforeEach(async ({ mockApi, page }) => {
    // Add mock routes for endpoints that data pages need (beyond the base mock set).
    // Without these, the catch-all returns {} which causes .map() crashes in components.
    await mockApi({
      routes: {
        '**/api/catalog/missions': async (route) => {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([]),
          });
        },
        '**/api/gallery': async (route) => {
          if (route.request().method() === 'GET') {
            await route.fulfill({
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify([]),
            });
          } else {
            await route.fulfill({
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify({ ok: true }),
            });
          }
        },
      },
    });
  });

  // --- Data Tools ---
  test('Data Tools page loads without crash', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));

    await page.goto('/data');
    await page.waitForLoadState('networkidle');

    await expect(page.getByTestId('app-header')).toBeVisible();

    // Log whatever is visible on the page body
    const bodyText = await page.locator('body').textContent();
    console.log(`Data Tools page content preview: ${bodyText?.slice(0, 200)}`);

    expect(errors).toEqual([]);
    await page.screenshot({ path: 'e2e/screenshots/data-tools.png', fullPage: true });
  });

  // --- Pipeline ---
  test('Pipeline page shows tabs', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));

    await page.goto('/pipeline');
    await page.waitForLoadState('networkidle');

    await expect(page.getByTestId('app-header')).toBeVisible();

    // Look for tab buttons (use exact match to avoid matching nav links)
    const replayBtn = page.getByRole('button', { name: 'Replay', exact: true });
    const pipelinesBtn = page.getByRole('button', { name: 'Pipelines', exact: true });

    const hasReplay = await replayBtn.isVisible().catch(() => false);
    const hasPipelines = await pipelinesBtn.isVisible().catch(() => false);

    console.log(`Pipeline tabs: Replay=${hasReplay}, Pipelines=${hasPipelines}`);

    // Try switching tabs if they exist
    if (hasPipelines) {
      await pipelinesBtn.click({ timeout: 3000 }).catch(() => {});
      await page.waitForTimeout(300);
    }
    if (hasReplay) {
      await replayBtn.click({ timeout: 3000 }).catch(() => {});
      await page.waitForTimeout(300);
    }

    expect(errors).toEqual([]);
    await page.screenshot({ path: 'e2e/screenshots/pipeline.png', fullPage: true });
  });

  // --- Gallery ---
  test('Gallery page loads without crash', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));

    await page.goto('/gallery');
    await page.waitForLoadState('networkidle');

    await expect(page.getByTestId('app-header')).toBeVisible();

    // Log whatever is visible on the page body
    const bodyText = await page.locator('body').textContent();
    console.log(`Gallery page content preview: ${bodyText?.slice(0, 200)}`);

    expect(errors).toEqual([]);
    await page.screenshot({ path: 'e2e/screenshots/gallery.png', fullPage: true });
  });

  // --- Memory ---
  test('Memory page shows tabs or no-session message', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));

    await page.goto('/memory');
    await page.waitForLoadState('networkidle');

    await expect(page.getByTestId('app-header')).toBeVisible();

    // Check for memory tabs or no-session state
    const memoriesTab = page.getByRole('button', { name: /memories/i });
    const reviewsTab = page.getByRole('button', { name: /reviews/i });

    const hasMemories = await memoriesTab.isVisible().catch(() => false);
    const hasReviews = await reviewsTab.isVisible().catch(() => false);

    console.log(`Memory tabs: Memories=${hasMemories}, Reviews=${hasReviews}`);

    // Try tab switching
    if (hasReviews) {
      await reviewsTab.click();
      await page.waitForTimeout(300);
    }
    if (hasMemories) {
      await memoriesTab.click();
      await page.waitForTimeout(300);
    }

    expect(errors).toEqual([]);
    await page.screenshot({ path: 'e2e/screenshots/memory.png', fullPage: true });
  });

  // --- Eureka ---
  test('Eureka page shows tabs or no-session message', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));

    await page.goto('/eureka');
    await page.waitForLoadState('networkidle');

    await expect(page.getByTestId('app-header')).toBeVisible();

    const chatTab = page.getByRole('button', { name: /chat/i });
    const findingsTab = page.getByRole('button', { name: /finding/i });

    const hasChat = await chatTab.isVisible().catch(() => false);
    const hasFindings = await findingsTab.isVisible().catch(() => false);

    console.log(`Eureka tabs: Chat=${hasChat}, Findings=${hasFindings}`);

    if (hasFindings) {
      await findingsTab.click();
      await page.waitForTimeout(300);
    }

    expect(errors).toEqual([]);
    await page.screenshot({ path: 'e2e/screenshots/eureka.png', fullPage: true });
  });
});
