import { test, expect } from '../../fixtures/base';

test.describe('Performance baseline @audit', () => {
  test('chat page loads within 5 seconds', async ({ page, mockApi }) => {
    await mockApi();
    const start = Date.now();
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - start;

    console.log(`Chat page load time: ${loadTime}ms`);
    expect(loadTime, 'Chat page should load within 5s').toBeLessThan(5000);
  });

  test('settings page loads within 5 seconds', async ({ page, mockApi }) => {
    await mockApi();
    const start = Date.now();
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - start;

    console.log(`Settings page load time: ${loadTime}ms`);
    expect(loadTime, 'Settings page should load within 5s').toBeLessThan(5000);
  });

  test('DOM node count is reasonable on chat page', async ({ page, mockApi }) => {
    await mockApi();
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const nodeCount = await page.evaluate(() =>
      document.querySelectorAll('*').length
    );

    console.log(`DOM nodes on Chat page: ${nodeCount}`);
    expect(nodeCount, 'Chat page DOM should not exceed 5000 nodes').toBeLessThan(5000);
  });

  test('DOM node count on all pages', async ({ page, mockApi }) => {
    await mockApi();
    const routes = [
      { path: '/', name: 'Chat' },
      { path: '/data', name: 'Data Tools' },
      { path: '/pipeline', name: 'Pipeline' },
      { path: '/gallery', name: 'Gallery' },
      { path: '/memory', name: 'Memory' },
      { path: '/eureka', name: 'Eureka' },
      { path: '/settings', name: 'Settings' },
    ];

    console.log('\n=== DOM node counts ===');
    for (const route of routes) {
      await page.goto(route.path);
      await page.waitForLoadState('networkidle');

      const nodeCount = await page.evaluate(() =>
        document.querySelectorAll('*').length
      );
      console.log(`  ${route.name}: ${nodeCount} nodes`);
    }
  });

  test('rapid navigation does not leak DOM nodes', async ({ page, mockApi }) => {
    await mockApi();
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    // Wait for the app to fully hydrate/render before measuring baseline
    await page.waitForTimeout(500);

    const initialCount = await page.evaluate(() =>
      document.querySelectorAll('*').length
    );

    // Navigate rapidly between pages 3 times
    const routes = ['/data', '/pipeline', '/memory', '/settings', '/eureka', '/gallery', '/'];
    for (let round = 0; round < 3; round++) {
      for (const route of routes) {
        await page.goto(route);
        await page.waitForLoadState('domcontentloaded');
      }
    }

    await page.waitForLoadState('networkidle');

    const finalCount = await page.evaluate(() =>
      document.querySelectorAll('*').length
    );

    console.log(`DOM nodes: initial=${initialCount}, after 21 navigations=${finalCount}`);
    const growth = finalCount - initialCount;
    console.log(`DOM growth: ${growth} nodes (${((growth / initialCount) * 100).toFixed(1)}%)`);

    // Allow up to 50% growth (some caching is expected)
    expect(finalCount, 'DOM should not grow excessively').toBeLessThan(initialCount * 1.5);
  });

  test('page weight and resource count', async ({ page, mockApi }) => {
    await mockApi();

    // Track all resources loaded
    const resources: { url: string; size: number; type: string }[] = [];
    page.on('response', async (response) => {
      try {
        const url = response.url();
        if (url.startsWith('data:')) return;
        const body = await response.body().catch(() => Buffer.from(''));
        resources.push({
          url: url.split('?')[0].split('/').slice(-2).join('/'),
          size: body.length,
          type: response.headers()['content-type'] || 'unknown',
        });
      } catch {}
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const totalSize = resources.reduce((sum, r) => sum + r.size, 0);

    console.log(`\n=== Page resources ===`);
    console.log(`  Total resources: ${resources.length}`);
    console.log(`  Total size: ${(totalSize / 1024).toFixed(0)} KB`);

    // Top 5 largest resources
    const sorted = [...resources].sort((a, b) => b.size - a.size);
    console.log(`  Top 5 largest:`);
    sorted.slice(0, 5).forEach(r => {
      console.log(`    ${(r.size / 1024).toFixed(0)} KB — ${r.url}`);
    });
  });
});
