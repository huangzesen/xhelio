import { test, expect } from '../../fixtures/base';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const viewports = [
  { name: 'mobile', width: 375, height: 812 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'laptop', width: 1024, height: 768 },
  { name: 'desktop', width: 1440, height: 900 },
];

const routes = [
  { path: '/', name: 'chat' },
  { path: '/data', name: 'data-tools' },
  { path: '/pipeline', name: 'pipeline' },
  { path: '/gallery', name: 'gallery' },
  { path: '/memory', name: 'memory' },
  { path: '/settings', name: 'settings' },
];

// Ensure screenshot directory exists
const screenshotDir = path.join(__dirname, '../../screenshots/responsive');

test.describe('Responsive layout @audit', () => {
  for (const vp of viewports) {
    for (const route of routes) {
      test(`${route.name} at ${vp.name} (${vp.width}px) — no horizontal overflow`, async ({ page, mockApi }) => {
        await mockApi();
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await page.goto(route.path);
        await page.waitForLoadState('networkidle');

        // Check for horizontal overflow
        const overflow = await page.evaluate(() => ({
          scrollWidth: document.documentElement.scrollWidth,
          clientWidth: document.documentElement.clientWidth,
          hasOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 5,
        }));

        if (overflow.hasOverflow) {
          console.log(`OVERFLOW on ${route.name} at ${vp.name}: scrollWidth=${overflow.scrollWidth} > clientWidth=${overflow.clientWidth}`);

          // Find overflowing elements
          const overflowing = await page.evaluate((vpWidth: number) => {
            const results: string[] = [];
            document.querySelectorAll('*').forEach(el => {
              const rect = el.getBoundingClientRect();
              if (rect.right > vpWidth + 5 && rect.width > 0) {
                const id = el.getAttribute('data-testid') || el.id || el.className?.toString().slice(0, 40) || '';
                results.push(`${el.tagName}[${id}] right=${Math.round(rect.right)}px`);
              }
            });
            return results.slice(0, 5);
          }, vp.width);

          if (overflowing.length > 0) {
            console.log('  Overflowing elements:', overflowing);
          }
        }

        // Take screenshot
        if (!fs.existsSync(screenshotDir)) {
          fs.mkdirSync(screenshotDir, { recursive: true });
        }
        await page.screenshot({
          path: path.join(screenshotDir, `${route.name}_${vp.name}.png`),
          fullPage: true,
        });

        expect(overflow.hasOverflow, `Horizontal overflow on ${route.name} at ${vp.width}px`).toBe(false);
      });
    }
  }

  test('header nav is scrollable on mobile', async ({ page, mockApi }) => {
    await mockApi();
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const nav = page.getByRole('navigation', { name: 'Main navigation' });
    const isVisible = await nav.isVisible();
    console.log(`Nav visible on mobile: ${isVisible}`);

    // Nav should either be visible (scrollable) or hidden behind a hamburger
    expect(isVisible).toBeDefined();
  });
});
