import { test, expect } from '../../fixtures/base';

const routes = [
  { path: '/', name: 'Chat' },
  { path: '/data', name: 'Data Tools' },
  { path: '/pipeline', name: 'Pipeline' },
  { path: '/gallery', name: 'Gallery' },
  { path: '/memory', name: 'Memory' },
  { path: '/eureka', name: 'Eureka' },
  { path: '/settings', name: 'Settings' },
  { path: '/settings/assets', name: 'Assets' },
];

for (const route of routes) {
  test(`no console errors on ${route.name} page @audit`, async ({ page, mockApi }) => {
    const errors: string[] = [];
    const warnings: string[] = [];

    page.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
      if (msg.type() === 'warning') warnings.push(msg.text());
    });
    page.on('pageerror', err => errors.push(`[PAGE ERROR] ${err.message}`));

    await mockApi();
    await page.goto(route.path);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Filter known benign errors
    const real = errors.filter(e =>
      !e.includes('favicon') &&
      !e.includes('[vite]') &&
      !e.includes('ERR_CONNECTION_REFUSED') &&
      !e.includes('net::')
    );

    if (real.length > 0) {
      console.log(`\n=== Console errors on ${route.name} (${route.path}) ===`);
      real.forEach(e => console.log(`  - ${e}`));
    }
    if (warnings.length > 0) {
      console.log(`\n=== Warnings on ${route.name} (${route.path}) ===`);
      warnings.forEach(w => console.log(`  - ${w}`));
    }

    expect(real, `Console errors on ${route.name}`).toEqual([]);
  });
}
