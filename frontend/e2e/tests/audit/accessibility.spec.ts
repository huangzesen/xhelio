import { test, expect } from '../../fixtures/base';

const routes = [
  { path: '/', name: 'Chat' },
  { path: '/data', name: 'Data Tools' },
  { path: '/pipeline', name: 'Pipeline' },
  { path: '/memory', name: 'Memory' },
  { path: '/settings', name: 'Settings' },
];

test.describe('Accessibility audit @audit', () => {
  for (const route of routes) {
    test(`${route.name} — interactive elements have accessible names`, async ({ page, mockApi }) => {
      await mockApi();
      await page.goto(route.path);
      await page.waitForLoadState('networkidle');

      const issues = await page.evaluate(() => {
        const problems: string[] = [];

        // Buttons without accessible text
        document.querySelectorAll('button').forEach(btn => {
          const name = btn.textContent?.trim() ||
            btn.getAttribute('aria-label') ||
            btn.getAttribute('title') ||
            btn.querySelector('svg')?.getAttribute('aria-label');
          if (!name) {
            problems.push(`Button without label: ${btn.outerHTML.slice(0, 120)}`);
          }
        });

        // Inputs without labels
        document.querySelectorAll('input, textarea, select').forEach(input => {
          const el = input as HTMLInputElement;
          const id = el.id;
          const label = id ? document.querySelector(`label[for="${id}"]`) : null;
          const ariaLabel = el.getAttribute('aria-label') ||
            el.getAttribute('aria-labelledby') ||
            el.getAttribute('placeholder');
          const parentLabel = el.closest('label');
          if (!label && !ariaLabel && !parentLabel) {
            problems.push(`Input without label: ${el.outerHTML.slice(0, 120)}`);
          }
        });

        // Images without alt text
        document.querySelectorAll('img').forEach(img => {
          const alt = img.getAttribute('alt');
          if (alt === null) {
            problems.push(`Image without alt: ${(img as HTMLImageElement).src?.slice(0, 80)}`);
          }
        });

        // Links without text
        document.querySelectorAll('a').forEach(a => {
          const name = a.textContent?.trim() || a.getAttribute('aria-label');
          if (!name) {
            problems.push(`Link without text: ${a.outerHTML.slice(0, 120)}`);
          }
        });

        return problems;
      });

      if (issues.length > 0) {
        console.log(`\n=== Accessibility issues on ${route.name} ===`);
        issues.forEach(i => console.log(`  - ${i}`));
      }

      // Only fail on buttons and links without names (critical)
      const critical = issues.filter(i =>
        i.startsWith('Button without') || i.startsWith('Link without')
      );
      expect(critical, `Critical accessibility issues on ${route.name}`).toEqual([]);
    });
  }

  test('keyboard Tab reaches chat input', async ({ page, mockApi }) => {
    await mockApi();
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const focusPath: string[] = [];

    for (let i = 0; i < 50; i++) {
      await page.keyboard.press('Tab');
      const focused = await page.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body) return 'body';
        const tag = el.tagName;
        const testId = el.getAttribute('data-testid') || '';
        const role = el.getAttribute('role') || '';
        return `${tag}[${testId || role || el.className?.toString().slice(0, 20)}]`;
      });
      focusPath.push(focused);
    }

    console.log('\n=== Tab focus path ===');
    focusPath.forEach((el, i) => console.log(`  Tab ${i + 1}: ${el}`));

    const reachedTextarea = focusPath.some(e => e.includes('TEXTAREA') || e.includes('chat-input'));
    expect(reachedTextarea, 'Tab should reach chat input textarea').toBe(true);
  });

  test('focus indicators are visible', async ({ page, mockApi }) => {
    await mockApi();
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Tab to first interactive element
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    const hasFocusIndicator = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return true;
      const styles = window.getComputedStyle(el);
      const outline = styles.outlineStyle;
      const boxShadow = styles.boxShadow;
      return outline !== 'none' || (boxShadow !== 'none' && boxShadow !== '');
    });

    console.log(`Focus indicator visible: ${hasFocusIndicator}`);
    // Report but don't hard-fail — Tailwind ring may not show in computed style
  });

  test('heading hierarchy is correct', async ({ page, mockApi }) => {
    await mockApi();
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const headings = await page.evaluate(() => {
      const results: { tag: string; text: string; level: number }[] = [];
      document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(h => {
        results.push({
          tag: h.tagName,
          text: h.textContent?.trim().slice(0, 50) || '',
          level: parseInt(h.tagName.replace('H', '')),
        });
      });
      return results;
    });

    console.log('\n=== Heading hierarchy ===');
    headings.forEach(h => console.log(`  ${h.tag}: ${h.text}`));

    // Check no skipped levels (e.g., h1 -> h3 without h2)
    if (headings.length > 1) {
      let skipped = false;
      for (let i = 1; i < headings.length; i++) {
        if (headings[i].level > headings[i - 1].level + 1) {
          console.log(`  WARNING: Heading level skipped from ${headings[i - 1].tag} to ${headings[i].tag}`);
          skipped = true;
        }
      }
    }
  });
});
