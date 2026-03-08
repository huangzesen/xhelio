import { test, expect } from '@playwright/test';
import {
  initApp, sendMessage, sendAndGetText, waitForStreamingDone,
  getActivityPanelText, recordIssue, LLM_TIMEOUT, DATA_TIMEOUT,
  WORKFLOW_TIMEOUT, collectConsoleErrors, screenshotOnFailure,
  apiGet,
} from './helpers';

test.describe('@e2e Eureka Agent', () => {
  test.skip(!process.env.XHELIO_E2E_REAL, 'Requires XHELIO_E2E_REAL=1 and real backend');

  let consoleErrors: string[];

  test.beforeEach(async ({ page }) => {
    consoleErrors = collectConsoleErrors(page);
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status !== 'passed') {
      await screenshotOnFailure(page, testInfo.title.replace(/\s+/g, '-'));
    }
  });

  test('1. Eureka triggers after data analysis', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // Fetch interesting data to trigger eureka
    await sendMessage(
      page,
      'Fetch ACE magnetic field magnitude and solar wind speed for 2024-01-15 to 2024-01-17 and plot them in a 2-panel chart.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, WORKFLOW_TIMEOUT);

    // Wait for eureka extraction to run (async background process)
    console.log('[EUREKA TEST 1] Waiting 45s for eureka extraction...');
    await page.waitForTimeout(15_000);

    // Check activity panel for eureka events
    const activityText = await getActivityPanelText(page);
    const hasEureka = activityText.toLowerCase().includes('eureka') ||
      activityText.toLowerCase().includes('finding') ||
      activityText.toLowerCase().includes('suggestion') ||
      activityText.toLowerCase().includes('discovery');

    // Also check API
    const eurekaResp = await apiGet(page, '/eureka').catch(() => ({ eurekas: [] }));
    const eurekas = eurekaResp?.eurekas ?? [];
    const eurekaCount = eurekas.length;

    if (!hasEureka && eurekaCount === 0) {
      recordIssue({
        feature: 'eureka',
        severity: 'major',
        title: 'No eureka findings after data analysis',
        detail: 'Neither activity panel nor API shows eureka findings after fetch+plot.',
      });
    }

    console.log('[EUREKA TEST 1] Activity eureka:', hasEureka, 'API count:', eurekaCount);
  });

  test('2. Eureka page renders and shows content', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // Navigate directly to Eureka page (no data fetch needed — just test rendering)
    await page.goto('/eureka');
    await page.waitForLoadState('networkidle', { timeout: 10_000 });

    const pageContent = await page.textContent('body');
    if (!pageContent || pageContent.length < 100) {
      recordIssue({
        feature: 'eureka',
        severity: 'major',
        title: 'Eureka page is empty or failed to render',
        detail: `Page content length: ${pageContent?.length ?? 0}`,
      });
    }

    // Check for any console errors on this page
    const pageErrors = consoleErrors.filter(e =>
      e.includes('Uncaught') || e.includes('TypeError'),
    );
    if (pageErrors.length > 0) {
      recordIssue({
        feature: 'eureka',
        severity: 'major',
        title: 'Console errors on Eureka page',
        detail: pageErrors.join('\n'),
      });
    }

    await screenshotOnFailure(page, 'eureka-page');
    console.log('[EUREKA TEST 2] Page content length:', pageContent?.length ?? 0);
  });

  test('3. Eureka API endpoints return valid data', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // Test eureka list endpoint
    let eurekaList: any;
    try {
      const eurekaResp = await apiGet(page, '/eureka');
      eurekaList = eurekaResp?.eurekas ?? [];
      expect(Array.isArray(eurekaList)).toBeTruthy();
      console.log('[EUREKA TEST 3] GET /eureka:', eurekaList.length, 'items');
    } catch (e: any) {
      recordIssue({
        feature: 'eureka',
        severity: 'major',
        title: 'GET /api/eureka endpoint error',
        detail: `Error: ${e.message}`,
      });
    }

    // Test suggestions endpoint
    try {
      const suggestionsResp = await apiGet(page, '/eureka/suggestions');
      const suggestions = suggestionsResp?.suggestions ?? [];
      expect(Array.isArray(suggestions)).toBeTruthy();
      console.log('[EUREKA TEST 3] GET /eureka/suggestions:', suggestions.length, 'items');
    } catch (e: any) {
      recordIssue({
        feature: 'eureka',
        severity: 'major',
        title: 'GET /api/eureka/suggestions endpoint error',
        detail: `Error: ${e.message}`,
      });
    }
  });

  test('4. Eureka findings have required fields', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // Generate data + wait for eureka
    await sendMessage(
      page,
      'Fetch ACE magnetic field for 2024-01-15 to 2024-01-17 and plot it.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, WORKFLOW_TIMEOUT);
    console.log('[EUREKA TEST 4] Waiting 45s for eureka extraction...');
    await page.waitForTimeout(15_000);

    const eurekaResp = await apiGet(page, '/eureka').catch(() => ({ eurekas: [] }));
    const eurekas = eurekaResp?.eurekas ?? [];
    if (Array.isArray(eurekas) && eurekas.length > 0) {
      for (const eureka of eurekas) {
        const requiredFields = ['title', 'observation', 'hypothesis', 'confidence', 'status'];
        for (const field of requiredFields) {
          if (!(field in eureka)) {
            recordIssue({
              feature: 'eureka',
              severity: 'major',
              title: `Eureka finding missing field: ${field}`,
              detail: `Finding keys: ${Object.keys(eureka).join(', ')}`,
            });
          }
        }
        if (typeof eureka.confidence === 'number' && (eureka.confidence < 0 || eureka.confidence > 1)) {
          recordIssue({
            feature: 'eureka',
            severity: 'minor',
            title: 'Eureka confidence out of range',
            detail: `Confidence: ${eureka.confidence}`,
          });
        }
      }
      console.log('[EUREKA TEST 4] Validated', eurekas.length, 'findings');
    } else {
      console.log('[EUREKA TEST 4] No eureka findings to validate (extraction may not have triggered)');
    }
  });

  test('5. Eureka suggestions have required fields and valid actions', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await sendMessage(
      page,
      'Fetch and plot ACE solar wind data for 2024-01-15 to 2024-01-17.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, WORKFLOW_TIMEOUT);
    console.log('[EUREKA TEST 5] Waiting 45s for eureka extraction...');
    await page.waitForTimeout(15_000);

    const suggestionsResp = await apiGet(page, '/eureka/suggestions').catch(() => ({ suggestions: [] }));
    const suggestions = suggestionsResp?.suggestions ?? [];
    if (Array.isArray(suggestions) && suggestions.length > 0) {
      for (const suggestion of suggestions) {
        const requiredFields = ['action', 'description', 'priority', 'status'];
        for (const field of requiredFields) {
          if (!(field in suggestion)) {
            recordIssue({
              feature: 'eureka',
              severity: 'major',
              title: `Eureka suggestion missing field: ${field}`,
              detail: `Suggestion keys: ${Object.keys(suggestion).join(', ')}`,
            });
          }
        }
        const validActions = ['fetch_data', 'compute', 'visualize'];
        if (suggestion.action && !validActions.includes(suggestion.action)) {
          recordIssue({
            feature: 'eureka',
            severity: 'minor',
            title: `Unexpected eureka suggestion action: ${suggestion.action}`,
            detail: `Expected one of: ${validActions.join(', ')}`,
          });
        }
      }
      console.log('[EUREKA TEST 5] Validated', suggestions.length, 'suggestions');
    } else {
      console.log('[EUREKA TEST 5] No suggestions to validate');
    }
  });
});
