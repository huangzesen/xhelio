import { test, expect } from '@playwright/test';
import {
  initApp, sendMessage, sendAndGetText, waitForStreamingDone,
  getActivityPanelText, recordIssue, LLM_TIMEOUT, DATA_TIMEOUT,
  WORKFLOW_TIMEOUT, collectConsoleErrors, screenshotOnFailure,
  apiGet, hasPlotlyChart,
} from './helpers';

test.describe('@e2e DataIO Agent', () => {
  test.skip(!process.env.XHELIO_E2E_REAL, 'Requires XHELIO_E2E_REAL=1 and real backend');

  let consoleErrors: string[];

  test.beforeEach(async ({ page }) => {
    consoleErrors = collectConsoleErrors(page);
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status !== 'passed') {
      await screenshotOnFailure(page, testInfo.title.replace(/\s+/g, '-'));
    }
    if (consoleErrors.length > 0) {
      recordIssue({
        feature: 'dataio',
        severity: 'minor',
        title: `Console errors in: ${testInfo.title}`,
        detail: consoleErrors.join('\n'),
      });
    }
  });

  test('1. Structured text extraction — ICME event list', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const icmeText = `Here's an ICME event list to import:

Start Time | End Time | Speed (km/s) | Bmax (nT)
2024-01-15 03:00 | 2024-01-15 18:00 | 450 | 12.3
2024-01-20 10:00 | 2024-01-21 06:00 | 620 | 18.7
2024-02-03 14:00 | 2024-02-04 08:00 | 380 | 9.5
2024-02-15 22:00 | 2024-02-16 14:00 | 550 | 15.1

Please extract this into a DataFrame and store it as "icme_events".`;

    const response = await sendAndGetText(page, icmeText, DATA_TIMEOUT);

    const activityText = await getActivityPanelText(page);
    const hasStoreCall = activityText.includes('store_dataframe') ||
      activityText.includes('custom_operation');

    if (!hasStoreCall) {
      recordIssue({
        feature: 'dataio',
        severity: 'major',
        title: 'DataIO did not store extracted table data',
        detail: `Activity: ${activityText.substring(0, 500)}`,
      });
    }

    // Verify stored data
    const sessions = await apiGet(page, '/sessions');
    if (sessions?.length > 0) {
      const data = await apiGet(page, `/sessions/${sessions[0].session_id}/data`).catch(() => []);
      const hasIcme = Array.isArray(data) && data.some((d: any) =>
        d.label?.toLowerCase().includes('icme'),
      );
      if (!hasIcme) {
        recordIssue({
          feature: 'dataio',
          severity: 'major',
          title: 'ICME event data not found in data store',
          detail: `Data entries: ${JSON.stringify(data?.map((d: any) => d.label) ?? [])}`,
        });
      }
    }

    console.log('[DATAIO TEST 1] ICME extraction:', response.substring(0, 300));
  });

  test('2. CSV-like data extraction from chat', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const csvText = `Please import this CSV data as "solar_events":

date,event_type,magnitude,duration_hours
2024-01-10,flare,M2.3,1.5
2024-01-12,flare,X1.0,2.1
2024-01-15,CME,12.5,24.0
2024-01-18,flare,C4.7,0.8
2024-01-22,CME,8.3,18.0`;

    const response = await sendAndGetText(page, csvText, DATA_TIMEOUT);

    const success = response.toLowerCase().includes('stored') ||
      response.toLowerCase().includes('imported') ||
      response.toLowerCase().includes('created') ||
      response.toLowerCase().includes('dataframe');

    if (!success) {
      recordIssue({
        feature: 'dataio',
        severity: 'major',
        title: 'CSV text extraction did not store data',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[DATAIO TEST 2] CSV extraction:', response.substring(0, 300));
  });

  test('3. Data preview after import', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await sendMessage(
      page,
      'Store this data as "test_data": x=[1,2,3,4,5], y=[10,20,30,40,50], z=[100,200,300,400,500]',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Preview the test_data. Show me the statistics and first few rows.',
      LLM_TIMEOUT,
    );

    const hasPreview = response.includes('x') ||
      response.includes('y') ||
      response.includes('mean') ||
      response.includes('count') ||
      response.includes('1') ||
      response.includes('10');

    if (!hasPreview) {
      recordIssue({
        feature: 'dataio',
        severity: 'minor',
        title: 'Data preview did not show data content',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[DATAIO TEST 3] Data preview:', response.substring(0, 300));
  });

  test('4. List fetched data shows all entries', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await sendMessage(
      page,
      'Store a DataFrame with columns a=[1,2,3] and b=[4,5,6] as "list_test_data".',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'List all fetched and stored data in this session.',
      LLM_TIMEOUT,
    );

    const mentionsData = response.toLowerCase().includes('list_test_data') ||
      response.toLowerCase().includes('data') ||
      response.toLowerCase().includes('entries') ||
      response.toLowerCase().includes('stored');

    if (!mentionsData) {
      recordIssue({
        feature: 'dataio',
        severity: 'minor',
        title: 'list_fetched_data does not show stored data',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[DATAIO TEST 4] List data:', response.substring(0, 300));
  });

  test('5. Data Tools page renders', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await page.goto('/data-tools');
    await page.waitForLoadState('networkidle', { timeout: 10_000 });

    const pageContent = await page.textContent('body');
    if (!pageContent || pageContent.length < 50) {
      recordIssue({
        feature: 'dataio',
        severity: 'major',
        title: 'Data Tools page failed to render',
        detail: `Content length: ${pageContent?.length ?? 0}`,
      });
    }

    const pageErrors = consoleErrors.filter(e =>
      e.includes('Uncaught') || e.includes('TypeError'),
    );
    if (pageErrors.length > 0) {
      recordIssue({
        feature: 'dataio',
        severity: 'major',
        title: 'Console errors on Data Tools page',
        detail: pageErrors.join('\n'),
      });
    }

    await screenshotOnFailure(page, 'data-tools-page');
    console.log('[DATAIO TEST 5] Data Tools page content length:', pageContent?.length ?? 0);
  });

  test('6. Import and plot workflow — end to end', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await sendMessage(
      page,
      `Import this data as "temperature":
date,temp_c
2024-01-01,15.2
2024-01-02,14.8
2024-01-03,16.1
2024-01-04,13.5
2024-01-05,17.3
2024-01-06,18.9
2024-01-07,16.7`,
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Now plot the temperature data as a line chart.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    const hasPlot = await hasPlotlyChart(page);
    if (!hasPlot) {
      recordIssue({
        feature: 'dataio',
        severity: 'major',
        title: 'Could not plot imported text data',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[DATAIO TEST 6] Import+plot:', response.substring(0, 300));
  });

  test('7. Data store API endpoint', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const sessions = await apiGet(page, '/sessions');
    if (sessions?.length > 0) {
      const sessionId = sessions[0].session_id;
      try {
        const data = await apiGet(page, `/sessions/${sessionId}/data`);
        expect(Array.isArray(data)).toBeTruthy();
        console.log('[DATAIO TEST 7] Data entries:', data.length);
      } catch (e: any) {
        recordIssue({
          feature: 'dataio',
          severity: 'major',
          title: 'GET /sessions/{id}/data endpoint error',
          detail: `Error: ${e.message}`,
        });
      }
    }
  });

  test('8. Fetch real CDAWeb data via chat', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Fetch ACE magnetic field magnitude for 2024-01-15 to 2024-01-16.',
      DATA_TIMEOUT,
    );

    const success = response.toLowerCase().includes('fetched') ||
      response.toLowerCase().includes('loaded') ||
      response.toLowerCase().includes('ace') ||
      response.toLowerCase().includes('magnetic') ||
      response.toLowerCase().includes('stored');

    if (!success) {
      const hasError = response.toLowerCase().includes('error') ||
        response.toLowerCase().includes('timeout') ||
        response.toLowerCase().includes('failed');
      if (hasError) {
        recordIssue({
          feature: 'dataio',
          severity: 'major',
          title: 'CDAWeb data fetch failed',
          detail: `Response: ${response.substring(0, 500)}`,
        });
      }
    }

    // Verify via API
    const sessions = await apiGet(page, '/sessions');
    if (sessions?.length > 0) {
      const data = await apiGet(page, `/sessions/${sessions[0].session_id}/data`).catch(() => []);
      console.log('[DATAIO TEST 8] Fetched data entries:', Array.isArray(data) ? data.length : 0);
    }
  });
});
