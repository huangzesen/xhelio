import { test, expect } from '@playwright/test';
import {
  initApp, sendMessage, sendAndGetText, waitForStreamingDone,
  getActivityPanelText, hasPlotlyChart, waitForPlot,
  recordIssue, LLM_TIMEOUT, DATA_TIMEOUT, WORKFLOW_TIMEOUT,
  collectConsoleErrors, screenshotOnFailure,
} from './helpers';

test.describe('@e2e Plotting Procedures', () => {
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
        feature: 'plotting',
        severity: 'minor',
        title: `Console errors in: ${testInfo.title}`,
        detail: consoleErrors.join('\n'),
      });
    }
  });

  test('1. Simple timeseries plot — single dataset', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Fetch ACE magnetic field magnitude data for 2024-01-15 to 2024-01-16 and plot it.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    try {
      await waitForPlot(page);
    } catch {
      // Plot didn't appear in time — will be caught by hasPlot check
    }
    const hasPlot = await hasPlotlyChart(page);
    if (!hasPlot) {
      recordIssue({
        feature: 'plotting',
        severity: 'critical',
        title: 'No Plotly chart rendered for simple timeseries',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    } else {
      const traceCount = await page.evaluate(() => {
        const plotEl = document.querySelector('.js-plotly-plot') as any;
        return plotEl?.data?.length ?? 0;
      });
      if (traceCount === 0) {
        recordIssue({
          feature: 'plotting',
          severity: 'critical',
          title: 'Plotly chart rendered but has no traces',
          detail: 'Plot element exists but data array is empty',
        });
      }
      console.log('[PLOT TEST 1] Trace count:', traceCount);
    }

    console.log('[PLOT TEST 1] Simple plot:', response.substring(0, 300));
  });

  test('2. Multi-panel plot — magnitude + components', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Fetch ACE magnetic field data (both magnitude and GSE components) for 2024-01-15 to 2024-01-16, then create a 2-panel plot: magnitude on top panel, Bx/By/Bz components on bottom panel.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    try {
      await waitForPlot(page);
    } catch {
      // Plot didn't appear in time — will be caught by hasPlot check
    }
    const hasPlot = await hasPlotlyChart(page);
    if (!hasPlot) {
      recordIssue({
        feature: 'plotting',
        severity: 'critical',
        title: 'No Plotly chart for multi-panel request',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    } else {
      const panelInfo = await page.evaluate(() => {
        const plotEl = document.querySelector('.js-plotly-plot') as any;
        const layout = plotEl?.layout ?? {};
        const yaxes = Object.keys(layout).filter((k: string) => k.startsWith('yaxis'));
        return { yaxisCount: yaxes.length, traceCount: plotEl?.data?.length ?? 0 };
      });

      if (panelInfo.yaxisCount < 2) {
        recordIssue({
          feature: 'plotting',
          severity: 'major',
          title: 'Multi-panel plot has only one y-axis',
          detail: `Expected 2+ yaxes, got ${panelInfo.yaxisCount}. Traces: ${panelInfo.traceCount}`,
        });
      }

      console.log('[PLOT TEST 2] Panels:', panelInfo.yaxisCount, 'Traces:', panelInfo.traceCount);
    }
  });

  test('3. Plot with computed data — smoothing', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await sendMessage(
      page,
      'Fetch ACE magnetic field magnitude for 2024-01-15 to 2024-01-16.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Compute a 30-minute rolling average of the magnetic field magnitude, then plot both the original and smoothed data together.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    try {
      await waitForPlot(page);
    } catch {
      // Plot didn't appear in time — will be caught by hasPlot check
    }
    const hasPlot = await hasPlotlyChart(page);
    if (!hasPlot) {
      recordIssue({
        feature: 'plotting',
        severity: 'major',
        title: 'No plot after compute + render pipeline',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    } else {
      const traceCount = await page.evaluate(() => {
        const plotEl = document.querySelector('.js-plotly-plot') as any;
        return plotEl?.data?.length ?? 0;
      });
      if (traceCount < 2) {
        recordIssue({
          feature: 'plotting',
          severity: 'minor',
          title: 'Expected 2 traces (original + smoothed), got fewer',
          detail: `Trace count: ${traceCount}`,
        });
      }
    }

    console.log('[PLOT TEST 3] Compute+plot:', response.substring(0, 300));
  });

  test('4. Plot modification — restyle existing plot', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await sendMessage(
      page,
      'Fetch and plot ACE magnetic field magnitude for 2024-01-15 to 2024-01-16.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);
    await waitForPlot(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Change the plot title to "ACE IMF Magnitude" and change the line color to red.',
      LLM_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    try {
      await waitForPlot(page);
    } catch {
      // Plot didn't appear in time — will be caught by hasPlot check
    }
    const hasPlot = await hasPlotlyChart(page);
    if (!hasPlot) {
      recordIssue({
        feature: 'plotting',
        severity: 'major',
        title: 'Plot disappeared after restyle request',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    } else {
      const title = await page.evaluate(() => {
        const plotEl = document.querySelector('.js-plotly-plot') as any;
        return plotEl?.layout?.title?.text ?? '';
      });
      console.log('[PLOT TEST 4] Updated title:', title);
    }
  });

  test('5. Fullscreen plot modal', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await sendMessage(
      page,
      'Fetch and plot ACE magnetic field magnitude for 2024-01-15 to 2024-01-16.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);
    await waitForPlot(page, DATA_TIMEOUT);

    // Look for expand/fullscreen button
    const expandBtn = page.locator('button').filter({ hasText: /expand|fullscreen|enlarge/i }).first();
    const svgExpandBtn = page.locator('button[title*="ull"], button[title*="xpand"], button[aria-label*="ull"]').first();

    let btnFound = false;
    for (const btn of [expandBtn, svgExpandBtn]) {
      try {
        if (await btn.isVisible({ timeout: 3_000 })) {
          await btn.click();
          btnFound = true;
          await page.waitForTimeout(1_000);

          // Look for enlarged plot
          const fullscreenPlot = page.locator('.js-plotly-plot');
          if (!(await fullscreenPlot.isVisible())) {
            recordIssue({
              feature: 'plotting',
              severity: 'minor',
              title: 'Fullscreen modal did not render plot',
              detail: 'Clicked expand button but no enlarged plot appeared',
            });
          }

          await page.keyboard.press('Escape');
          await page.waitForTimeout(500);
          break;
        }
      } catch {
        continue;
      }
    }

    if (!btnFound) {
      recordIssue({
        feature: 'plotting',
        severity: 'observation',
        title: 'No fullscreen/expand button found on plot',
        detail: 'Could not locate an expand button on the rendered plot card',
      });
    }

    console.log('[PLOT TEST 5] Fullscreen test complete, btnFound:', btnFound);
  });

  test('6. Multi-source plot — two different datasets', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Fetch ACE magnetic field magnitude and ACE solar wind speed for 2024-01-15 to 2024-01-16, then create a 2-panel plot with B magnitude on top and wind speed on bottom.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    try {
      await waitForPlot(page);
    } catch {
      // Plot didn't appear in time — will be caught by hasPlot check
    }
    const hasPlot = await hasPlotlyChart(page);
    if (!hasPlot) {
      recordIssue({
        feature: 'plotting',
        severity: 'major',
        title: 'Multi-source plot not rendered',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[PLOT TEST 6] Multi-source:', response.substring(0, 300));
  });

  test('7. Plot zoom — manage_plot to change time range', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await sendMessage(
      page,
      'Fetch and plot ACE magnetic field magnitude for 2024-01-15 to 2024-01-17.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);
    await waitForPlot(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Zoom the plot to show only January 16, 2024.',
      LLM_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    try {
      await waitForPlot(page);
    } catch {
      // Plot didn't appear in time — will be caught by hasPlot check
    }
    const hasPlot = await hasPlotlyChart(page);
    if (!hasPlot) {
      recordIssue({
        feature: 'plotting',
        severity: 'major',
        title: 'Plot disappeared after zoom request',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[PLOT TEST 7] Zoom:', response.substring(0, 300));
  });

  test('8. Empty data handling — future date range', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Fetch and plot ACE magnetic field for 2030-01-01 to 2030-01-02.',
      DATA_TIMEOUT,
    );

    // Should handle gracefully — error message, not crash
    const crashed = consoleErrors.some(e =>
      e.includes('Uncaught') || e.includes('TypeError') || e.includes('Cannot read'),
    );
    if (crashed) {
      recordIssue({
        feature: 'plotting',
        severity: 'critical',
        title: 'App crashed on empty/future data range',
        detail: `Console errors: ${consoleErrors.join('; ')}`,
      });
    }

    console.log('[PLOT TEST 8] Empty data:', response.substring(0, 300));
  });
});
