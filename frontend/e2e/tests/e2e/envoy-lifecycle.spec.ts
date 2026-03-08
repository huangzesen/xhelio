import { test, expect } from '@playwright/test';
import {
  initApp, sendMessage, sendAndGetText, waitForStreamingDone,
  getActivityPanelText, recordIssue, LLM_TIMEOUT,
  DATA_TIMEOUT, WORKFLOW_TIMEOUT, collectConsoleErrors,
  screenshotOnFailure,
} from './helpers';

test.describe('@e2e Envoy Lifecycle (Package Envoy)', () => {
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
        feature: 'envoy',
        severity: 'minor',
        title: `Console errors in: ${testInfo.title}`,
        detail: consoleErrors.join('\n'),
      });
    }
  });

  test('1. Add envoy — introspect a Python package', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // scipy is guaranteed installed in the venv
    const response = await sendAndGetText(
      page,
      'Add a new package envoy for scipy.signal. Use add_envoy tool with package_name "scipy.signal".',
      DATA_TIMEOUT,
    );

    const activityText = await getActivityPanelText(page);

    if (!activityText.includes('add_envoy')) {
      recordIssue({
        feature: 'envoy',
        severity: 'critical',
        title: 'add_envoy tool not called',
        detail: `Agent did not call add_envoy. Response: ${response.substring(0, 500)}`,
      });
    }

    const mentionsApi = response.toLowerCase().includes('function') ||
      response.toLowerCase().includes('api') ||
      response.toLowerCase().includes('signal');
    if (!mentionsApi) {
      recordIssue({
        feature: 'envoy',
        severity: 'major',
        title: 'add_envoy did not return API surface',
        detail: `Response doesn't mention functions/API: ${response.substring(0, 500)}`,
      });
    }

    console.log('[ENVOY TEST 1] add_envoy response length:', response.length);
  });

  test('2. Save envoy — persist package envoy definition', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await sendMessage(
      page,
      'Add a package envoy for scipy.signal using add_envoy, then save it with save_envoy using envoy_id "SCIPY_SIGNAL", name "SciPy Signal Processing", imports [{"import_path": "scipy.signal", "sandbox_alias": "signal"}].',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, WORKFLOW_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Did the save succeed? What file was created?',
      LLM_TIMEOUT,
    );

    const saveSuccess = response.toLowerCase().includes('save') ||
      response.toLowerCase().includes('created') ||
      response.toLowerCase().includes('success');
    if (!saveSuccess) {
      recordIssue({
        feature: 'envoy',
        severity: 'critical',
        title: 'save_envoy may have failed',
        detail: `No save confirmation in response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[ENVOY TEST 2] save_envoy response:', response.substring(0, 500));
  });

  test('3. List envoys — verify saved envoy appears', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'List all package envoys using the list_envoys tool.',
      LLM_TIMEOUT,
    );

    const found = response.toLowerCase().includes('scipy') ||
      response.toLowerCase().includes('signal');
    if (!found) {
      recordIssue({
        feature: 'envoy',
        severity: 'major',
        title: 'list_envoys does not show saved envoy',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[ENVOY TEST 3] list_envoys response:', response.substring(0, 500));
  });

  test('4. Use package envoy — run sandboxed code via custom_operation', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Using the SCIPY_SIGNAL envoy, generate a test signal: create a 1-second chirp signal using signal.chirp with f0=1, f1=50, t1=1, and 1000 sample points. Store the result as a DataFrame.',
      DATA_TIMEOUT,
    );

    const activityText = await getActivityPanelText(page);
    if (!activityText.includes('custom_operation')) {
      recordIssue({
        feature: 'envoy',
        severity: 'major',
        title: 'custom_operation not called via package envoy',
        detail: `Activity panel: ${activityText.substring(0, 500)}`,
      });
    }

    const hasError = response.toLowerCase().includes('error') &&
      !response.toLowerCase().includes('no error');
    if (hasError) {
      recordIssue({
        feature: 'envoy',
        severity: 'critical',
        title: 'custom_operation execution error in package envoy',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[ENVOY TEST 4] envoy custom_operation:', response.substring(0, 500));
  });

  test('5. Remove envoy — clean up test envoy', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Remove the SCIPY_SIGNAL package envoy using remove_envoy.',
      LLM_TIMEOUT,
    );

    const removed = response.toLowerCase().includes('removed') ||
      response.toLowerCase().includes('deleted') ||
      response.toLowerCase().includes('success');
    if (!removed) {
      recordIssue({
        feature: 'envoy',
        severity: 'major',
        title: 'remove_envoy may have failed',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    // Verify it's gone from list
    const listResponse = await sendAndGetText(
      page,
      'List all package envoys again.',
      LLM_TIMEOUT,
    );

    // Check if SCIPY_SIGNAL is mentioned as still active (not in removal context)
    const lr = listResponse.toLowerCase();
    const stillListed = lr.includes('scipy_signal') &&
      !lr.includes('no ') &&
      !lr.includes('empty') &&
      !lr.includes('removed') &&
      !lr.includes('0 ') &&
      !lr.includes('none');
    if (stillListed) {
      recordIssue({
        feature: 'envoy',
        severity: 'critical',
        title: 'Envoy still listed after removal',
        detail: `list_envoys still shows SCIPY_SIGNAL: ${listResponse.substring(0, 500)}`,
      });
    }

    console.log('[ENVOY TEST 5] remove_envoy:', response.substring(0, 300));
  });

  test('6. Envoy with invalid package — error handling', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Add a package envoy for "nonexistent_fake_package_xyz" using add_envoy.',
      LLM_TIMEOUT,
    );

    const handlesError = response.toLowerCase().includes('error') ||
      response.toLowerCase().includes('not found') ||
      response.toLowerCase().includes('import') ||
      response.toLowerCase().includes('install') ||
      response.toLowerCase().includes('cannot') ||
      response.toLowerCase().includes("doesn't exist");
    if (!handlesError) {
      recordIssue({
        feature: 'envoy',
        severity: 'major',
        title: 'No error message for invalid package',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[ENVOY TEST 6] invalid package:', response.substring(0, 300));
  });

  test('7. Envoy ID validation — special characters rejected', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Save a package envoy with envoy_id "bad/id/../hack" and imports [{"import_path": "os"}]. Use save_envoy directly.',
      DATA_TIMEOUT,
    );

    const lr = response.toLowerCase();
    const rejected = lr.includes('invalid') ||
      lr.includes('error') ||
      lr.includes('alphanumeric') ||
      lr.includes('rejected') ||
      lr.includes('must contain') ||
      lr.includes('refuse') ||
      lr.includes("can't") ||
      lr.includes("won't") ||
      lr.includes('security') ||
      lr.includes('malicious') ||
      lr.includes('traversal') ||
      lr.includes('block') ||
      lr.includes('not allowed') ||
      lr.includes('special character');
    if (!rejected) {
      recordIssue({
        feature: 'envoy',
        severity: 'critical',
        title: 'Malicious envoy_id not rejected',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[ENVOY TEST 7] ID validation:', response.substring(0, 300));
  });
});
