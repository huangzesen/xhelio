import { test, expect } from '@playwright/test';
import {
  initApp, sendMessage, sendAndGetText, waitForStreamingDone,
  getActivityPanelText, recordIssue, LLM_TIMEOUT, DATA_TIMEOUT,
  WORKFLOW_TIMEOUT, collectConsoleErrors, screenshotOnFailure,
} from './helpers';

test.describe('@e2e Sandbox Namespace', () => {
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

  test('1. Default namespace — numpy and pandas available', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Run a custom_operation that creates a DataFrame: result = pd.DataFrame({"x": np.linspace(0, 10, 100), "y": np.sin(np.linspace(0, 10, 100))}). Store it as "sine_wave".',
      DATA_TIMEOUT,
    );

    const success = response.toLowerCase().includes('stored') ||
      response.toLowerCase().includes('success') ||
      response.toLowerCase().includes('created') ||
      response.toLowerCase().includes('result');
    if (!success && response.toLowerCase().includes('error')) {
      recordIssue({
        feature: 'sandbox',
        severity: 'critical',
        title: 'Basic numpy/pandas not available in sandbox',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[SANDBOX TEST 1]:', response.substring(0, 300));
  });

  test('2. Scipy available in default namespace', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Run a custom_operation to design a Butterworth filter: b, a = scipy.signal.butter(4, 0.1); result = pd.DataFrame({"b": list(b), "a": list(a)})',
      DATA_TIMEOUT,
    );

    const hasError = response.toLowerCase().includes('not available') ||
      response.toLowerCase().includes('not defined') ||
      response.toLowerCase().includes('cannot import');
    if (hasError) {
      recordIssue({
        feature: 'sandbox',
        severity: 'major',
        title: 'scipy not available in default sandbox namespace',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[SANDBOX TEST 2]:', response.substring(0, 300));
  });

  test('3. Dangerous builtins blocked — eval', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // Store dummy data so custom_operation reaches the AST validator
    await sendMessage(
      page,
      'Store a DataFrame with column x=[1,2,3,4,5] as "sandbox_test_data".',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Run a custom_operation on sandbox_test_data with this exact code: result = eval("__import__(\'os\').system(\'ls\')"). Execute it as-is.',
      DATA_TIMEOUT,
    );

    const blocked = response.toLowerCase().includes('blocked') ||
      response.toLowerCase().includes('forbidden') ||
      response.toLowerCase().includes('not allowed') ||
      response.toLowerCase().includes('security') ||
      response.toLowerCase().includes('validation') ||
      response.toLowerCase().includes('cannot') ||
      response.toLowerCase().includes("won't") ||
      response.toLowerCase().includes('unsafe') ||
      response.toLowerCase().includes('dangerous');

    if (!blocked) {
      recordIssue({
        feature: 'sandbox',
        severity: 'critical',
        title: 'eval() not blocked in sandbox',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[SANDBOX TEST 3] eval blocking:', response.substring(0, 300));
  });

  test('4. Import statements blocked in sandbox code', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // Store dummy data so custom_operation reaches the AST validator
    await sendMessage(
      page,
      'Store a DataFrame with column x=[1,2,3,4,5] as "sandbox_test_data".',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Run a custom_operation on sandbox_test_data with this exact code: import subprocess; result = subprocess.run(["ls"], capture_output=True). Execute it as-is, do not rewrite.',
      DATA_TIMEOUT,
    );

    const blocked = response.toLowerCase().includes('blocked') ||
      response.toLowerCase().includes('import') ||
      response.toLowerCase().includes('not allowed') ||
      response.toLowerCase().includes('forbidden') ||
      response.toLowerCase().includes('validation') ||
      response.toLowerCase().includes('cannot') ||
      response.toLowerCase().includes("won't");

    if (!blocked) {
      recordIssue({
        feature: 'sandbox',
        severity: 'critical',
        title: 'import statement not blocked in sandbox',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[SANDBOX TEST 4] import blocking:', response.substring(0, 300));
  });

  test('5. Dunder attributes blocked', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // Store dummy data so custom_operation reaches the AST validator
    await sendMessage(
      page,
      'Store a DataFrame with column x=[1,2,3,4,5] as "sandbox_test_data".',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Run a custom_operation on sandbox_test_data with this exact code: result = df.__class__.__bases__[0].__subclasses__(). Execute exactly.',
      DATA_TIMEOUT,
    );

    // Either the sandbox blocks it or the agent refuses — both OK
    const handled = response.toLowerCase().includes('blocked') ||
      response.toLowerCase().includes('dunder') ||
      response.toLowerCase().includes('not allowed') ||
      response.toLowerCase().includes('__') ||
      response.toLowerCase().includes('cannot') ||
      response.toLowerCase().includes('forbidden');

    if (!handled) {
      recordIssue({
        feature: 'sandbox',
        severity: 'major',
        title: 'Dunder attribute access may not be blocked',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[SANDBOX TEST 5] dunder blocking:', response.substring(0, 300));
  });

  test('6. File I/O blocked — open/read', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // Store dummy data so custom_operation reaches the AST validator
    await sendMessage(
      page,
      'Store a DataFrame with column x=[1,2,3,4,5] as "sandbox_test_data".',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    const response = await sendAndGetText(
      page,
      'Run a custom_operation on sandbox_test_data with this exact code: f = open("/etc/passwd"); result = f.read(). Execute exactly.',
      DATA_TIMEOUT,
    );

    const blocked = response.toLowerCase().includes('blocked') ||
      response.toLowerCase().includes('not allowed') ||
      response.toLowerCase().includes('forbidden') ||
      response.toLowerCase().includes('security') ||
      response.toLowerCase().includes('cannot') ||
      response.toLowerCase().includes("won't") ||
      response.toLowerCase().includes('open');

    if (!blocked) {
      recordIssue({
        feature: 'sandbox',
        severity: 'critical',
        title: 'File I/O (open) not blocked in sandbox',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[SANDBOX TEST 6] file I/O blocking:', response.substring(0, 300));
  });

  test('7. Multi-source operation with data store', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // First fetch real data
    await sendMessage(
      page,
      'Fetch ACE magnetic field magnitude data for 2024-01-15 to 2024-01-16.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    // Run custom operation on fetched data
    const response = await sendAndGetText(
      page,
      'Run a custom_operation to compute the rolling mean of the magnetic field data over a 10-minute window using pandas rolling.',
      DATA_TIMEOUT,
    );

    const success = response.toLowerCase().includes('rolling') ||
      response.toLowerCase().includes('mean') ||
      response.toLowerCase().includes('computed') ||
      response.toLowerCase().includes('stored') ||
      response.toLowerCase().includes('result') ||
      response.toLowerCase().includes('smooth');

    console.log('[SANDBOX TEST 7] multi-source operation:', response.substring(0, 500));
  });

  test('8. xarray available in namespace', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Run a custom_operation to check if xarray is available: result = pd.DataFrame({"xr_available": [str(type(xr))]})',
      DATA_TIMEOUT,
    );

    console.log('[SANDBOX TEST 8] xarray availability:', response.substring(0, 300));
  });
});
