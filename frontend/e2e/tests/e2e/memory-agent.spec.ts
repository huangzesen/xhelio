import { test, expect } from '@playwright/test';
import {
  initApp, sendMessage, sendAndGetText, waitForStreamingDone,
  getActivityPanelText, recordIssue, LLM_TIMEOUT, DATA_TIMEOUT,
  WORKFLOW_TIMEOUT, collectConsoleErrors, screenshotOnFailure,
  apiGet, apiPost,
} from './helpers';

test.describe('@e2e Memory Agent', () => {
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

  test('1. Memory extraction triggers after conversation', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    // Get session ID
    const sessions = await apiGet(page, '/sessions');
    const sessionId = sessions?.[0]?.session_id ?? '';

    // Have a substantive conversation
    await sendMessage(
      page,
      'Fetch ACE magnetic field magnitude for 2024-01-15 to 2024-01-16.',
      DATA_TIMEOUT,
    );
    await waitForStreamingDone(page, DATA_TIMEOUT);

    await sendMessage(
      page,
      'I prefer logarithmic y-axis for magnetic field plots. Remember this preference.',
      LLM_TIMEOUT,
    );
    await waitForStreamingDone(page, LLM_TIMEOUT);

    // Wait for memory extraction (async)
    console.log('[MEMORY TEST 1] Waiting 30s for memory extraction...');
    await page.waitForTimeout(30_000);

    // Check if memories were created
    if (sessionId) {
      try {
        const listResponse = await apiGet(page, `/sessions/${sessionId}/memories/list`);
        const memories = listResponse?.memories ?? [];
        if (Array.isArray(memories) && memories.length > 0) {
          console.log('[MEMORY TEST 1] Memories found:', memories.length);
          const hasPreference = memories.some((m: any) => m.type === 'preference');
          if (!hasPreference) {
            recordIssue({
              feature: 'memory',
              severity: 'major',
              title: 'No preference memory extracted from explicit preference',
              detail: `Found ${memories.length} memories but none of type "preference". Types: ${memories.map((m: any) => m.type).join(', ')}`,
            });
          }
        } else {
          console.log('[MEMORY TEST 1] No memories found yet (may need more turns)');
        }
      } catch (e: any) {
        console.log('[MEMORY TEST 1] Memory list error:', e.message);
      }
    }
  });

  test('2. Memory page displays and renders', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    await page.goto('/memory');
    await page.waitForLoadState('networkidle', { timeout: 10_000 });

    const pageContent = await page.textContent('body');
    if (!pageContent || pageContent.length < 50) {
      recordIssue({
        feature: 'memory',
        severity: 'major',
        title: 'Memory page failed to render',
        detail: `Content length: ${pageContent?.length ?? 0}`,
      });
    }

    const pageErrors = consoleErrors.filter(e =>
      e.includes('Uncaught') || e.includes('TypeError'),
    );
    if (pageErrors.length > 0) {
      recordIssue({
        feature: 'memory',
        severity: 'major',
        title: 'Console errors on Memory page',
        detail: pageErrors.join('\n'),
      });
    }

    await screenshotOnFailure(page, 'memory-page');
    console.log('[MEMORY TEST 2] Memory page content length:', pageContent?.length ?? 0);
  });

  test('3. Memory search API works', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const sessions = await apiGet(page, '/sessions');
    const sessionId = sessions?.[0]?.session_id ?? '';

    if (sessionId) {
      try {
        const searchResponse = await apiGet(
          page,
          `/sessions/${sessionId}/memories/search?q=magnetic+field`,
        );
        const searchResults = searchResponse?.results ?? [];
        expect(Array.isArray(searchResults)).toBeTruthy();
        console.log('[MEMORY TEST 3] Search results:', searchResults.length);
      } catch (e: any) {
        recordIssue({
          feature: 'memory',
          severity: 'major',
          title: 'Memory search API error',
          detail: `Error: ${e.message}`,
        });
      }
    }
  });

  test('4. Memory toggle global works', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const sessions = await apiGet(page, '/sessions');
    const sessionId = sessions?.[0]?.session_id ?? '';

    if (sessionId) {
      try {
        // Disable memory
        await apiPost(page, `/sessions/${sessionId}/memories/toggle-global`, { enabled: false });
        console.log('[MEMORY TEST 4] Disabled memory globally');

        // Re-enable
        await apiPost(page, `/sessions/${sessionId}/memories/toggle-global`, { enabled: true });
        console.log('[MEMORY TEST 4] Re-enabled memory globally');
      } catch (e: any) {
        recordIssue({
          feature: 'memory',
          severity: 'major',
          title: 'Memory toggle-global API error',
          detail: `Error: ${e.message}`,
        });
      }
    }
  });

  test('5. Memory archived list endpoint', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const sessions = await apiGet(page, '/sessions');
    const sessionId = sessions?.[0]?.session_id ?? '';

    if (sessionId) {
      try {
        const archivedResponse = await apiGet(page, `/sessions/${sessionId}/memories/archived`);
        const archived = archivedResponse?.archived ?? [];
        expect(Array.isArray(archived)).toBeTruthy();
        console.log('[MEMORY TEST 5] Archived memories:', archived.length);
      } catch (e: any) {
        recordIssue({
          feature: 'memory',
          severity: 'minor',
          title: 'Memory archived endpoint error',
          detail: `Error: ${e.message}`,
        });
      }
    }
  });

  test('6. Memory entries have required fields', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const sessions = await apiGet(page, '/sessions');
    const sessionId = sessions?.[0]?.session_id ?? '';

    if (sessionId) {
      const listResp = await apiGet(page, `/sessions/${sessionId}/memories/list`).catch(() => ({ memories: [] }));
      const memories = listResp?.memories ?? [];
      if (Array.isArray(memories) && memories.length > 0) {
        for (const mem of memories) {
          const requiredFields = ['id', 'type', 'scopes', 'content', 'enabled'];
          for (const field of requiredFields) {
            if (!(field in mem)) {
              recordIssue({
                feature: 'memory',
                severity: 'major',
                title: `Memory entry missing field: ${field}`,
                detail: `Memory keys: ${Object.keys(mem).join(', ')}`,
              });
            }
          }
          const validTypes = ['preference', 'summary', 'pitfall', 'reflection', 'review'];
          if (mem.type && !validTypes.includes(mem.type)) {
            recordIssue({
              feature: 'memory',
              severity: 'minor',
              title: `Unexpected memory type: ${mem.type}`,
              detail: `Expected one of: ${validTypes.join(', ')}`,
            });
          }
          if (mem.scopes && !Array.isArray(mem.scopes)) {
            recordIssue({
              feature: 'memory',
              severity: 'major',
              title: 'Memory scopes is not an array',
              detail: `Type: ${typeof mem.scopes}, value: ${JSON.stringify(mem.scopes)}`,
            });
          }
        }
        console.log('[MEMORY TEST 6] Validated', memories.length, 'memories');
      } else {
        console.log('[MEMORY TEST 6] No memories to validate');
      }
    }
  });

  test('7. Memory injection visible to agent', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const response = await sendAndGetText(
      page,
      'Do you have any long-term memories or preferences stored? What do you remember from previous sessions?',
      LLM_TIMEOUT,
    );

    const mentionsMemory = response.toLowerCase().includes('memory') ||
      response.toLowerCase().includes('remember') ||
      response.toLowerCase().includes('preference') ||
      response.toLowerCase().includes('previous session') ||
      response.toLowerCase().includes('long-term');

    if (!mentionsMemory) {
      recordIssue({
        feature: 'memory',
        severity: 'minor',
        title: 'Agent does not acknowledge memory system',
        detail: `Response: ${response.substring(0, 500)}`,
      });
    }

    console.log('[MEMORY TEST 7] Memory acknowledgment:', response.substring(0, 300));
  });

  test('8. Memory refresh endpoint', async ({ page }) => {
    test.setTimeout(WORKFLOW_TIMEOUT);
    await initApp(page);

    const sessions = await apiGet(page, '/sessions');
    const sessionId = sessions?.[0]?.session_id ?? '';

    if (sessionId) {
      try {
        const refreshResult = await apiPost(page, `/sessions/${sessionId}/memories/refresh`);
        console.log('[MEMORY TEST 8] Refresh result:', JSON.stringify(refreshResult).substring(0, 200));
      } catch (e: any) {
        recordIssue({
          feature: 'memory',
          severity: 'minor',
          title: 'Memory refresh API error',
          detail: `Error: ${e.message}`,
        });
      }
    }
  });
});
