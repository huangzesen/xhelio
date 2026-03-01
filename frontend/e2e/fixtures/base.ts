import { test as base } from '@playwright/test';
import { setupMockAPI, setupMockSSE, type ResponseOverrides } from './mock-api';
import { AppPage } from '../pages/app.page';
import { ChatPage } from '../pages/chat.page';
import { SettingsPage } from '../pages/settings.page';

/**
 * Extended test fixture that provides:
 * - Page objects (appPage, chatPage, settingsPage)
 * - Mock API setup (mockApi) — also clears localStorage and injects mock EventSource
 */
type Fixtures = {
  appPage: AppPage;
  chatPage: ChatPage;
  settingsPage: SettingsPage;
  mockApi: (overrides?: ResponseOverrides) => Promise<{ currentSessionId: string }>;
};

export const test = base.extend<Fixtures>({
  appPage: async ({ page }, use) => {
    await use(new AppPage(page));
  },

  chatPage: async ({ page }, use) => {
    await use(new ChatPage(page));
  },

  settingsPage: async ({ page }, use) => {
    await use(new SettingsPage(page));
  },

  mockApi: async ({ page }, use) => {
    const setup = async (overrides?: ResponseOverrides) => {
      // Clear any persisted state from previous test runs / real usage.
      // addInitScript runs before any page script, ensuring the app starts clean.
      await page.addInitScript(() => {
        localStorage.clear();
        sessionStorage.clear();
      });
      await setupMockSSE(page);
      return setupMockAPI(page, overrides);
    };
    await use(setup);
  },
});

export { expect } from '@playwright/test';
