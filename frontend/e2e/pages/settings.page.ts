import type { Page } from '@playwright/test';

/**
 * Page object for the Settings page.
 */
export class SettingsPage {
  readonly page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  async goto() {
    await this.page.goto('/settings');
  }

  /** Get the page heading */
  async getHeading() {
    return this.page.getByRole('heading').first().textContent();
  }
}
