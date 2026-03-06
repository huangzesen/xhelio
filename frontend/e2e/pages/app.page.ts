import type { Page, Locator } from '@playwright/test';

/**
 * Page object for the App shell — loading state, setup screen, error state, navigation.
 */
export class AppPage {
  readonly page: Page;
  readonly loadingSpinner: Locator;
  readonly errorContainer: Locator;
  readonly setupScreen: Locator;
  readonly header: Locator;
  readonly mainContent: Locator;

  constructor(page: Page) {
    this.page = page;
    this.loadingSpinner = page.getByTestId('app-loading');
    this.errorContainer = page.getByTestId('app-error');
    this.setupScreen = page.getByTestId('setup-screen');
    this.header = page.getByTestId('app-header');
    this.mainContent = page.locator('#main-content');
  }

  async goto(path = '/') {
    await this.page.goto(path);
  }

  /** Wait for the app to fully load (header visible, no loading spinner) */
  async waitForReady() {
    await this.header.waitFor({ state: 'visible', timeout: 10000 });
  }

  /** Check if the loading spinner is visible */
  async isLoading() {
    return this.loadingSpinner.isVisible();
  }

  /** Check if the setup screen is shown */
  async isShowingSetupScreen() {
    return this.setupScreen.isVisible();
  }

  /** Check if the error container is shown */
  async isShowingError() {
    return this.errorContainer.isVisible();
  }

  /** Get the error message text */
  async getErrorText() {
    return this.errorContainer.textContent();
  }

  /** Click a navigation link by its label */
  async navigateTo(label: string) {
    await this.page.getByRole('navigation', { name: 'Main navigation' }).getByText(label).click();
  }

  /** Toggle the sidebar via the header button */
  async toggleSidebar() {
    await this.page.getByLabel('Toggle sidebar').click();
  }

  /** Toggle the activity panel via the header button */
  async toggleActivity() {
    await this.page.getByLabel('Toggle activity panel').click();
  }

  /** Toggle the theme (dark/light) */
  async toggleTheme() {
    await this.page.getByLabel(/Switch to .* mode/).click();
  }
}
