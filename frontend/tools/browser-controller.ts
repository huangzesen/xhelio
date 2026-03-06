import { chromium, Browser, Page, BrowserContext, CDPSession } from 'playwright';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';

export interface BrowserControllerConfig {
  headless?: boolean;
  viewport?: { width: number; height: number };
  screenshotDir?: string;
}

export interface NavigateOptions {
  waitUntil?: 'load' | 'domcontentloaded' | 'networkidle';
  timeout?: number;
}

export interface ElementInfo {
  tag: string;
  id?: string;
  classes?: string;
  text?: string;
  xpath?: string;
  selector?: string;
  visible: boolean;
  enabled: boolean;
  boundingBox?: { x: number; y: number; width: number; height: number };
}

export class BrowserController {
  private browser: Browser | null = null;
  private context: BrowserContext | null = null;
  private page: Page | null = null;
  private cdpSession: CDPSession | null = null;
  private screenshotDir: string;

  constructor(config: BrowserControllerConfig = {}) {
    const downloadsDir = path.join(os.homedir(), 'Downloads', 'xhelio-screenshots');
    this.screenshotDir = config.screenshotDir || downloadsDir;
    if (!fs.existsSync(this.screenshotDir)) {
      fs.mkdirSync(this.screenshotDir, { recursive: true });
    }
  }

  async launch(headless = false): Promise<void> {
    this.browser = await chromium.launch({ headless });
    this.context = await this.browser.newContext({
      viewport: { width: 1400, height: 900 },
    });
    this.page = await this.context.newPage();
  }

  async connectToExisting(): Promise<void> {
    this.browser = await chromium.launch({ headless: true });
    this.context = await this.browser.newContext({
      viewport: { width: 1400, height: 900 },
    });
    this.page = await this.context.newPage();
  }

  async navigate(url: string, options: NavigateOptions = {}): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.goto(url, {
      waitUntil: options.waitUntil || 'networkidle',
      timeout: options.timeout || 30000,
    });
  }

  async click(selector: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.click(selector);
  }

  async fill(selector: string, value: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.fill(selector, value);
  }

  async type(selector: string, text: string, options?: { delay?: number }): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.type(selector, text, options);
  }

  async press(selector: string, key: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.press(selector, key);
  }

  async getText(selector: string): Promise<string> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.textContent(selector) || '';
  }

  async getAttribute(selector: string, attribute: string): Promise<string | null> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.getAttribute(selector, attribute);
  }

  async isVisible(selector: string): Promise<boolean> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.isVisible(selector);
  }

  async isEnabled(selector: string): Promise<boolean> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.isEnabled(selector);
  }

  async waitForSelector(selector: string, options?: { timeout?: number; state?: 'visible' | 'hidden' | 'attached' }): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.waitForSelector(selector, {
      timeout: options?.timeout || 10000,
      state: options?.state || 'visible',
    });
  }

  async waitForNavigation(urlPattern?: string | RegExp, options?: { timeout?: number }): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    if (urlPattern) {
      await this.page.waitForURL(urlPattern, { timeout: options?.timeout || 30000 });
    } else {
      await this.page.waitForLoadState('networkidle', { timeout: options?.timeout || 30000 });
    }
  }

  async getAllElements(selector: string): Promise<ElementInfo[]> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.evaluate((sel) => {
      const elements = document.querySelectorAll(sel);
      return Array.from(elements).map((el) => {
        const rect = el.getBoundingClientRect();
        return {
          tag: el.tagName.toLowerCase(),
          id: el.id || undefined,
          classes: el.className || undefined,
          text: el.textContent?.trim().substring(0, 100) || undefined,
          selector: sel,
          visible: el instanceof HTMLElement && el.offsetParent !== null,
          enabled: !(el as HTMLElement).disabled,
          boundingBox: rect.width > 0 && rect.height > 0 ? {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
          } : undefined,
        };
      });
    }, selector);
  }

  async getElementAtPosition(x: number, y: number): Promise<ElementInfo | null> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.evaluate(([mx, my]) => {
      const el = document.elementFromPoint(mx, my);
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      return {
        tag: el.tagName.toLowerCase(),
        id: el.id || undefined,
        classes: el.className || undefined,
        text: el.textContent?.trim().substring(0, 100) || undefined,
        visible: el instanceof HTMLElement && el.offsetParent !== null,
        enabled: !(el as HTMLElement).disabled,
        boundingBox: rect.width > 0 && rect.height > 0 ? {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height,
        } : undefined,
      };
    }, [x, y]);
  }

  async getPageState(): Promise<{
    url: string;
    title: string;
    route: string;
    loading: boolean;
  }> {
    if (!this.page) throw new Error('Browser not launched');
    const url = this.page.url();
    const title = await this.page.title();
    const hash = url.split('#')[1] || '/';
    return {
      url,
      title,
      route: hash || '/',
      loading: false,
    };
  }

  async getConsoleLogs(): Promise<{ type: string; text: string; time: number }[]> {
    if (!this.page) throw new Error('Browser not launched');
    return [];
  }

  async getJsErrors(): Promise<string[]> {
    if (!this.page) throw new Error('Browser not launched');
    return [];
  }

  async getPageHTML(): Promise<string> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.content();
  }

  async getElementHtml(selector: string): Promise<string> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.innerHTML(selector);
  }

  async evaluate<T>(fn: string | (() => T)): Promise<T> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.evaluate(fn);
  }

  async getUrl(): Promise<string> {
    if (!this.page) throw new Error('Browser not launched');
    return this.page.url();
  }

  async reload(): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.reload();
  }

  async goBack(): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.goBack();
  }

  async goForward(): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.goForward();
  }

  async scrollTo(selector: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.evaluate((sel) => {
      const el = document.querySelector(sel);
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, selector);
  }

  async scrollBy(x: number, y: number): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.evaluate(([dx, dy]) => {
      window.scrollBy(dx, dy);
    }, [x, y]);
  }

  async hover(selector: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.hover(selector);
  }

  async focus(selector: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.focus(selector);
  }

  async getInputValue(selector: string): Promise<string> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.inputValue(selector);
  }

  async selectOption(selector: string, value: string | string[]): Promise<string[]> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.selectOption(selector, value);
  }

  async check(selector: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.check(selector);
  }

  async uncheck(selector: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');
    await this.page.uncheck(selector);
  }

  async isChecked(selector: string): Promise<boolean> {
    if (!this.page) throw new Error('Browser not launched');
    return await this.page.isChecked(selector);
  }

  getPage(): Page | null {
    return this.page;
  }

  getContext(): BrowserContext | null {
    return this.context;
  }

  getScreenshotDir(): string {
    return this.screenshotDir;
  }

  async close(): Promise<void> {
    if (this.cdpSession) {
      await this.cdpSession.detach();
    }
    if (this.page) {
      await this.page.close();
    }
    if (this.context) {
      await this.context.close();
    }
    if (this.browser) {
      await this.browser.close();
    }
    this.page = null;
    this.context = null;
    this.browser = null;
    this.cdpSession = null;
  }
}

let globalController: BrowserController | null = null;

export async function getBrowserController(config?: BrowserControllerConfig): Promise<BrowserController> {
  if (!globalController) {
    globalController = new BrowserController(config);
    await globalController.launch(true);
  }
  return globalController;
}

export async function closeBrowserController(): Promise<void> {
  if (globalController) {
    await globalController.close();
    globalController = null;
  }
}
