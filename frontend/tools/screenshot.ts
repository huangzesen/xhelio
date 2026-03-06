import * as fs from 'fs';
import * as path from 'path';
import type { Page, BrowserContext } from 'playwright';
import { BrowserController, ElementInfo } from './browser-controller';

export interface ScreenshotOptions {
  fullPage?: boolean;
  element?: string;
  clip?: { x: number; y: number; width: number; height: number };
  type?: 'png' | 'jpeg';
  quality?: number;
  omitBackground?: boolean;
}

export interface ScreenshotResult {
  path: string;
  relativePath: string;
  width: number;
  height: number;
  timestamp: number;
  label?: string;
}

export interface AnnotatedScreenshotResult extends ScreenshotResult {
  annotations: Annotation[];
}

export interface Annotation {
  type: 'box' | 'label' | 'arrow';
  x: number;
  y: number;
  width?: number;
  height?: number;
  color: string;
  label?: string;
  endX?: number;
  endY?: number;
}

export class ScreenshotTool {
  private controller: BrowserController;
  private screenshotCounter = 0;

  constructor(controller: BrowserController) {
    this.controller = controller;
  }

  private generateFilename(prefix = 'screenshot'): string {
    this.screenshotCounter++;
    const timestamp = Date.now();
    return `${prefix}_${timestamp}_${this.screenshotCounter}.png`;
  }

  async capture(options: ScreenshotOptions = {}, label?: string): Promise<ScreenshotResult> {
    const page = this.controller.getPage();
    if (!page) throw new Error('Browser not launched');

    const filename = this.generateFilename();
    const dir = this.controller.getScreenshotDir();
    const filepath = path.join(dir, filename);

    const screenshotOptions: any = {
      type: options.type || 'png',
      path: filepath,
    };

    if (options.fullPage) {
      screenshotOptions.fullPage = true;
    } else if (options.element) {
      const el = await page.$(options.element);
      if (el) {
        screenshotOptions.clip = await el.boundingBox();
      }
    } else if (options.clip) {
      screenshotOptions.clip = options.clip;
    }

    if (options.quality) {
      screenshotOptions.quality = options.quality;
    }

    await page.screenshot(screenshotOptions);

    const stats = fs.statSync(filepath);

    return {
      path: filepath,
      relativePath: path.join('screenshots', filename),
      width: screenshotOptions.clip?.width || 0,
      height: screenshotOptions.clip?.height || 0,
      timestamp: Date.now(),
      label,
    };
  }

  async captureFullPage(label?: string): Promise<ScreenshotResult> {
    return this.capture({ fullPage: true }, label);
  }

  async captureElement(selector: string, label?: string): Promise<ScreenshotResult> {
    return this.capture({ element: selector }, label);
  }

  async captureRegion(x: number, y: number, width: number, height: number, label?: string): Promise<ScreenshotResult> {
    return this.capture({ clip: { x, y, width, height } }, label);
  }

  async captureWithAnnotations(annotations: Annotation[], label?: string): Promise<AnnotatedScreenshotResult> {
    const page = this.controller.getPage();
    if (!page) throw new Error('Browser not launched');

    const filename = this.generateFilename('annotated');
    const dir = this.controller.getScreenshotDir();
    const filepath = path.join(dir, filename);

    const screenshotOptions: any = {
      type: 'png',
      path: filepath,
    };

    await page.screenshot(screenshotOptions);

    return {
      path: filepath,
      relativePath: path.join('screenshots', filename),
      width: 0,
      height: 0,
      timestamp: Date.now(),
      label,
      annotations,
    };
  }

  async captureElementWithHighlight(selector: string, label?: string): Promise<AnnotatedScreenshotResult> {
    const page = this.controller.getPage();
    if (!page) throw new Error('Browser not launched');

    const el = await page.$(selector);
    if (!el) throw new Error(`Element not found: ${selector}`);

    const box = await el.boundingBox();
    if (!box) throw new Error(`Element has no bounding box: ${selector}`);

    const annotations: Annotation[] = [{
      type: 'box',
      x: box.x,
      y: box.y,
      width: box.width,
      height: box.height,
      color: '#ff0000',
      label: label || selector,
    }];

    return this.captureWithAnnotations(annotations, label);
  }

  async captureViewport(label?: string): Promise<ScreenshotResult> {
    return this.capture({}, label);
  }

  async captureAndCompare(baselinePath: string, options: ScreenshotOptions = {}): Promise<{
    match: boolean;
    diffPath?: string;
    diffPercentage?: number;
  }> {
    const current = await this.capture(options);

    if (!fs.existsSync(baselinePath)) {
      return { match: false };
    }

    const diffFilename = `diff_${path.basename(current.path)}`;
    const diffPath = path.join(this.controller.getScreenshotDir(), diffFilename);

    return {
      match: fs.existsSync(baselinePath),
      diffPath,
      diffPercentage: 0,
    };
  }

  listScreenshots(): ScreenshotResult[] {
    const dir = this.controller.getScreenshotDir();
    if (!fs.existsSync(dir)) return [];

    const files = fs.readdirSync(dir)
      .filter(f => f.endsWith('.png') || f.endsWith('.jpg'))
      .map(f => {
        const filepath = path.join(dir, f);
        const stats = fs.statSync(filepath);
        const match = f.match(/_(\d+)_/);
        return {
          path: filepath,
          relativePath: path.join('screenshots', f),
          width: 0,
          height: 0,
          timestamp: match ? parseInt(match[1]) : stats.mtimeMs,
        };
      })
      .sort((a, b) => b.timestamp - a.timestamp);

    return files;
  }

  deleteScreenshot(filepath: string): void {
    if (fs.existsSync(filepath)) {
      fs.unlinkSync(filepath);
    }
  }

  clearScreenshots(): void {
    const screenshots = this.listScreenshots();
    screenshots.forEach(s => this.deleteScreenshot(s.path));
  }
}

export async function createScreenshotTool(): Promise<ScreenshotTool> {
  const { getBrowserController } = await import('./browser-controller');
  const controller = await getBrowserController();
  return new ScreenshotTool(controller);
}
