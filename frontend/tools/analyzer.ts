import { BrowserController, ElementInfo } from './browser-controller';

export interface UIComponent {
  id: string;
  type: string;
  role?: string;
  label?: string;
  text?: string;
  visible: boolean;
  enabled: boolean;
  children?: UIComponent[];
}

export interface PageAnalysis {
  url: string;
  route: string;
  title: string;
  components: UIComponent[];
  forms: FormInfo[];
  links: LinkInfo[];
  buttons: ButtonInfo[];
  inputs: InputInfo[];
  errors: string[];
  warnings: string[];
}

export interface FormInfo {
  selector: string;
  action?: string;
  method?: string;
  inputs: string[];
  buttons: string[];
}

export interface LinkInfo {
  href: string;
  text: string;
  visible: boolean;
}

export interface ButtonInfo {
  selector: string;
  text?: string;
  visible: boolean;
  enabled: boolean;
  type?: string;
}

export interface InputInfo {
  selector: string;
  type: string;
  name?: string;
  id?: string;
  placeholder?: string;
  value?: string;
  required?: boolean;
  disabled?: boolean;
}

export interface AccessibilityIssue {
  type: 'error' | 'warning' | 'info';
  element: string;
  message: string;
  selector: string;
  suggestion?: string;
}

export class UIAnalyzer {
  private controller: BrowserController;

  constructor(controller: BrowserController) {
    this.controller = controller;
  }

  async analyzePage(): Promise<PageAnalysis> {
    const page = this.controller.getPage();
    if (!page) throw new Error('Browser not launched');

    const state = await this.controller.getPageState();
    const components = await this.extractComponents();
    const forms = await this.findForms();
    const links = await this.findLinks();
    const buttons = await this.findButtons();
    const inputs = await this.findInputs();
    const errors = await this.findErrors();
    const warnings = await this.findWarnings();

    return {
      url: state.url,
      route: state.route,
      title: state.title,
      components,
      forms,
      links,
      buttons,
      inputs,
      errors,
      warnings,
    };
  }

  private async extractComponents(): Promise<UIComponent[]> {
    return await this.controller.evaluate(() => {
      const components: UIComponent[] = [];
      const interactiveSelectors = [
        'button', 'a', 'input', 'select', 'textarea',
        '[role="button"]', '[role="link"]', '[role="textbox"]',
        '[role="combobox"]', '[role="menuitem"]', 'nav', 'header', 'main', 'section',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span',
      ];

      const seen = new Set<Element>();

      interactiveSelectors.forEach(selector => {
        try {
          document.querySelectorAll(selector).forEach(el => {
            if (seen.has(el)) return;
            seen.add(el);

            const role = el.getAttribute('role');
            const ariaLabel = el.getAttribute('aria-label');
            const ariaLabelledby = el.getAttribute('aria-labelledby');
            const text = el.textContent?.trim().substring(0, 50);

            components.push({
              id: el.id || `el-${components.length}`,
              type: el.tagName.toLowerCase(),
              role: role || undefined,
              label: ariaLabel || (ariaLabelledby ? `#${ariaLabelledby}` : undefined),
              text: text || undefined,
              visible: el instanceof HTMLElement && el.offsetParent !== null,
              enabled: !(el as HTMLElement).disabled,
            });
          });
        } catch {
          // Ignore invalid selectors
        }
      });

      return components;
    });
  }

  private async findForms(): Promise<FormInfo[]> {
    return await this.controller.evaluate(() => {
      const forms = document.querySelectorAll('form');
      return Array.from(forms).map(form => {
        const inputs = Array.from(form.querySelectorAll('input, select, textarea'))
          .map(el => el.name || el.id || el.tagName.toLowerCase());
        const buttons = Array.from(form.querySelectorAll('button, input[type="submit"]'))
          .map(el => el.textContent?.trim() || el.getAttribute('value') || 'submit');

        return {
          selector: form.id ? `#${form.id}` : 'form',
          action: form.action || undefined,
          method: form.method || undefined,
          inputs,
          buttons,
        };
      });
    });
  }

  private async findLinks(): Promise<LinkInfo[]> {
    return await this.controller.evaluate(() => {
      const links = document.querySelectorAll('a[href]');
      return Array.from(links).map(link => ({
        href: link.getAttribute('href') || '',
        text: link.textContent?.trim() || '',
        visible: link instanceof HTMLElement && link.offsetParent !== null,
      }));
    });
  }

  private async findButtons(): Promise<ButtonInfo[]> {
    return await this.controller.evaluate(() => {
      const buttons = document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]');
      return Array.from(buttons).map(btn => ({
        selector: btn.id ? `#${btn.id}` : '',
        text: btn.textContent?.trim() || btn.getAttribute('value') || undefined,
        visible: btn instanceof HTMLElement && btn.offsetParent !== null,
        enabled: !(btn as HTMLButtonElement).disabled,
        type: btn.getAttribute('type') || undefined,
      })).filter(b => b.selector || b.text);
    });
  }

  private async findInputs(): Promise<InputInfo[]> {
    return await this.controller.evaluate(() => {
      const inputs = document.querySelectorAll('input, select, textarea');
      return Array.from(inputs).map(input => {
        const tag = input.tagName.toLowerCase();
        return {
          selector: input.id ? `#${input.id}` : input.name ? `[name="${input.name}"]` : tag,
          type: (input as HTMLInputElement).type || tag,
          name: input.name || undefined,
          id: input.id || undefined,
          placeholder: (input as HTMLInputElement).placeholder || undefined,
          value: (input as HTMLInputElement).value || undefined,
          required: input.hasAttribute('required'),
          disabled: (input as HTMLInputElement).disabled,
        };
      });
    });
  }

  private async findErrors(): Promise<string[]> {
    return await this.controller.evaluate(() => {
      const errors: string[] = [];

      document.querySelectorAll('[role="alert"], .error, .Error, [aria-invalid="true"]').forEach(el => {
        const text = el.textContent?.trim();
        if (text) errors.push(text);
      });

      return errors;
    });
  }

  private async findWarnings(): Promise<string[]> {
    return await this.controller.evaluate(() => {
      const warnings: string[] = [];

      document.querySelectorAll('.warning, .Warning, [role="warning"]').forEach(el => {
        const text = el.textContent?.trim();
        if (text) warnings.push(text);
      });

      return warnings;
    });
  }

  async checkAccessibility(): Promise<AccessibilityIssue[]> {
    const page = this.controller.getPage();
    if (!page) throw new Error('Browser not launched');

    const issues: AccessibilityIssue[] = [];

    const results = await page.evaluate(() => {
      const issues: AccessibilityIssue[] = [];

      const interactiveElements = document.querySelectorAll('button, a, input, select, textarea, [role]');
      interactiveElements.forEach(el => {
        const tag = el.tagName.toLowerCase();
        const hasAriaLabel = el.hasAttribute('aria-label') || el.hasAttribute('aria-labelledby');
        const hasLabel = el.id && document.querySelector(`label[for="${el.id}"]`);

        if (!hasAriaLabel && !hasLabel && el.textContent?.trim() === '') {
          issues.push({
            type: 'warning',
            element: `${tag}${el.id ? `#${el.id}` : ''}`,
            message: 'Interactive element lacks accessible name',
            selector: el.id ? `#${el.id}` : tag,
            suggestion: 'Add aria-label or associate with a label',
          });
        }

        if (el.getAttribute('role') === 'img' && !el.hasAttribute('alt')) {
          issues.push({
            type: 'error',
            element: `${tag}#${el.id || ''}`,
            message: 'Image missing alt text',
            selector: el.id ? `#${el.id}` : tag,
            suggestion: 'Add alt attribute for accessibility',
          });
        }
      });

      return issues;
    });

    return results;
  }

  async findElementByText(text: string, exact = false): Promise<ElementInfo[]> {
    const page = this.controller.getPage();
    if (!page) throw new Error('Browser not launched');

    const escapedText = text.replace(/'/g, "\\'");
    const selector = exact
      ? `*:text-is('${escapedText}')`
      : `*:text('${escapedText}')`;

    return this.controller.getAllElements(selector);
  }

  async findElementByRole(role: string): Promise<ElementInfo[]> {
    return this.controller.getAllElements(`[role="${role}"]`);
  }

  async findElementByTestId(testId: string): Promise<ElementInfo[]> {
    return this.controller.getAllElements(`[data-testid="${testId}"]`);
  }

  async getElementInfo(selector: string): Promise<ElementInfo | null> {
    const elements = await this.controller.getAllElements(selector);
    return elements[0] || null;
  }

  async getComputedStyle(selector: string, property: string): Promise<string | null> {
    const page = this.controller.getPage();
    if (!page) throw new Error('Browser not launched');

    return await page.evaluate(([sel, prop]) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      return window.getComputedStyle(el).getPropertyValue(prop);
    }, [selector, property]);
  }

  async getElementPosition(selector: string): Promise<{ x: number; y: number } | null> {
    const info = await this.getElementInfo(selector);
    if (!info?.boundingBox) return null;
    return { x: info.boundingBox.x, y: info.boundingBox.y };
  }

  async isElementInViewport(selector: string): Promise<boolean> {
    const page = this.controller.getPage();
    if (!page) throw new Error('Browser not launched');

    return await page.evaluate((sel) => {
      const el = document.querySelector(sel);
      if (!el) return false;
      const rect = el.getBoundingClientRect();
      return rect.top >= 0 && rect.left >= 0 &&
        rect.bottom <= window.innerHeight &&
        rect.right <= window.innerWidth;
    }, selector);
  }

  async getZIndex(selector: string): Promise<number> {
    const style = await this.getComputedStyle(selector, 'z-index');
    return style ? parseInt(style) : 0;
  }

  async getColorPalette(): Promise<string[]> {
    const page = this.controller.getPage();
    if (!page) throw new Error('Browser not launched');

    return await page.evaluate(() => {
      const colors = new Set<string>();
      const elements = document.querySelectorAll('*');
      elements.forEach(el => {
        const style = window.getComputedStyle(el);
        const bg = style.backgroundColor;
        const color = style.color;
        if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') colors.add(bg);
        if (color && color !== 'rgba(0, 0, 0, 0)' && color !== 'transparent') colors.add(color);
      });
      return Array.from(colors).slice(0, 20);
    });
  }
}

export async function createUIAnalyzer(): Promise<UIAnalyzer> {
  const { getBrowserController } = await import('./browser-controller');
  const controller = await getBrowserController();
  return new UIAnalyzer(controller);
}
