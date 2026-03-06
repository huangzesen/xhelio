import { BrowserController } from './browser-controller';
import { UIAnalyzer, PageAnalysis } from './analyzer';

export interface BugReport {
  id: string;
  timestamp: number;
  severity: 'critical' | 'high' | 'medium' | 'low';
  category: string;
  title: string;
  description: string;
  selector?: string;
  screenshotPath?: string;
  stackTrace?: string;
  suggestedFix?: string;
}

export interface NavigationBug {
  type: 'broken-link' | 'dead-end' | 'missing-route' | 'redirect-loop';
  from: string;
  to?: string;
  message: string;
}

export interface StateBug {
  type: 'stale-data' | 'memory-leak' | 'race-condition' | 'state-mismatch';
  description: string;
  evidence: string;
}

export class BugDetector {
  private controller: BrowserController;
  private analyzer: UIAnalyzer;

  constructor(controller: BrowserController, analyzer: UIAnalyzer) {
    this.controller = controller;
    this.analyzer = analyzer;
  }

  async detectAll(): Promise<BugReport[]> {
    const bugs: BugReport[] = [];

    const pageBugs = await this.detectPageIssues();
    bugs.push(...pageBugs);

    const navBugs = await this.detectNavigationIssues();
    bugs.push(...navBugs);

    const stateBugs = await this.detectStateIssues();
    bugs.push(...stateBugs);

    const accessibilityBugs = await this.detectAccessibilityIssues();
    bugs.push(...accessibilityBugs);

    const performanceBugs = await this.detectPerformanceIssues();
    bugs.push(...performanceBugs);

    return bugs;
  }

  private async detectPageIssues(): Promise<BugReport[]> {
    const bugs: BugReport[] = [];
    const page = this.controller.getPage();
    if (!page) return bugs;

    const consoleErrors = await page.evaluate(() => {
      return (window as any).__consoleErrors || [];
    });

    if (consoleErrors.length > 0) {
      consoleErrors.forEach((err: string) => {
        bugs.push({
          id: `console-error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          timestamp: Date.now(),
          severity: 'high',
          category: 'runtime-error',
          title: 'Console Error Detected',
          description: err,
          stackTrace: err,
        });
      });
    }

    const brokenImages = await this.analyzer.findElementByRole('img').then(imgs => 
      imgs.filter(img => !img.classes?.includes('hidden'))
    );

    if (brokenImages.length > 0) {
      bugs.push({
        id: `broken-images-${Date.now()}`,
        timestamp: Date.now(),
        severity: 'low',
        category: 'broken-resource',
        title: 'Potential Broken Images',
        description: `Found ${brokenImages.length} images to verify`,
      });
    }

    return bugs;
  }

  private async detectNavigationIssues(): Promise<BugReport[]> {
    const bugs: BugReport[] = [];
    const analysis = await this.analyzer.analyzePage();

    for (const link of analysis.links) {
      if (link.href === '#' || link.href === 'javascript:void(0)' || link.href === '') {
        const matchingButtons = analysis.buttons.filter(b => 
          b.text?.toLowerCase().includes(link.text.toLowerCase())
        );

        if (matchingButtons.length === 0) {
          bugs.push({
            id: `dead-link-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp: Date.now(),
            severity: 'medium',
            category: 'navigation',
            title: 'Non-functional link',
            description: `Link "${link.text}" has no valid href`,
            suggestedFix: 'Add proper href or convert to button',
          });
        }
      }

      if (link.href.startsWith('/') || link.href.startsWith('#')) {
        const state = await this.controller.getPageState();
        const targetUrl = link.href.startsWith('#') 
          ? state.url.split('#')[0] + link.href 
          : link.href;

        if (link.href === state.route || link.href === '#' + state.route) {
          bugs.push({
            id: `self-link-${Date.now()}`,
            timestamp: Date.now(),
            severity: 'low',
            category: 'navigation',
            title: 'Link points to current page',
            description: `Link "${link.text}" points to the current route`,
          });
        }
      }
    }

    const brokenButtons = analysis.buttons.filter(b => b.visible && !b.enabled);
    if (brokenButtons.length > 0) {
      bugs.push({
        id: `disabled-buttons-${Date.now()}`,
        timestamp: Date.now(),
        severity: 'low',
        category: 'ui-state',
        title: 'Disabled interactive elements',
        description: `Found ${brokenButtons.length} visible but disabled buttons`,
      });
    }

    return bugs;
  }

  private async detectStateIssues(): Promise<BugReport[]> {
    const bugs: BugReport[] = [];
    const page = this.controller.getPage();
    if (!page) return bugs;

    const staleElements = await page.evaluate(() => {
      const stale: string[] = [];
      const inputs = document.querySelectorAll('input, select, textarea');
      inputs.forEach(input => {
        if (input.classList.contains('stale') || input.dataset.stale === 'true') {
          stale.push(input.id || input.name || input.tagName);
        }
      });
      return stale;
    });

    if (staleElements.length > 0) {
      bugs.push({
        id: `stale-data-${Date.now()}`,
        timestamp: Date.now(),
        severity: 'medium',
        category: 'state',
        title: 'Stale data detected',
        description: `Found stale elements: ${staleElements.join(', ')}`,
      });
    }

    const loadingStates = await page.evaluate(() => {
      const stuck: string[] = [];
      const spinners = document.querySelectorAll('[class*="spinner"], [class*="loading"], [data-loading="true"]');
      spinners.forEach(el => {
        const style = window.getComputedStyle(el);
        if (style.display !== 'none' && style.visibility !== 'hidden') {
          stuck.push(el.className || el.tagName);
        }
      });
      return stuck;
    });

    if (loadingStates.length > 0) {
      bugs.push({
        id: `stuck-loading-${Date.now()}`,
        timestamp: Date.now(),
        severity: 'high',
        category: 'state',
        title: 'Stuck loading state',
        description: `Found potentially stuck loading indicators: ${loadingStates.join(', ')}`,
      });
    }

    return bugs;
  }

  private async detectAccessibilityIssues(): Promise<BugReport[]> {
    const bugs: BugReport[] = [];
    const issues = await this.analyzer.checkAccessibility();

    issues.forEach(issue => {
      bugs.push({
        id: `a11y-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: Date.now(),
        severity: issue.type === 'error' ? 'high' : 'medium',
        category: 'accessibility',
        title: `Accessibility: ${issue.message}`,
        description: issue.message,
        selector: issue.selector,
        suggestedFix: issue.suggestion,
      });
    });

    return bugs;
  }

  private async detectPerformanceIssues(): Promise<BugReport[]> {
    const bugs: BugReport[] = [];
    const page = this.controller.getPage();
    if (!page) return bugs;

    const metrics = await page.evaluate(() => {
      const perfData = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      const resources = performance.getEntriesByType('resource');
      
      return {
        domContentLoaded: perfData?.domContentLoadedEventEnd - perfData?.domContentLoadedEventStart || 0,
        loadComplete: perfData?.loadEventEnd - perfData?.fetchStart || 0,
        slowResources: resources.filter(r => r.duration > 3000).map(r => ({
          name: r.name,
          duration: r.duration,
        })),
      };
    });

    if (metrics.domContentLoaded > 5000) {
      bugs.push({
        id: `slow-dom-${Date.now()}`,
        timestamp: Date.now(),
        severity: 'medium',
        category: 'performance',
        title: 'Slow DOM content loaded',
        description: `DOM content loaded in ${(metrics.domContentLoaded / 1000).toFixed(2)}s`,
      });
    }

    if (metrics.loadComplete > 10000) {
      bugs.push({
        id: `slow-page-${Date.now()}`,
        timestamp: Date.now(),
        severity: 'high',
        category: 'performance',
        title: 'Slow page load',
        description: `Page fully loaded in ${(metrics.loadComplete / 1000).toFixed(2)}s`,
      });
    }

    metrics.slowResources.forEach(resource => {
      bugs.push({
        id: `slow-resource-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: Date.now(),
        severity: 'low',
        category: 'performance',
        title: 'Slow resource load',
        description: `${resource.name} took ${(resource.duration / 1000).toFixed(2)}s`,
      });
    });

    return bugs;
  }

  async testFormValidation(): Promise<BugReport[]> {
    const bugs: BugReport[] = [];
    const analysis = await this.analyzer.analyzePage();

    for (const form of analysis.forms) {
      const requiredInputs = form.inputs.filter((_, i) => {
        const input = analysis.inputs[i];
        return input?.required;
      });

      if (requiredInputs.length > 0) {
        const submitButton = form.buttons.find(b => b.toLowerCase().includes('submit'));
        if (!submitButton) {
          bugs.push({
            id: `no-submit-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp: Date.now(),
            severity: 'medium',
            category: 'form-validation',
            title: 'Form has required fields but no submit button',
            description: `Form with ${requiredInputs.length} required fields lacks a submit mechanism`,
            selector: form.selector,
          });
        }
      }
    }

    return bugs;
  }

  async testResponsiveBreakpoints(): Promise<{ width: number; bugs: BugReport[] }[]> {
    const breakpoints = [375, 768, 1024, 1440];
    const results: { width: number; bugs: BugReport[] }[] = [];

    const context = this.controller.getContext();
    if (!context) return results;

    for (const width of breakpoints) {
      await context.setViewportSize({ width, height: 900 });
      await this.controller.reload();

      const bugs = await this.detectAll();
      results.push({ width, bugs });

      if (bugs.length > 0) {
        console.log(`Found ${bugs.length} issues at ${width}px width`);
      }
    }

    await context.setViewportSize({ width: 1400, height: 900 });

    return results;
  }

  generateReport(bugs: BugReport[]): string {
    const bySeverity = {
      critical: bugs.filter(b => b.severity === 'critical'),
      high: bugs.filter(b => b.severity === 'high'),
      medium: bugs.filter(b => b.severity === 'medium'),
      low: bugs.filter(b => b.severity === 'low'),
    };

    let report = `# Bug Report\n\n`;
    report += `Generated: ${new Date().toISOString()}\n`;
    report += `Total Issues: ${bugs.length}\n\n`;

    report += `## Summary\n`;
    report += `- Critical: ${bySeverity.critical.length}\n`;
    report += `- High: ${bySeverity.high.length}\n`;
    report += `- Medium: ${bySeverity.medium.length}\n`;
    report += `- Low: ${bySeverity.low.length}\n\n`;

    if (bugs.length > 0) {
      report += `## Issues\n\n`;
      bugs.forEach((bug, i) => {
        report += `### ${i + 1}. [${bug.severity.toUpperCase()}] ${bug.title}\n`;
        report += `**Category:** ${bug.category}\n`;
        report += `**Description:** ${bug.description}\n`;
        if (bug.selector) report += `**Selector:** \`${bug.selector}\`\n`;
        if (bug.suggestedFix) report += `**Suggested Fix:** ${bug.suggestedFix}\n`;
        if (bug.stackTrace) report += `\`\`\`\n${bug.stackTrace}\n\`\`\`\n`;
        report += `\n`;
      });
    }

    return report;
  }
}

export async function createBugDetector(): Promise<BugDetector> {
  const { getBrowserController } = await import('./browser-controller');
  const { createUIAnalyzer } = await import('./analyzer');
  
  const controller = await getBrowserController();
  const analyzer = await createUIAnalyzer();
  
  return new BugDetector(controller, analyzer);
}
