import type { ChatMessage, PlotlyFigure } from '../api/types';

interface ExportOptions {
  sessionName?: string | null;
  tokenUsage?: Record<string, number>;
}

/**
 * Render a PlotlyFigure to a base64 PNG data URL using an offscreen div.
 * Returns null if Plotly is not available or rendering fails.
 */
async function figureToBase64(figure: PlotlyFigure): Promise<string | null> {
  const Plotly = (window as unknown as Record<string, unknown>).Plotly as {
    newPlot: (el: HTMLElement, data: Plotly.Data[], layout: Partial<Plotly.Layout>) => Promise<unknown>;
    toImage: (el: HTMLElement, opts: Record<string, unknown>) => Promise<string>;
    purge: (el: HTMLElement) => void;
  } | undefined;
  if (!Plotly) return null;

  const div = document.createElement('div');
  div.style.position = 'absolute';
  div.style.left = '-9999px';
  div.style.width = '800px';
  div.style.height = '500px';
  document.body.appendChild(div);

  try {
    await Plotly.newPlot(div, figure.data, {
      ...figure.layout,
      width: 800,
      height: 500,
    });
    const dataUrl = await Plotly.toImage(div, {
      format: 'png',
      width: 800,
      height: 500,
    });
    return dataUrl; // "data:image/png;base64,..."
  } catch {
    return null;
  } finally {
    try { Plotly.purge(div); } catch { /* ignore */ }
    document.body.removeChild(div);
  }
}

export async function exportSessionAsMarkdown(
  messages: ChatMessage[],
  sessionId: string,
  options?: ExportOptions,
): Promise<void> {
  const lines: string[] = [];

  // Header
  const title = options?.sessionName || 'XHelio Session Export';
  lines.push(`# ${title}`);
  lines.push('');
  lines.push(`Session: ${sessionId}`);
  lines.push(`Exported: ${new Date().toISOString()}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  // Messages
  for (const msg of messages) {
    if (msg.role === 'user') {
      lines.push(`## User`);
      lines.push('');
      lines.push(msg.content);
      lines.push('');
    } else if (msg.role === 'agent') {
      lines.push(`## XHelio`);
      lines.push('');
      lines.push(msg.content);
      lines.push('');
    } else if (msg.role === 'thinking') {
      lines.push(`<details><summary>Agent thinking</summary>`);
      lines.push('');
      lines.push(msg.content);
      lines.push('');
      lines.push('</details>');
      lines.push('');
    } else if (msg.role === 'plot' && msg.figure) {
      const dataUrl = await figureToBase64(msg.figure);
      if (dataUrl) {
        lines.push(`![Plot](${dataUrl})`);
        lines.push('');
      }
    }
    lines.push('---');
    lines.push('');
  }

  // Token usage footer
  const usage = options?.tokenUsage;
  if (usage && Object.keys(usage).length > 0) {
    lines.push('## Token Usage');
    lines.push('');
    lines.push('| Metric | Count |');
    lines.push('|--------|-------|');
    for (const [key, value] of Object.entries(usage)) {
      const label = key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
      lines.push(`| ${label} | ${value.toLocaleString()} |`);
    }
    lines.push('');
  }

  const content = lines.join('\n');
  const blob = new Blob([content], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `xhelio-session-${sessionId.slice(0, 8)}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
