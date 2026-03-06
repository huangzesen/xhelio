import type { ChatMessage } from '../api/types';

export type ExportFormat = 'base64' | 'local';

interface ExportOptions {
  sessionName?: string | null;
  tokenUsage?: Record<string, unknown>;
  format?: ExportFormat;
}

function getImageSrc(msg: ChatMessage): string | null {
  if (msg.figure_url) return msg.figure_url;
  if (msg.thumbnailUrl) return msg.thumbnailUrl;
  
  if (msg.mplImageUrl) return msg.mplImageUrl;
  
  if (msg.jsxSessionId && msg.jsxScriptId) {
    return `/api/sessions/${msg.jsxSessionId}/jsx-screenshots/${msg.jsxScriptId}.png`;
  }
  
  return null;
}

async function fetchImageAsBase64(url: string): Promise<string | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    const blob = await response.blob();
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

export async function exportSessionAsMarkdown(
  messages: ChatMessage[],
  sessionId: string,
  options?: ExportOptions,
): Promise<void> {
  const lines: string[] = [];
  const format = options?.format ?? 'base64';
  const imageUrls: Map<string, string> = new Map();

  const title = options?.sessionName || 'Xhelio Session Export';
  lines.push(`# ${title}`);
  lines.push('');
  lines.push(`Session: ${sessionId}`);
  lines.push(`Exported: ${new Date().toISOString()}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    if (msg.role === 'user') {
      lines.push(`## User`);
      lines.push('');
      lines.push(msg.content);
      lines.push('');
    } else if (msg.role === 'agent') {
      lines.push(`## Xhelio`);
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
    } else if (msg.role === 'plot') {
      const src = getImageSrc(msg);
      if (!src) {
        continue;
      }
      
      let imageRef: string;
      
      if (format === 'base64') {
        if (!imageUrls.has(src)) {
          const base64 = await fetchImageAsBase64(src);
          if (base64) {
            imageUrls.set(src, base64);
          }
        }
        const dataUrl = imageUrls.get(src);
        if (dataUrl) {
          imageRef = dataUrl;
        } else {
          continue;
        }
      } else {
        imageRef = src;
      }
      
      lines.push(`![Figure](${imageRef})`);
      lines.push('');
    }
    lines.push('---');
    lines.push('');
  }

  const usage = options?.tokenUsage;
  if (usage && Object.keys(usage).length > 0) {
    lines.push('## Token Usage');
    lines.push('');
    lines.push('| Metric | Count |');
    lines.push('|--------|-------|');
    for (const [key, value] of Object.entries(usage)) {
      const label = key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
      lines.push(`| ${label} | ${Number(value).toLocaleString()} |`);
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
