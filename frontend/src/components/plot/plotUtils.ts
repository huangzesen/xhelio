import type { PlotlyFigure } from '../../api/types';

/** Apply dark theme overrides to all axes in the layout. */
export function applyDarkTheme(layout: Partial<Plotly.Layout>): Partial<Plotly.Layout> {
  const overrides: Record<string, unknown> = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#94a3b8' },
  };
  for (const key in layout) {
    if (key.startsWith('xaxis') || key.startsWith('yaxis')) {
      overrides[key] = {
        ...(layout as Record<string, unknown>)[key] as object,
        gridcolor: '#1e293b',
        zerolinecolor: '#334155',
      };
    }
  }
  return overrides as Partial<Plotly.Layout>;
}

/** Extract the plot title string from a Plotly layout. */
export function getPlotTitle(figure: PlotlyFigure): string {
  const { title } = figure.layout;
  if (!title) return '';
  if (typeof title === 'string') return title;
  if (typeof title === 'object' && 'text' in title) return title.text ?? '';
  return '';
}

/** Count the number of y-axis panels in the layout. */
export function countPanels(layout: Partial<Plotly.Layout>): number {
  let count = 0;
  for (const key in layout) {
    if (/^yaxis\d*$/.test(key)) count++;
  }
  return Math.max(count, 1);
}
