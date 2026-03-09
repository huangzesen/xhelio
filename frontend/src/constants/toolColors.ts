/** Tool -> badge color mapping. Single source of truth for pipeline UI. */
export const TOOL_COLORS: Record<string, string> = {
  fetch_data: 'bg-badge-blue-bg text-badge-blue-text',
  run_code: 'bg-badge-orange-bg text-badge-orange-text',
  render_plotly_json: 'bg-badge-pink-bg text-badge-pink-text',
  generate_mpl_script: 'bg-badge-pink-bg text-badge-pink-text',
  generate_jsx_component: 'bg-badge-pink-bg text-badge-pink-text',
  manage_plot: 'bg-badge-gray-bg text-badge-gray-text',
};

/** Render/presentation tool names. Must match backend rendering.registry.RENDER_TOOL_NAMES. */
export const RENDER_TOOL_NAMES = new Set([
  'render_plotly_json',
  'generate_mpl_script',
  'generate_jsx_component',
]);
