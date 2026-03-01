/**
 * HTML template builder for sandboxed JSX/Recharts component rendering.
 *
 * Builds a self-contained HTML document with React + Recharts from CDN,
 * data injected as a window global, and auto-height via postMessage.
 */

// Pinned CDN versions for reproducibility
const REACT_VERSION = '18.3.1';
const RECHARTS_VERSION = '2.15.3';

/**
 * Build a self-contained HTML page that renders a compiled JSX component.
 *
 * @param bundleCode - The compiled ESM bundle code (from esbuild)
 * @param dataJson - JSON string of the data to inject
 * @returns Complete HTML string for use as iframe srcDoc
 */
export function buildIframeHtml(bundleCode: string, dataJson: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: transparent;
      overflow: hidden;
    }
    #root { width: 100%; min-height: 100px; }
    .error-boundary {
      padding: 16px;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 8px;
      color: #991b1b;
      font-size: 14px;
    }
    .error-boundary h3 { margin-bottom: 8px; font-size: 15px; }
    .error-boundary pre {
      white-space: pre-wrap;
      font-size: 12px;
      background: #fff;
      padding: 8px;
      border-radius: 4px;
      margin-top: 8px;
    }
  </style>
</head>
<body>
  <div id="root"></div>

  <!-- Data injection -->
  <script>
    window.__XHELIO_DATA__ = ${dataJson};
  </script>

  <!-- React + ReactDOM from CDN -->
  <script crossorigin src="https://unpkg.com/react@${REACT_VERSION}/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@${REACT_VERSION}/umd/react-dom.production.min.js"></script>

  <!-- Recharts from CDN (requires React as global) -->
  <script crossorigin src="https://unpkg.com/recharts@${RECHARTS_VERSION}/umd/Recharts.js"></script>

  <script type="module">
    // Map external imports to globals (esbuild --external)
    const React = window.React;
    const ReactDOM = window.ReactDOM;
    const Recharts = window.Recharts;

    // Create module shims for the ESM bundle
    const moduleShims = {
      'react': { ...React, default: React, createElement: React.createElement, createContext: React.createContext, useContext: React.useContext, useState: React.useState, useEffect: React.useEffect, useMemo: React.useMemo, useCallback: React.useCallback, useRef: React.useRef },
      'react-dom': { ...ReactDOM, default: ReactDOM },
      'react-dom/client': { ...ReactDOM, default: ReactDOM },
      'react/jsx-runtime': {
        jsx: React.createElement,
        jsxs: React.createElement,
        Fragment: React.Fragment,
      },
      'recharts': Recharts,
    };

    // Data context hooks (match the wrapper injected by jsx_sandbox.py)
    const DataContext = React.createContext(window.__XHELIO_DATA__ || {});

    function useData(label) {
      const data = React.useContext(DataContext);
      return data[label] || [];
    }

    function useAllLabels() {
      const data = React.useContext(DataContext);
      return Object.keys(data);
    }

    // Make hooks globally available for the bundle
    window.__useData = useData;
    window.__useAllLabels = useAllLabels;

    // Error boundary component
    class ErrorBoundary extends React.Component {
      constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
      }
      static getDerivedStateFromError(error) {
        return { hasError: true, error };
      }
      render() {
        if (this.state.hasError) {
          return React.createElement('div', { className: 'error-boundary' },
            React.createElement('h3', null, 'Component Error'),
            React.createElement('pre', null, String(this.state.error))
          );
        }
        return this.props.children;
      }
    }

    // Load and render the component
    try {
      // The bundle code is inlined and uses external markers for react/recharts.
      // We need to execute it with our module shims.
      const bundleCode = ${JSON.stringify(bundleCode)};

      // Replace external imports with our shims
      let execCode = bundleCode;

      // For ESM bundles, esbuild generates import statements for externals.
      // We need to replace them with variable declarations from our shims.
      const importReplacements = [
        [/import\\s*\\{([^}]+)\\}\\s*from\\s*["']react["'];?/g, (_, names) => {
          return names.split(',').map(n => {
            const trimmed = n.trim();
            const parts = trimmed.split(/\\s+as\\s+/);
            const src = parts[0].trim();
            const alias = parts.length > 1 ? parts[1].trim() : src;
            return \`var \${alias} = moduleShims['react'].\${src};\`;
          }).join('\\n');
        }],
        [/import\\s*\\*\\s*as\\s+(\\w+)\\s*from\\s*["']react["'];?/g, (_, name) => \`var \${name} = moduleShims['react'];\`],
        [/import\\s+(\\w+)\\s*from\\s*["']react["'];?/g, (_, name) => \`var \${name} = moduleShims['react'];\`],
        [/import\\s*\\{([^}]+)\\}\\s*from\\s*["']recharts["'];?/g, (_, names) => {
          return names.split(',').map(n => {
            const trimmed = n.trim();
            const parts = trimmed.split(/\\s+as\\s+/);
            const src = parts[0].trim();
            const alias = parts.length > 1 ? parts[1].trim() : src;
            return \`var \${alias} = moduleShims['recharts'].\${src};\`;
          }).join('\\n');
        }],
        [/import\\s*\\*\\s*as\\s+(\\w+)\\s*from\\s*["']recharts["'];?/g, (_, name) => \`var \${name} = moduleShims['recharts'];\`],
        [/import\\s*\\{([^}]+)\\}\\s*from\\s*["']react\\/jsx-runtime["'];?/g, (_, names) => {
          return names.split(',').map(n => {
            const trimmed = n.trim();
            const parts = trimmed.split(/\\s+as\\s+/);
            const src = parts[0].trim();
            const alias = parts.length > 1 ? parts[1].trim() : src;
            return \`var \${alias} = moduleShims['react/jsx-runtime'].\${src};\`;
          }).join('\\n');
        }],
        [/import\\s*\\{([^}]+)\\}\\s*from\\s*["']react-dom\\/client["'];?/g, (_, names) => {
          return names.split(',').map(n => {
            const trimmed = n.trim();
            const parts = trimmed.split(/\\s+as\\s+/);
            const src = parts[0].trim();
            const alias = parts.length > 1 ? parts[1].trim() : src;
            return \`var \${alias} = moduleShims['react-dom/client'].\${src};\`;
          }).join('\\n');
        }],
        [/import\\s*\\{([^}]+)\\}\\s*from\\s*["']react-dom["'];?/g, (_, names) => {
          return names.split(',').map(n => {
            const trimmed = n.trim();
            const parts = trimmed.split(/\\s+as\\s+/);
            const src = parts[0].trim();
            const alias = parts.length > 1 ? parts[1].trim() : src;
            return \`var \${alias} = moduleShims['react-dom'].\${src};\`;
          }).join('\\n');
        }],
      ];

      for (const [pattern, replacer] of importReplacements) {
        execCode = execCode.replace(pattern, replacer);
      }

      // Replace export default with a variable assignment
      execCode = execCode.replace(/export\\s*\\{\\s*(\\w+)\\s+as\\s+default\\s*\\};?/, 'var __default_export = $1;');
      execCode = execCode.replace(/export\\s+default\\s+/, 'var __default_export = ');

      // Inject useData and useAllLabels
      execCode = \`var useData = window.__useData; var useAllLabels = window.__useAllLabels;\\n\` + execCode;

      // Execute the transformed code
      const fn = new Function('moduleShims', 'React', 'Recharts', execCode + '\\nreturn __default_export;');
      const Component = fn(moduleShims, React, Recharts);

      if (!Component) {
        throw new Error('Component did not export a default function/class');
      }

      // Render
      const root = document.getElementById('root');
      const element = React.createElement(
        ErrorBoundary, null,
        React.createElement(Component)
      );

      if (ReactDOM.createRoot) {
        ReactDOM.createRoot(root).render(element);
      } else {
        ReactDOM.render(element, root);
      }

      // Auto-height via ResizeObserver -> postMessage
      const observer = new ResizeObserver(() => {
        const height = root.scrollHeight;
        window.parent.postMessage({ type: 'xhelio-resize', height }, '*');
      });
      observer.observe(root);

      // Initial height report
      requestAnimationFrame(() => {
        window.parent.postMessage({ type: 'xhelio-resize', height: root.scrollHeight }, '*');
      });

    } catch (err) {
      console.error('JSX render error:', err);
      const root = document.getElementById('root');
      root.innerHTML = '<div class="error-boundary"><h3>Render Error</h3><pre>' +
        String(err.message || err).replace(/</g, '&lt;') + '</pre></div>';
      window.parent.postMessage({ type: 'xhelio-resize', height: root.scrollHeight }, '*');
    }
  </script>
</body>
</html>`;
}
