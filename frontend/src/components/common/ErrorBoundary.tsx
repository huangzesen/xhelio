import { Component, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  resetKey: number;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, resetKey: 0 };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  handleReset = () => {
    // Increment resetKey to force React to unmount and remount children
    this.setState((s) => ({ hasError: false, error: null, resetKey: s.resetKey + 1 }));
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-md">
            <div className="text-status-error-text text-lg font-semibold mb-2">Something went wrong</div>
            <p className="text-sm text-text-muted mb-4">{this.state.error?.message}</p>
            <button
              onClick={this.handleReset}
              className="px-4 py-2 rounded-lg bg-primary text-white text-sm hover:bg-primary-dark transition-colors"
            >
              Try again
            </button>
          </div>
        </div>
      );
    }

    // Wrapping children in a keyed div forces React to unmount/remount
    // all children when resetKey changes, clearing their internal state.
    // Using display:contents so this wrapper doesn't break flex/grid layouts.
    return <div key={this.state.resetKey} style={{ display: 'contents' }}>{this.props.children}</div>;
  }
}
