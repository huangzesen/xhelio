import { useEffect } from 'react';

interface ShortcutHandlers {
  onToggleSidebar?: () => void;
  onToggleActivity?: () => void;
  onNewChat?: () => void;
  onCommandPalette?: () => void;
  onFocusInput?: () => void;
  onShortcutsHelp?: () => void;
}

export function useKeyboardShortcuts(handlers: ShortcutHandlers) {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const meta = e.metaKey || e.ctrlKey;
      const target = e.target as HTMLElement;
      const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT';

      // Cmd+B — toggle sidebar
      if (meta && e.key === 'b') {
        e.preventDefault();
        handlers.onToggleSidebar?.();
        return;
      }

      // Cmd+\ — toggle activity panel
      if (meta && e.key === '\\') {
        e.preventDefault();
        handlers.onToggleActivity?.();
        return;
      }

      // Cmd+N — new chat
      if (meta && e.key === 'n') {
        e.preventDefault();
        handlers.onNewChat?.();
        return;
      }

      // Cmd+K — command palette
      if (meta && e.key === 'k') {
        e.preventDefault();
        handlers.onCommandPalette?.();
        return;
      }

      // Cmd+? — keyboard shortcuts help
      if (meta && e.key === '?') {
        e.preventDefault();
        handlers.onShortcutsHelp?.();
        return;
      }

      // / — focus chat input (only when not already in an input)
      if (e.key === '/' && !isInput) {
        e.preventDefault();
        handlers.onFocusInput?.();
        return;
      }
    }

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handlers]);
}
