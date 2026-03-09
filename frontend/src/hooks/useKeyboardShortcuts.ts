import { useEffect, useRef } from 'react';

interface ShortcutHandlers {
  onToggleSidebar?: () => void;
  onToggleActivity?: () => void;
  onNewChat?: () => void;
  onCommandPalette?: () => void;
  onFocusInput?: () => void;
  onShortcutsHelp?: () => void;
}

export function useKeyboardShortcuts(handlers: ShortcutHandlers) {
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const meta = e.metaKey || e.ctrlKey;
      const target = e.target as HTMLElement;
      const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT';

      if (meta && e.key === 'b') { e.preventDefault(); handlersRef.current.onToggleSidebar?.(); return; }
      if (meta && e.key === '\\') { e.preventDefault(); handlersRef.current.onToggleActivity?.(); return; }
      if (meta && e.key === 'n') { e.preventDefault(); handlersRef.current.onNewChat?.(); return; }
      if (meta && e.key === 'k') { e.preventDefault(); handlersRef.current.onCommandPalette?.(); return; }
      if (meta && e.key === '?') { e.preventDefault(); handlersRef.current.onShortcutsHelp?.(); return; }
      if (e.key === '/' && !isInput) { e.preventDefault(); handlersRef.current.onFocusInput?.(); return; }
    }

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps -- handlers accessed via ref
}
