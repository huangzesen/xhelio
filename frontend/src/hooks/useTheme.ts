import { useState, useEffect, useCallback, useSyncExternalStore } from 'react';

type Theme = 'light' | 'dark';

function getInitialTheme(): Theme {
  const stored = localStorage.getItem('theme') as Theme | null;
  if (stored === 'light' || stored === 'dark') return stored;
  return 'dark'; // default to dark
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
    _notifyDark();
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setThemeState((t) => (t === 'dark' ? 'light' : 'dark'));
  }, []);

  return { theme, toggleTheme };
}

// Reactive dark-mode check — re-renders when theme toggles
const _darkSubs = new Set<() => void>();

function _notifyDark() {
  _darkSubs.forEach((cb) => cb());
}

const _darkStore = {
  subscribe: (cb: () => void) => {
    _darkSubs.add(cb);
    return () => { _darkSubs.delete(cb); };
  },
  getSnapshot: () => document.documentElement.classList.contains('dark'),
};

/** Reactive hook — re-renders components when dark/light mode toggles */
export function useIsDark(): boolean {
  return useSyncExternalStore(_darkStore.subscribe, _darkStore.getSnapshot);
}
