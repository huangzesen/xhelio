import { NavLink, Outlet } from 'react-router-dom';
import { Key, Layers, Search, Settings, Save, Loader2, Check } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

export function SettingsLayout() {
  const { t } = useTranslation(['settings', 'common']);

  const NAV_ITEMS = [
    { to: '/settings/api-keys', icon: Key, label: t('nav.apiKeys') },
    { to: '/settings/models', icon: Layers, label: t('nav.models') },
    { to: '/settings/data', icon: Search, label: t('nav.data') },
    { to: '/settings/advanced', icon: Settings, label: t('nav.advanced') },
  ];
  const { loading, saving, error, saved, sessionSwitched, loadConfig, saveConfig } =
    useSettingsStore();

  useEffect(() => {
    loadConfig();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex-1 overflow-hidden flex flex-col bg-surface">
      {/* Header */}
      <div className="px-6 pt-4 max-w-6xl mx-auto w-full">
        <h1 className="text-xl font-semibold text-text">{t('title')}</h1>

        {error && (
          <div className="mt-3 px-4 py-2 rounded-lg bg-status-error-bg text-status-error-text text-sm">
            {error}
          </div>
        )}

        {saved && !error && (
          <div className="mt-3 px-4 py-2 rounded-lg bg-status-success-bg text-status-success-text text-sm flex items-center gap-2">
            <Check size={16} />
            <span>
              {sessionSwitched ? t('savedWithSession') : t('saved')}
            </span>
          </div>
        )}
      </div>

      {/* Main: nav + content */}
      <div className="flex-1 flex overflow-hidden max-w-6xl mx-auto w-full">
        {/* Desktop vertical nav */}
        <nav className="hidden md:flex flex-col w-44 shrink-0 pr-4 pt-2 border-r border-border">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-r-md text-sm transition-colors ${
                  isActive
                    ? 'bg-primary/10 text-primary border-l-2 border-primary font-medium'
                    : 'text-text-muted hover:text-text hover:bg-panel border-l-2 border-transparent'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Mobile horizontal tabs */}
        <div className="md:hidden flex gap-1 px-4 pb-3 overflow-x-auto">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm whitespace-nowrap transition-colors ${
                  isActive
                    ? 'bg-primary/10 text-primary font-medium'
                    : 'text-text-muted hover:text-text hover:bg-panel'
                }`
              }
            >
              <Icon size={14} />
              {label}
            </NavLink>
          ))}
        </div>

        {/* Content area */}
        <div className="flex-1 overflow-y-auto px-6 pb-24">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={24} className="animate-spin text-text-muted" />
            </div>
          ) : (
            <Outlet />
          )}
        </div>
      </div>

      {/* Sticky save footer */}
      <div className="sticky bottom-0 bg-surface border-t border-border px-6 py-3">
        <div className="max-w-6xl mx-auto flex justify-end">
          <button
            onClick={saveConfig}
            disabled={saving || loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {saving ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Save size={16} />
            )}
            {saving ? t('common:status.saving') : t('common:actions.save')}
          </button>
        </div>
      </div>
    </div>
  );
}
