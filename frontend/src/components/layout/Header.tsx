import { NavLink } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, Sun, Moon } from 'lucide-react';
import { HelionLogo } from '../common/HelionLogo';
import { Button } from '@/components/ui/button';

interface Props {
  sidebarOpen: boolean;
  activityOpen: boolean;
  onToggleSidebar: () => void;
  onToggleActivity: () => void;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
}

export function Header({ sidebarOpen, activityOpen, onToggleSidebar, onToggleActivity, theme, onToggleTheme }: Props) {
  const { t } = useTranslation('common');

  const navItems = [
    { to: '/', label: t('nav.chat') },
    { to: '/data', label: t('nav.dataTools') },
    { to: '/pipeline', label: t('nav.pipeline') },
    { to: '/memory', label: t('nav.memory') },
    { to: '/eureka', label: t('nav.eureka') },
    { to: '/settings', label: t('nav.settings') },
  ];

  return (
    <header data-testid="app-header" className="bg-panel border-b border-border flex items-center justify-between px-4 h-12 shrink-0">
      <div className="flex items-center gap-3 min-w-0 flex-1 overflow-hidden">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onToggleSidebar}
          className=""
          aria-label={t('aria.toggleSidebar')}
        >
          {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
        </Button>
        <HelionLogo size={24} className="text-primary" />
        <span className="font-semibold text-lg text-text hidden sm:inline">XHelio</span>

        <nav className="flex items-center gap-1 ml-4 overflow-x-auto scrollbar-none min-w-0" aria-label="Main navigation">
          {navItems.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `px-3 py-1 rounded-md text-sm whitespace-nowrap transition-colors ${
                  isActive
                    ? 'bg-primary/15 text-primary font-medium'
                    : 'text-text-muted hover:text-text hover:bg-hover-bg'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onToggleTheme}
          aria-label={theme === 'dark' ? t('aria.switchToLightMode') : t('aria.switchToDarkMode')}
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onToggleActivity}
          aria-label={t('aria.toggleActivityPanel')}
        >
          {activityOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
        </Button>
      </div>
    </header>
  );
}
