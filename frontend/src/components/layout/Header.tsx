import { NavLink } from 'react-router-dom';
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

const navItems = [
  { to: '/', label: 'Chat' },
  { to: '/data', label: 'Data Tools' },
  { to: '/pipeline', label: 'Pipeline' },
  { to: '/memory', label: 'Memory' },
  { to: '/eureka', label: 'Eureka' },
  { to: '/settings', label: 'Settings' },
];

export function Header({ sidebarOpen, activityOpen, onToggleSidebar, onToggleActivity, theme, onToggleTheme }: Props) {
  return (
    <header data-testid="app-header" className="bg-panel border-b border-border flex items-center justify-between px-4 h-12 shrink-0">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onToggleSidebar}
          className=""
          aria-label="Toggle sidebar"
        >
          {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
        </Button>
        <HelionLogo size={24} className="text-primary" />
        <span className="font-semibold text-lg text-text hidden sm:inline">XHelio</span>

        <nav className="flex items-center gap-1 ml-4 overflow-x-auto scrollbar-none" aria-label="Main navigation">
          {navItems.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `px-3 py-1 rounded-md text-sm transition-colors ${
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
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onToggleActivity}
          aria-label="Toggle activity panel"
        >
          {activityOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
        </Button>
      </div>
    </header>
  );
}
