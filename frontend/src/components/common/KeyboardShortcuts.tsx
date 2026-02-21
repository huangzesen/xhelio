import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const isMac = typeof navigator !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.userAgent);
const mod = isMac ? '⌘' : 'Ctrl';

const shortcuts = [
  {
    group: 'General',
    items: [
      { keys: [`${mod}+K`], label: 'Open command palette' },
      { keys: [`${mod}+B`], label: 'Toggle sidebar' },
      { keys: [`${mod}+\\`], label: 'Toggle activity panel' },
      { keys: [`${mod}+N`], label: 'New chat' },
      { keys: [`${mod}+?`], label: 'Keyboard shortcuts' },
      { keys: ['/'], label: 'Focus chat input' },
    ],
  },
  {
    group: 'Chat Input',
    items: [
      { keys: ['Enter'], label: 'Send message' },
      { keys: ['Shift+Enter'], label: 'New line' },
      { keys: ['↑ / ↓'], label: 'Browse input history' },
      { keys: ['Tab'], label: 'Accept autocomplete' },
      { keys: ['Esc'], label: 'Dismiss autocomplete' },
    ],
  },
  {
    group: 'Slash Commands',
    items: [
      { keys: ['/help'], label: 'Show available commands' },
      { keys: ['/reset'], label: 'Reset session' },
      { keys: ['/branch'], label: 'Fork into new branch' },
      { keys: ['/status'], label: 'Session info' },
      { keys: ['/data'], label: 'List data in memory' },
      { keys: ['/retry'], label: 'Retry failed plan task' },
    ],
  },
];

export function KeyboardShortcuts({ open, onOpenChange }: Props) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 mt-2">
          {shortcuts.map((section) => (
            <div key={section.group}>
              <h3 className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">
                {section.group}
              </h3>
              <div className="space-y-1">
                {section.items.map((item) => (
                  <div
                    key={item.label}
                    className="flex items-center justify-between py-1"
                  >
                    <span className="text-sm text-text">{item.label}</span>
                    <div className="flex items-center gap-1">
                      {item.keys.map((key) => (
                        <kbd
                          key={key}
                          className="px-1.5 py-0.5 text-xs font-mono bg-surface-elevated border border-border rounded text-text-muted"
                        >
                          {key}
                        </kbd>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
